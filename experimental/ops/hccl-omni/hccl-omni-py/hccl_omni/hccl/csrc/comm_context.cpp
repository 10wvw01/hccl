// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

/*!
 * \file comm_context.cpp
 * \brief Simplified CommContextManager for HcclOmniRun.
 *
 * Unlike mega_moe's 550-line CommContextManager (which manages KFC direct-IPC
 * engine contexts, channel descriptors, and IPC buffer exchange), this version
 * is ~150 lines and only does what HcclOmniRun needs:
 *   - Resolve HcclComm handle from group name (via HcomGetCommHandleByGroup)
 *   - Detect backend mode from SoC name (KFC vs Channel)
 *   - Wrap comm + stream handles in a context struct
 *
 * HcclOmniRun delegates to standard HCCL AlltoAllV, which manages its own
 * engine contexts, channels, and buffers internally.
 */

// No CANN headers needed at compile time — all HCCL symbols resolved via dlsym at runtime.


#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

#include <dlfcn.h>

#include "torch/extension.h"
#include "acl/acl_rt.h"
#include "hccl/hccl_types.h"

namespace hccl_ops {

// =========================================================================
// Context struct — fits in a 1D int32 tensor on NPU
// =========================================================================

struct OmniRunContext {
    uint64_t comm_handle;    // HcclComm cast to uint64
    uint64_t stream_handle;  // aclrtStream cast to uint64
    int64_t  world_size;
    int64_t  rank_id;
    char     backend[32];    // "kfc" or "channel", null-terminated
};

static_assert(sizeof(OmniRunContext) % sizeof(int32_t) == 0,
              "OmniRunContext size must be int32-aligned for tensor copy");

static int64_t context_tensor_size()
{
    return static_cast<int64_t>(sizeof(OmniRunContext) / sizeof(int32_t));
}

// =========================================================================
// Backend detection via SoC name
// =========================================================================

enum class BackendMode : uint8_t { UNINITIALIZED, KFC, CHANNEL };

static const char* GetSocName()
{
    static const char* socName = aclrtGetSocName();
    return socName;
}

static BackendMode ResolveBackend(const pybind11::object& backend)
{
    // String mode: direct "kfc" or "channel"
    if (pybind11::isinstance<pybind11::str>(backend)) {
        auto mode = backend.cast<std::string>();
        if (mode == "channel") return BackendMode::CHANNEL;
        if (mode == "kfc")     return BackendMode::KFC;
        TORCH_CHECK(false, "backend string must be 'kfc' or 'channel', got '", mode, "'");
    }

    // Dict mode: match SoC name against keys → resolve value
    if (pybind11::isinstance<pybind11::dict>(backend)) {
        auto dict = backend.cast<pybind11::dict>();
        TORCH_CHECK(dict.size() > 0, "backend dict must not be empty");

        const char* socName = GetSocName();
        TORCH_CHECK(socName != nullptr, "aclrtGetSocName returned nullptr");

        for (auto item : dict) {
            std::string key   = pybind11::cast<std::string>(item.first);
            std::string value = pybind11::cast<std::string>(item.second);
            if (value == "channel" || value == "kfc") {
                if (std::strstr(socName, key.c_str()) != nullptr) {
                    return value == "channel" ? BackendMode::CHANNEL : BackendMode::KFC;
                }
            } else {
                TORCH_CHECK(false, "backend dict value must be 'kfc' or 'channel', got '", value, "'");
            }
        }
        TORCH_CHECK(false, "No matching SoC name found for '", socName, "' in backend dict");
    }

    TORCH_CHECK(false, "backend must be a string ('kfc'/'channel') or a dict");
}

static const char* BackendModeName(BackendMode mode)
{
    switch (mode) {
        case BackendMode::KFC:     return "kfc";
        case BackendMode::CHANNEL: return "channel";
        default:                   return "unknown";
    }
}

// =========================================================================
// HcclComm resolution from group name (dlsym from libhccl.so)
// =========================================================================

using HcomGetCommHandleByGroup_t = HcclResult (*)(const char*, HcclComm*);

static HcomGetCommHandleByGroup_t g_hcomGetCommHandleByGroup = nullptr;

static void* TryDlopen(const char* path, int flags)
{
    void* h = dlopen(path, flags);
    return h;
}

static void* ResolveLibhccl()
{
    // Strategy: try RTLD_NOLOAD first (libhccl.so already loaded by torch.distributed
    // or torch_npu), then try full paths from CANN env vars, finally try loading by
    // soname. This covers three scenarios:
    //   1. torch.distributed pre-loaded libhccl.so     → NOLOAD hits immediately
    //   2. CANN installed, env vars set                 → full path from ASCEND_HOME_PATH
    //   3. CANN installed, ldconfig aware               → soname via RTLD_LAZY
    //   4. User overrides with HCCL_LIBRARY_PATH         → explicit path

    const char* env_path = std::getenv("HCCL_LIBRARY_PATH");
    if (env_path != nullptr) {
        void* h = TryDlopen(env_path, RTLD_LAZY);
        if (h != nullptr) return h;
    }

    // 1. Already loaded? (torch.distributed / torch_npu)
    void* handle = TryDlopen("libhccl.so", RTLD_NOLOAD | RTLD_LAZY);
    if (handle != nullptr) return handle;
    handle = TryDlopen("libhccl.so.9", RTLD_NOLOAD | RTLD_LAZY);
    if (handle != nullptr) return handle;

    // 2. Full path from CANN env vars
    const char* cann_paths[] = {
        std::getenv("ASCEND_HOME_PATH"),
        std::getenv("ASCEND_CANN_PACKAGE_PATH"),
    };
    const char* sonames[] = {"libhccl.so", "libhccl.so.9"};
    for (auto cann_root : cann_paths) {
        if (cann_root == nullptr) continue;
        for (auto soname : sonames) {
            std::string full = std::string(cann_root) + "/lib64/" + soname;
            handle = TryDlopen(full.c_str(), RTLD_LAZY);
            if (handle != nullptr) return handle;
            // Also try <cann_root>/lib/ (some install layouts)
            full = std::string(cann_root) + "/lib/" + soname;
            handle = TryDlopen(full.c_str(), RTLD_LAZY);
            if (handle != nullptr) return handle;
        }
    }

    // 3. Soname via ld.so search (last resort)
    handle = TryDlopen("libhccl.so", RTLD_LAZY);
    if (handle != nullptr) return handle;

    return nullptr;
}

static void InitHcclFunctions()
{
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    void* handle = ResolveLibhccl();
    TORCH_CHECK(handle != nullptr,
        "Cannot find libhccl.so. Ensure one of:\n"
        "  - torch.distributed is initialized (pre-loads libhccl.so)\n"
        "  - ASCEND_HOME_PATH or ASCEND_CANN_PACKAGE_PATH is set\n"
        "  - HCCL_LIBRARY_PATH points to libhccl.so\n"
        "  - libhccl.so is on LD_LIBRARY_PATH");

    g_hcomGetCommHandleByGroup = reinterpret_cast<HcomGetCommHandleByGroup_t>(
        dlsym(handle, "HcomGetCommHandleByGroup"));
    TORCH_CHECK(g_hcomGetCommHandleByGroup != nullptr,
                "Failed to resolve HcomGetCommHandleByGroup from libhccl.so: ", dlerror());
}

// =========================================================================
// CommContextManager
// =========================================================================

class CommContextManager {
public:
    CommContextManager(const std::string& group_name,
                       int64_t world_size,
                       int64_t rank_id,
                       const pybind11::object& backend = pybind11::str("channel"))
        : group_name_(group_name)
        , world_size_(world_size)
        , rank_id_(rank_id)
        , backend_(backend)
        , mode_(BackendMode::UNINITIALIZED)
        , comm_handle_(nullptr)
    {}

    // ---- Context creation (from group name) ----
    at::Tensor create_context()
    {
        EnsureResolved();
        ResolveComm();

        OmniRunContext ctx = {};
        ctx.comm_handle   = reinterpret_cast<uint64_t>(comm_handle_);
        ctx.stream_handle = 0;  // filled by caller via set_stream()
        ctx.world_size    = world_size_;
        ctx.rank_id       = rank_id_;
        std::strncpy(ctx.backend, BackendModeName(mode_), sizeof(ctx.backend) - 1);

        at::Tensor context = at::empty({context_tensor_size()},
            at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1));

        at::Tensor host = at::from_blob(&ctx, {context_tensor_size()}, at::kInt);
        context.copy_(host);

        return context;
    }

    // ---- Direct HcclComm handle path (COMM-03) ----
    static CommContextManager from_handle(int64_t comm_handle, int64_t world_size, int64_t rank_id,
                                          const pybind11::object& backend = pybind11::str("channel"))
    {
        CommContextManager mgr("", world_size, rank_id, backend);
        mgr.mode_ = mgr.ResolveBackendInternal(backend);
        mgr.comm_handle_ = reinterpret_cast<HcclComm>(comm_handle);
        return mgr;
    }

    // ---- Accessors ----
    int64_t get_comm_handle() const { return reinterpret_cast<int64_t>(comm_handle_); }
    int64_t get_rank_id()     const { return rank_id_; }
    int64_t get_world_size()  const { return world_size_; }
    std::string get_backend() const { return BackendModeName(mode_); }
    std::string get_group_name() const { return group_name_; }

private:
    void EnsureResolved()
    {
        if (mode_ == BackendMode::UNINITIALIZED) {
            mode_ = ResolveBackendInternal(backend_);
        }
    }

    BackendMode ResolveBackendInternal(const pybind11::object& backend)
    {
        return ResolveBackend(backend);
    }

    void ResolveComm()
    {
        if (comm_handle_ != nullptr) return;  // already resolved

        InitHcclFunctions();

        HcclResult ret = g_hcomGetCommHandleByGroup(group_name_.c_str(), &comm_handle_);
        TORCH_CHECK(ret == 0,
                    "Failed to get HcclComm handle for group '", group_name_, "', ret=", static_cast<int>(ret));

        TORCH_CHECK(static_cast<uint32_t>(world_size_) >= rank_id_ + 1,
                    "world_size (", world_size_, ") inconsistent with rank_id (", rank_id_, ")");
    }

    std::string        group_name_;
    int64_t            world_size_;
    pybind11::object   backend_;
    BackendMode        mode_;
    HcclComm           comm_handle_;
    int64_t            rank_id_;
};

} // namespace hccl_ops

// =========================================================================
// Python module registration
// =========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<hccl_ops::CommContextManager>(m, "CommContextManager")
        .def(pybind11::init<const std::string&, int64_t, int64_t, const pybind11::object&>(),
             pybind11::arg("group_name"),
             pybind11::arg("world_size"),
             pybind11::arg("rank_id"),
             pybind11::arg("backend") = std::string("channel"))
        .def_static("from_handle", &hccl_ops::CommContextManager::from_handle,
             pybind11::arg("comm_handle"),
             pybind11::arg("world_size"),
             pybind11::arg("rank_id"),
             pybind11::arg("backend") = std::string("channel"))
        .def("create_context", &hccl_ops::CommContextManager::create_context)
        .def_property_readonly("comm_handle", &hccl_ops::CommContextManager::get_comm_handle)
        .def_property_readonly("rank_id", &hccl_ops::CommContextManager::get_rank_id)
        .def_property_readonly("world_size", &hccl_ops::CommContextManager::get_world_size)
        .def_property_readonly("backend", &hccl_ops::CommContextManager::get_backend)
        .def_property_readonly("group_name", &hccl_ops::CommContextManager::get_group_name);
}
