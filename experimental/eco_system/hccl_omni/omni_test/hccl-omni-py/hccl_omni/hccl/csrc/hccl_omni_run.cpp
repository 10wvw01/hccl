// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

// No CANN headers needed at compile time — all HCCL/ACL symbols resolved via dlsym at runtime.
// Type definitions below mirror <hccl.h> + <acl/acl.h> ABI for this extension only.

#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <string>

#include "torch/extension.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"


namespace hccl_ops {

// =========================================================================
// dlsym-resolved HCCL function pointers (consistent with comm_context.cpp)
// =========================================================================

// New HcclOmniRun signature (9 parameters):
//   HcclOmniRun(sendBuf, recvBuf, sendType, recvType,
//               xmlPath, opParam, opParamSize,
//               comm, stream)
// Counts/displs are now read from the opParam blob by the C++ implementation.
using HcclOmniRun_t        = HcclResult (*)(const void*, const void*,
                                           HcclDataType, HcclDataType,
                                           const char*, const void*, uint64_t,
                                           HcclComm, aclrtStream);
using AclrtSynchronizeStream_t = aclError (*)(aclrtStream);

static HcclOmniRun_t             g_HcclOmniRun = nullptr;
static AclrtSynchronizeStream_t  g_AclrtSync   = nullptr;

static void* TryDlopen(const char* path, int flags) { return dlopen(path, flags); }

static void* ResolveLibhccl()
{
    const char* env_path = std::getenv("HCCL_LIBRARY_PATH");
    if (env_path != nullptr) { void* h = TryDlopen(env_path, RTLD_LAZY); if (h) return h; }

    void* h = TryDlopen("libhccl.so", RTLD_NOLOAD | RTLD_LAZY);  if (h) return h;
    h = TryDlopen("libhccl.so.9", RTLD_NOLOAD | RTLD_LAZY);      if (h) return h;

    const char* cann_paths[] = {
        std::getenv("ASCEND_HOME_PATH"), std::getenv("ASCEND_CANN_PACKAGE_PATH")};
    const char* sonames[] = {"libhccl.so", "libhccl.so.9"};
    for (auto root : cann_paths) {
        if (!root) continue;
        for (auto soname : sonames) {
            std::string full = std::string(root) + "/lib64/" + soname;
            h = TryDlopen(full.c_str(), RTLD_LAZY); if (h) return h;
            full = std::string(root) + "/lib/" + soname;
            h = TryDlopen(full.c_str(), RTLD_LAZY); if (h) return h;
        }
    }
    return TryDlopen("libhccl.so", RTLD_LAZY);
}

static void InitHcclFunctions()
{
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    void* handle = ResolveLibhccl();
    TORCH_CHECK(handle != nullptr,
        "Cannot find libhccl.so. Ensure ASCEND_HOME_PATH is set, "
        "or torch.distributed is initialized, or HCCL_LIBRARY_PATH points to libhccl.so.");

    g_HcclOmniRun = reinterpret_cast<HcclOmniRun_t>(dlsym(handle, "HcclOmniRun"));
    TORCH_CHECK(g_HcclOmniRun != nullptr,
        "Failed to resolve HcclOmniRun from libhccl.so: ", dlerror());

    // aclrtSynchronizeStream is in libacl_rt.so, but available via global symbol table
    g_AclrtSync = reinterpret_cast<AclrtSynchronizeStream_t>(dlsym(RTLD_DEFAULT, "aclrtSynchronizeStream"));
    TORCH_CHECK(g_AclrtSync != nullptr,
        "Failed to resolve aclrtSynchronizeStream: ", dlerror());
}

// =========================================================================
// PyTorch dtype -> HcclDataType lookup table
// =========================================================================

static HcclDataType torchDtypeToHcclDataType(at::ScalarType scalar_type) {
    switch (scalar_type) {
        case at::kByte:     return HCCL_DATA_TYPE_INT8;
        case at::kChar:     return HCCL_DATA_TYPE_INT8;
        case at::kShort:    return HCCL_DATA_TYPE_INT16;
        case at::kInt:      return HCCL_DATA_TYPE_INT32;
        case at::kLong:     return HCCL_DATA_TYPE_INT64;
        case at::kHalf:     return HCCL_DATA_TYPE_FP16;
        case at::kBFloat16: return HCCL_DATA_TYPE_BFP16;
        case at::kFloat:    return HCCL_DATA_TYPE_FP32;
        case at::kDouble:   return HCCL_DATA_TYPE_FP64;
        case at::kBool:     return HCCL_DATA_TYPE_UINT8;
        default:
            TORCH_CHECK(false, "[hccl_omni_run] Unsupported dtype: ", scalar_type);
            return HCCL_DATA_TYPE_INT8;
    }
}

// =========================================================================
// HcclResult check — 0 = success, non-zero = error (raw int propagated)
// =========================================================================

static void checkHcclResult(int ret, const char* op_name) {
    TORCH_CHECK(ret == 0, "[", op_name, "] HCCL error, ret=", ret);
}

// =========================================================================
// SAFE-01: Comprehensive input validation
// =========================================================================

// Minimum op_param blob size: 4104 bytes (8-byte header + 4 × 1024-byte int64[128] arrays)
static constexpr uint64_t OP_PARAM_BLOB_MIN_SIZE = 4104;

static void validateInputs(
    const at::Tensor& send_buf,
    const at::Tensor& recv_buf,
    int64_t send_type, int64_t recv_type,
    int64_t comm_handle, int64_t stream_handle,
    int64_t world_size,
    const c10::optional<at::Tensor>& op_param)
{
    const char* fn = "hccl_omni_run";

    // Data buffers (send_buf, recv_buf) must be on NPU device
    auto check_device = [&](const at::Tensor& t, const char* name) {
        TORCH_CHECK(t.device().is_privateuseone(),
            "[", fn, "] ", name, " must be on NPU device, got device=", t.device());
    };
    check_device(send_buf, "send_buf");
    check_device(recv_buf, "recv_buf");

    // Contiguity
    auto check_contiguous = [&](const at::Tensor& t, const char* name) {
        TORCH_CHECK(t.is_contiguous(), "[", fn, "] ", name,
            " must be contiguous. Use .contiguous() before calling.");
    };
    check_contiguous(send_buf, "send_buf");
    check_contiguous(recv_buf, "recv_buf");

    // Non-empty
    TORCH_CHECK(send_buf.numel() > 0, "[", fn, "] send_buf must be non-empty");
    TORCH_CHECK(recv_buf.numel() > 0, "[", fn, "] recv_buf must be non-empty");

    // In-place not supported
    TORCH_CHECK(send_buf.const_data_ptr() != recv_buf.const_data_ptr(),
        "[", fn, "] send_buf and recv_buf must be different tensors "
        "(in-place communication is not supported).");

    // op_param validation: must be a uint8 tensor on CPU with >= 4104 bytes
    if (op_param.has_value() && op_param->defined()) {
        TORCH_CHECK(op_param->device().is_cpu(),
            "[", fn, "] op_param must be on CPU, got device=", op_param->device());
        TORCH_CHECK(op_param->is_contiguous(),
            "[", fn, "] op_param must be contiguous");
        TORCH_CHECK(op_param->scalar_type() == at::kByte,
            "[", fn, "] op_param must be uint8 (torch.uint8), got ", op_param->scalar_type());
        TORCH_CHECK(static_cast<uint64_t>(op_param->nbytes()) >= OP_PARAM_BLOB_MIN_SIZE,
            "[", fn, "] op_param must be >= ", OP_PARAM_BLOB_MIN_SIZE,
            " bytes (got ", op_param->nbytes(), "). "
            "The blob must contain the full OpParamBlob layout: "
            "8-byte header + 4 x int64[128] arrays.");
    } else {
        // op_param is now required — counts/displs are read from it
        TORCH_CHECK(false,
            "[", fn, "] op_param is required. "
            "Counts/displs are read from the op_param blob by the C++ implementation.");
    }

    // Handle non-null
    TORCH_CHECK(comm_handle != 0, "[", fn, "] comm_handle must be non-zero");

    // send_buf same device as recv_buf
    if (send_buf.device() != recv_buf.device()) {
        TORCH_CHECK(false,
            "[", fn, "] send_buf and recv_buf must be on the same device, "
            "got ", send_buf.device(), " vs ", recv_buf.device(), ".");
    }
}

// =========================================================================
// SAFE-03: Context validity helper
// =========================================================================

bool is_valid_comm_handle(int64_t comm_handle) { return comm_handle != 0; }

// =========================================================================
// Main operator: torch.ops.npu.hccl_omni_run()
// =========================================================================

at::Tensor hccl_omni_run(
    const at::Tensor& send_buf,
    const at::Tensor& recv_buf,
    int64_t send_type,
    int64_t recv_type,
    const std::string& xml_path,
    const c10::optional<at::Tensor>& op_param,
    int64_t comm_handle,
    bool synchronize,
    int64_t world_size)
{
    InitHcclFunctions();

    int64_t stream_handle = reinterpret_cast<int64_t>(c10_npu::getCurrentNPUStream().stream());

    validateInputs(send_buf, recv_buf, send_type, recv_type,
                   comm_handle, stream_handle, world_size, op_param);

    const void* sendBuf = send_buf.const_data_ptr();
    const void* recvBuf = recv_buf.const_data_ptr();

    HcclDataType hcclSendType = torchDtypeToHcclDataType(
        (send_type >= 0) ? static_cast<at::ScalarType>(send_type) : send_buf.scalar_type());
    HcclDataType hcclRecvType = torchDtypeToHcclDataType(
        (recv_type >= 0) ? static_cast<at::ScalarType>(recv_type) : recv_buf.scalar_type());

    HcclComm      comm   = reinterpret_cast<HcclComm>(comm_handle);
    aclrtStream   stream = reinterpret_cast<aclrtStream>(stream_handle);

    // --- Extract op_param data ---
    const void* opParamData = nullptr;
    uint64_t    opParamSize = 0;
    if (op_param.has_value() && op_param->defined()) {
        opParamData = op_param->const_data_ptr();
        opParamSize = static_cast<uint64_t>(op_param->nbytes());
    }

    // --- Call HCCL via dlsym-resolved function pointer (new 9-param signature) ---
    HcclResult ret = g_HcclOmniRun(
        sendBuf, recvBuf,
        hcclSendType, hcclRecvType,
        xml_path.empty() ? nullptr : xml_path.c_str(),
        opParamData, opParamSize,
        comm, stream);

    checkHcclResult(ret, "hccl_omni_run");

    if (synchronize) {
        aclError sync_ret = g_AclrtSync(stream);
        TORCH_CHECK(sync_ret == ACL_SUCCESS,
            "[hccl_omni_run] aclrtSynchronizeStream failed, ret=", sync_ret);
    }

    return recv_buf;
}

} // namespace hccl_ops

// =========================================================================
// PyTorch extension module registration
// =========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hccl_omni_run", &hccl_ops::hccl_omni_run,
        "HCCL OmniRun custom collective communication operator.\n"
        "All HCCL symbols resolved via dlsym at runtime — no link-time dependency on libhccl.so.\n"
        "Counts/displs are read from the op_param blob by the C++ implementation.\n"
        "Parameter order matches C++ HcclOmniRun API.",
        pybind11::arg("send_buf"),
        pybind11::arg("recv_buf"),
        pybind11::arg("send_type"),
        pybind11::arg("recv_type"),
        pybind11::arg("xml_path"),
        pybind11::arg("op_param") = pybind11::none(),
        pybind11::arg("comm_handle"),
        pybind11::arg("synchronize") = false,
        pybind11::arg("world_size") = -1);

    m.def("is_valid_comm_handle", &hccl_ops::is_valid_comm_handle,
        "Check if a comm_handle is non-zero (SAFE-03 lifecycle safety).",
        pybind11::arg("comm_handle"));
}
