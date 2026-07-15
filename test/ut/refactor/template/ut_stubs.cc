/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Template UT stub functions.
 */

#include <cstring>
#include <cstdarg>
#include <vector>
#include <string>
#include <set>
#include <cstdint>

#include <hccl/hccl_types.h>
#include <hccl/hccl_rank_graph.h>
#include "test_helpers.h"
#include "hcomm_primitives_dl.h"
#include "exec_timeout_manager.h"
#include "channel_request.h"

namespace ops_hccl {
namespace testing {
std::vector<TemplateHcall> g_tmplRecords;
std::vector<std::vector<uint8_t>> g_tmplCaptured;
bool g_tmplFailNext = false;
int32_t g_tmplFailRet = 0;
std::string g_tmplFailName;
} // namespace testing
} // namespace ops_hccl

using ops_hccl::testing::g_tmplRecords;
using ops_hccl::testing::g_tmplCaptured;
using ops_hccl::testing::g_tmplFailNext;
using ops_hccl::testing::g_tmplFailRet;
using ops_hccl::testing::g_tmplFailName;

// ───────────── log stub ─────────────
bool IsErrorToWarn() { return false; }
bool HcclCheckLogLevel(int, int) { return false; }

extern "C" errno_t memset_s(void *dest, size_t destMax, int c, size_t count)
{
    if (dest == nullptr || count > destMax) { return 1; }
    (void)memset(dest, c, count);
    return 0;
}

// snprintf_s 桩
int snprintf_s(char *dest, size_t destMax, size_t count, const char *format, ...)
{
    (void)destMax; (void)count;
    va_list args;
    va_start(args, format);
    int ret = vsnprintf(dest, destMax, format, args);
    va_end(args);
    return ret;
}

// ───────────── ExecTimeoutManager stub ─────────────
namespace ops_hccl {
ExecTimeoutManager &ExecTimeoutManager::Instance()
{
    static ExecTimeoutManager inst;
    return inst;
}
ExecTimeoutManager::ExecTimeoutManager() : execTimeout_(1000), timeoutSet_(false) {}
ExecTimeoutManager::~ExecTimeoutManager() = default;
void ExecTimeoutManager::SetExecTimeout(u32 t) { execTimeout_ = t; timeoutSet_ = true; }
u32 ExecTimeoutManager::GetExecTimeout() { return execTimeout_.load(); }
} // namespace ops_hccl

// ───────────── HcclAlgorithm 序列化桩 (C++ 链接) ─────────────
namespace ops_hccl {
void HcclAlgorithm::SerializeTo(BinaryStream &) const {}
void HcclAlgorithm::DeserializeFrom(BinaryStream &) {}
void AlgoExecDesc::Serialize(BinaryStream &, const AlgoExecDesc &) {}
AlgoExecDesc AlgoExecDesc::Deserialize(BinaryStream &) { return AlgoExecDesc{}; }
} // namespace ops_hccl

// ───────────── mock 控制 ─────────────
static inline bool shouldTmplFail(const char *name)
{
    if (g_tmplFailNext) { return true; }
    if (!g_tmplFailName.empty() && g_tmplFailName == name) { return true; }
    return false;
}

static inline int32_t tmplRetCode(const char *name)
{
    return shouldTmplFail(name) ? (g_tmplFailRet ? g_tmplFailRet : -1) : 0;
}

static inline void captureTmplBytes(const void *src, uint64_t len)
{
    std::vector<uint8_t> buf(len);
    if (len && src) { memcpy(buf.data(), src, len); }
    g_tmplCaptured.push_back(std::move(buf));
}

// ───────────── HcclRankGraph 桩 ─────────────
extern "C" HcclResult HcclRankGraphGetLayers(HcclComm, uint32_t **netLayers, uint32_t *netLayerNum)
{
    static uint32_t layer = 0;
    *netLayers = &layer;
    *netLayerNum = 1;
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclRankGraphGetLinks(HcclComm, uint32_t, uint32_t, uint32_t, CommLink **links, uint32_t *linkNum)
{
    static CommLink link;
    link.linkAttr.linkProtocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    *links = &link;
    *linkNum = 1;
    return HCCL_SUCCESS;
}

// ───────────── CalcNHRChannelConnect 桩 (C++ 链接, ops_hccl 命名空间) ─────────────
namespace ops_hccl {
HcclResult CalcNHRChannelConnect(u32, u32 rankSize, u32, std::set<u32> &connectRanks)
{
    if (rankSize <= 1) { return HCCL_SUCCESS; }
    for (u32 i = 0; i < rankSize; ++i) { connectRanks.insert(i); }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

// ───────────── Hcomm* 桩 (LocalCopy / NotifyRecord / NotifyWait) ─────────────
extern "C" {

int32_t HcommLocalCopyOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t len)
{
    g_tmplRecords.push_back({"LocalCopy", (unsigned long)thread, 0, 0xFFFFFFFF, dst, src, len});
    captureTmplBytes(src, len);
    if (len && dst && src) { memcpy(dst, src, len); }
    return tmplRetCode("LocalCopy");
}

int32_t HcommThreadNotifyRecordOnThread(ThreadHandle mainThread, ThreadHandle subThread, uint32_t idx)
{
    g_tmplRecords.push_back({"NotifyRecord", (unsigned long)mainThread, (unsigned long)subThread, idx,
                             nullptr, nullptr, 0});
    return tmplRetCode("NotifyRecord");
}

int32_t HcommThreadNotifyWaitOnThread(ThreadHandle mainThread, uint32_t idx, uint32_t /*timeout*/)
{
    g_tmplRecords.push_back({"NotifyWait", (unsigned long)mainThread, 0, idx, nullptr, nullptr, 0});
    return tmplRetCode("NotifyWait");
}

} // extern "C"
