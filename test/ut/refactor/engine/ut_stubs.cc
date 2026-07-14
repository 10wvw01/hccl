/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * AICPU Engine Send UT stub functions.
 * 捕获 Hcomm* 搬移原语的调用序列与传输字节内容, 供 aicpu_engine_test.cc 断言。
 */

#include <cstring>
#include <iostream>
#include <vector>

#include "test_helpers.h"
#include "exec_timeout_manager.h"
#include "hcomm_primitives_dl.h"   // ThreadHandle / ChannelHandle / HcommDataType / HcommReduceOp

namespace ops_hccl {
namespace testing {
std::vector<Hcall> g_records;
std::vector<std::vector<uint8_t>> g_captured;
bool g_failNext = false;
int32_t g_failRet = 0;
std::string g_failName;
} // namespace testing
} // namespace ops_hccl

using ops_hccl::testing::g_records;
using ops_hccl::testing::g_captured;
using ops_hccl::testing::g_failNext;
using ops_hccl::testing::g_failRet;
using ops_hccl::testing::g_failName;
using ops_hccl::testing::Hcall;

// ───────────── log stub (src/common/log.cc 依赖) ─────────────
// IsErrorToWarn / HcclCheckLogLevel 为 C++ 链接, 不可用 extern "C"; 返回类型 bool
bool IsErrorToWarn() { return false; }
bool HcclCheckLogLevel(int, int) { return false; }
extern "C" errno_t memset_s(void *dest, size_t destMax, int c, size_t count)
{
    if (dest == nullptr || count > destMax) { return 1; }
    (void)memset(dest, c, count);
    return 0;
}

// ───────────── ExecTimeoutManager (提供单例实现, 避免链接 exec_timeout_manager.cc) ─────────────
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

// ───────────── Send 路径不涉及的符号 (仅满足链接) ─────────────
HcclResult LoadAICPUKernel() { return HCCL_SUCCESS; }
HcclResult HcclLaunchAicpuKernel(const OpParam &, AlgResourceCtxSerializable &)
{
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

// ───────────── mock 控制 ─────────────
static inline bool shouldFail(const char *name)
{
    if (g_failNext) { return true; }
    if (!g_failName.empty() && g_failName == name) { return true; }
    return false;
}

static inline int32_t retCode(const char *name)
{
    return shouldFail(name) ? (g_failRet ? g_failRet : -1) : 0;
}

static inline void captureBytes(const void *src, uint64_t len)
{
    std::vector<uint8_t> buf(len);
    if (len && src) { memcpy(buf.data(), src, len); }
    g_captured.push_back(std::move(buf));
}

// ───────────── Hcomm* 搬移/同步原语 stub (extern "C", 与 aicpu_engine.cc 调用匹配) ─────────────
extern "C" {

int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t idx)
{
    g_records.push_back({"NotifyRecord", (unsigned long)thread, (unsigned long)channel, idx,
                         nullptr, nullptr, 0, 0, 0});
    return retCode("NotifyRecord");
}

int32_t HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t idx,
                                       uint32_t /*timeout*/)
{
    g_records.push_back({"NotifyWait", (unsigned long)thread, (unsigned long)channel, idx,
                         nullptr, nullptr, 0, 0, 0});
    return retCode("NotifyWait");
}

int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
                           uint64_t len)
{
    g_records.push_back({"Write", (unsigned long)thread, (unsigned long)channel, 0xFFFFFFFF,
                         dst, src, len, 0, 0});
    captureBytes(src, len);
    return retCode("Write");
}

int32_t HcommWriteReduceOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
                                 uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    int dt = static_cast<int>(dataType);
    int op = static_cast<int>(reduceOp);
    g_records.push_back({"WriteReduce", (unsigned long)thread, (unsigned long)channel, 0xFFFFFFFF,
                         dst, src, count, dt, op});
    captureBytes(src, count * ops_hccl::testing::DataTypeSize(static_cast<HcclDataType>(dt)));
    return retCode("WriteReduce");
}

int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
                          uint64_t len)
{
    g_records.push_back({"Read", (unsigned long)thread, (unsigned long)channel, 0xFFFFFFFF,
                         dst, src, len, 0, 0});
    captureBytes(src, len);
    return retCode("Read");
}

int32_t HcommReadReduceOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
                                uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    int dt = static_cast<int>(dataType);
    int op = static_cast<int>(reduceOp);
    g_records.push_back({"ReadReduce", (unsigned long)thread, (unsigned long)channel, 0xFFFFFFFF,
                         dst, src, count, dt, op});
    captureBytes(src, count * ops_hccl::testing::DataTypeSize(static_cast<HcclDataType>(dt)));
    return retCode("ReadReduce");
}

// CreateRes 调用 (C 链接符号), UT 不测 CreateRes, 仅满足链接
// 类型为全局 typedef (hccl_res.h), 不可加 ops_hccl:: 限定
HcclResult HcclChannelAcquire(HcclComm, CommEngine,
                              const HcclChannelDesc *, uint32_t,
                              ChannelHandle *)
{
    return HCCL_SUCCESS;
}

// CreateRes 调用 (C 链接符号), 获取 CCL buffer, 仅满足链接
HcclResult HcclGetHcclBuffer(HcclComm, void **buffer, uint64_t *size)
{
    if (buffer) { *buffer = nullptr; }
    if (size) { *size = 0; }
    return HCCL_SUCCESS;
}

} // extern "C"
