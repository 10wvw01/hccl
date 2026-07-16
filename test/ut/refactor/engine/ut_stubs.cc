/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * AICPU Engine Send UT stub functions.
 * 捕获 Hcomm* 搬移原语的调用序列与传输字节内容, 供 aicpu_engine_test.cc 断言。
 */

#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <iostream>
#include <vector>

#include "test_helpers.h"
#include "exec_timeout_manager.h"
#include "hcomm_primitives_dl.h"   // ThreadHandle / ChannelHandle / HcommDataType / HcommReduceOp
#include "hccl_res_dl.h"
#include "hcomm_diag_dl.h"
#include "hcomm_device_profiling_dl.h"
#include "hccl_device_comm_dl.h"
#include "dlhcomm_function.h"
#include "load_kernel.h"
#include "hccl_algorithm.h"
#include "ops_executor.h"
#include "binary_stream.h"
#include "hccl_rank_graph_dl.h"    // EndpointDesc / EndpointAttr

// hcomm_host_profiling_dl.h 与 hcomm_device_profiling_dl.h 冲突（重复定义 HcomProInfoTmp），
// 此处仅前向声明 ut_stubs.cc 实际用到的函数。
extern "C" {
uint64_t HcommGetProfilingSysCycleTime();
HcclResult HcclReportAicpuKernel(HcclComm, uint64_t, char*);
}

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

// CreateRes 调用 alg.SerializeTo，UT Send 测试不走此路径，仅满足链接
void HcclAlgorithm::SerializeTo(BinaryStream &) const {}
void HcclAlgorithm::DeserializeFrom(BinaryStream &) {}
void AlgoExecDesc::Serialize(BinaryStream &, const AlgoExecDesc &) {}
AlgoExecDesc AlgoExecDesc::Deserialize(BinaryStream &) { return AlgoExecDesc{}; }

// HcclAlgorithm::GetExecutor / OpsExecutor stubs（kernel_launch.cc HcclLaunchAicpuKernel 路径引用，UT 不测试该路径）
std::unique_ptr<OpsExecutor> HcclAlgorithm::GetExecutor(OpParam &)
{
    return nullptr;
}
OpsExecutor::OpsExecutor(HcclAlgorithm &, OpParam &) {}
OpsExecutor::~OpsExecutor() = default;
HcclResult OpsExecutor::Orchestrate(AlgResourceCtxSerializable &) { return HCCL_SUCCESS; }
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

// HcclLaunchAicpuKernel 由 kernel_launch.cc 提供真实实现，UT 仅测试简单函数。

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

// CreateRes 调用 (C 链接符号), 创建线程, 仅满足链接
HcclResult HcclThreadAcquire(HcclComm, CommEngine, uint32_t, uint32_t, ThreadHandle *threads)
{
    if (threads) { *threads = 0; }
    return HCCL_SUCCESS;
}

HcclResult HcclThreadAcquireWithStream(HcclComm, CommEngine, void *, uint32_t, ThreadHandle *thread)
{
    if (thread) { *thread = 0; }
    return HCCL_SUCCESS;
}

// CreateRes 调用 (C 链接符号), 创建引擎上下文, 仅满足链接
HcclResult HcclEngineCtxCreate(HcclComm, const char *, CommEngine, uint64_t size, void **ctx)
{
    if (ctx) { *ctx = (size > 0) ? calloc(1, size) : nullptr; }
    return HCCL_SUCCESS;
}

// CreateRes 调用 (C 链接符号), 获取远端 CCL buffer, 仅满足链接
HcclResult HcclChannelGetHcclBuffer(HcclComm, ChannelHandle, void **buffer, uint64_t *size)
{
    if (buffer) { *buffer = nullptr; }
    if (size) { *size = 0; }
    return HCCL_SUCCESS;
}

// snprintf_s 桩（securec 库符号, UT 环境可能缺失）
int snprintf_s(char *dest, size_t destMax, size_t count, const char *format, ...)
{
    (void)destMax; (void)count;
    va_list args;
    va_start(args, format);
    int ret = vsnprintf(dest, destMax, format, args);
    va_end(args);
    return ret;
}

// strncpy_s 桩（securec 库符号, UT 环境可能缺失）
#ifndef EOK
#define EOK 0
#endif
#ifndef EINVAL
#define EINVAL 22
#endif
errno_t strncpy_s(char *strDest, size_t destMax, const char *strSrc, size_t count)
{
    if (strDest == nullptr || strSrc == nullptr || destMax == 0) { return EINVAL; }
    size_t copyLen = (count < destMax) ? count : destMax - 1;
    (void)memcpy(strDest, strSrc, copyLen);
    strDest[copyLen] = '\0';
    return EOK;
}

// ───────────── LaunchKernel 路径 stub ─────────────

// HcclTaskRegister (dlsym weak, UT 环境无 libHcommHandle)
int32_t HcclTaskRegister(HcclComm, const char*, Callback)
{
    return 0;
}

// HcclThreadExportToCommEngine (dlsym weak, UT 环境无 libHcommHandle)
HcclResult HcclThreadExportToCommEngine(HcclComm, uint32_t, const ThreadHandle*, CommEngine, ThreadHandle* exportedThreads)
{
    if (exportedThreads) { *exportedThreads = 0; }
    return HCCL_SUCCESS;
}

// HcommIsSupportHcclThreadExportToCommEngine (auto-generated by DECL_SUPPORT_FLAG)
bool HcommIsSupportHcclThreadExportToCommEngine(void)
{
    return false;
}

// HcommThreadNotifyRecordOnThread (dlsym weak)
int32_t HcommThreadNotifyRecordOnThread(ThreadHandle, ThreadHandle, uint32_t)
{
    return 0;
}

// HcommThreadNotifyWaitOnThread (dlsym weak)
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle, uint32_t, uint32_t)
{
    return 0;
}

// HcommThreadNotifyWaitOnThreadWithDefaultTimeout (dlsym weak)
int32_t HcommThreadNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle, uint32_t)
{
    return 0;
}

// HcommSetNotifyWaitTimeOut (dlsym weak)
int32_t HcommSetNotifyWaitTimeOut(uint32_t)
{
    return 0;
}

// HcommIsSupportHcommSetNotifyWaitTimeOut (auto-generated by DECL_SUPPORT_FLAG)
bool HcommIsSupportHcommSetNotifyWaitTimeOut(void)
{
    return false;
}

// HcommIsSupportHcommThreadNotifyWaitOnThreadWithDefaultTimeout (auto-generated by DECL_SUPPORT_FLAG)
bool HcommIsSupportHcommThreadNotifyWaitOnThreadWithDefaultTimeout(void)
{
    return false;
}

// HcommGetProfilingSysCycleTime (dlsym weak)
uint64_t HcommGetProfilingSysCycleTime()
{
    return 0;
}

// HcclReportAicpuKernel (dlsym weak)
HcclResult HcclReportAicpuKernel(HcclComm, uint64_t, char*)
{
    return HCCL_SUCCESS;
}

// ACL runtime stubs
aclError aclrtBinaryGetFunction(aclrtBinHandle, const char*, aclrtFuncHandle*)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInit(aclrtFuncHandle, aclrtArgsHandle*)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle, void*, size_t, aclrtParamHandle*)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle, uint32_t, aclrtStream,
    aclrtLaunchKernelCfg*, aclrtArgsHandle, void*)
{
    return ACL_SUCCESS;
}

// ───────────── kernel_launch.cc 路径 stub (extern "C") ─────────────

// HcommAcquireComm / HcommReleaseComm (hcomm_primitives.h)
int32_t HcommAcquireComm(const char *) { return HCCL_SUCCESS; }
int32_t HcommReleaseComm(const char *) { return HCCL_SUCCESS; }

// HcommBatchModeStart / HcommBatchModeEnd (hcomm_primitives.h)
int32_t HcommBatchModeStart(const char *) { return HCCL_SUCCESS; }
int32_t HcommBatchModeEnd(const char *) { return HCCL_SUCCESS; }

// HcommRegOpInfo / HcommIsSupportHcommRegOpInfo (hcomm_diag_dl.h)
HcclResult HcommRegOpInfo(const char *, void *, size_t) { return HCCL_SUCCESS; }
bool HcommIsSupportHcommRegOpInfo(void) { return false; }

// HcommRegOpTaskException / HcommIsSupportHcommRegOpTaskException (hcomm_diag_dl.h)
HcclResult HcommRegOpTaskException(const char *, HcommGetOpInfoCallback) { return HCCL_SUCCESS; }
bool HcommIsSupportHcommRegOpTaskException(void) { return false; }

// HcclCommGetStatus / HcommIsSupportHcclCommGetStatus (hccl_device_comm_dl.h)
HcclResult HcclCommGetStatus(const char *, HcclCommStatus *status)
{
    if (status) { *status = HCCL_COMM_STATUS_READY; }
    return HCCL_SUCCESS;
}
bool HcommIsSupportHcclCommGetStatus(void) { return false; }

// HcclThreadResAcquireTimeOut (hcomm_primitives_dl.h)
HcclResult HcclThreadResAcquireTimeOut(uint32_t) { return HCCL_SUCCESS; }

// HcommThreadResAcquireTimeOut / HcommIsSupportHcommThreadResAcquireTimeOut (hcomm_primitives_dl.h via DEFINE_WEAK_FUNC)
int32_t HcommThreadResAcquireTimeOut(uint32_t) { return 0; }
bool HcommIsSupportHcommThreadResAcquireTimeOut(void) { return false; }

// HcommProfilingReportKernelStartTask / HcommProfilingReportKernelEndTask (hcomm_device_profiling_dl.h)
HcclResult HcommProfilingReportKernelStartTask(uint64_t, const char *) { return HCCL_SUCCESS; }
HcclResult HcommProfilingReportKernelEndTask(uint64_t, const char *) { return HCCL_SUCCESS; }

// HcclDfxRegOpInfoByCommId (hcomm_device_profiling_dl.h)
HcclResult HcclDfxRegOpInfoByCommId(char *, void *) { return HCCL_SUCCESS; }

// HcommProfilingReportDeviceOp (hcomm_device_profiling_dl.h)
HcclResult HcommProfilingReportDeviceOp(const char *) { return HCCL_SUCCESS; }

} // extern "C"

// ───────────── C++ 链接 stub ─────────────

// IsHcommDefaultTimeoutSupported / HcclSetNotifyWaitTimeOut / HcclThreadNotifyWaitOnThreadDefault
// 这三个是 HCCL 内部 C++ wrapper, 非 extern "C"
bool IsHcommDefaultTimeoutSupported() { return false; }
HcclResult HcclSetNotifyWaitTimeOut(uint32_t) { return HCCL_E_NOT_SUPPORT; }
HcclResult HcclThreadNotifyWaitOnThreadDefault(ThreadHandle, uint32_t, uint32_t) { return HCCL_SUCCESS; }

// g_binKernelHandle (load_kernel.cc 中定义, UT 不编译该文件)
namespace ops_hccl {
aclrtBinHandle g_binKernelHandle = nullptr;
}

// HcclRankGraphGetEndpointInfo stub (hccl_rank_graph_dl.cc 中定义, UT 不编译该文件)
HcclResult HcclRankGraphGetEndpointInfo(HcclComm, uint32_t, const EndpointDesc*,
    EndpointAttr, uint32_t, void*) { return HCCL_SUCCESS; }

// DlHcommFunction::GetInstance() stub
namespace ops_hccl {
DlHcommFunction &DlHcommFunction::GetInstance()
{
    static DlHcommFunction inst;
    return inst;
}
DlHcommFunction::~DlHcommFunction() = default;
HcclResult DlHcommFunction::DlHcommFunctionInit() { return HCCL_SUCCESS; }
HcclResult DlHcommFunction::DlHcommFunctionInterInit() { return HCCL_SUCCESS; }
}
