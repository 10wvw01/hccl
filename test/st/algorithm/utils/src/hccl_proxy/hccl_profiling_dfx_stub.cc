/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_common.h"
#include "hcomm_diag.h"
#include "hcomm_host_profiling_dl.h"
#include <atomic>

namespace {
std::atomic<int32_t> g_p2pDfxRegRet{HCCL_SUCCESS};
std::atomic<int32_t> g_p2pProfilingStartRet{HCCL_SUCCESS};
std::atomic<int32_t> g_p2pProfilingEndRet{HCCL_SUCCESS};
std::atomic<int32_t> g_p2pProfilingDeviceRet{HCCL_SUCCESS};
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

uint64_t HcommGetProfilingSysCycleTime()
{
    return 0;
}

HcclResult HcommProfilingRegThread(HcomProInfoTmp profInfo, ThreadHandle* threads)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingUnRegThread(HcomProInfoTmp profInfo, ThreadHandle* threads)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingReportKernel(uint64_t beginTime, const char *profName)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingReportOp(HcomProInfoTmp profInfo)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcommRegOpInfo(const char* commId, void* opInfo, size_t size)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcommRegOpTaskException(const char* commId, HcommGetOpInfoCallback callback)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcclDfxRegOpInfoByCommId(char* commId, void* hcclDfxOpInfo)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return static_cast<HcclResult>(g_p2pDfxRegRet.load(std::memory_order_relaxed));
}

HcclResult HcclReportAivKernel(HcclComm comm, uint64_t beginTime)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcclReportAicpuKernel(HcclComm comm, uint64_t beginTime, char *kernelName)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcclProfilingReportOp(HcclComm comm, uint64_t beginTime)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingReportDeviceOp(const char* groupname)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return static_cast<HcclResult>(g_p2pProfilingDeviceRet.load(std::memory_order_relaxed));
}
HcclResult HcommProfilingReportKernelStartTask(uint64_t thread, const char* groupname)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return static_cast<HcclResult>(g_p2pProfilingStartRet.load(std::memory_order_relaxed));
}

HcclResult HcommProfilingReportKernelEndTask(uint64_t thread, const char* groupname)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return static_cast<HcclResult>(g_p2pProfilingEndRet.load(std::memory_order_relaxed));
}

/**
 * 重置P2P DFX与Profiling桩的返回值
 * @return 无
 */
void ResetP2pDfxProfilingStub()
{
    g_p2pDfxRegRet.store(HCCL_SUCCESS, std::memory_order_relaxed);
    g_p2pProfilingStartRet.store(HCCL_SUCCESS, std::memory_order_relaxed);
    g_p2pProfilingEndRet.store(HCCL_SUCCESS, std::memory_order_relaxed);
    g_p2pProfilingDeviceRet.store(HCCL_SUCCESS, std::memory_order_relaxed);
}

/**
 * 设置P2P DFX与Profiling桩的返回值
 * @param dfxRegRet DFX注册返回值
 * @param profilingStartRet Profiling起始任务上报返回值
 * @param profilingEndRet Profiling结束任务上报返回值
 * @param profilingDeviceRet Profiling设备算子上报返回值
 * @return 无
 */
void SetP2pDfxProfilingStub(HcclResult dfxRegRet, HcclResult profilingStartRet, HcclResult profilingEndRet,
    HcclResult profilingDeviceRet)
{
    g_p2pDfxRegRet.store(dfxRegRet, std::memory_order_relaxed);
    g_p2pProfilingStartRet.store(profilingStartRet, std::memory_order_relaxed);
    g_p2pProfilingEndRet.store(profilingEndRet, std::memory_order_relaxed);
    g_p2pProfilingDeviceRet.store(profilingDeviceRet, std::memory_order_relaxed);
}

HcclResult HcommProfilingReportMainStreamAndFirstTask(ThreadHandle thread)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingReportMainStreamAndLastTask(ThreadHandle thread)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingEnd(ThreadHandle* threads, uint32_t threadNum)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

bool HcommIsSupportHcommRegOpInfo()
{
    return false;
}

bool HcommIsSupportHcommRegOpTaskException()
{
    return false;
}
#ifdef __cplusplus
}
#endif  // __cplusplus
