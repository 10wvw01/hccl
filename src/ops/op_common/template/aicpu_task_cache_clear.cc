/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_comm.h"
#include "load_kernel.h"
#include "log.h"

using namespace ops_hccl;

static inline HcclResult LaunchKernelAndSyncStream_(
    aclrtFuncHandle funcHandle, aclrtArgsHandle argsHandle, aclrtStream stream)
{
    // 下发kernel
    constexpr u16 kernelLaunchTimeout = 27 * 68; // 单位秒
    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = kernelLaunchTimeout;
    aclrtLaunchKernelCfg cfg{};
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;
    aclError ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, stream, &cfg, argsHandle, nullptr);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[%s][aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed",
        __func__, ret), HCCL_E_RUNTIME);

    constexpr u16 streamTimeout = 60; // 单位毫秒
    ret = aclrtSynchronizeStreamWithTimeout(stream, streamTimeout);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[%s] sync stream failed, errNo[0x%016llx]", __func__, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult AicpuCacheEvitKernelLaunch(HcclComm comm)
{
    const char kernelName[] = "HcclLaunchAicpuCacheEvitKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;

    // 获取function handle
    aclError ret = aclrtBinaryGetFunction(g_binKernelHandle, kernelName, &funcHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtBinaryGetFunction]errNo[0x%016llx] kernelName:%s", ret, kernelName),
        HCCL_E_RUNTIME);

    // 初始化和准备参数
    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] kernelName:%s", ret, kernelName),
        HCCL_E_RUNTIME);
    aclrtParamHandle paraHandle;
    ret = aclrtKernelArgsAppend(argsHandle, &comm, sizeof(comm), &paraHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsAppend]errNo[0x%016llx] kernelName:%s", ret, kernelName),
        HCCL_E_RUNTIME);
    ret = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsFinalize]errNo[0x%016llx] kernelName:%s", ret, kernelName),
        HCCL_E_RUNTIME);

    // 创建流
    aclrtStream stream;
    ret = aclrtCreateStreamWithConfig(&stream, 0, ACL_STREAM_FAST_SYNC);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] create stream failed, errNo[0x%016llx]", __func__, ret), HCCL_E_RUNTIME);

    HCCL_INFO("[%s]launch kernel[%s] comm[%p]", __func__, kernelName, comm);
    HcclResult result = LaunchKernelAndSyncStream_(funcHandle, argsHandle, stream);

    // 销毁流
    CHK_PRT_RET(aclrtDestroyStream(stream) != ACL_SUCCESS,
        HCCL_ERROR("[%s] destroy stream failed", __func__), HCCL_E_RUNTIME);
    return result;
}

HcclResult AicpuTaskCacheCommStateCallback(HcclComm comm, HcclCommStatePhase state, void *args) 
{
    (void)args;
    HCCL_INFO("[%s] comm[%p] state[%d]", __func__, comm, state);
    if (state == HCCL_COMM_STATE_PHASE_DESTROY_POST || state == HCCL_COMM_STATE_PHASE_RESUME_POST) {
        // 通信域销毁或者N秒快恢时，调用device接口，清理通信域相关的task缓存
        CHK_PRT(AicpuCacheEvitKernelLaunch(comm));
    }

    return HCCL_SUCCESS;
}

__attribute__((constructor)) void RegisterAicpuTaskCacheCallback()
{
    const char REG_NAME[] = "aicpu_task_cache_callback";
    HCCL_INFO("[%s] start register comm state callback", __func__);
    uint64_t args = 1u; // unused
    CHK_PRT(HcclCommRegCommStateCallback(REG_NAME, AicpuTaskCacheCommStateCallback, reinterpret_cast<void *>(args)));
}
