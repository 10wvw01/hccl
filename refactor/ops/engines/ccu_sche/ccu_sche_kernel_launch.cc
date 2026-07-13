/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_sche_kernel_launch.h"

#include "alg_param.h"
#include "ops_executor.h"
#include "hcomm_primitives.h"
#include "dfx/task_exception_fun.h"
#include "hcomm_diag_dl.h"
#include "hcomm_device_profiling_dl.h"
#include "hccl_device_comm_dl.h"
#include "log.h"

namespace ops_hccl {

HcclResult CcuScheLaunchKernel(const OpParam &param, OpsExecutor &executor, AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("Entry-%s, commName[%s], tag[%s], algTag[%s]", __func__, param.commName, param.tag, param.algTag);
    if (HcommAcquireComm(param.commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param.commName);
        return HCCL_E_INTERNAL;
    }

    // CCU 无需 AICPU 的 ScatterOpInfo 注册与 RestoreVarData 变长数据还原

    // 1. 获取 Device 侧主 thread
    ThreadHandle thread = resCtx.threads[0];
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuScheLaunchKernel] failed set batch mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    // 2. 注册 DFX op 信息（需在第一个 task 之前上报）
    HcclDfxOpInfoCompat dfxOpInfo{};
    if (ConvertToHcclDfxOpInfo(const_cast<OpParam *>(&param), &dfxOpInfo) != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuScheLaunchKernel] ConvertToHcclDfxOpInfo fail, commName is %s, tag is %s",
                   param.commName, param.algTag);
        return HCCL_E_INTERNAL;
    }
    if (HcclDfxRegOpInfoByCommId(const_cast<char *>(param.commName), reinterpret_cast<void *>(&dfxOpInfo)) != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuScheLaunchKernel] HcclDfxRegOpInfoByCommId fail, commName is %s, tag is %s",
                   param.commName, param.algTag);
        return HCCL_E_INTERNAL;
    }

    // 3. 上报主流和第一个 task（wait 之前）
    if (HcommProfilingReportKernelStartTask(thread, param.commName) != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuScheLaunchKernel] failed to report MainStream And FirstTask, thread %lu, commName %s.",
                   thread, param.commName);
        return HCCL_E_INTERNAL;
    }

    // 4. 主 thread 等待 Host stream 的 notify 通知
    ThreadHandle exportedCcuTsThread = param.opThread;
    u32 maxNotifyNum = resCtx.notifyNumOnMainThread;
    for (u32 i = 0; i < resCtx.notifyNumPerThread.size(); i++) {
        if (resCtx.notifyNumPerThread[i] > maxNotifyNum) {
            maxNotifyNum = resCtx.notifyNumPerThread[i];
        }
    }
    HCCL_DEBUG("[%s] Notify wait on thread[%llu], maxNotifyNum[%u], timeout[%u]",
               __func__, thread, maxNotifyNum, CUSTOM_TIMEOUT);
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, maxNotifyNum, CUSTOM_TIMEOUT)));

    // 5. 执行算法编排：使用调用方传入的 executor
    if (executor.Orchestrate(resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuScheLaunchKernel] orchestrate failed for alg:%s", param.algName);
        return HCCL_E_INTERNAL;
    }

    // 6. 上报 device op profiling
    if (HcommProfilingReportDeviceOp(param.commName) != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuScheLaunchKernel] HcommProfilingReportDeviceOp fail, commName[%s]", param.commName);
        return HCCL_E_INTERNAL;
    }

    // 7. 主 thread 通知 Host stream 完成
    constexpr u32 DEFAULT_NOTIFY_IDX = 0;
    HCCL_DEBUG("[%s] Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]",
               __func__, thread, exportedCcuTsThread, DEFAULT_NOTIFY_IDX);
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        thread, exportedCcuTsThread, DEFAULT_NOTIFY_IDX)));

    // 8. 上报主流和最后一个 task（notify 之后）
    if (HcommProfilingReportKernelEndTask(thread, param.commName) != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuScheLaunchKernel] failed to report MainStream And LastTask, thread %lu, commName %s.",
                   thread, param.commName);
        return HCCL_E_INTERNAL;
    }

    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuScheLaunchKernel] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommReleaseComm(param.commName) != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuScheLaunchKernel] HcommReleaseComm fail, commName[%s]", param.commName);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[CcuScheLaunchKernel] success, tag[%s], algTag[%s], commName[%s]",
              param.tag, param.algTag, param.commName);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
