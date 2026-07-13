/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <sstream>
#include <memory>

#include "alg_param.h"
#include "ops_executor.h"
#include "kernel_launch.h"
#include "hcomm_primitives.h"
#include "hcomm_primitives_dl.h"
#include "dfx/task_exception_fun.h"
#include "hcomm_diag_dl.h"
#include "hcomm_device_profiling_dl.h"
#include "hccl_device_comm_dl.h"
#include "exec_timeout_manager.h"
#include "log.h"

namespace ops_hccl {

/**
 * AICPU kernel 下发入口。
 * 工作流程：
 *   1. 获取通信域句柄（HcommAcquireComm）；
 *   2. 根据 opType 还原变长数据（如 AllGatherV 的 counts/displs）；
 *   3. 设置 batch mode，注册 DFX op 信息与 profiling；
 *   4. 主 thread 等待 Host stream 的 notify 通知；
 *   5. 调用 executor.Orchestrate 驱动算法编排；
 *   6. 上报 profiling，通知 Host stream 完成，结束 batch mode；
 *   7. 释放通信域句柄（HcommReleaseComm）。
 * 与旧实现差异：
 *   - resCtx 由 AiCpuLauncher::CreateRes 创建并直接传入，无需从 param->resCtx 反序列化；
 *   - executor 由调用方（AiCpuLauncher::LaunchKernel）传入，无需通过 CollAlgExecRegistry 重新获取。
 * 输入参数：
 *   - param: 算子参数
 *   - executor: 执行器引用，提供 Orchestrate 接口
 *   - resCtx: 已创建的资源上下文
 */
HcclResult HcclLaunchAicpuKernel(const OpParam &param, OpsExecutor &executor, AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("Entry-%s, commName[%s], tag[%s], algTag[%s]", __func__, param.commName, param.tag, param.algTag);
    if (HcommAcquireComm(param.commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param.commName);
        return HCCL_E_INTERNAL;
    }

    HcclResult ret = HCCL_SUCCESS;

// 非主线设备类型注册 ScatterOpInfo（保持原有设备类型分发逻辑）
#ifdef MACRO_DEV_TYPE_NEW
    if (param.deviceType != DevType::DEV_TYPE_950) {
#else
    if (param.deviceType != DevType::DEV_TYPE_910_95) {
#endif
        ScatterOpInfo opInfo;
        if (CreateScatter(const_cast<OpParam *>(&param), &opInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("%s CreateScatter fail", __func__);
            return HCCL_E_INTERNAL;
        }
        if (HcommIsSupportHcommRegOpInfo()
            && HcommRegOpInfo(param.commName, reinterpret_cast<void *>(&opInfo), sizeof(ScatterOpInfo))
                   != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommRegOpInfo fail, commName[%s], algTag[%s], size[%u]", __func__, param.commName,
                opInfo.algTag, sizeof(ScatterOpInfo));
            return HCCL_E_INTERNAL;
        }
        if (HcommIsSupportHcommRegOpTaskException()
            && HcommRegOpTaskException(param.commName, ops_hccl::GetScatterOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s HcommRegOpTaskException fail, commName[%s], algTag[%s]", __func__, param.commName, param.algTag);
            return HCCL_E_INTERNAL;
        }
    }

    // 1. 还原变长数据指针（resCtx 直接使用传入的，无需反序列化）
    if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        ret = RestoreVarDataBatchSendRecv(const_cast<OpParam &>(param));
    } else if (param.opType == HCCL_CMD_ALLTOALLV || param.opType == HCCL_CMD_ALLTOALLVC
               || param.opType == HCCL_CMD_ALLTOALL) {
        ret = RestoreVarDataAlltoAllV(const_cast<OpParam &>(param), resCtx);
    } else if (param.opType == HCCL_CMD_REDUCE_SCATTER_V) {
        ret = RestoreVarDataReduceScatterV(const_cast<OpParam &>(param), resCtx);
    } else if (param.opType == HCCL_CMD_ALLGATHER_V) {
        ret = RestoreVarDataAllGatherV(const_cast<OpParam &>(param), resCtx);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to restore optype [%d] data and counts.", param.opType);
        return ret;
    }

    // 2. 获取 Device 侧主 thread
    ThreadHandle thread = resCtx.threads[0];
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set batch mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    // 3. 注册 DFX op 信息（需在第一个 task 之前上报）
    HcclDfxOpInfoCompat dfxOpInfo{};
    if (ConvertToHcclDfxOpInfo(const_cast<OpParam *>(&param), &dfxOpInfo) != HCCL_SUCCESS) {
        HCCL_ERROR("ConvertToHcclDfxOpInfo fail, commName is %s, tag is %s", param.commName, param.algTag);
        return HCCL_E_INTERNAL;
    }
    if (HcclDfxRegOpInfoByCommId(const_cast<char *>(param.commName), reinterpret_cast<void *>(&dfxOpInfo)) != HCCL_SUCCESS) {
        HCCL_ERROR("HcclDfxRegOpInfoByCommId fail, commName is %s, tag is %s", param.commName, param.algTag);
        return HCCL_E_INTERNAL;
    }

    // 4. 上报主流和第一个 task（wait 之前）
    if (HcommProfilingReportKernelStartTask(thread, param.commName) != HCCL_SUCCESS) {
        HCCL_ERROR(
            "%s failed to report MainStream And FirstTask, thread %lu, commName %s.", __func__, thread, param.commName);
        return HCCL_E_INTERNAL;
    }

    // 5. 主 thread 等待 Host stream 的 notify 通知
    ThreadHandle exportedAicpuTsThread = param.opThread;
    u32 maxNotifyNum = resCtx.notifyNumOnMainThread;
    for (u32 i = 0; i < resCtx.notifyNumPerThread.size(); i++) {
        if (resCtx.notifyNumPerThread[i] > maxNotifyNum) {
            maxNotifyNum = resCtx.notifyNumPerThread[i];
        }
    }
    HCCL_DEBUG("[%s]Notify wait on thread[%llu], maxNotifyNum[%u], timeout[%u]", __func__, thread, maxNotifyNum,
        CUSTOM_TIMEOUT);
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, maxNotifyNum, CUSTOM_TIMEOUT)));

    // 6. 执行算法编排：使用调用方传入的 executor，无需通过 registry 重新获取
    if (executor.Orchestrate(resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("orchestrate failed for alg:%s", param.algName);
        return HCCL_E_INTERNAL;
    }

    // 7. 上报 device op profiling
    if (HcommProfilingReportDeviceOp(param.commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommProfilingReportDeviceOp fail, commName[%s]", __func__, param.commName);
        return HCCL_E_INTERNAL;
    }

    // 8. 主 thread 通知 Host stream 完成
    constexpr u32 DEFAULT_NOTIFY_IDX = 0;
    HCCL_DEBUG("[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread,
        exportedAicpuTsThread, DEFAULT_NOTIFY_IDX);
    CHK_RET(
        static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX)));

    // 9. 上报主流和最后一个 task（notify 之后）
    if (HcommProfilingReportKernelEndTask(thread, param.commName) != HCCL_SUCCESS) {
        HCCL_ERROR(
            "%s failed to report MainStream And LastTask, thread %lu, commName %s.", __func__, thread, param.commName);
        return HCCL_E_INTERNAL;
    }

    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommReleaseComm(param.commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param.commName);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("%s success, tag[%s], algTag[%s], commName[%s]", __func__, param.tag, param.algTag, param.commName);
    return HCCL_SUCCESS;
}

HcclResult RestoreVarDataBatchSendRecv(OpParam &param)
{
    u64 sendRecvItemSize = static_cast<u64>(sizeof(HcclSendRecvItem));
    u64 itemNum = static_cast<u64>(param.batchSendRecvDataDes.itemNum);
    if (param.varMemSize != itemNum * sendRecvItemSize) {
        HCCL_ERROR("param.varMemSize[%lu] is not equal to itemNum[%lu] multiply [HcclSendRecvItem] size[%lu]."
                   "Failed to restore end recv info for BatchSendRecv!",
            param.varMemSize, itemNum, sendRecvItemSize);
        return HCCL_E_PARA;
    }
    param.batchSendRecvDataDes.sendRecvItemsPtr = reinterpret_cast<HcclSendRecvItem *>(param.varData);
    return HCCL_SUCCESS;
}

HcclResult RestoreVarDataAlltoAllV(OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    CHK_PRT_RET(param.varMemSize != ALL_TO_ALL_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR("[RestoreVarDataAlltoAllV] param.varMemSize [%llu] is invalid,"
                   " ALL_TO_ALL_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize, ALL_TO_ALL_V_VECTOR_NUM, rankSize, sizeof(u64)),
        HCCL_E_PARA);

    constexpr u32 ALL_TO_ALL_V_OFFSET_SCOUNTS = 0;
    constexpr u32 ALL_TO_ALL_V_OFFSET_RECV_COUNTS = 1;
    constexpr u32 ALL_TO_ALL_V_OFFSET_SDISPLS = 2;
    constexpr u32 ALL_TO_ALL_V_OFFSET_RDISPLS = 3;

    u64 *data = reinterpret_cast<u64 *>(param.varData);
    param.all2AllVDataDes.sendCounts = data;
    param.all2AllVDataDes.recvCounts = data + ALL_TO_ALL_V_OFFSET_RECV_COUNTS * rankSize;
    param.all2AllVDataDes.sdispls = data + ALL_TO_ALL_V_OFFSET_SDISPLS * rankSize;
    param.all2AllVDataDes.rdispls = data + ALL_TO_ALL_V_OFFSET_RDISPLS * rankSize;

    return HCCL_SUCCESS;
}

HcclResult RestoreVarDataReduceScatterV(OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    HCCL_INFO("rankSize:%u", rankSize);
    CHK_PRT_RET(param.varMemSize != REDUCE_SCATTER_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR("[RestoreVarDataReduceScatterV] param.varMemSize [%llu] is invalid,"
                   "REDUCE_SCATTER_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize, REDUCE_SCATTER_V_VECTOR_NUM, rankSize, sizeof(u64)),
        HCCL_E_PARA);

    u64 *data = reinterpret_cast<u64 *>(param.varData);
    param.vDataDes.counts = data;
    param.vDataDes.displs = data + rankSize;
    return HCCL_SUCCESS;
}

HcclResult RestoreVarDataAllGatherV(OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    HCCL_INFO("rankSize:%u", rankSize);
    CHK_PRT_RET(param.varMemSize != ALL_GATHER_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR("[RestoreVarDataAllGatherV] param.varMemSize [%llu] is invalid,"
                   "ALL_GATHER_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize, ALL_GATHER_V_VECTOR_NUM, rankSize, sizeof(u64)),
        HCCL_E_PARA);

    u64 *data = reinterpret_cast<u64 *>(param.varData);
    param.vDataDes.counts = data;
    for (u64 i = 0; i < rankSize; i++) {
        HCCL_INFO("param.vDataDes.counts[%u]:%u", i, reinterpret_cast<u64 *>(param.vDataDes.counts)[i]);
    }
    param.vDataDes.displs = data + rankSize;
    for (u64 i = 0; i < rankSize; i++) {
        HCCL_INFO("param.vDataDes.displs[%u]:%u", i, reinterpret_cast<u64 *>(param.vDataDes.displs)[i]);
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
