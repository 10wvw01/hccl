/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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
#include "hccl_algorithm.h"
#include "kernel_launch.h"
#include "hcomm_primitives.h"
#include "hcomm_primitives_dl.h"
#include "dfx/task_exception_fun.h"
#include "hcomm_diag_dl.h"
#include "hcomm_device_profiling_dl.h"
#include "hccl_device_comm_dl.h"
#include "exec_timeout_manager.h"
#include "binary_stream.h"
#include "log.h"

using namespace ops_hccl;

namespace {

/**
 * 从 param->resCtx 反序列化 AlgResourceCtxSerializable。
 * param->resCtx 指向序列化后的字节流，param->ctxSize 为字节流长度。
 */
std::unique_ptr<AlgResourceCtxSerializable> DeserializeResCtx(const OpParam *param)
{
    auto resCtx = std::make_unique<AlgResourceCtxSerializable>();
    char *ctx = static_cast<char *>(param->resCtx);
    std::vector<char> seq(ctx, ctx + param->ctxSize);
    resCtx->DeSerialize(seq);
    return resCtx;
}

} // anonymous namespace

namespace ops_hccl {

HcclResult RestoreVarDataBatchSendRecv(OpParam &param)
{
    u64 sendRecvItemSize = static_cast<u64>(sizeof(HcclSendRecvItem));
    u64 itemNum = static_cast<u64>(param.batchSendRecvDataDes.itemNum);
    if (param.varMemSize != itemNum * sendRecvItemSize) {
        HCCL_ERROR("param.varMemSize[%lu] is not equal to itemNum[%lu] multiply [HcclSendRecvItem] size[%lu]."
                   "Failed to restore end recv info for BatchSendRecv!",
            param.varMemSize,
            itemNum,
            sendRecvItemSize);
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
            param.varMemSize,
            ALL_TO_ALL_V_VECTOR_NUM,
            rankSize,
            sizeof(u64)),
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
            param.varMemSize,
            REDUCE_SCATTER_V_VECTOR_NUM,
            rankSize,
            sizeof(u64)),
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
            param.varMemSize,
            ALL_GATHER_V_VECTOR_NUM,
            rankSize,
            sizeof(u64)),
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

/**
 * AICPU kernel 下发入口（C 接口，与 src 侧保持一致）。
 * 在 device 侧执行，通过 aclrt kernel launch 下发。
 */
extern "C" unsigned int HcclLaunchAicpuKernel(OpParam *param)
{
    if (param == nullptr) {
        HCCL_ERROR("%s param is nullptr", __func__);
        return 1;
    }
    HCCL_INFO("Entry-%s, commName[%s], tag[%s], algTag[%s]", __func__, param->commName, param->tag, param->algTag);
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }

    // 1. 从 param->resCtx 反序列化 AlgResourceCtxSerializable
    auto resCtx = DeserializeResCtx(param);
    AlgResourceCtxSerializable *resCtxPtr = resCtx.get();

    // 2. 从 resCtx->algoSerialData 反序列化 HcclAlgorithm（hcclCmdType/engineType/algoExecDesc），重建 executor
    HcclAlgorithm alg;
    BinaryStream algoBs(resCtxPtr->algoSerialData);
    alg.DeserializeFrom(algoBs);
    auto executor = alg.GetExecutor(*param);

    // 非主线设备类型注册 ScatterOpInfo（保持原有设备类型分发逻辑）
#ifdef MACRO_DEV_TYPE_NEW
    if (param->deviceType != DevType::DEV_TYPE_950) {
#else
    if (param->deviceType != DevType::DEV_TYPE_910_95) {
#endif
        ScatterOpInfo opInfo;
        if (CreateScatter(param, &opInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("%s CreateScatter fail", __func__);
            return 1;
        }
        if (HcommIsSupportHcommRegOpInfo()
            && HcommRegOpInfo(param->commName, reinterpret_cast<void *>(&opInfo), sizeof(ScatterOpInfo))
                   != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommRegOpInfo fail, commName[%s], algTag[%s], size[%u]", __func__, param->commName,
                opInfo.algTag, sizeof(ScatterOpInfo));
            return 1;
        }
        if (HcommIsSupportHcommRegOpTaskException()
            && HcommRegOpTaskException(param->commName, ops_hccl::GetScatterOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s HcommRegOpTaskException fail, commName[%s], algTag[%s]", __func__, param->commName, param->algTag);
            return 1;
        }
    }

    // 3. 还原变长数据指针
    HcclResult ret = HCCL_SUCCESS;
    if (param->opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        ret = ops_hccl::RestoreVarDataBatchSendRecv(*param);
    } else if (param->opType == HCCL_CMD_ALLTOALLV || param->opType == HCCL_CMD_ALLTOALLVC
               || param->opType == HCCL_CMD_ALLTOALL) {
        ret = ops_hccl::RestoreVarDataAlltoAllV(*param, *resCtxPtr);
    } else if (param->opType == HCCL_CMD_REDUCE_SCATTER_V) {
        ret = ops_hccl::RestoreVarDataReduceScatterV(*param, *resCtxPtr);
    } else if (param->opType == HCCL_CMD_ALLGATHER_V) {
        ret = ops_hccl::RestoreVarDataAllGatherV(*param, *resCtxPtr);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to restore optype [%d] data and counts.", param->opType);
        return 1;
    }

    // 4. 获取 Device 侧主 thread，设置 batch mode
    ThreadHandle thread = resCtxPtr->threads[0];
    if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
        return 1;
    }

    // 5. 注册 DFX op 信息（需在第一个 task 之前上报）
    HcclDfxOpInfoCompat dfxOpInfo{};
    if (ConvertToHcclDfxOpInfo(param, &dfxOpInfo) != HCCL_SUCCESS) {
        HCCL_ERROR("ConvertToHcclDfxOpInfo fail, commName is %s, tag is %s", param->commName, param->algTag);
        return 1;
    }
    if (HcclDfxRegOpInfoByCommId(param->commName, reinterpret_cast<void *>(&dfxOpInfo)) != HCCL_SUCCESS) {
        HCCL_ERROR("HcclDfxRegOpInfoByCommId fail, commName is %s, tag is %s", param->commName, param->algTag);
        return 1;
    }

    // 6. 上报主流和第一个 task（wait 之前）
    if (HcommProfilingReportKernelStartTask(thread, param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR(
            "%s failed to report MainStream And FirstTask, thread %lu, commName %s.", __func__, thread, param->commName);
        return 1;
    }

    // 7. 主 thread 等待 Host stream 的 notify 通知
    ThreadHandle exportedAicpuTsThread = param->opThread;
    u32 hostToDeviceNotifyIdx = resCtxPtr->notifyNumOnMainThread - 1;
    u32 maxNotifyNum = resCtxPtr->notifyNumOnMainThread;
    for (u32 i = 0; i < resCtxPtr->notifyNumPerThread.size(); i++) {
        if (resCtxPtr->notifyNumPerThread[i] > maxNotifyNum) {
            maxNotifyNum = resCtxPtr->notifyNumPerThread[i];
        }
    }
    HCCL_DEBUG("[%s]Notify wait on thread[%llu], notifyIdx[%u], timeout[%u]", __func__, thread, hostToDeviceNotifyIdx,
        CUSTOM_TIMEOUT);
    if (HcommThreadNotifyWaitOnThread(thread, hostToDeviceNotifyIdx, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommThreadNotifyWaitOnThread failed", __func__);
        return 1;
    }

    // 8. 执行算法编排
    if (executor->Orchestrate(*resCtxPtr) != HCCL_SUCCESS) {
        HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
        return 1;
    }

    // 9. 上报 device op profiling
    if (HcommProfilingReportDeviceOp(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommProfilingReportDeviceOp fail, commName[%s]", __func__, param->commName);
        return 1;
    }

    // 10. 主 thread 通知 Host stream 完成
    constexpr u32 DEFAULT_NOTIFY_IDX = 0;
    HCCL_DEBUG("[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread,
        exportedAicpuTsThread, DEFAULT_NOTIFY_IDX);
    if (HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommThreadNotifyRecordOnThread failed", __func__);
        return 1;
    }

    // 11. 上报主流和最后一个 task（notify 之后）
    if (HcommProfilingReportKernelEndTask(thread, param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR(
            "%s failed to report MainStream And LastTask, thread %lu, commName %s.", __func__, thread, param->commName);
        return 1;
    }

    if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
        return 1;
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s success, tag[%s], algTag[%s], commName[%s]", __func__, param->tag, param->algTag, param->commName);
    return 0;
}
