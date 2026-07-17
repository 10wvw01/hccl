/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reducescatter_mesh.h"

#include "base_engine.h"
#include "primitives/mesh_primitives.h"

namespace ops_hccl {

HcclResult ReduceScatterMeshTemplate::RunAlgorithm(TemplateResource &templateResource,
                                                 std::vector<TxRxSlicesList> &txRxSlicesLists,
                                                 std::vector<u32> &ranksForOutputData)
{
    (void)templateResource;
    HCCL_INFO("[ReduceScatterMeshTemplate][RunAlgorithm] start, myRank[%u], rankSize[%u].",
              myRank_, templateRankSize_);

    CHK_RET(RunMeshReduceScatter(tempAlgParams_, ranks_, myRank_, ranksForOutputData, txRxSlicesLists));

    HCCL_INFO("[ReduceScatterMeshTemplate][RunAlgorithm] end.");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshTemplate::PreCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[ReduceScatterMeshTemplate][PreCopy] start, myRank[%u].", myRank_);
    if (threads.empty()) {
        HCCL_ERROR("[ReduceScatterMeshTemplate][PreCopy] threads is empty.");
        return HCCL_E_INTERNAL;
    }
    if (tempAlgParams_.ranksForInputData.empty()) {
        HCCL_ERROR("[ReduceScatterMeshTemplate][PreCopy] ranksForInputData is empty.");
        return HCCL_E_INTERNAL;
    }
    if (tempAlgParams_.cclBufferPtr == nullptr) {
        HCCL_ERROR("[ReduceScatterMeshTemplate][PreCopy] cclBufferPtr is null.");
        return HCCL_E_INTERNAL;
    }
    if (tempAlgParams_.inputBufferPtr == nullptr) {
        HCCL_ERROR("[ReduceScatterMeshTemplate][PreCopy] inputBufferPtr is null.");
        return HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams_.dataType];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = tempAlgParams_.tailCount * dataTypeSize;

    // 找 myRank 在 ranksForInputData 中的索引。
    auto myIt = std::find(tempAlgParams_.ranksForInputData.begin(),
                          tempAlgParams_.ranksForInputData.end(), myRank_);
    CHK_PRT_RET(myIt == tempAlgParams_.ranksForInputData.end(),
                HCCL_ERROR("[ReduceScatterMeshTemplate][PreCopy] myRank[%u] not in ranksForInputData.", myRank_),
                HCCL_E_PARA);
    const u64 myIdx = static_cast<u64>(std::distance(tempAlgParams_.ranksForInputData.begin(), myIt));
    const size_t lastInputIdx = tempAlgParams_.ranksForInputData.size() - 1;
    // ReduceScatter 中 myRank 的输出切片在 input 中大小为 sliceSize + tailSize（若 myIdx 是尾 rank）。
    const u64 curSliceSize = (tailSize > 0 && myIdx == lastInputIdx) ? sliceSize + tailSize : sliceSize;
    if (curSliceSize == 0) {
        return HCCL_SUCCESS;
    }
    const u64 sliceCount = curSliceSize / dataTypeSize;

    const u64 inOff = tempAlgParams_.dataOffset + tempAlgParams_.sliceOffset + myIdx * tempAlgParams_.dataStride;
    const u64 cclOff = tempAlgParams_.sliceOffset + static_cast<u64>(myRank_) * tempAlgParams_.scratchStride;
    DataSlice srcSlice(tempAlgParams_.inputBufferPtr, inOff, curSliceSize, sliceCount);
    DataSlice dstSlice(tempAlgParams_.cclBufferPtr, cclOff, curSliceSize, sliceCount);
    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    HCCL_INFO("[ReduceScatterMeshTemplate][PreCopy] end.");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshTemplate::SendAll(BaseEngine &engine,
    const std::vector<TxRxSlicesList> &txRxSlicesLists, TemplateResource &templateResource,
    const std::vector<ThreadHandle> &threads)
{
    // 1. 通信阶段：纯搬运（reduceOp=RESERVED），不 inline reduce。
    for (size_t i = 0; i < txRxSlicesLists.size(); ++i) {
        TransferContext ctx;
        ctx.enableRemoteMemAccess = tempAlgParams_.enableRemoteMemAccess;
        ctx.buffType = tempAlgParams_.cclBufferType;
        ctx.txRxSlicesList = txRxSlicesLists[i];
        ctx.templateRes = templateResource;
        ctx.dataType = tempAlgParams_.dataType;
        ctx.reduceOp = HCCL_REDUCE_RESERVED;
        CHK_RET(engine.Send(ctx));
    }

    // 2. 通信完成后：LocalReduce 本卡 ccl buffer 各 rank 槽位到 ccl[myRank 槽]。
    //    ccl[myRank 槽] 已由 PreCopy 填入本卡贡献作为初始值，跳过 myRank 自身，
    //    其余 rank 槽逐个 LocalReduce 累加到 ccl[myRank 槽]。
    //    ranksForInputData 已是 [0,1,2,...] 递增，按顺序遍历保证确定性。
    if (threads.empty()) {
        HCCL_ERROR("[ReduceScatterMeshTemplate][SendAll] threads is empty for LocalReduce.");
        return HCCL_E_INTERNAL;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams_.dataType];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = (tempAlgParams_.tailCount == 0) ? sliceSize :
                         (sliceSize + tempAlgParams_.tailCount * dataTypeSize);
    // myRank 是尾 rank 时归约 slice 大小为 tailSize，否则 sliceSize（与 RunMeshReduceScatter 的 rxSliceSize 对齐）。
    const bool iAmTail = (myRank_ == ranks_[ranks_.size() - 1]);
    const u64 curSliceSize = iAmTail ? tailSize : sliceSize;
    if (curSliceSize == 0) {
        return HCCL_SUCCESS;
    }
    const u64 sliceCount = curSliceSize / dataTypeSize;
    const u64 dstCclOff = tempAlgParams_.sliceOffset + static_cast<u64>(myRank_) * tempAlgParams_.scratchStride;
    for (size_t srcIdx = 0; srcIdx < tempAlgParams_.ranksForInputData.size(); ++srcIdx) {
        u32 sourceRank = tempAlgParams_.ranksForInputData[srcIdx];
        if (sourceRank == myRank_) {
            continue;
        }
        const u64 srcCclOff = tempAlgParams_.sliceOffset +
                              static_cast<u64>(sourceRank) * tempAlgParams_.scratchStride;
        DataSlice srcSlice(tempAlgParams_.cclBufferPtr, srcCclOff, curSliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.cclBufferPtr, dstCclOff, curSliceSize, sliceCount);
        CHK_RET(LocalReduce(threads[0], srcSlice, dstSlice,
                            tempAlgParams_.dataType, tempAlgParams_.reduceOp));
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
