/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
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
    if (tempAlgParams_.inputBufferType != BufferType::INPUT) {
        return HCCL_SUCCESS;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams_.dataType];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = (tempAlgParams_.tailCount == 0) ? sliceSize :
                         (sliceSize + tempAlgParams_.tailCount * dataTypeSize);
    const u32 rankSize = static_cast<u32>(ranks_.size());
    const u32 tailRankId = ranks_[rankSize - 1];
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    MeshRsLayoutInfo layoutInfo;
    CHK_RET(InitMeshRsLayoutInfo(tempAlgParams_, ranks_, myRank_, layoutInfo));
    const u32 myAlgRank = layoutInfo.myAlgRank;

    const size_t inputSize = tempAlgParams_.ranksForInputData.size();
    for (size_t idx = 0; idx < inputSize; ++idx) {
        if (static_cast<u32>(idx) % rankSize != myAlgRank) {
            continue;
        }
        const u32 rank = tempAlgParams_.ranksForInputData[idx];
        const u64 curSliceSize = (rank == tailRankId) ? tailSize : sliceSize;
        if (curSliceSize == 0) {
            continue;
        }
        const u64 sliceCount = curSliceSize / dataTypeSize;
        DataSlice srcSlice(tempAlgParams_.inputBufferPtr, GetMeshRsInputOffset(tempAlgParams_, idx),
                           curSliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.cclBufferPtr,
                           GetMeshRsFinalCclOffset(tempAlgParams_, layoutInfo, idx, rank),
                           curSliceSize, sliceCount);
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    HCCL_INFO("[ReduceScatterMeshTemplate][PreCopy] end.");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshTemplate::SendAll(BaseEngine &engine,
    const std::vector<TxRxSlicesList> &txRxSlicesLists, TemplateResource &templateResource,
    const std::vector<ThreadHandle> &threads)
{
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

    if (threads.empty()) {
        HCCL_ERROR("[ReduceScatterMeshTemplate][SendAll] threads is empty for LocalReduce.");
        return HCCL_E_INTERNAL;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams_.dataType];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = (tempAlgParams_.tailCount == 0) ? sliceSize :
                         (sliceSize + tempAlgParams_.tailCount * dataTypeSize);
    const u32 rankSize = static_cast<u32>(ranks_.size());
    const u32 tailRankId = ranks_[rankSize - 1];
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    MeshRsLayoutInfo layoutInfo;
    CHK_RET(InitMeshRsLayoutInfo(tempAlgParams_, ranks_, myRank_, layoutInfo));
    const u32 myAlgRank = layoutInfo.myAlgRank;
    const size_t inputSize = tempAlgParams_.ranksForInputData.size();

    for (size_t groupIdx = 0; groupIdx < (inputSize + rankSize - 1) / rankSize; ++groupIdx) {
        const size_t tgtIdx = groupIdx * rankSize + myAlgRank;
        if (tgtIdx >= inputSize) {
            break;
        }
        const u32 tgtRank = tempAlgParams_.ranksForInputData[tgtIdx];
        const u64 tgtSliceSize = (tgtRank == tailRankId) ? tailSize : sliceSize;
        if (tgtSliceSize == 0) {
            continue;
        }
        const u64 dstCclOff = GetMeshRsFinalCclOffset(tempAlgParams_, layoutInfo, tgtIdx, tgtRank);
        for (u32 algRank = 0; algRank < rankSize; ++algRank) {
            if (algRank == myAlgRank) {
                continue;
            }
            const size_t srcIdx = groupIdx * rankSize + algRank;
            u64 srcSliceSize = tgtSliceSize;
            if (!layoutInfo.reuseCclBuffer) {
                if (srcIdx >= inputSize) {
                    break;
                }
                const u32 srcRank = tempAlgParams_.ranksForInputData[srcIdx];
                srcSliceSize = (srcRank == tailRankId) ? tailSize : sliceSize;
                if (srcSliceSize != tgtSliceSize) {
                    continue;
                }
            }
            DataSlice srcSlice(tempAlgParams_.cclBufferPtr,
                               GetMeshRsTempCclOffset(tempAlgParams_, layoutInfo, srcIdx, algRank),
                               srcSliceSize, srcSliceSize / dataTypeSize);
            DataSlice dstSlice(tempAlgParams_.cclBufferPtr, dstCclOff, tgtSliceSize, tgtSliceSize / dataTypeSize);
            CHK_RET(LocalReduce(threads[0], srcSlice, dstSlice,
                                tempAlgParams_.dataType, tempAlgParams_.reduceOp));
        }
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
