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
    // inputBufferType==HCCL_BUFFER 时 input 与 ccl 共用，无需 PreCopy。
    if (tempAlgParams_.inputBufferType != BufferType::INPUT) {
        return HCCL_SUCCESS;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams_.dataType];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = (tempAlgParams_.tailCount == 0) ? sliceSize :
                         (sliceSize + tempAlgParams_.tailCount * dataTypeSize);
    const u32 rankSize = static_cast<u32>(ranks_.size());
    const u32 tailRankId = ranks_[rankSize - 1];
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, ranks_, myAlgRank));
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    // 本卡只拷贝 ranksForInputData 中 idx % rankSize == myAlgRank 的切片到 ccl 槽位
    // (g*rankSize + myAlgRank)（按 algRank 维度寻址）。INPUT 场景下 ccl buffer 仅覆盖本
    // subComm 域（slot 数 = rankSize，无空位），tx/rx/LocalReduce 沿用同一 algRank 维度布局。
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
        const u64 inOff = tempAlgParams_.dataOffset + tempAlgParams_.sliceOffset +
                          static_cast<u64>(idx) * tempAlgParams_.dataStride;
        const u64 slotIdx = (static_cast<u64>(idx) / rankSize) * rankSize + myAlgRank;
        const u64 cclOff = tempAlgParams_.sliceOffset + slotIdx * tempAlgParams_.scratchStride;
        DataSlice srcSlice(tempAlgParams_.inputBufferPtr, inOff, curSliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.cclBufferPtr, cclOff, curSliceSize, sliceCount);
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

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

    // 2. 通信完成后：LocalReduce 把对端发来的数据 reduce 到本卡归约目标槽位。
    //    - reuseCclBuffer 场景：对端数据落到空位 slot（emptySlots[对端 algRank]），
    //      LocalReduce 把空位 slot reduce 到归约目标 slot ranksForInputData[idx%rankSize==myAlgRank]
    //      （按 rank 值寻址，与 NHR/AicpuBaseTemplate 一致）。
    //    - INPUT 场景：tx/rx 目标按 algRank 维度 (g*rankSize + srcAlgRank) 寻址，
    //      LocalReduce 把同组其他 algRank slot reduce 到 (g*rankSize + myAlgRank)。
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
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, ranks_, myAlgRank));
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    const size_t inputSize = tempAlgParams_.ranksForInputData.size();
    const bool reuseCclBuffer = (tempAlgParams_.inputBufferType == BufferType::HCCL_BUFFER);

    // reuseCclBuffer 场景：收集空位 slot（不在 ranksForInputData 中的 rank）。
    // INPUT 场景不需要 CollectEmptySlots，空位 = 非 myAlgRank 的 algRank 维度 slot (g*rankSize + c)。
    std::vector<u32> emptySlots;
    if (reuseCclBuffer) {
        CollectEmptySlots(ranks_, tempAlgParams_.ranksForInputData, rankSize, emptySlots);
    }

    // 计算归约目标 slot 的 ccl 偏移：
    //   reuseCclBuffer：按 rank 值寻址 ccl[tgtRank]；
    //   INPUT：按 algRank 维度寻址 ccl[g*rankSize + myAlgRank]。
    auto tgtCclOff = [&](u32 tgtRank, size_t g) -> u64 {
        const u64 slot = reuseCclBuffer ? tgtRank : (g * rankSize + myAlgRank);
        return tempAlgParams_.sliceOffset + slot * tempAlgParams_.scratchStride;
    };
    // 计算对端来源 slot 的 ccl 偏移：
    //   reuseCclBuffer：空位 emptySlots[c]（rank 值）；
    //   INPUT：algRank 维度 slot (g*rankSize + c)。
    auto srcCclOff = [&](u32 c, size_t g) -> u64 {
        const u64 slot = reuseCclBuffer ? emptySlots[c < emptySlots.size() ? c : 0]
                                        : (g * rankSize + c);
        return tempAlgParams_.sliceOffset + slot * tempAlgParams_.scratchStride;
    };

    // LocalReduce：把对端发来的数据（空位 slot）reduce 到归约目标 slot。
    for (size_t g = 0; g < (inputSize + rankSize - 1) / rankSize; ++g) {
        const size_t tgtIdx = g * rankSize + myAlgRank;
        if (tgtIdx >= inputSize) {
            break;
        }
        const u32 tgtRank = tempAlgParams_.ranksForInputData[tgtIdx];
        const u64 tgtSliceSize = (tgtRank == tailRankId) ? tailSize : sliceSize;
        if (tgtSliceSize == 0) {
            continue;
        }
        const u64 dstCclOff = tgtCclOff(tgtRank, g);
        for (u32 c = 0; c < rankSize; ++c) {
            if (c == myAlgRank) {
                continue;
            }
            // src 切片大小：reuseCclBuffer 与 tgt 一致；INPUT 需按 srcRank 判断 tail。
            u64 srcSliceSize = tgtSliceSize;
            if (!reuseCclBuffer) {
                const size_t srcIdx = g * rankSize + c;
                if (srcIdx >= inputSize) {
                    break;
                }
                const u32 srcRank = tempAlgParams_.ranksForInputData[srcIdx];
                srcSliceSize = (srcRank == tailRankId) ? tailSize : sliceSize;
                if (srcSliceSize != tgtSliceSize) {
                    continue;
                }
            }
            DataSlice srcSlice(tempAlgParams_.cclBufferPtr, srcCclOff(c, g), srcSliceSize,
                               srcSliceSize / dataTypeSize);
            DataSlice dstSlice(tempAlgParams_.cclBufferPtr, dstCclOff, tgtSliceSize, tgtSliceSize / dataTypeSize);
            CHK_RET(LocalReduce(threads[0], srcSlice, dstSlice,
                                tempAlgParams_.dataType, tempAlgParams_.reduceOp));
        }
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
