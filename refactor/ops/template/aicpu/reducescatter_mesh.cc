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
    // 只有 input 来源是 user input 时才需要 PreCopy（input→ccl）；
    // inputBufferType==HCCL_BUFFER 时 input 与 ccl 共用，无需拷贝。
    if (tempAlgParams_.inputBufferType != BufferType::INPUT) {
        HCCL_INFO("[ReduceScatterMeshTemplate][PreCopy] inputBufferType=%d != INPUT, skip PreCopy.",
            static_cast<int>(tempAlgParams_.inputBufferType));
        return HCCL_SUCCESS;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams_.dataType];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = (tempAlgParams_.tailCount == 0) ? sliceSize :
                         (sliceSize + tempAlgParams_.tailCount * dataTypeSize);
    // mesh 子域大小即 ranks_.size()；尾 rank 用 ranks_ 最后一个。
    const u32 rankSize = static_cast<u32>(ranks_.size());
    const u32 tailRankId = ranks_[rankSize - 1];
    // myAlgRank：本卡在 mesh 子域内的算法 rank。
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, ranks_, myAlgRank));
    if (rankSize <= 1) {
        // 单 rank 时 ranksForInputData 即全部输出，按基类语义 PreCopy（此处不发生，KernelRun 已短路）。
        return HCCL_SUCCESS;
    }

    // PreCopy：本卡只拷贝 ranksForInputData 中 idx % rankSize == myAlgRank 的切片到
    //   ccl 槽位 idx（按 idx 寻址：cclOff = sliceOffset + idx * scratchStride）。
    //   该切片是本卡（myAlgRank）对 ranksForInputData[idx] 的归约贡献；
    //   slot layout 与 RunMeshReduceScatter 的 tx/rx slice、SendAll 的 LocalReduce 保持一致。
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
        const u64 cclOff = tempAlgParams_.sliceOffset + static_cast<u64>(idx) * tempAlgParams_.scratchStride;
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

    // 2. 通信完成后：LocalReduce 把对端 algRank 槽位的数据 reduce 到本卡归约目标槽位。
    //    ccl 布局：槽位 idx 存放 ranksForInputData[idx] 切片，来源 algRank = idx % rankSize。
    //    本卡归约目标槽 = idx % rankSize == myAlgRank；对每个目标槽 tgtIdx，
    //      把同组（tgtIdx/rankSize 相同）的其他 algRank 槽位 reduce 进来。
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
    const size_t groupNum = (inputSize + rankSize - 1) / rankSize;
    for (size_t g = 0; g < groupNum; ++g) {
        const size_t tgtIdx = g * rankSize + myAlgRank;
        if (tgtIdx >= inputSize) {
            break;
        }
        const u32 tgtRank = tempAlgParams_.ranksForInputData[tgtIdx];
        const u64 tgtSliceSize = (tgtRank == tailRankId) ? tailSize : sliceSize;
        if (tgtSliceSize == 0) {
            continue;
        }
        const u64 tgtSliceCount = tgtSliceSize / dataTypeSize;
        const u64 dstCclOff = tempAlgParams_.sliceOffset +
                              static_cast<u64>(tgtIdx) * tempAlgParams_.scratchStride;
        for (u32 c = 0; c < rankSize; ++c) {
            if (c == myAlgRank) {
                continue;
            }
            const size_t srcIdx = g * rankSize + c;
            if (srcIdx >= inputSize) {
                break;
            }
            const u32 srcRank = tempAlgParams_.ranksForInputData[srcIdx];
            const u64 srcSliceSize = (srcRank == tailRankId) ? tailSize : sliceSize;
            if (srcSliceSize == 0 || srcSliceSize != tgtSliceSize) {
                continue;
            }
            const u64 srcCclOff = tempAlgParams_.sliceOffset +
                                  static_cast<u64>(srcIdx) * tempAlgParams_.scratchStride;
            DataSlice srcSlice(tempAlgParams_.cclBufferPtr, srcCclOff, srcSliceSize,
                               srcSliceSize / dataTypeSize);
            DataSlice dstSlice(tempAlgParams_.cclBufferPtr, dstCclOff, tgtSliceSize, tgtSliceCount);
            CHK_RET(LocalReduce(threads[0], srcSlice, dstSlice,
                                tempAlgParams_.dataType, tempAlgParams_.reduceOp));
        }
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
