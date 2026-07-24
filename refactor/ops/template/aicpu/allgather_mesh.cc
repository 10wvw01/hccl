/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_mesh.h"

#include "base_engine.h"
#include "primitives/mesh_primitives.h"
#include "utils/utils.h"

#include "log.h"

namespace ops_hccl {

HcclResult AllGatherMeshTemplate::RunAlgorithm(TemplateResource &templateResource,
                                                 std::vector<TxRxSlicesList> &txRxSlicesLists,
                                                 std::vector<u32> &ranksForOutputData)
{
    (void)templateResource;
    HCCL_INFO("[AllGatherMeshTemplate][RunAlgorithm] start, myRank[%u], rankSize[%u].",
              myRank_, templateRankSize_);

    CHK_RET(RunMeshAllGather(tempAlgParams_, ranks_, myRank_, ranksForOutputData, txRxSlicesLists));

    HCCL_INFO("[AllGatherMeshTemplate][RunAlgorithm] end.");
    return HCCL_SUCCESS;
}

// 判断是否走"rx 直接落 output"的优化路径：output 为 userBuffer 时启用。
static inline bool DirectToOutputMode(const TemplateDataParams &params)
{
    return params.outputBufferType == BufferType::OUTPUT;
}

HcclResult AllGatherMeshTemplate::SendAll(BaseEngine &engine, const std::vector<TxRxSlicesList> &txRxSlicesLists,
                                           TemplateResource &templateResource, const std::vector<ThreadHandle> &threads)
{
    // outputBufferType==OUTPUT：对端无法直接写本端 output，必须本端主动 Read。
    // 设 enableRemoteMemAccess=true + buffType=OUTPUT 让 engine 走 READ 方向（SendRecvRead）。
    // 否则交由基类默认 SendAll（WRITE 方向）。
    if (!DirectToOutputMode(tempAlgParams_)) {
        return AicpuBaseTemplate::SendAll(engine, txRxSlicesLists, templateResource, threads);
    }

    (void)threads;
    for (size_t i = 0; i < txRxSlicesLists.size(); ++i) {
        TransferContext ctx;
        // 触发 engine READ 分支：本端主动从对端 ccl buffer 读取数据到本地 output。
        ctx.enableRemoteMemAccess = true;
        ctx.buffType = BufferType::OUTPUT;
        ctx.txRxSlicesList = txRxSlicesLists[i];
        ctx.templateRes = templateResource;
        ctx.dataType = tempAlgParams_.dataType;
        ctx.reduceOp = tempAlgParams_.reduceOp;
        CHK_RET(engine.Send(ctx));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherMeshTemplate::PostCopy(const std::vector<ThreadHandle> &threads)
{
    // outputBufferType==OUTPUT：rx 已在 SendAll 中直接写入 output（peer 数据就位）。
    // 但 myRank 的本地数据仅在 PreCopy 写入 ccl[myRank]，尚未落到 output[myRank]，这里补搬一次。
    // peer 数据已在 output，不能复用基类 PostCopy（会从 ccl[peer] 读垃圾覆盖 output[peer]）。
    // outputBufferType==HCCL_BUFFER：交由基类默认 PostCopy（ccl buffer -> output）。
    if (!DirectToOutputMode(tempAlgParams_)) {
        return AicpuBaseTemplate::PostCopy(threads);
    }

    HCCL_INFO("[AllGatherMeshTemplate][PostCopy] directToOutput mode, copy myRank only, myRank[%u].", myRank_);
    if (threads.empty()) {
        HCCL_ERROR("[AllGatherMeshTemplate][PostCopy] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams_.dataType];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = tempAlgParams_.tailCount * dataTypeSize;
    // myRank 的数据在 ranksForInputData 中（AllGather 下 ranksForInputData={myRank}）。
    // 尾块语义与 PreCopy 对齐：最后一个 input rank 承担尾块。
    const size_t lastInputIdx = tempAlgParams_.ranksForInputData.empty() ? 0 :
                                tempAlgParams_.ranksForInputData.size() - 1;
    for (size_t idx = 0; idx < tempAlgParams_.ranksForInputData.size(); ++idx) {
        u32 rank = tempAlgParams_.ranksForInputData[idx];
        u64 curSliceSize = (tailSize > 0 && idx == lastInputIdx) ? sliceSize + tailSize : sliceSize;
        if (curSliceSize == 0) {
            continue;
        }
        const u64 sliceCount = curSliceSize / dataTypeSize;
        // ccl[myRank] -> output[myRank slot]：output 按 rankId 布局
        const u64 cclOff = tempAlgParams_.sliceOffset + static_cast<u64>(rank) * tempAlgParams_.scratchStride;
        const u64 outOff = tempAlgParams_.dataOffset + tempAlgParams_.sliceOffset +
                           static_cast<u64>(rank) * tempAlgParams_.dataStride;
        DataSlice srcSlice(tempAlgParams_.cclBufferPtr, cclOff, curSliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.outputBufferPtr, outOff, curSliceSize, sliceCount);
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
