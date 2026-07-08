/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mesh_primitives.h"
#include "../../executor/ops_executor.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

namespace {

u64 CalcRankDataOffset(u64 baseOffset, const std::vector<u64> &rankDataSizeByAlgRank, u32 layoutAlgRank,
                       u32 dataAlgRank)
{
    const u32 rankSize = static_cast<u32>(rankDataSizeByAlgRank.size());
    const u32 dataPosition = (dataAlgRank + rankSize - layoutAlgRank) % rankSize;
    u64 offset = baseOffset;
    for (u32 position = 0; position < dataPosition; ++position) {
        const u32 rankIdx = (layoutAlgRank + position) % rankSize;
        offset += rankDataSizeByAlgRank[rankIdx];
    }
    return offset;
}

} // namespace

HcclResult RunMeshAllGather(const TemplateDataParam &tempAlgParams, TemplateResource &templateResource,
                            const std::vector<u32> &ranks, u32 myRank, std::vector<u32> &ranksForOutputData,
                            std::vector<SendRecvInfo> &sendRecvInfos)
{
    sendRecvInfos.clear();
    if (ranks.size() <= 1 || templateResource.channels.empty()) {
        ranksForOutputData = tempAlgParams.ranksForInputData;
        return HCCL_SUCCESS;
    }

    const u32 rankSize = static_cast<u32>(ranks.size());
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank, ranks, myAlgRank));

    const HcclDataType dataType = tempAlgParams.dataType;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];

    std::vector<u32> ranksForInputData = tempAlgParams.ranksForInputData;
    if (ranksForInputData.empty()) {
        ranksForInputData.emplace_back(myRank);
    }
    const u32 blockNum = static_cast<u32>(ranksForInputData.size());

    // 当前 scratch 中的 block 归属用 rankId 表示，计算环状布局时先转换为当前通信域内的 algRank。
    std::vector<u32> algRanksForInputData;
    algRanksForInputData.reserve(blockNum);
    for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
        u32 dataAlgRank = 0;
        CHK_RET(GetAlgRank(ranksForInputData[blockIdx], ranks, dataAlgRank));
        algRanksForInputData.emplace_back(dataAlgRank);
    }

    // 每个 algRank 当前持有一组 block；tailCount 只影响最后一个 algRank 对应 block 的字节数。
    std::vector<u64> rankDataSizeByAlgRank(rankSize, 0);
    for (u32 dataAlgRank = 0; dataAlgRank < rankSize; ++dataAlgRank) {
        const u32 dataPosition = (dataAlgRank + rankSize - myAlgRank) % rankSize;
        for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
            const u32 blockDataAlgRank = (algRanksForInputData[blockIdx] + dataPosition) % rankSize;
            const u64 blockCount = (tempAlgParams.tailCount > 0 && blockDataAlgRank == rankSize - 1) ?
                tempAlgParams.tailCount : tempAlgParams.sliceCount;
            rankDataSizeByAlgRank[dataAlgRank] += blockCount * dataTypeSize;
        }
    }

    ranksForOutputData.resize(static_cast<size_t>(rankSize) * blockNum);
    for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
        ranksForOutputData[blockIdx] = ranksForInputData[blockIdx];
    }

    for (u32 threadIdx = 0; threadIdx < rankSize - 1; ++threadIdx) {
        const u32 connectedAlgRank = (myAlgRank + 1 + threadIdx) % rankSize;
        const u32 connectedRank = ranks[connectedAlgRank];
        CHK_PRT_RET(templateResource.channels.count(connectedRank) == 0 ||
                        templateResource.channels.at(connectedRank).empty(),
                    HCCL_ERROR("[RunMeshAllGather] connectedRank[%u] has no link.", connectedRank), HCCL_E_PARA);
        const ChannelInfo &linkRemote = templateResource.channels.at(connectedRank)[0];

        const u64 txDataSize = rankDataSizeByAlgRank[myAlgRank];
        const u64 rxDataSize = rankDataSizeByAlgRank[connectedAlgRank];
        const u64 txSliceCount = txDataSize / dataTypeSize;
        const u64 rxSliceCount = rxDataSize / dataTypeSize;
        const u64 txSrcOffset = tempAlgParams.cclBufferOffset;
        const u64 rxSrcOffset = tempAlgParams.cclBufferOffset;
        const u64 txDstOffset = CalcRankDataOffset(tempAlgParams.cclBufferOffset, rankDataSizeByAlgRank,
                                                   connectedAlgRank, myAlgRank);
        const u64 rxDstOffset = CalcRankDataOffset(tempAlgParams.cclBufferOffset, rankDataSizeByAlgRank,
                                                   myAlgRank, connectedAlgRank);
        const u32 connectedDataPosition = (connectedAlgRank + rankSize - myAlgRank) % rankSize;

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;

        txSrcSlicesAll.emplace_back(tempAlgParams.cclBufferPtr, txSrcOffset, txDataSize, txSliceCount);
        txDstSlicesAll.emplace_back(linkRemote.remoteCclMem.addr, txDstOffset, txDataSize, txSliceCount);
        rxSrcSlicesAll.emplace_back(linkRemote.remoteCclMem.addr, rxSrcOffset, rxDataSize, rxSliceCount);
        rxDstSlicesAll.emplace_back(tempAlgParams.cclBufferPtr, rxDstOffset, rxDataSize, rxSliceCount);

        HCCL_DEBUG("[RunMeshAllGather] rankId [%u] connectedRank [%u] threadIdx [%u] txSrcSlices: "
                   "offset[%llu] sliceSize[%llu] count[%llu].",
                   myRank, connectedRank, threadIdx, txSrcOffset, txDataSize, txSliceCount);
        HCCL_DEBUG("[RunMeshAllGather] rankId [%u] connectedRank [%u] threadIdx [%u] txDstSlices: "
                   "offset[%llu] sliceSize[%llu] count[%llu].",
                   myRank, connectedRank, threadIdx, txDstOffset, txDataSize, txSliceCount);
        HCCL_DEBUG("[RunMeshAllGather] rankId [%u] connectedRank [%u] threadIdx [%u] rxSrcSlices: "
                   "offset[%llu] sliceSize[%llu] count[%llu].",
                   myRank, connectedRank, threadIdx, rxSrcOffset, rxDataSize, rxSliceCount);
        HCCL_DEBUG("[RunMeshAllGather] rankId [%u] connectedRank [%u] threadIdx [%u] rxDstSlices: "
                   "offset[%llu] sliceSize[%llu] count[%llu].",
                   myRank, connectedRank, threadIdx, rxDstOffset, rxDataSize, rxSliceCount);

        // ranksForOutputData 必须和 rxDst 的物理写入位置保持一致，供 PostCopy 解释 scratch 布局。
        for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
            const u32 dataAlgRank = (algRanksForInputData[blockIdx] + connectedDataPosition) % rankSize;
            ranksForOutputData[connectedDataPosition * blockNum + blockIdx] = ranks[dataAlgRank];
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        sendRecvInfos.emplace_back(sendRecvChannels, sendRecvSlicesList, dataType);
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
