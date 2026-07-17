/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nhr_primitives.h"
#include "base_template.h"
#include "utils/utils.h"

#include <algorithm>
#include <limits>

namespace ops_hccl {

namespace {

    // 根据当前输入数据来源和目标 rank 的全局偏移，生成该目标 rank 当前持有的数据来源列表。
    inline HcclResult GetNhrConnectedInputRanks(
        const std::vector<u32> &ranksForInputData, long long connectedOffset, std::vector<u32> &connectedInputRanks)
    {
        connectedInputRanks.clear();
        connectedInputRanks.reserve(ranksForInputData.size());
        for (u32 rankId : ranksForInputData) {
            const long long connectedRankId = static_cast<long long>(rankId) + connectedOffset;
            CHK_PRT_RET(
                connectedRankId < 0 || connectedRankId > static_cast<long long>(std::numeric_limits<u32>::max()),
                HCCL_ERROR("[RunNhrAllGather] connectedRankId is invalid, rankId=%u, connectedOffset=%lld.", rankId,
                    connectedOffset),
                HCCL_E_PARA);
            HCCL_DEBUG("[RunNhrAllGather] GetNhrConnectedInputRanks: rankId=%u, connectedOffset=%lld, "
                       "connectedRankId=%lld",
                rankId, connectedOffset, connectedRankId);
            connectedInputRanks.emplace_back(static_cast<u32>(connectedRankId));
        }
        return HCCL_SUCCESS;
    }

    // 按全局 rankId 槽位追加 NHR AllGather 的一组 src/dst DataSlice。
    inline void AddNhrRankDataSlices(const TemplateDataParams &tempAlgParams, const std::vector<u32> &rankIds,
        u32 tailRankId, NhrAllGatherSlicePair &slicePair)
    {
        const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
        const u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
        const u64 tailSize = tempAlgParams.tailCount * dataTypeSize;
        for (u32 rankId : rankIds) {
            const u64 offset = tempAlgParams.sliceOffset + static_cast<u64>(rankId) * tempAlgParams.scratchStride;
            const u64 dataSize = (tailSize > 0 && rankId == tailRankId) ? tailSize : sliceSize;
            HCCL_DEBUG("[RunNhrAllGather] AddNhrRankDataSlices: rankId=%u, sliceOffset=%lu, stride=%lu, "
                       "offset=%lu, size=%lu, count=%lu",
                rankId, tempAlgParams.sliceOffset, tempAlgParams.scratchStride, offset, dataSize,
                dataSize / dataTypeSize);
            slicePair.srcSlices.emplace_back(slicePair.srcPtr, offset, dataSize, dataSize / dataTypeSize);
            slicePair.dstSlices.emplace_back(slicePair.dstPtr, offset, dataSize, dataSize / dataTypeSize);
        }
    }

} // namespace

HcclResult RunNhrAllGather(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks, u32 myRank,
    std::vector<u32> &ranksForOutputData, std::vector<TxRxSlicesList> &txRxSlicesLists)
{
    txRxSlicesLists.clear();
    CHK_RET(CheckInputDataRanks(tempAlgParams, "RunNhrAllGather"));

    u32 rankSize = static_cast<u32>(ranks.size());
    if (rankSize <= 1) {
        ranksForOutputData = tempAlgParams.ranksForInputData;
        HCCL_INFO("[RunNhrAllGather] no sendRecv needed, ranksForOutputDataNum=%zu", ranksForOutputData.size());
        return HCCL_SUCCESS;
    }
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank, ranks, myAlgRank));

    HcclDataType dataType = tempAlgParams.dataType;
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    u64 tailSize = tempAlgParams.tailCount * dataTypeSize;
    std::vector<u32> ranksForInputData = tempAlgParams.ranksForInputData;
    u32 nSteps = 0;
    // 计算ceil(log2(rankSize))
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_INFO("[RunNhrAllGather] myAlgRank=%u, dataTypeSize=%u, sliceSize=%lu, tailSize=%lu, nSteps=%u", myAlgRank,
        dataTypeSize, sliceSize, tailSize, nSteps);

    ranksForOutputData = ranksForInputData;
    for (u32 step = 0; step < nSteps; ++step) {
        // delta 从 rankSize/2 开始，每次减半，直到 1
        u32 delta = 1 << (nSteps - 1 - step);
        u32 sendToAlgRank = (myAlgRank + delta) % rankSize;
        u32 recvFromAlgRank = (myAlgRank + rankSize - delta) % rankSize;
        // 每个rank上的数据来源的间隔步长
        u32 algRankStep = 1 << (nSteps - step);
        u32 nSlices = (rankSize - 1 + delta) / algRankStep;
        HCCL_INFO("[RunNhrAllGather] step=%u, delta=%u, sendToAlgRank=%u, sendToRank=%u, "
                  "recvFromAlgRank=%u, recvFromRank=%u, algRankStep=%u, nSlices=%u",
            step, delta, sendToAlgRank, ranks[sendToAlgRank], recvFromAlgRank, ranks[recvFromAlgRank], algRankStep,
            nSlices);

        std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;
        // tx: 本端持有的数据发给对端；rx: 对端持有的数据发给本端。
        u32 txAlgRank = myAlgRank;
        u32 rxAlgRank = recvFromAlgRank;
        std::vector<u32> txRankIds;
        std::vector<u32> rxRankIds;
        const u32 tailRankId = ranks[rankSize - 1];
        for (u32 i = 0; i < nSlices; ++i) {
            const long long txConnectedOffset
                = static_cast<long long>(ranks[txAlgRank]) - static_cast<long long>(myRank);
            const long long rxConnectedOffset
                = static_cast<long long>(ranks[rxAlgRank]) - static_cast<long long>(myRank);
            CHK_RET(GetNhrConnectedInputRanks(ranksForInputData, txConnectedOffset, txRankIds));
            CHK_RET(GetNhrConnectedInputRanks(ranksForInputData, rxConnectedOffset, rxRankIds));
            HCCL_DEBUG("[RunNhrAllGather] slice=%u, txAlgRank=%u, txConnectedOffset=%lld, "
                       "rxAlgRank=%u, rxConnectedOffset=%lld, txRankNum=%zu, rxRankNum=%zu",
                i, txAlgRank, txConnectedOffset, rxAlgRank, rxConnectedOffset, txRankIds.size(), rxRankIds.size());
            NhrAllGatherSlicePair txSlicePair{tempAlgParams.cclBufferPtr, nullptr, txSrc, txDst};
            AddNhrRankDataSlices(tempAlgParams, txRankIds, tailRankId, txSlicePair);
            NhrAllGatherSlicePair rxSlicePair{nullptr, tempAlgParams.cclBufferPtr, rxSrc, rxDst};
            AddNhrRankDataSlices(tempAlgParams, rxRankIds, tailRankId, rxSlicePair);
            ranksForOutputData.insert(ranksForOutputData.end(), rxRankIds.begin(), rxRankIds.end());
            txAlgRank = (txAlgRank + rankSize - algRankStep) % rankSize;
            rxAlgRank = (rxAlgRank + rankSize - algRankStep) % rankSize;
        }
        txRxSlicesLists.emplace_back(
            SlicesList(txSrc, txDst), SlicesList(rxSrc, rxDst), ranks[recvFromAlgRank], ranks[sendToAlgRank]);
    }
    std::sort(ranksForOutputData.begin(), ranksForOutputData.end());
    HCCL_INFO("[RunNhrAllGather] end: txRxSlicesListNum=%zu, ranksForOutputDataNum=%zu", txRxSlicesLists.size(),
        ranksForOutputData.size());
    return HCCL_SUCCESS;
}

HcclResult RunNhrReduceScatter(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks, u32 myRank,
    std::vector<u32> &ranksForOutputData, std::vector<TxRxSlicesList> &txRxSlicesLists)
{
    txRxSlicesLists.clear();
    CHK_RET(CheckInputDataRanks(tempAlgParams, "RunNhrReduceScatter"));

    const u32 rankSize = static_cast<u32>(ranks.size());
    // ReduceScatter 语义：本 rank 最终只持有自己负责归约的那一块，输出归属恒为 myRank。
    ranksForOutputData = {myRank};
    if (rankSize <= 1) {
        HCCL_INFO("[RunNhrReduceScatter] no sendRecv needed, ranksForOutputDataNum=%zu", ranksForOutputData.size());
        return HCCL_SUCCESS;
    }
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank, ranks, myAlgRank));

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    const u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    const u64 tailSize = tempAlgParams.tailCount * dataTypeSize;
    const u32 tailRankId = ranks[rankSize - 1];
    u32 nSteps = 0;
    // 计算ceil(log2(rankSize))
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_INFO("[RunNhrReduceScatter] myAlgRank=%u, rankSize=%u, dataTypeSize=%u, sliceSize=%lu, tailSize=%lu, "
              "nSteps=%u",
        myAlgRank, rankSize, dataTypeSize, sliceSize, tailSize, nSteps);

    for (u32 step = 0; step < nSteps; ++step) {
        // ReduceScatter 与 AllGather 相反：delta 从 1 开始每步翻倍，直到 rankSize/2
        const u32 delta = 1 << step;
        const u32 sendToAlgRank = (myAlgRank + rankSize - delta) % rankSize;
        const u32 recvFromAlgRank = (myAlgRank + delta) % rankSize;
        // 本步参与收发的槽位间隔，随步数翻倍
        const u32 algRankStep = 1 << (step + 1);
        const u32 nSlices = (rankSize - 1 + delta) / algRankStep;
        HCCL_INFO("[RunNhrReduceScatter] step=%u, delta=%u, sendToAlgRank=%u, sendToRank=%u, "
                  "recvFromAlgRank=%u, recvFromRank=%u, algRankStep=%u, nSlices=%u",
            step, delta, sendToAlgRank, ranks[sendToAlgRank], recvFromAlgRank, ranks[recvFromAlgRank], algRankStep,
            nSlices);

        // tx: 发送对端归约集合所需的槽位（从 sendToAlgRank 起、间隔 algRankStep）；
        // rx: 对端数据落入本端归约集合槽位（从 myAlgRank 起、间隔 algRankStep），由执行层与本端数据归约。
        std::vector<u32> txRankIds;
        std::vector<u32> rxRankIds;
        u32 txAlgRank = sendToAlgRank;
        u32 rxAlgRank = myAlgRank;
        for (u32 i = 0; i < nSlices; ++i) {
            txRankIds.emplace_back(ranks[txAlgRank]);
            rxRankIds.emplace_back(ranks[rxAlgRank]);
            txAlgRank = (txAlgRank + rankSize - algRankStep) % rankSize;
            rxAlgRank = (rxAlgRank + rankSize - algRankStep) % rankSize;
        }

        // 槽位按全局 rankId 寻址：offset = sliceOffset + rankId * scratchStride；远端地址由执行层按 channel 解析。
        std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;
        NhrAllGatherSlicePair txSlicePair{tempAlgParams.cclBufferPtr, nullptr, txSrc, txDst};
        AddNhrRankDataSlices(tempAlgParams, txRankIds, tailRankId, txSlicePair);
        NhrAllGatherSlicePair rxSlicePair{nullptr, tempAlgParams.cclBufferPtr, rxSrc, rxDst};
        AddNhrRankDataSlices(tempAlgParams, rxRankIds, tailRankId, rxSlicePair);
        txRxSlicesLists.emplace_back(SlicesList(txSrc, txDst), SlicesList(rxSrc, rxDst), ranks[recvFromAlgRank],
            ranks[sendToAlgRank]);
        HCCL_INFO("[RunNhrReduceScatter] Build TxRxSlicesList: step=%u, txRankNum=%zu, rxRankNum=%zu, "
                  "txRxSlicesListNum=%zu",
            step, txRankIds.size(), rxRankIds.size(), txRxSlicesLists.size());
    }
    HCCL_INFO("[RunNhrReduceScatter] end: txRxSlicesListNum=%zu", txRxSlicesLists.size());
    return HCCL_SUCCESS;
}

HcclResult RunNhrScatter(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks, u32 myRank,
    std::vector<u32> &ranksForOutputData, std::vector<TxRxSlicesList> &txRxSlicesLists)
{
    txRxSlicesLists.clear();
    CHK_RET(CheckInputDataRanks(tempAlgParams, "RunNhrScatter"));

    u32 rankSize = static_cast<u32>(ranks.size());
    // Scatter outputs only the local rank block, so PostCopy only needs myRank.
    ranksForOutputData = {myRank};
    if (rankSize <= 1) {
        HCCL_INFO("[RunNhrScatter] no sendRecv needed, ranksForOutputDataNum=%zu", ranksForOutputData.size());
        return HCCL_SUCCESS;
    }
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank, ranks, myAlgRank));
    u32 rootAlgRank = 0;
    CHK_RET(GetAlgRank(tempAlgParams.root, ranks, rootAlgRank));

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    const u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    const u64 tailSize = tempAlgParams.tailCount * dataTypeSize;
    const u32 tailRankId = ranks[rankSize - 1];
    // myRelRoot 是在以 root 为基准的排序中的 rank 位置
    const u32 myRelRoot = (myAlgRank + rankSize - rootAlgRank) % rankSize;
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }

    for (u32 step = 0; step < nSteps; ++step) {
        const u32 delta = 1U << (nSteps - 1 - step);
        // 每一步都将以 root 为基准的 ranks 分为左半部分和右半部分。
        const u32 groupSize = delta << 1;
        const u32 groupOffset = myRelRoot % groupSize;

        std::vector<DataSlice> txSrc;
        std::vector<DataSlice> txDst;
        std::vector<DataSlice> rxSrc;
        std::vector<DataSlice> rxDst;
        // 左半部分的领头 rank 将所有右半部分的目标数据块发送给右半部分的领头 rank
        if (groupOffset == 0 && myRelRoot + delta < rankSize) {
            const u32 sendToRel = myRelRoot + delta;
            const u32 sendToAlgRank = (rootAlgRank + sendToRel) % rankSize;
            // 最后一组可能不完整，限制在 rankSize 范围内
            const u32 lastRel = std::min(myRelRoot + groupSize, rankSize);
            std::vector<u32> txRankIds;
            for (u32 rel = sendToRel; rel < lastRel; ++rel) {
                txRankIds.emplace_back(ranks[(rootAlgRank + rel) % rankSize]);
            }
            // txRankIds 是对端子树所拥有的所有数据块；偏移量仍使用全局 rank 槽位
            NhrAllGatherSlicePair txSlicePair{tempAlgParams.cclBufferPtr, nullptr, txSrc, txDst};
            AddNhrRankDataSlices(tempAlgParams, txRankIds, tailRankId, txSlicePair);
            txRxSlicesLists.emplace_back(SlicesList(txSrc, txDst), SlicesList({}, {}),
                ranks[sendToAlgRank], ranks[sendToAlgRank]);
            HCCL_INFO("[RunNhrScatter] Build tx TxRxSlicesList: step=%u, remoteRank=%u, txRankNum=%zu",
                step, ranks[sendToAlgRank], txRankIds.size());
            continue;
        }
        // 右半部分的领头 rank 从左半部分的领头 rank 接收其子树的数据块
        if (groupOffset == delta) {
            const u32 recvFromRel = myRelRoot - delta;
            const u32 recvFromAlgRank = (rootAlgRank + recvFromRel) % rankSize;
            const u32 lastRel = std::min(myRelRoot + delta, rankSize);
            std::vector<u32> rxRankIds;
            for (u32 rel = myRelRoot; rel < lastRel; ++rel) {
                rxRankIds.emplace_back(ranks[(rootAlgRank + rel) % rankSize]);
            }
            // rxRankIds 是本子树所拥有的数据块；后续步骤可能会转发其中的一部分
            NhrAllGatherSlicePair rxSlicePair{nullptr, tempAlgParams.cclBufferPtr, rxSrc, rxDst};
            AddNhrRankDataSlices(tempAlgParams, rxRankIds, tailRankId, rxSlicePair);
            txRxSlicesLists.emplace_back(SlicesList({}, {}), SlicesList(rxSrc, rxDst),
                ranks[recvFromAlgRank], ranks[recvFromAlgRank]);
            HCCL_INFO("[RunNhrScatter] Build rx TxRxSlicesList: step=%u, remoteRank=%u, rxRankNum=%zu",
                step, ranks[recvFromAlgRank], rxRankIds.size());
        }
    }
    // Scatter 语义：输出仅对应本 rank。
    ranksForOutputData = {myRank};
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
