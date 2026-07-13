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

namespace ops_hccl {

HcclResult RunNhrReduceScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                               const std::vector<u32> &ranks, u32 myRank,
                               std::vector<SendRecvInfo> &sendRecvInfos)
{
    sendRecvInfos.clear();

    u32 rankSize = static_cast<u32>(ranks.size());
    if (rankSize <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }
    u32 myRankIdx = 0;
    for (u32 i = 0; i < rankSize; ++i) {
        if (ranks[i] == myRank) {
            myRankIdx = i;
            break;
        }
    }
    HcclDataType dataType = tempAlgParams.dataType;
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    u64 tailSize = tempAlgParams.tailCount * dataTypeSize;
    u64 sliceOffset = tempAlgParams.sliceOffset;
    u64 stride = tempAlgParams.stride;
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }

    u32 channelsPerRank = 1;
    if (!templateResource.channels.empty()) {
        channelsPerRank = static_cast<u32>(templateResource.channels.begin()->second.size());
    }

    for (u32 channelIdx = 0; channelIdx < channelsPerRank; ++channelIdx) {
        for (u32 step = 0; step < nSteps; ++step) {
            u32 delta = 1 << step;
            u32 sendToIdx = (myRankIdx + rankSize - delta) % rankSize;
            u32 recvFromIdx = (myRankIdx + delta) % rankSize;
            u32 deltaSliceIndex = 1 << (step + 1);
            u32 nSlices = (rankSize - 1 + delta) / deltaSliceIndex;
            u32 txSliceIdx = sendToIdx;
            u32 rxSliceIdx = myRankIdx;

            const ChannelInfo &linkSend = templateResource.channels.at(ranks[sendToIdx])[channelIdx];
            const ChannelInfo &linkRecv = templateResource.channels.at(ranks[recvFromIdx])[channelIdx];

            std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;
            for (u32 i = 0; i < nSlices; ++i) {
                bool txTail = (txSliceIdx == rankSize - 1 && tailSize > 0);
                bool rxTail = (rxSliceIdx == rankSize - 1 && tailSize > 0);
                u64 txSz = txTail ? tailSize : sliceSize;
                u64 rxSz = rxTail ? tailSize : sliceSize;
                u64 txOff = sliceOffset + txSliceIdx * stride;
                u64 rxOff = sliceOffset + rxSliceIdx * stride;
                txSrc.emplace_back(tempAlgParams.cclBufferPtr, txOff, txSz, txSz / dataTypeSize);
                txDst.emplace_back(linkSend.remoteCclMem.addr, txOff, txSz, txSz / dataTypeSize);
                rxSrc.emplace_back(linkRecv.remoteCclMem.addr, rxOff, rxSz, rxSz / dataTypeSize);
                rxDst.emplace_back(tempAlgParams.cclBufferPtr, rxOff, rxSz, rxSz / dataTypeSize);
                txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
                rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
            }

            TxRxSlicesList sendRecvSlicesList({txSrc, txDst}, {rxSrc, rxDst});
            TxRxChannels sendRecvChannels(linkSend, linkRecv);
            sendRecvInfos.emplace_back(sendRecvChannels, sendRecvSlicesList, dataType);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunNhrAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                           const std::vector<u32> &ranks, u32 myRank,
                           std::vector<SendRecvInfo> &sendRecvInfos)
{
    sendRecvInfos.clear();

    u32 rankSize = static_cast<u32>(ranks.size());
    if (rankSize <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }
    u32 myRankIdx = 0;
    for (u32 i = 0; i < rankSize; ++i) {
        if (ranks[i] == myRank) {
            myRankIdx = i;
            break;
        }
    }
    HcclDataType dataType = tempAlgParams.dataType;
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    u64 tailSize = tempAlgParams.tailCount * dataTypeSize;
    u64 sliceOffset = tempAlgParams.sliceOffset;
    u64 stride = tempAlgParams.stride;
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }

    u32 channelsPerRank = 1;
    if (!templateResource.channels.empty()) {
        channelsPerRank = static_cast<u32>(templateResource.channels.begin()->second.size());
    }

    for (u32 channelIdx = 0; channelIdx < channelsPerRank; ++channelIdx) {
        for (u32 step = 0; step < nSteps; ++step) {
            u32 delta = 1 << (nSteps - 1 - step);
            u32 sendToIdx = (myRankIdx + delta) % rankSize;
            u32 recvFromIdx = (myRankIdx + rankSize - delta) % rankSize;
            u32 deltaSliceIndex = 1 << (nSteps - step);
            u32 nSlices = (rankSize - 1 + delta) / deltaSliceIndex;
            u32 txSliceIdx = myRankIdx;
            u32 rxSliceIdx = (myRankIdx + rankSize - delta) % rankSize;

            const ChannelInfo &linkSend = templateResource.channels.at(ranks[sendToIdx])[channelIdx];
            const ChannelInfo &linkRecv = templateResource.channels.at(ranks[recvFromIdx])[channelIdx];

            std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;
            for (u32 i = 0; i < nSlices; ++i) {
                bool txTail = (txSliceIdx == rankSize - 1 && tailSize > 0);
                bool rxTail = (rxSliceIdx == rankSize - 1 && tailSize > 0);
                u64 txSz = txTail ? tailSize : sliceSize;
                u64 rxSz = rxTail ? tailSize : sliceSize;
                u64 txOff = sliceOffset + txSliceIdx * stride;
                u64 rxOff = sliceOffset + rxSliceIdx * stride;
                txSrc.emplace_back(tempAlgParams.cclBufferPtr, txOff, txSz, txSz / dataTypeSize);
                txDst.emplace_back(linkSend.remoteCclMem.addr, txOff, txSz, txSz / dataTypeSize);
                rxSrc.emplace_back(linkRecv.remoteCclMem.addr, rxOff, rxSz, rxSz / dataTypeSize);
                rxDst.emplace_back(tempAlgParams.cclBufferPtr, rxOff, rxSz, rxSz / dataTypeSize);
                txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
                rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
            }

            TxRxSlicesList sendRecvSlicesList({txSrc, txDst}, {rxSrc, rxDst});
            TxRxChannels sendRecvChannels(linkSend, linkRecv);
            sendRecvInfos.emplace_back(sendRecvChannels, sendRecvSlicesList, dataType);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunNhrScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                         const std::vector<u32> &ranks, u32 myRank,
                         std::vector<SendRecvInfo> &sendRecvInfos)
{
    sendRecvInfos.clear();

    u32 rankSize = static_cast<u32>(ranks.size());
    if (rankSize <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }
    u32 myRankIdx = 0;
    u32 rootIdx = 0;
    for (u32 i = 0; i < rankSize; ++i) {
        if (ranks[i] == myRank) {
            myRankIdx = i;
        }
        if (ranks[i] == tempAlgParams.root) {
            rootIdx = i;
        }
    }
    HcclDataType dataType = tempAlgParams.dataType;
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    u64 sliceOffset = tempAlgParams.sliceOffset;
    u64 stride = tempAlgParams.stride;
    u32 myRel = (myRankIdx + rankSize - rootIdx) % rankSize;
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }

    u32 channelsPerRank = 1;
    if (!templateResource.channels.empty()) {
        channelsPerRank = static_cast<u32>(templateResource.channels.begin()->second.size());
    }

    for (u32 channelIdx = 0; channelIdx < channelsPerRank; ++channelIdx) {
        for (u32 step = 0; step < nSteps; ++step) {
            u32 delta = 1 << (nSteps - 1 - step);
            u32 sendToIdx = ((myRel + delta) % rankSize + rootIdx) % rankSize;
            u32 recvFromIdx = ((myRel + rankSize - delta) % rankSize + rootIdx) % rankSize;

            const ChannelInfo &linkSend = templateResource.channels.at(ranks[sendToIdx])[channelIdx];
            const ChannelInfo &linkRecv = templateResource.channels.at(ranks[recvFromIdx])[channelIdx];
            u64 off = sliceOffset + myRel * stride;
            u64 sz = sliceSize;

            std::vector<DataSlice> txSrc{{tempAlgParams.cclBufferPtr, off, sz, sz / dataTypeSize}};
            std::vector<DataSlice> txDst{{linkSend.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
            std::vector<DataSlice> rxSrc{{linkRecv.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
            std::vector<DataSlice> rxDst{{tempAlgParams.cclBufferPtr, off, sz, sz / dataTypeSize}};

            TxRxSlicesList sendRecvSlicesList({txSrc, txDst}, {rxSrc, rxDst});
            TxRxChannels sendRecvChannels(linkSend, linkRecv);
            sendRecvInfos.emplace_back(sendRecvChannels, sendRecvSlicesList, dataType);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunNhrBarrier(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                         const std::vector<u32> &ranks, u32 myRank,
                         std::vector<SendRecvInfo> &sendRecvInfos)
{
    sendRecvInfos.clear();

    u32 rankSize = static_cast<u32>(ranks.size());
    if (rankSize <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }
    u32 myRankIdx = 0;
    for (u32 i = 0; i < rankSize; ++i) {
        if (ranks[i] == myRank) {
            myRankIdx = i;
            break;
        }
    }
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    std::vector<DataSlice> empty;

    for (u32 step = 0; step < nSteps; ++step) {
        u32 delta = 1 << step;
        u32 peerIdx = (myRankIdx + delta) % rankSize;
        const ChannelInfo &link = templateResource.channels.at(ranks[peerIdx])[0];
        TxRxSlicesList sendRecvSlicesList({empty, empty}, {empty, empty});
        TxRxChannels sendRecvChannels(link, link);
        sendRecvInfos.emplace_back(sendRecvChannels, sendRecvSlicesList, tempAlgParams.dataType);
    }
    return HCCL_SUCCESS;
}

}
