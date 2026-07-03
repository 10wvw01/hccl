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
#include "../../executor/base_executor.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

namespace {

bool IsPcieProtocol(const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    for (const auto &rankChannels : channels) {
        if (!rankChannels.second.empty() &&
            rankChannels.second[0].protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
            return true;
        }
    }
    return false;
}

}  // namespace

HcclResult RunMeshAllGather(const ::TemplateDataParam &tempAlgParams, TemplateResource &templateResource,
                            const std::vector<u32> &ranks, u32 myRank)
{
    if (ranks.size() <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }

    const u32 rankSize = static_cast<u32>(ranks.size());
    u32 myAlgRank = 0;
    bool rankFound = false;
    for (u32 rankIdx = 0; rankIdx < rankSize; ++rankIdx) {
        if (ranks[rankIdx] == myRank) {
            myAlgRank = rankIdx;
            rankFound = true;
            break;
        }
    }
    CHK_PRT_RET(!rankFound, HCCL_ERROR("[RunMeshAllGather] rank[%u] is not in ranks.", myRank), HCCL_E_PARA);

    const HcclDataType dataType = tempAlgParams.dataType;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    const bool variableCount = (tempAlgParams.sliceMode == TemplateDataSliceMode::VARIABLE_COUNT);
    CHK_PRT_RET(variableCount && tempAlgParams.rankSliceCounts.size() < rankSize,
                HCCL_ERROR("[RunMeshAllGather] rankSliceCounts size[%u] is smaller than rankSize[%u].",
                           static_cast<u32>(tempAlgParams.rankSliceCounts.size()), rankSize),
                HCCL_E_PARA);

    const u32 blockNum = tempAlgParams.ranksForInputData.empty() ?
        1 : static_cast<u32>(tempAlgParams.ranksForInputData.size());
    const u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    const u64 tailSize = tempAlgParams.tailCount * dataTypeSize;

    for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
        u32 threadIdx = 0;
        const u32 inputBlockIdx = tempAlgParams.ranksForInputData.empty() ?
            blockIdx : (tempAlgParams.ranksForInputData[blockIdx] % blockNum);
        for (u32 i = 1; i < rankSize; ++i) {
            const u32 connectedAlgRank = (myAlgRank + i) % rankSize;
            const u32 connectedRank = ranks[connectedAlgRank];
            CHK_PRT_RET(templateResource.channels.count(connectedRank) == 0 ||
                            templateResource.channels.at(connectedRank).empty(),
                        HCCL_ERROR("[RunMeshAllGather] connectedRank[%u] has no link.", connectedRank),
                        HCCL_E_PARA);
            CHK_PRT_RET(threadIdx >= templateResource.threads.size(),
                        HCCL_ERROR("[RunMeshAllGather] invalid transfer task."), HCCL_E_PARA);
            const ChannelInfo &linkRemote = templateResource.channels.at(connectedRank)[0];

            // AllGather 的 scratch 采用 rank-major 布局：
            // slot = algRank * blockNum + ranksForInputData[blockIdx] % blockNum。
            // 这样同一个 rank 的所有 input block 连续摆放，替代旧 stride 语义。
            u64 txRankOffset = (static_cast<u64>(myAlgRank) * blockNum + inputBlockIdx) * sliceSize;
            u64 rxRankOffset = (static_cast<u64>(connectedAlgRank) * blockNum + inputBlockIdx) * sliceSize;
            u64 txSliceSize = (myAlgRank == rankSize - 1 && tailSize > 0) ? tailSize : sliceSize;
            u64 rxSliceSize = (connectedAlgRank == rankSize - 1 && tailSize > 0) ? tailSize : sliceSize;
            u64 txSliceCount = txSliceSize / dataTypeSize;
            u64 rxSliceCount = rxSliceSize / dataTypeSize;
            if (variableCount) {
                txRankOffset = 0;
                rxRankOffset = 0;
                for (u32 algRank = 0; algRank < myAlgRank; ++algRank) {
                    txRankOffset += tempAlgParams.rankSliceCounts[algRank] * blockNum * dataTypeSize;
                }
                for (u32 algRank = 0; algRank < connectedAlgRank; ++algRank) {
                    rxRankOffset += tempAlgParams.rankSliceCounts[algRank] * blockNum * dataTypeSize;
                }
                txSliceCount = tempAlgParams.rankSliceCounts[myAlgRank];
                rxSliceCount = tempAlgParams.rankSliceCounts[connectedAlgRank];
                txSliceSize = txSliceCount * dataTypeSize;
                rxSliceSize = rxSliceCount * dataTypeSize;
                txRankOffset += static_cast<u64>(inputBlockIdx) * txSliceSize;
                rxRankOffset += static_cast<u64>(inputBlockIdx) * rxSliceSize;
            }

            const u64 txScratchOffset = tempAlgParams.cclBufferOffset + txRankOffset;
            const u64 rxScratchOffset = tempAlgParams.cclBufferOffset + rxRankOffset;
            std::vector<DataSlice> txSrcSlices{
                DataSlice(tempAlgParams.cclBufferPtr, txScratchOffset, txSliceSize, txSliceCount)};
            std::vector<DataSlice> txDstSlices{
                DataSlice(linkRemote.remoteCclMem.addr, txScratchOffset, txSliceSize, txSliceCount)};
            std::vector<DataSlice> rxSrcSlices{
                DataSlice(linkRemote.remoteCclMem.addr, rxScratchOffset, rxSliceSize, rxSliceCount)};
            std::vector<DataSlice> rxDstSlices{
                DataSlice(tempAlgParams.cclBufferPtr, rxScratchOffset, rxSliceSize, rxSliceCount)};
            SendRecvInfo sendRecvInfo{{linkRemote, linkRemote},
                                      {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}},
                                      dataType};
            if (variableCount) {
                CHK_RET(SendRecvWrite(sendRecvInfo, templateResource.threads[threadIdx]));
            } else {
                CHK_RET(SendRecvRead(sendRecvInfo, templateResource.threads[threadIdx]));
            }
            ++threadIdx;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshReduceScatter(const ::TemplateDataParam &tempAlgParams, TemplateResource &templateResource,
                                const std::vector<u32> &ranks, u32 myRank)
{
    if (ranks.size() <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }

    const u32 rankSize = static_cast<u32>(ranks.size());
    CHK_PRT_RET(!tempAlgParams.ranksForInputData.empty() &&
                    tempAlgParams.ranksForInputData.size() % rankSize != 0,
                HCCL_ERROR("[RunMeshReduceScatter] ranksForInputData size[%u] is not aligned to rankSize[%u].",
                           static_cast<u32>(tempAlgParams.ranksForInputData.size()), rankSize),
                HCCL_E_PARA);

    u32 myAlgRank = 0;
    bool rankFound = false;
    for (u32 rankIdx = 0; rankIdx < rankSize; ++rankIdx) {
        if (ranks[rankIdx] == myRank) {
            myAlgRank = rankIdx;
            rankFound = true;
            break;
        }
    }
    CHK_PRT_RET(!rankFound, HCCL_ERROR("[RunMeshReduceScatter] rank[%u] is not in ranks.", myRank), HCCL_E_PARA);

    const HcclDataType dataType = tempAlgParams.dataType;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    const bool variableCount = (tempAlgParams.sliceMode == TemplateDataSliceMode::VARIABLE_COUNT);
    CHK_PRT_RET(variableCount && tempAlgParams.rankSliceCounts.size() < rankSize,
                HCCL_ERROR("[RunMeshReduceScatter] rankSliceCounts size[%u] is smaller than rankSize[%u].",
                           static_cast<u32>(tempAlgParams.rankSliceCounts.size()), rankSize),
                HCCL_E_PARA);

    bool isDmaRead = false;
    for (const auto &rankChannels : templateResource.channels) {
        if (!rankChannels.second.empty() &&
            rankChannels.second[0].protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
            isDmaRead = true;
            break;
        }
    }
    const u32 groupNum = tempAlgParams.ranksForInputData.empty() ?
        1 : static_cast<u32>(tempAlgParams.ranksForInputData.size() / rankSize);
    const u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    const u64 tailSize = tempAlgParams.tailCount * dataTypeSize;

    for (u32 groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
        u32 threadIdx = 0;
        const u32 myInputIdx = tempAlgParams.ranksForInputData.empty() ?
            groupIdx : (tempAlgParams.ranksForInputData[groupIdx * rankSize + myAlgRank] % groupNum);
        for (u32 i = 1; i < rankSize; ++i) {
            const u32 connectedAlgRank = (myAlgRank + i) % rankSize;
            const u32 connectedRank = ranks[connectedAlgRank];
            CHK_PRT_RET(templateResource.channels.count(connectedRank) == 0 ||
                            templateResource.channels.at(connectedRank).empty(),
                        HCCL_ERROR("[RunMeshReduceScatter] connectedRank[%u] has no link.", connectedRank),
                        HCCL_E_PARA);
            CHK_PRT_RET(threadIdx >= templateResource.threads.size(),
                        HCCL_ERROR("[RunMeshReduceScatter] invalid transfer task."), HCCL_E_PARA);
            const ChannelInfo &linkRemote = templateResource.channels.at(connectedRank)[0];

            // ReduceScatter/Scatter 类输入是 N * groupNum 个分片，输出只保留 myAlgRank 对应分片。
            // scratch 同样按 rank-major 布局，组内下标来自 ranksForInputData。
            const u32 connectedInputIdx = tempAlgParams.ranksForInputData.empty() ?
                groupIdx : (tempAlgParams.ranksForInputData[groupIdx * rankSize + connectedAlgRank] % groupNum);
            u64 myRankOffset = (static_cast<u64>(myAlgRank) * groupNum + myInputIdx) * sliceSize;
            u64 connectedRankOffset = (static_cast<u64>(connectedAlgRank) * groupNum + connectedInputIdx) * sliceSize;
            u64 txSliceSize = (connectedAlgRank == rankSize - 1 && tailSize > 0) ? tailSize : sliceSize;
            u64 rxSliceSize = (myAlgRank == rankSize - 1 && tailSize > 0) ? tailSize : sliceSize;
            u64 txSliceCount = txSliceSize / dataTypeSize;
            u64 rxSliceCount = rxSliceSize / dataTypeSize;
            if (variableCount) {
                myRankOffset = 0;
                connectedRankOffset = 0;
                for (u32 algRank = 0; algRank < myAlgRank; ++algRank) {
                    myRankOffset += tempAlgParams.rankSliceCounts[algRank] * groupNum * dataTypeSize;
                }
                for (u32 algRank = 0; algRank < connectedAlgRank; ++algRank) {
                    connectedRankOffset += tempAlgParams.rankSliceCounts[algRank] * groupNum * dataTypeSize;
                }
                txSliceCount = tempAlgParams.rankSliceCounts[connectedAlgRank];
                rxSliceCount = tempAlgParams.rankSliceCounts[myAlgRank];
                txSliceSize = txSliceCount * dataTypeSize;
                rxSliceSize = rxSliceCount * dataTypeSize;
                myRankOffset += static_cast<u64>(myInputIdx) * rxSliceSize;
                connectedRankOffset += static_cast<u64>(connectedInputIdx) * txSliceSize;
            }

            const u64 txSrcOffset = tempAlgParams.cclBufferOffset + connectedRankOffset;
            const u64 txDstOffset = tempAlgParams.cclBufferOffset + myRankOffset;
            const u64 rxSrcOffset = tempAlgParams.cclBufferOffset + myRankOffset;
            const u64 rxDstOffset = tempAlgParams.cclBufferOffset + myRankOffset;
            std::vector<DataSlice> txSrcSlices{
                DataSlice(tempAlgParams.cclBufferPtr, txSrcOffset, txSliceSize, txSliceCount)};
            std::vector<DataSlice> txDstSlices{
                DataSlice(linkRemote.remoteCclMem.addr, txDstOffset, txSliceSize, txSliceCount)};
            std::vector<DataSlice> rxSrcSlices{
                DataSlice(linkRemote.remoteCclMem.addr, rxSrcOffset, rxSliceSize, rxSliceCount)};
            std::vector<DataSlice> rxDstSlices{
                DataSlice(tempAlgParams.cclBufferPtr, rxDstOffset, rxSliceSize, rxSliceCount)};
            SendRecvReduceInfo sendRecvInfo{{linkRemote, linkRemote},
                                            {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}},
                                            dataType, tempAlgParams.reduceOp};
            CHK_RET(isDmaRead ? SendRecvReadReduce(sendRecvInfo, templateResource.threads[threadIdx]) :
                                SendRecvBatchWriteReduce(sendRecvInfo, templateResource.threads[threadIdx]));
            ++threadIdx;
        }
    }
    return HCCL_SUCCESS;
}
HcclResult RunMeshScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                          EngineType engineType, const std::vector<u32> &ranks, u32 myRank)
{
    (void)engineType;
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
    u64 sliceSize = tempAlgParams.sliceSize;
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u64 base = tempAlgParams.buffInfo.hcclBuffBaseOff;

    const std::vector<ChannelInfo> &portGroup = templateResource.channels.begin()->second;
    u32 channelsPerRank = static_cast<u32>(portGroup.size());
    std::vector<u64> ec, sizeOut, elemOffset;
    CHK_RET(CalcDataSplitByPortGroupCommon(sliceSize / dataTypeSize, dataTypeSize, portGroup, ec, sizeOut, elemOffset, channelsPerRank));

    u32 t = 0;
    if (myRankIdx == rootIdx) {
        for (u32 r = 0; r < rankSize; ++r) {
            if (r == rootIdx) {
                continue;
            }
            for (u32 ch = 0; ch < channelsPerRank; ++ch) {
                const ChannelInfo &link = templateResource.channels.at(ranks[r])[ch];
                u64 off = base + sliceSize * r + elemOffset[ch];
                u64 sz = sizeOut[ch];
                std::vector<DataSlice> txSrc{{tempAlgParams.buffInfo.inputPtr, off, sz, sz / dataTypeSize}};
                std::vector<DataSlice> txDst{{link.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
                std::vector<DataSlice> empty;
                SendRecvInfo info{{link, link}, {{txSrc, txDst}, {empty, empty}}, dataType};
                CHK_RET(isDmaRead ? SendRecvRead(info, templateResource.threads[t]) :
                                    SendRecvBatchWrite(info, templateResource.threads[t]));
                t++;
            }
        }
    } else {
        for (u32 ch = 0; ch < channelsPerRank; ++ch) {
            const ChannelInfo &link = templateResource.channels.at(ranks[rootIdx])[ch];
            u64 off = base + sliceSize * myRankIdx + elemOffset[ch];
            u64 sz = sizeOut[ch];
            std::vector<DataSlice> rxSrc{{link.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
            std::vector<DataSlice> rxDst{{tempAlgParams.buffInfo.outputPtr, off, sz, sz / dataTypeSize}};
            std::vector<DataSlice> empty;
            SendRecvInfo info{{link, link}, {{empty, empty}, {rxSrc, rxDst}}, dataType};
            CHK_RET(isDmaRead ? SendRecvRead(info, templateResource.threads[t]) :
                                SendRecvBatchWrite(info, templateResource.threads[t]));
            t++;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                         EngineType engineType, const std::vector<u32> &ranks, u32 myRank)
{
    (void)engineType;
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
    u64 sliceSize = tempAlgParams.sliceSize;
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u64 base = tempAlgParams.buffInfo.hcclBuffBaseOff;

    const std::vector<ChannelInfo> &portGroup = templateResource.channels.begin()->second;
    u32 channelsPerRank = static_cast<u32>(portGroup.size());
    std::vector<u64> ec, sizeOut, elemOffset;
    CHK_RET(CalcDataSplitByPortGroupCommon(sliceSize / dataTypeSize, dataTypeSize, portGroup, ec, sizeOut, elemOffset, channelsPerRank));

    u32 t = 0;
    if (myRankIdx == rootIdx) {
        for (u32 r = 0; r < rankSize; ++r) {
            if (r == rootIdx) {
                continue;
            }
            for (u32 ch = 0; ch < channelsPerRank; ++ch) {
                const ChannelInfo &link = templateResource.channels.at(ranks[r])[ch];
                u64 off = base + sliceSize * r + elemOffset[ch];
                u64 sz = sizeOut[ch];
                std::vector<DataSlice> rxSrc{{link.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
                std::vector<DataSlice> rxDst{{tempAlgParams.buffInfo.outputPtr, off, sz, sz / dataTypeSize}};
                std::vector<DataSlice> empty;
                SendRecvInfo info{{link, link}, {{empty, empty}, {rxSrc, rxDst}}, dataType};
                CHK_RET(isDmaRead ? SendRecvRead(info, templateResource.threads[t]) :
                                    SendRecvBatchWrite(info, templateResource.threads[t]));
                t++;
            }
        }
    } else {
        for (u32 ch = 0; ch < channelsPerRank; ++ch) {
            const ChannelInfo &link = templateResource.channels.at(ranks[rootIdx])[ch];
            u64 off = base + sliceSize * myRankIdx + elemOffset[ch];
            u64 sz = sizeOut[ch];
            std::vector<DataSlice> txSrc{{tempAlgParams.buffInfo.inputPtr, off, sz, sz / dataTypeSize}};
            std::vector<DataSlice> txDst{{link.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
            std::vector<DataSlice> empty;
            SendRecvInfo info{{link, link}, {{txSrc, txDst}, {empty, empty}}, dataType};
            CHK_RET(isDmaRead ? SendRecvRead(info, templateResource.threads[t]) :
                                SendRecvBatchWrite(info, templateResource.threads[t]));
            t++;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshAllToAll(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                           EngineType engineType, const std::vector<u32> &ranks, u32 myRank)
{
    (void)engineType;
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
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u32 channelsPerRank = static_cast<u32>(templateResource.channels.begin()->second.size());

    u32 t = 0;
    for (u32 i = 1; i < rankSize; ++i) {
        u32 peer = ranks[(myRankIdx + i) % rankSize];
        std::vector<u64> sendEc, sendSz, sendOff;
        std::vector<u64> recvEc, recvSz, recvOff;
        const std::vector<ChannelInfo> &peerCh = templateResource.channels.at(peer);
        CHK_RET(CalcDataSplitByPortGroupCommon(tempAlgParams.sendCounts[peer], dataTypeSize, peerCh, sendEc, sendSz, sendOff, channelsPerRank));
        CHK_RET(CalcDataSplitByPortGroupCommon(tempAlgParams.recvCounts[peer], dataTypeSize, peerCh, recvEc, recvSz, recvOff, channelsPerRank));
        for (u32 ch = 0; ch < channelsPerRank; ++ch) {
            const ChannelInfo &link = peerCh[ch];
            u64 txOff = tempAlgParams.sdispls[peer] * dataTypeSize + sendOff[ch];
            u64 rxOff = tempAlgParams.rdispls[peer] * dataTypeSize + recvOff[ch];
            std::vector<DataSlice> txSrc{{tempAlgParams.buffInfo.inputPtr, txOff, sendSz[ch], sendEc[ch]}};
            std::vector<DataSlice> txDst{{link.remoteCclMem.addr, txOff, sendSz[ch], sendEc[ch]}};
            std::vector<DataSlice> rxSrc{{link.remoteCclMem.addr, rxOff, recvSz[ch], recvEc[ch]}};
            std::vector<DataSlice> rxDst{{tempAlgParams.buffInfo.outputPtr, rxOff, recvSz[ch], recvEc[ch]}};
            SendRecvInfo info{{link, link}, {{txSrc, txDst}, {rxSrc, rxDst}}, dataType};
            CHK_RET(isDmaRead ? SendRecvRead(info, templateResource.threads[t]) :
                                SendRecvBatchWrite(info, templateResource.threads[t]));
            t++;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshBarrier(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                          EngineType engineType, const std::vector<u32> &ranks, u32 myRank)
{
    (void)engineType;
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
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    std::vector<DataSlice> empty;

    u32 t = 0;
    for (u32 i = 1; i < rankSize; ++i) {
        u32 peerIdx = (myRankIdx + i) % rankSize;
        const ChannelInfo &link = templateResource.channels.at(ranks[peerIdx])[0];
        SendRecvInfo info{{link, link}, {{empty, empty}, {empty, empty}}, tempAlgParams.dataType};
        CHK_RET(isDmaRead ? SendRecvRead(info, templateResource.threads[t]) :
                            SendRecvBatchWrite(info, templateResource.threads[t]));
        t++;
    }
    return HCCL_SUCCESS;
}

}
