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
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

HcclResult RunMeshAllGather(const ::TemplateDataParam &tempAlgParams, TemplateResource &templateResource,
                            const std::vector<u32> &ranks, u32 myRank, std::vector<u32> &ranksForOutputData,
                            std::vector<SendRecvInfo> &sendRecvInfos)
{
    sendRecvInfos.clear();
    if (ranks.size() <= 1 || templateResource.channels.empty()) {
        // 无 peer 通信时，输出数据归属不变。
        ranksForOutputData = tempAlgParams.ranksForInputData;
        return HCCL_SUCCESS;
    }

    const u32 rankSize = static_cast<u32>(ranks.size());
    u32 myAlgRank = 0;
    bool rankFound = false;
    for (u32 i = 0; i < rankSize; ++i) {
        if (ranks[i] == myRank) {
            myAlgRank = i;
            rankFound = true;
            break;
        }
    }
    CHK_PRT_RET(!rankFound, HCCL_ERROR("[RunMeshAllGather] rank[%u] is not in ranks.", myRank), HCCL_E_PARA);

    const HcclDataType dataType = tempAlgParams.dataType;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];

    std::vector<u32> ranksForInputData = tempAlgParams.ranksForInputData;
    if (ranksForInputData.empty()) {
        // 兼容上层暂未传 ranksForInputData：默认当前 scratch 只有本 rank 一块数据。
        ranksForInputData.emplace_back(myRank);
    }
    const u32 blockNum = static_cast<u32>(ranksForInputData.size());

    // self-first 环状布局：本 rank group 固定在 position 0，其余按 algRank 环状排列。
    // 整段 group append：与 peer 通信时搬整段 group，不要求全局严格环状物理布局，
    // PostCopy 按 ranksForOutputData 解释 scratch 实际顺序。
    auto positionOf = [rankSize, myAlgRank](u32 algRank) -> u32 {
        return (algRank + rankSize - myAlgRank) % rankSize;
    };
    // 前置不变量：ranksForInputData 的每个 rankId 都在 ranks 内（scratch 只持有通信域内 rank 的数据）。
    auto algRankOfRankId = [&](u32 rankId) -> u32 {
        for (u32 i = 0; i < rankSize; ++i) {
            if (ranks[i] == rankId) {
                return i;
            }
        }
        return rankSize; // 不应到达
    };
    // 推导 ownerAlgRank 的 group 内第 blockIdx 块的真实 rank 归属。
    // 依整段 append 的同构平移假设：peer 同位置 block 是本 rank 该 block 按 position 距离旋转后的 rank。
    auto inferDataRank = [&](u32 blockIdx, u32 ownerAlgRank) -> u32 {
        const u32 dataAlgRank = algRankOfRankId(ranksForInputData[blockIdx]);
        const u32 dist = positionOf(ownerAlgRank);
        return ranks[(dataAlgRank + dist) % rankSize];
    };
    // 定长 + 尾块：最后一个 algRank 的数据块用 tailCount，其余用 sliceCount。
    auto blockSizeOf = [&](u32 dataRank) -> u64 {
        const u64 cnt = (tempAlgParams.tailCount > 0 && algRankOfRankId(dataRank) == rankSize - 1)
                            ? tempAlgParams.tailCount
                            : tempAlgParams.sliceCount;
        return cnt * dataTypeSize;
    };

    // 预计算每个 owner algRank 的 group 总大小（视角无关，依赖同构平移假设）。
    std::vector<u64> groupSizeForOwner(rankSize, 0);
    for (u32 a = 0; a < rankSize; ++a) {
        for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
            groupSizeForOwner[a] += blockSizeOf(inferDataRank(blockIdx, a));
        }
    }
    // algRank 顺序全局前缀和，用于 O(1) 环状区间求和（替代旧的 O(rankSize²) 逐 peer 重算）。
    std::vector<u64> prefix(rankSize + 1, 0);
    for (u32 a = 0; a < rankSize; ++a) {
        prefix[a + 1] = prefix[a] + groupSizeForOwner[a];
    }
    auto cyclicSum = [&](u32 start, u32 len) -> u64 {
        const u32 end = start + len;
        if (end <= rankSize) {
            return prefix[end] - prefix[start];
        }
        return (prefix[rankSize] - prefix[start]) + prefix[end - rankSize];
    };
    // layoutSelf 视角下 ownerAlgRank 的 group 起始偏移 = 该视角下 position(owner) 之前的 group 累计大小。
    auto groupOffsetIn = [&](u32 ownerAlgRank, u32 layoutSelfAlgRank) -> u64 {
        return cyclicSum(layoutSelfAlgRank, (ownerAlgRank + rankSize - layoutSelfAlgRank) % rankSize);
    };

    // ranksForOutputData：position 0 区段是本 rank 自己的 group，顺序沿用 ranksForInputData。
    ranksForOutputData.resize(static_cast<size_t>(rankSize) * blockNum);
    for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
        ranksForOutputData[blockIdx] = ranksForInputData[blockIdx];
    }

    for (u32 i = 1; i < rankSize; ++i) {
        const u32 connectedAlgRank = (myAlgRank + i) % rankSize;
        const u32 connectedRank = ranks[connectedAlgRank];
        CHK_PRT_RET(templateResource.channels.count(connectedRank) == 0 ||
                        templateResource.channels.at(connectedRank).empty(),
                    HCCL_ERROR("[RunMeshAllGather] connectedRank[%u] has no link.", connectedRank), HCCL_E_PARA);
        const ChannelInfo &linkRemote = templateResource.channels.at(connectedRank)[0];

        const u64 txGroupSize = groupSizeForOwner[myAlgRank];
        const u64 rxGroupSize = groupSizeForOwner[connectedAlgRank];
        // 本 rank group 在自己视角 position 0；对端 group 在对端视角 position 0。
        const u64 txSrcOffset = tempAlgParams.cclBufferOffset;
        const u64 rxSrcOffset = tempAlgParams.cclBufferOffset;
        // 本 rank group 落到对端 scratch 的 position(myAlgRank, connectedAlgRank)；
        // 对端 group 落到本 rank scratch 的 position(connectedAlgRank, myAlgRank)。
        const u64 txDstOffset = tempAlgParams.cclBufferOffset + groupOffsetIn(myAlgRank, connectedAlgRank);
        const u64 rxDstOffset = tempAlgParams.cclBufferOffset + groupOffsetIn(connectedAlgRank, myAlgRank);

        // 四组 DataSlice 含义：
        // txSrc：本地 scratch 当前已有 group；txDst：写到对端 scratch 中“我”对应的位置。
        // rxSrc：对端 scratch 当前已有 group；rxDst：写到本地 scratch 中“peer”对应的位置。
        std::vector<DataSlice> txSrcSlices{
            DataSlice(tempAlgParams.cclBufferPtr, txSrcOffset, txGroupSize, txGroupSize / dataTypeSize)};
        std::vector<DataSlice> txDstSlices{
            DataSlice(linkRemote.remoteCclMem.addr, txDstOffset, txGroupSize, txGroupSize / dataTypeSize)};
        std::vector<DataSlice> rxSrcSlices{
            DataSlice(linkRemote.remoteCclMem.addr, rxSrcOffset, rxGroupSize, rxGroupSize / dataTypeSize)};
        std::vector<DataSlice> rxDstSlices{
            DataSlice(tempAlgParams.cclBufferPtr, rxDstOffset, rxGroupSize, rxGroupSize / dataTypeSize)};

        // 同步 ranksForOutputData：对端 group 落在 position(connectedAlgRank) 区段，
        // group 内 block 顺序沿用对端当前已有顺序（由 inferDataRank 推导）。slot 与 rxDst 物理布局一致。
        const u32 peerPos = positionOf(connectedAlgRank);
        for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
            ranksForOutputData[peerPos * blockNum + blockIdx] = inferDataRank(blockIdx, connectedAlgRank);
        }

        // primitive 只产描述符，实际 SendRecvRead/Write 由 template 遍历 sendRecvInfos 执行。
        sendRecvInfos.emplace_back(SendRecvInfo{{linkRemote, linkRemote},
                                                {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}},
                                                dataType});
    }
    return HCCL_SUCCESS;
}

#if 0 // 以下算子暂缓：待定长 RunMeshAllGather 样例审过后依次迁移
// 恢复时需一并恢复 IsPcieProtocol helper（原位于本文件匿名 namespace）。
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

#endif // 以下算子暂缓：待定长 RunMeshAllGather 样例审过后依次迁移（恢复时需一并恢复 IsPcieProtocol helper）

}
