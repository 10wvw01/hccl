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
                            const std::vector<u32> &ranks, u32 myRank, std::vector<u32> &ranksForOutputData,
                            std::vector<SendRecvInfo> &sendRecvInfos)
{
    sendRecvInfos.clear();
    if (ranks.size() <= 1 || templateResource.channels.empty()) {
        // 没有实际 peer 通信时，输出数据归属不变。
        ranksForOutputData = tempAlgParams.ranksForInputData;
        return HCCL_SUCCESS;
    }

    ranksForOutputData.clear();

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

    std::vector<u32> ranksForInputData = tempAlgParams.ranksForInputData;
    if (ranksForInputData.empty()) {
        // 兼容上层暂未传 ranksForInputData 的场景：默认当前 scratch 只有本 rank 一块数据。
        ranksForInputData.emplace_back(myRank);
    }
    const u32 blockNum = static_cast<u32>(ranksForInputData.size());
    // ranksForOutputData 描述本轮 AllGather 后，本地 scratch 每个连续 block 属于哪个真实 rank。
    // 布局按环状 self-first 展开：每个 rank 视角下自己的 block group 永远在最前。
    ranksForOutputData.resize(static_cast<size_t>(rankSize) * blockNum);
    for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
        ranksForOutputData[blockIdx] = ranksForInputData[blockIdx];
    }

    // 环状 self-first 布局：自己固定在 0 号区域，后面按 ranks 环状顺序排列。
    // 例如 rank0 视角是 0,1,2,3；rank2 视角是 2,3,0,1。
    auto getPosition = [&](u32 algRank, u32 selfAlgRank) -> u32 {
        return (algRank + rankSize - selfAlgRank) % rankSize;
    };

    auto getAlgRankByRankId = [&](u32 rankId, u32 &algRank) -> HcclResult {
        for (u32 rankIdx = 0; rankIdx < rankSize; ++rankIdx) {
            if (ranks[rankIdx] == rankId) {
                algRank = rankIdx;
                return HCCL_SUCCESS;
            }
        }
        HCCL_ERROR("[RunMeshAllGather] data rank[%u] is not in ranks.", rankId);
        return HCCL_E_PARA;
    };

    auto inferDataRank = [&](u32 blockIdx, u32 ownerAlgRank, u32 &dataRank) -> HcclResult {
        // ranksForInputData 是“我视角”下当前已有 block 的归属。
        // 推导 peer 视角的同位置 block 时，不能用 global rank 直接加减；
        // 需要先转成 algRank，再按 ownerAlgRank 相对 myAlgRank 的环状距离旋转。
        u32 dataAlgRank = 0;
        CHK_RET(getAlgRankByRankId(ranksForInputData[blockIdx], dataAlgRank));
        const u32 algRankDistance = getPosition(ownerAlgRank, myAlgRank);
        const u32 inferredAlgRank = (dataAlgRank + algRankDistance) % rankSize;
        dataRank = ranks[inferredAlgRank];
        return HCCL_SUCCESS;
    };

    auto getBlockSize = [&](u32 dataRank, u64 &blockSize) -> HcclResult {
        // 当前只处理定长场景：大多数 rank 使用 sliceCount；最后一个 rank 可用 tailCount 表示尾块。
        u32 dataAlgRank = 0;
        CHK_RET(getAlgRankByRankId(dataRank, dataAlgRank));
        u64 blockCount = tempAlgParams.sliceCount;
        if (tempAlgParams.tailCount > 0 && dataAlgRank == rankSize - 1) {
            blockCount = tempAlgParams.tailCount;
        }
        blockSize = blockCount * dataTypeSize;
        return HCCL_SUCCESS;
    };

    auto getGroupSize = [&](u32 ownerAlgRank, u64 &groupSize) -> HcclResult {
        // 一个 group 表示某个 owner rank 当前已经拥有的整段 block。
        // AllGather 和 peer 通信时搬的是整段 group，不是单个 block。
        groupSize = 0;
        for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
            u32 dataRank = 0;
            u64 blockSize = 0;
            CHK_RET(inferDataRank(blockIdx, ownerAlgRank, dataRank));
            CHK_RET(getBlockSize(dataRank, blockSize));
            groupSize += blockSize;
        }
        return HCCL_SUCCESS;
    };

    auto getAlgRankByPosition = [&](u32 position, u32 selfAlgRank) -> u32 {
        return (selfAlgRank + position) % rankSize;
    };

    auto getGroupOffset = [&](u32 ownerAlgRank, u32 layoutSelfAlgRank, u64 &groupOffset) -> HcclResult {
        // 计算 ownerAlgRank 这一整段 group 在 layoutSelfAlgRank 视角的 scratch 起始偏移。
        // 因为 tail block 可能让最后一个 rank 大小不同，所以这里仍按前面 group 的实际大小累加。
        groupOffset = 0;
        const u32 ownerPosition = getPosition(ownerAlgRank, layoutSelfAlgRank);
        for (u32 position = 0; position < ownerPosition; ++position) {
            const u32 prevOwnerAlgRank = getAlgRankByPosition(position, layoutSelfAlgRank);
            u64 prevGroupSize = 0;
            CHK_RET(getGroupSize(prevOwnerAlgRank, prevGroupSize));
            groupOffset += prevGroupSize;
        }
        return HCCL_SUCCESS;
    };

    u32 threadIdx = 0;
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

        u64 txGroupSize = 0;
        u64 rxGroupSize = 0;
        u64 txSrcOffset = 0;
        u64 txDstOffset = 0;
        u64 rxSrcOffset = 0;
        u64 rxDstOffset = 0;
        CHK_RET(getGroupSize(myAlgRank, txGroupSize));
        CHK_RET(getGroupSize(connectedAlgRank, rxGroupSize));
        CHK_RET(getGroupOffset(myAlgRank, myAlgRank, txSrcOffset));
        CHK_RET(getGroupOffset(myAlgRank, connectedAlgRank, txDstOffset));
        CHK_RET(getGroupOffset(connectedAlgRank, connectedAlgRank, rxSrcOffset));
        CHK_RET(getGroupOffset(connectedAlgRank, myAlgRank, rxDstOffset));

        // 新布局不再使用 outputSliceStride/repeatStride。
        // 每个 repeat 对应当前 scratch 中一个连续 block，ranksForInputData 描述这些 block 的归属和顺序；
        // AllGather 对一个 peer 通信时，直接把当前已有 block 组成的整段数据搬到对端/本端对应区域。
        txSrcOffset += tempAlgParams.cclBufferOffset;
        txDstOffset += tempAlgParams.cclBufferOffset;
        rxSrcOffset += tempAlgParams.cclBufferOffset;
        rxDstOffset += tempAlgParams.cclBufferOffset;

        // 四组 DataSlice 与旧模板含义一致：
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

        const u32 position = getPosition(connectedAlgRank, myAlgRank);
        for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
            // 同步更新输出归属表。这里的 slot 必须和 rxDst 的物理布局一致：
            // peer 的 group 放在 position 区域，group 内 block 顺序沿用对端当前已有顺序。
            u32 dataRank = 0;
            CHK_RET(inferDataRank(blockIdx, connectedAlgRank, dataRank));
            const u32 slot = position * blockNum + blockIdx;
            ranksForOutputData[slot] = dataRank;
        }

        // primitive 只把四组 slice 打包成 SendRecvInfo 返回给模板；
        // 实际 SendRecvRead/Write 由模板按 sendRecvInfos 顺序选择线程执行。
        SendRecvInfo sendRecvInfo{{linkRemote, linkRemote},
                                  {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}},
                                  dataType};
        sendRecvInfos.emplace_back(sendRecvInfo);
        ++threadIdx;
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
