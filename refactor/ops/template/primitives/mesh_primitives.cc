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

// 按环状 scratch 布局计算 dataAlgRank 这段数据在 dstAlgRank 下的起始偏移。
u64 CalcDataOffsetForScratch(u64 baseOffset, const std::vector<u64> &rankDataSizeByAlgRank, u32 dstAlgRank, u32 dataAlgRank)
{
    const u32 rankSize = static_cast<u32>(rankDataSizeByAlgRank.size());
    const u32 dataPosition = (dataAlgRank + rankSize - dstAlgRank) % rankSize;
    u64 offset = baseOffset;
    for (u32 position = 0; position < dataPosition; ++position) {
        const u32 rankIdx = (dstAlgRank + position) % rankSize;
        offset += rankDataSizeByAlgRank[rankIdx];
    }
    return offset;
}

//  rankId 转换成 algRank。
HcclResult GetInputDataAlgRanks(const std::vector<u32> &ranksForInputData, const std::vector<u32> &ranks,
                                std::vector<u32> &algRanksForInputData)
{
    algRanksForInputData.clear();
    algRanksForInputData.reserve(ranksForInputData.size());
    for (u32 blockIdx = 0; blockIdx < ranksForInputData.size(); ++blockIdx) {
        u32 dataAlgRank = 0;
        CHK_RET(GetAlgRank(ranksForInputData[blockIdx], ranks, dataAlgRank));
        algRanksForInputData.emplace_back(dataAlgRank);
    }
    return HCCL_SUCCESS;
}

// 计算每个 algRank 数据总字节数。
inline void CalcRankDataSize(const TemplateDataParam &tempAlgParams, const std::vector<u32> &algRanksForInputData,
                             u32 rankSize, u32 myAlgRank, u32 dataTypeSize,
                             std::vector<u64> &rankDataSizeByAlgRank)
{
    rankDataSizeByAlgRank.assign(rankSize, 0);
    for (u32 dataAlgRank = 0; dataAlgRank < rankSize; ++dataAlgRank) {
        const u32 dataPosition = (dataAlgRank + rankSize - myAlgRank) % rankSize;
        for (u32 blockIdx = 0; blockIdx < algRanksForInputData.size(); ++blockIdx) {
            const u32 blockDataAlgRank = (algRanksForInputData[blockIdx] + dataPosition) % rankSize;
            const u64 blockCount = (tempAlgParams.tailCount > 0 && blockDataAlgRank == rankSize - 1) ?
                tempAlgParams.tailCount : tempAlgParams.sliceCount;
            rankDataSizeByAlgRank[dataAlgRank] += blockCount * dataTypeSize;
        }
    }
}

// 初始化输出数据归属表，position 0 固定保留本 rank 当前已有数据。
inline void InitOutputDataRanks(const std::vector<u32> &ranksForInputData, u32 rankSize,
                                std::vector<u32> &ranksForOutputData)
{
    ranksForOutputData.resize(static_cast<size_t>(rankSize) * ranksForInputData.size());
    for (u32 blockIdx = 0; blockIdx < ranksForInputData.size(); ++blockIdx) {
        ranksForOutputData[blockIdx] = ranksForInputData[blockIdx];
    }
}

inline void UpdateOutputDataRanks(const std::vector<u32> &ranks, const std::vector<u32> &algRanksForInputData,
                                  u32 connectedDataPosition, std::vector<u32> &ranksForOutputData)
{
    const u32 rankSize = static_cast<u32>(ranks.size());
    const u32 blockNum = static_cast<u32>(algRanksForInputData.size());
    for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
        const u32 dataAlgRank = (algRanksForInputData[blockIdx] + connectedDataPosition) % rankSize;
        ranksForOutputData[connectedDataPosition * blockNum + blockIdx] = ranks[dataAlgRank];
    }
}

// 获取对端的通信链路。
inline HcclResult GetConnectedLink(TemplateResource &templateResource, const std::vector<u32> &ranks,
                                   u32 connectedAlgRank, const ChannelInfo *&linkRemote)
{
    const u32 connectedRank = ranks[connectedAlgRank];
    CHK_PRT_RET(templateResource.channels.count(connectedRank) == 0 ||
                    templateResource.channels.at(connectedRank).empty(),
                HCCL_ERROR("[RunMeshAllGather] connectedRank[%u] has no link.", connectedRank), HCCL_E_PARA);
    linkRemote = &templateResource.channels.at(connectedRank)[0];
    return HCCL_SUCCESS;
}

// 组装四组 DataSlice 
inline void GetSendRecvInfo(const TemplateDataParam &tempAlgParams, const ChannelInfo &linkRemote, u64 txDataSize,
                              u64 rxDataSize, u64 txDstOffset, u64 rxDstOffset, HcclDataType dataType,
                              u32 dataTypeSize, std::vector<SendRecvInfo> &sendRecvInfos)
{
    const u64 txSliceCount = txDataSize / dataTypeSize;
    const u64 rxSliceCount = rxDataSize / dataTypeSize;
    std::vector<DataSlice> txSrcSlicesAll;
    std::vector<DataSlice> txDstSlicesAll;
    std::vector<DataSlice> rxSrcSlicesAll;
    std::vector<DataSlice> rxDstSlicesAll;

    txSrcSlicesAll.emplace_back(tempAlgParams.cclBufferPtr, tempAlgParams.cclBufferOffset, txDataSize, txSliceCount);
    txDstSlicesAll.emplace_back(linkRemote.remoteCclMem.addr, txDstOffset, txDataSize, txSliceCount);
    rxSrcSlicesAll.emplace_back(linkRemote.remoteCclMem.addr, tempAlgParams.cclBufferOffset, rxDataSize, rxSliceCount);
    rxDstSlicesAll.emplace_back(tempAlgParams.cclBufferPtr, rxDstOffset, rxDataSize, rxSliceCount);

    TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
    TxRxChannels sendRecvChannels(linkRemote, linkRemote);
    sendRecvInfos.emplace_back(sendRecvChannels, sendRecvSlicesList, dataType);
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

    std::vector<u32> ranksForInputData = tempAlgParams.ranksForInputData;
    if (ranksForInputData.empty()) {
        ranksForInputData.emplace_back(myRank);
    }

    std::vector<u32> algRanksForInputData;
    CHK_RET(GetInputDataAlgRanks(ranksForInputData, ranks, algRanksForInputData));

    const HcclDataType dataType = tempAlgParams.dataType;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    std::vector<u64> rankDataSizeByAlgRank;
    CalcRankDataSize(tempAlgParams, algRanksForInputData, rankSize, myAlgRank, dataTypeSize,
                     rankDataSizeByAlgRank);
    InitOutputDataRanks(ranksForInputData, rankSize, ranksForOutputData);

    for (u32 threadIdx = 0; threadIdx < rankSize - 1; ++threadIdx) {
        const u32 connectedAlgRank = (myAlgRank + 1 + threadIdx) % rankSize;
        const u32 connectedRank = ranks[connectedAlgRank];
        const ChannelInfo *linkRemote = nullptr;
        CHK_RET(GetConnectedLink(templateResource, ranks, connectedAlgRank, linkRemote));

        const u64 txDataSize = rankDataSizeByAlgRank[myAlgRank];
        const u64 rxDataSize = rankDataSizeByAlgRank[connectedAlgRank];
        const u64 txDstOffset = CalcDataOffsetForScratch(tempAlgParams.cclBufferOffset, rankDataSizeByAlgRank, connectedAlgRank,
                                               myAlgRank);
        const u64 rxDstOffset = CalcDataOffsetForScratch(tempAlgParams.cclBufferOffset, rankDataSizeByAlgRank, myAlgRank,
                                               connectedAlgRank);
        GetSendRecvInfo(tempAlgParams, *linkRemote, txDataSize, rxDataSize, txDstOffset, rxDstOffset, dataType,
                          dataTypeSize, sendRecvInfos);
        const u32 connectedDataPosition = (connectedAlgRank + rankSize - myAlgRank) % rankSize;
        UpdateOutputDataRanks(ranks, algRanksForInputData, connectedDataPosition, ranksForOutputData);
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
