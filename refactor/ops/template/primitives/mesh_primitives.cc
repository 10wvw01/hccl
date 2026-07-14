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
#include "base_template.h"
#include "utils/utils.h"

namespace ops_hccl {

namespace {

// 构造 Mesh AllGather 通信描述前的参数检查。
inline HcclResult PreCheckMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                                        const std::vector<u32> &ranks, u32 myRank)
{
    HCCL_INFO("[RunMeshAllGather] PreCheck: myRank=%u, rankSize=%zu, inputRankNum=%zu, channelRankNum=%zu",
              myRank, ranks.size(), tempAlgParams.ranksForInputData.size(), templateResource.channels.size());
    CHK_PRT_RET(tempAlgParams.ranksForInputData.empty(),
                HCCL_ERROR("[RunMeshAllGather] ranksForInputData is empty."), HCCL_E_PARA);
    CHK_PRT_RET(ranks.size() > 1 && templateResource.channels.empty(),
                HCCL_ERROR("[RunMeshAllGather] channels is empty."), HCCL_E_PARA);
    for (u32 rank : ranks) {
        if (rank == myRank) {
            continue;
        }
        CHK_PRT_RET(templateResource.channels.count(rank) == 0 ||
                        templateResource.channels.at(rank).empty(),
                    HCCL_ERROR("[RunMeshAllGather] connectedRank[%u] has no link.", rank), HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}


// 将输入数据来源 rankId 转成当前 mesh 通信域内的 algRank。
inline HcclResult GetAlgRanksForInputData(const std::vector<u32> &ranks,
                                          const std::vector<u32> &ranksForInputData,
                                          std::vector<u32> &algRanksForInputData)
{
    algRanksForInputData.clear();
    algRanksForInputData.reserve(ranksForInputData.size());
    for (u32 rankId : ranksForInputData) {
        u32 srcAlgRank = 0;
        CHK_RET(GetAlgRank(rankId, ranks, srcAlgRank));
        HCCL_DEBUG("[RunMeshAllGather] GetAlgRanksForInputData: rankId=%u, srcAlgRank=%u", rankId, srcAlgRank);
        algRanksForInputData.emplace_back(srcAlgRank);
    }
    return HCCL_SUCCESS;
}

// 根据本端输入数据来源和对端相对偏移，生成对端输入数据来源列表。
// 例如 ranks=[4,8,12,16]，本端已有 rankId=[4,8] 时，先转成 algRank=[0,1]；
// 若 connectedOffset=2，则对端来源为 algRank=[2,3]，再映射回 rankId=[12,16]。
inline HcclResult GetConnectedInputRanks(const std::vector<u32> &ranks,
                                         const std::vector<u32> &algRanksForInputData,
                                         u32 rankSize, u32 connectedOffset,
                                         std::vector<u32> &connectedInputRanks)
{
    connectedInputRanks.clear();
    connectedInputRanks.reserve(algRanksForInputData.size());
    for (u32 blockIdx = 0; blockIdx < algRanksForInputData.size(); ++blockIdx) {
        u32 rankId = 0;
        const u32 curAlgRank = algRanksForInputData[blockIdx];
        const u32 connectedSrcAlgRank = (curAlgRank + connectedOffset) % rankSize;
        rankId = ranks[connectedSrcAlgRank];
        HCCL_DEBUG("[RunMeshAllGather] GetConnectedInputRanks: blockIdx=%u, curAlgRank=%u, "
                   "connectedOffset=%u, rankSize=%u, connectedSrcAlgRank=%u, rankId=%u",
                   blockIdx, curAlgRank, connectedOffset, rankSize, connectedSrcAlgRank, rankId);
        connectedInputRanks.emplace_back(rankId);
    }
    return HCCL_SUCCESS;
}

// 追加两组共享 rankId 偏移规则的 DataSlice。
inline void AddRankDataSlices(const MeshAllGatherSliceInfo &sliceInfo, const std::vector<u32> &rankIds,
                              MeshAllGatherSlicePair &slicePair)
{
    for (u32 rankId : rankIds) {
        const u64 dataSize = sliceInfo.sliceSize;
        const u64 dataOffset = sliceInfo.tempAlgParams.sliceOffset + static_cast<u64>(rankId) * sliceInfo.stride;
        HCCL_DEBUG("[RunMeshAllGather] AddRankDataSlices: rankId=%u, sliceOffset=%lu, stride=%lu, "
                   "dataOffset=%lu, dataSize=%lu, sliceCount=%lu",
                   rankId, sliceInfo.tempAlgParams.sliceOffset, sliceInfo.stride, dataOffset, dataSize,
                   sliceInfo.tempAlgParams.sliceCount);
        slicePair.firstSlices.emplace_back(slicePair.firstBufferPtr, dataOffset, dataSize,
                                           sliceInfo.tempAlgParams.sliceCount);
        slicePair.secondSlices.emplace_back(slicePair.secondBufferPtr, dataOffset, dataSize,
                                            sliceInfo.tempAlgParams.sliceCount);
    }
}

// 根据 tx/rx 数据来源列表组装一个对端的 SendRecvInfo。
inline void GetSendRecvInfo(const MeshAllGatherSliceInfo &sliceInfo, const std::vector<u32> &txRankIds,
                            const std::vector<u32> &rxRankIds, std::vector<SendRecvInfo> &sendRecvInfos)
{
    std::vector<DataSlice> txSrcSlicesAll;
    std::vector<DataSlice> txDstSlicesAll;
    std::vector<DataSlice> rxSrcSlicesAll;
    std::vector<DataSlice> rxDstSlicesAll;
    // tx/rx 的本端和远端 buffer 方向相反，但使用相同的 rankId 槽位规则。
    MeshAllGatherSlicePair txSlicePair{sliceInfo.tempAlgParams.cclBufferPtr,
                                       sliceInfo.linkRemote.remoteCclMem.addr,
                                       txSrcSlicesAll, txDstSlicesAll};
    AddRankDataSlices(sliceInfo, txRankIds, txSlicePair);
    MeshAllGatherSlicePair rxSlicePair{sliceInfo.linkRemote.remoteCclMem.addr,
                                       sliceInfo.tempAlgParams.cclBufferPtr,
                                       rxSrcSlicesAll, rxDstSlicesAll};
    AddRankDataSlices(sliceInfo, rxRankIds, rxSlicePair);
    TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
    TxRxChannels sendRecvChannels(sliceInfo.linkRemote, sliceInfo.linkRemote);
    sendRecvInfos.emplace_back(sendRecvChannels, sendRecvSlicesList, sliceInfo.tempAlgParams.dataType);
    HCCL_INFO("[RunMeshAllGather] Build SendRecvInfo: dataType=%d, sliceSize=%lu, stride=%lu, "
              "txRankNum=%zu, rxRankNum=%zu, sendRecvInfoNum=%zu",
              static_cast<int>(sliceInfo.tempAlgParams.dataType), sliceInfo.sliceSize, sliceInfo.stride,
              txRankIds.size(), rxRankIds.size(), sendRecvInfos.size());
}

} // namespace

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            const std::vector<u32> &ranks, u32 myRank, std::vector<u32> &ranksForOutputData,
                            std::vector<SendRecvInfo> &sendRecvInfos)
{

    sendRecvInfos.clear();
    HCCL_INFO("[RunMeshAllGather] start: myRank=%u, rankSize=%zu, dataType=%d, sliceCount=%lu, "
              "sliceOffset=%lu, stride=%lu, inputRankNum=%zu",
              myRank, ranks.size(), static_cast<int>(tempAlgParams.dataType), tempAlgParams.sliceCount,
              tempAlgParams.sliceOffset, tempAlgParams.stride, tempAlgParams.ranksForInputData.size());
    for (size_t i = 0; i < tempAlgParams.ranksForInputData.size(); ++i) {
        HCCL_INFO("[RunMeshAllGather] ranksForInputData[%zu]=%u", i, tempAlgParams.ranksForInputData[i]);
    }

    CHK_RET(PreCheckMeshAllGather(tempAlgParams, templateResource, ranks, myRank));
    if (ranks.size() <= 1 || templateResource.channels.empty()) {
        ranksForOutputData = tempAlgParams.ranksForInputData;
        HCCL_INFO("[RunMeshAllGather] no sendRecv needed, ranksForOutputDataNum=%zu",
                  ranksForOutputData.size());
        return HCCL_SUCCESS;
    }

    const u32 rankSize = static_cast<u32>(ranks.size());
    const u64 stride = tempAlgParams.stride;
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank, ranks, myAlgRank));
    std::vector<u32> ranksForInputData;
    ranksForInputData = tempAlgParams.ranksForInputData;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    const u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    std::vector<u32> algRanksForInputData;
    CHK_RET(GetAlgRanksForInputData(ranks, ranksForInputData, algRanksForInputData));
    HCCL_INFO("[RunMeshAllGather] myAlgRank=%u, rankSize=%u, dataTypeSize=%u, sliceSize=%lu",
              myAlgRank, rankSize, dataTypeSize, sliceSize);
    for (size_t i = 0; i < ranks.size(); ++i) {
        HCCL_INFO("[RunMeshAllGather] ranks[%zu]=%u", i, ranks[i]);
    }
    for (size_t i = 0; i < algRanksForInputData.size(); ++i) {
        HCCL_INFO("[RunMeshAllGather] algRanksForInputData[%zu]=%u", i, algRanksForInputData[i]);
    }

    for (u32 connectedIdx = 0; connectedIdx < rankSize; ++connectedIdx) {
        const u32 connectedRank = ranks[connectedIdx];
        if (connectedRank == myRank) {
            continue;
        }
        const u32 connectedOffset = (connectedIdx + rankSize - myAlgRank) % rankSize;
        const ChannelInfo *linkRemote = nullptr;
        linkRemote = &templateResource.channels.at(connectedRank)[0];
        // 对端数据来源通过本端数据来源在 algRank 空间内环状平移得到。
        std::vector<u32> connectedInputRanks;
        CHK_RET(GetConnectedInputRanks(ranks, algRanksForInputData, rankSize, connectedOffset, connectedInputRanks));
        HCCL_INFO("[RunMeshAllGather] connectedRank=%u, connectedAlgRank=%u, connectedOffset=%u, "
                  "connectedInputRankNum=%zu",
                  connectedRank, connectedIdx, connectedOffset, connectedInputRanks.size());
        for (size_t i = 0; i < connectedInputRanks.size(); ++i) {
            HCCL_INFO("[RunMeshAllGather] connectedInputRanks[%zu]=%u", i, connectedInputRanks[i]);
        }
        const MeshAllGatherSliceInfo sliceInfo{tempAlgParams, *linkRemote, sliceSize, stride};
        GetSendRecvInfo(sliceInfo, ranksForInputData, connectedInputRanks, sendRecvInfos);
    }
    HCCL_INFO("[RunMeshAllGather] end: sendRecvInfoNum=%zu", sendRecvInfos.size());
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
