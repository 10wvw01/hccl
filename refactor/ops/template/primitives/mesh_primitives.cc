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

#include <algorithm>
#include <limits>

namespace ops_hccl {

namespace {

// 根据本端输入数据归属和对端全局 rank 偏移，生成对端当前持有的数据来源列表。
inline HcclResult GetConnectedInputRanks(const std::vector<u32> &ranksForInputData,
                                         long long connectedOffset,
                                         std::vector<u32> &connectedInputRanks)
{
    connectedInputRanks.clear();
    connectedInputRanks.reserve(ranksForInputData.size());
    for (u32 rankId : ranksForInputData) {
        const long long connectedRankId = static_cast<long long>(rankId) + connectedOffset;
        CHK_PRT_RET(connectedRankId < 0 ||
                        connectedRankId > static_cast<long long>(std::numeric_limits<u32>::max()),
                    HCCL_ERROR("[RunMeshAllGather] connectedRankId is invalid, rankId=%u, connectedOffset=%lld.",
                               rankId, connectedOffset),
                    HCCL_E_PARA);
        HCCL_DEBUG("[RunMeshAllGather] GetConnectedInputRanks: rankId=%u, connectedOffset=%lld, "
                   "connectedRankId=%lld",
                   rankId, connectedOffset, connectedRankId);
        connectedInputRanks.emplace_back(static_cast<u32>(connectedRankId));
    }
    return HCCL_SUCCESS;
}

// 按全局 rankId 槽位追加一组 src/dst DataSlice。
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

// 只拼四组 DataSlice 与对端 rank，channel/线程选择留给执行层。
} // namespace

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks, u32 myRank,
                            std::vector<u32> &ranksForOutputData, std::vector<TxRxSlicesList> &txRxSlicesLists)
{
    txRxSlicesLists.clear();
    HCCL_INFO("[RunMeshAllGather] start: myRank=%u, rankSize=%zu, dataType=%d, sliceCount=%lu, "
              "sliceOffset=%lu, stride=%lu, inputRankNum=%zu",
              myRank, ranks.size(), static_cast<int>(tempAlgParams.dataType), tempAlgParams.sliceCount,
              tempAlgParams.sliceOffset, tempAlgParams.stride, tempAlgParams.ranksForInputData.size());
    for (size_t i = 0; i < tempAlgParams.ranksForInputData.size(); ++i) {
        HCCL_INFO("[RunMeshAllGather] ranksForInputData[%zu]=%u", i, tempAlgParams.ranksForInputData[i]);
    }

    CHK_RET(CheckInputDataRanks(tempAlgParams, "RunMeshAllGather"));
    if (ranks.size() <= 1) {
        ranksForOutputData = tempAlgParams.ranksForInputData;
        HCCL_INFO("[RunMeshAllGather] no sendRecv needed, ranksForOutputDataNum=%zu",
                  ranksForOutputData.size());
        return HCCL_SUCCESS;
    }

    const u32 rankSize = static_cast<u32>(ranks.size());
    const u64 stride = tempAlgParams.stride;
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank, ranks, myAlgRank));
    std::vector<u32> ranksForInputData = tempAlgParams.ranksForInputData;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    const u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    HCCL_INFO("[RunMeshAllGather] myAlgRank=%u, rankSize=%u, dataTypeSize=%u, sliceSize=%lu",
              myAlgRank, rankSize, dataTypeSize, sliceSize);
    for (size_t i = 0; i < ranks.size(); ++i) {
        HCCL_INFO("[RunMeshAllGather] ranks[%zu]=%u", i, ranks[i]);
    }

    ranksForOutputData = ranksForInputData;
    for (u32 connectedIdx = 0; connectedIdx < rankSize; ++connectedIdx) {
        const u32 connectedRank = ranks[connectedIdx];
        if (connectedRank == myRank) {
            continue;
        }
        const long long connectedOffset = static_cast<long long>(connectedRank) - static_cast<long long>(myRank);
        std::vector<u32> connectedInputRanks;
        CHK_RET(GetConnectedInputRanks(ranksForInputData, connectedOffset, connectedInputRanks));
        HCCL_INFO("[RunMeshAllGather] connectedRank=%u, connectedAlgRank=%u, connectedOffset=%lld, "
                  "connectedInputRankNum=%zu",
                  connectedRank, connectedIdx, connectedOffset, connectedInputRanks.size());
        for (size_t i = 0; i < connectedInputRanks.size(); ++i) {
            HCCL_INFO("[RunMeshAllGather] connectedInputRanks[%zu]=%u", i, connectedInputRanks[i]);
        }
        const MeshAllGatherSliceInfo sliceInfo{tempAlgParams, sliceSize, stride};
        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        MeshAllGatherSlicePair txSlicePair{tempAlgParams.cclBufferPtr, nullptr, txSrcSlicesAll, txDstSlicesAll};
        AddRankDataSlices(sliceInfo, ranksForInputData, txSlicePair);
        MeshAllGatherSlicePair rxSlicePair{nullptr, tempAlgParams.cclBufferPtr, rxSrcSlicesAll, rxDstSlicesAll};
        AddRankDataSlices(sliceInfo, connectedInputRanks, rxSlicePair);
        txRxSlicesLists.emplace_back(SlicesList(txSrcSlicesAll, txDstSlicesAll),
                                     SlicesList(rxSrcSlicesAll, rxDstSlicesAll), connectedRank, connectedRank);
        HCCL_INFO("[RunMeshAllGather] Build TxRxSlicesList: dataType=%d, sliceSize=%lu, stride=%lu, "
                  "connectedRank=%u, txRankNum=%zu, rxRankNum=%zu, txRxSlicesListNum=%zu",
                  static_cast<int>(tempAlgParams.dataType), sliceSize, stride, connectedRank,
                  ranksForInputData.size(), connectedInputRanks.size(), txRxSlicesLists.size());
        ranksForOutputData.insert(ranksForOutputData.end(), connectedInputRanks.begin(), connectedInputRanks.end());
    }
    std::sort(ranksForOutputData.begin(), ranksForOutputData.end());
    HCCL_INFO("[RunMeshAllGather] end: txRxSlicesListNum=%zu", txRxSlicesLists.size());
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
