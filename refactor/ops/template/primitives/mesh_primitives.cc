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
inline void AddRankDataSlices(const MeshSliceInfo &sliceInfo, const std::vector<u32> &rankIds,
                              MeshSlicePair &slicePair)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[sliceInfo.tempAlgParams.dataType];
    for (u32 rankId : rankIds) {
        const u64 dataSize = (sliceInfo.tailSize > 0 && rankId == sliceInfo.tailRankId) ?
            sliceInfo.tailSize : sliceInfo.sliceSize;
        const u64 dataOffset = sliceInfo.tempAlgParams.sliceOffset + static_cast<u64>(rankId) * sliceInfo.stride;
        HCCL_DEBUG("[RunMeshAllGather] AddRankDataSlices: rankId=%u, sliceOffset=%lu, stride=%lu, "
                   "dataOffset=%lu, dataSize=%lu, count=%lu",
                   rankId, sliceInfo.tempAlgParams.sliceOffset, sliceInfo.stride, dataOffset, dataSize,
                   dataSize / dataTypeSize);
        slicePair.firstSlices.emplace_back(slicePair.firstBufferPtr, dataOffset, dataSize, dataSize / dataTypeSize);
        slicePair.secondSlices.emplace_back(slicePair.secondBufferPtr, dataOffset, dataSize, dataSize / dataTypeSize);
    }
}

// 只拼四组 DataSlice 与对端 rank，channel/线程选择留给执行层。
} // namespace

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks, u32 myRank,
                            std::vector<u32> &ranksForOutputData, std::vector<TxRxSlicesList> &txRxSlicesLists)
{
    txRxSlicesLists.clear();

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
    const u64 tailSize = tempAlgParams.tailCount * dataTypeSize;
    const u32 tailRankId = ranks[rankSize - 1];
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
        const MeshSliceInfo sliceInfo{tempAlgParams, sliceSize, tailSize, stride, tailRankId};
        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        MeshSlicePair txSlicePair{tempAlgParams.cclBufferPtr, nullptr, txSrcSlicesAll, txDstSlicesAll};
        AddRankDataSlices(sliceInfo, ranksForInputData, txSlicePair);
        MeshSlicePair rxSlicePair{nullptr, tempAlgParams.cclBufferPtr, rxSrcSlicesAll, rxDstSlicesAll};
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

HcclResult RunMeshScatter(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks, u32 myRank,
                          std::vector<u32> &ranksForOutputData, std::vector<TxRxSlicesList> &txRxSlicesLists)
{
    txRxSlicesLists.clear();
    CHK_RET(CheckInputDataRanks(tempAlgParams, "RunMeshScatter"));

    const u32 rankSize = static_cast<u32>(ranks.size());
    ranksForOutputData = {myRank};
    if (rankSize <= 1) {
        HCCL_INFO("[RunMeshScatter] no send/recv needed, ranksForOutputDataNum=%zu", ranksForOutputData.size());
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
    const MeshSliceInfo sliceInfo{tempAlgParams, sliceSize, tailSize, tempAlgParams.stride, tailRankId};
    HCCL_INFO("[RunMeshScatter] myAlgRank=%u, rootAlgRank=%u, rankSize=%u, dataTypeSize=%u, sliceSize=%lu",
              myAlgRank, rootAlgRank, rankSize, dataTypeSize, sliceSize);
    // root 只发不收
    if (myRank == tempAlgParams.root) {
        for (u32 algRank = 0; algRank < rankSize; ++algRank) {
            const u32 remoteRank = ranks[algRank];
            if (remoteRank == myRank) {
                continue;
            }
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            MeshSlicePair txSlicePair{tempAlgParams.cclBufferPtr, nullptr, txSrcSlices, txDstSlices};
            AddRankDataSlices(sliceInfo, {remoteRank}, txSlicePair);
            txRxSlicesLists.emplace_back(SlicesList(txSrcSlices, txDstSlices), SlicesList({}, {}), remoteRank, remoteRank);
            HCCL_INFO("[RunMeshScatter] Build tx TxRxSlicesList: remoteRank=%u, offset=%lu, size=%lu, "
                      "txRxSlicesListNum=%zu",
                      remoteRank, txSrcSlices[0].offset_, txSrcSlices[0].size_, txRxSlicesLists.size());
        }
        return HCCL_SUCCESS;
    }
    // 其他 只收不发
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    MeshSlicePair rxSlicePair{nullptr, tempAlgParams.cclBufferPtr, rxSrcSlices, rxDstSlices};
    AddRankDataSlices(sliceInfo, {myRank}, rxSlicePair);
    txRxSlicesLists.emplace_back(SlicesList({}, {}), SlicesList(rxSrcSlices, rxDstSlices),
                                 tempAlgParams.root, tempAlgParams.root);
    HCCL_INFO("[RunMeshScatter] Build rx TxRxSlicesList: root=%u, offset=%lu, size=%lu",
              tempAlgParams.root, rxDstSlices[0].offset_, rxDstSlices[0].size_);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
