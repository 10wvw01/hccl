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
// src/dst 偏移统一按 ccl buffer 布局（sliceOffset + rankId * stride）。
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

// outputBufferType==OUTPUT 场景：rx 直接落 output buffer。
// rx 源仍从对端 ccl buffer 读（offset 按 ccl 布局：sliceOffset + rankId * stride），
// rx 目标写入本地 output（offset 按 output 布局：dataOffset + sliceOffset + rankId * dataStride）。
// AllGather 下 output 按 rankId 排列，dataStride 通常等于 scratchStride。
inline void AddRankDataSlicesToOutput(const MeshSliceInfo &sliceInfo, const std::vector<u32> &rankIds,
                                       MeshSlicePair &slicePair)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[sliceInfo.tempAlgParams.dataType];
    for (u32 rankId : rankIds) {
        const u64 dataSize = (sliceInfo.tailSize > 0 && rankId == sliceInfo.tailRankId) ?
            sliceInfo.tailSize : sliceInfo.sliceSize;
        // rx 源：对端 ccl buffer 按 ccl 布局寻址（addr=nullptr，engine 用 channel.remoteCclMem.addr 解析）
        const u64 srcOffset = sliceInfo.tempAlgParams.sliceOffset + static_cast<u64>(rankId) * sliceInfo.stride;
        // rx 目标：本地 output 按 output 布局寻址（dataOffset + sliceOffset + rankId * dataStride）
        const u64 dstOffset = sliceInfo.tempAlgParams.dataOffset + sliceInfo.tempAlgParams.sliceOffset +
                              static_cast<u64>(rankId) * sliceInfo.tempAlgParams.dataStride;
        HCCL_DEBUG("[RunMeshAllGather] AddRankDataSlicesToOutput: rankId=%u, srcOffset=%lu, dstOffset=%lu, "
                   "dataSize=%lu, count=%lu",
                   rankId, srcOffset, dstOffset, dataSize, dataSize / dataTypeSize);
        slicePair.firstSlices.emplace_back(slicePair.firstBufferPtr, srcOffset, dataSize, dataSize / dataTypeSize);
        slicePair.secondSlices.emplace_back(slicePair.secondBufferPtr, dstOffset, dataSize, dataSize / dataTypeSize);
    }
}

// 只拼四组 DataSlice 与对端 rank，channel/线程选择留给执行层。
} // namespace

// 收集 reuseCclBuffer 场景下的空位 slot：ccl buffer 按 rank 值寻址，空位 = 不在
// ranksForInputData 中的 rank。ranks 是 subComm 域 ranks，ranksForInputData 是本 template
// 要归约的 rank 列表。空位来源：
//   1. ranks 中不在 ranksForInputData 的 rank（subComm 域内空位）；
//   2. 不足时（ranksForInputData 包含 subComm 全部 rank）遍历 [0, ranks.back()+rankSize+1)
//      找不在 ranksForInputData 的 rank（其他 subComm 域的 slot）。
// 不越界：ReduceScatter 的 ccl buffer 覆盖全局 rankSize 个 slot
// （scratchMultiple_=1, scratchStride=sliceCount*dataTypeSize, cclBufferSize≥globalRankSize*scratchStride），
// 空位一定在 [0, globalRankSize) 内。
void CollectEmptySlots(const std::vector<u32> &ranks, const std::vector<u32> &ranksForInputData,
                       u32 rankSize, std::vector<u32> &emptySlots)
{
    emptySlots.clear();
    for (u32 r : ranks) {
        if (std::find(ranksForInputData.begin(), ranksForInputData.end(), r) == ranksForInputData.end()) {
            emptySlots.emplace_back(r);
        }
    }
    if (emptySlots.size() < rankSize) {
        u32 scanEnd = ranks.back() + rankSize + 1;
        for (u32 r = 0; r < scanEnd && emptySlots.size() < rankSize; ++r) {
            if (std::find(ranksForInputData.begin(), ranksForInputData.end(), r) == ranksForInputData.end() &&
                std::find(emptySlots.begin(), emptySlots.end(), r) == emptySlots.end()) {
                emptySlots.emplace_back(r);
            }
        }
    }
}

HcclResult InitMeshRsLayoutInfo(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks,
                                u32 myRank, MeshRsLayoutInfo &layoutInfo)
{
    layoutInfo = MeshRsLayoutInfo{};
    layoutInfo.rankSize = static_cast<u32>(ranks.size());
    layoutInfo.reuseCclBuffer = (tempAlgParams.inputBufferType == BufferType::HCCL_BUFFER);
    CHK_RET(GetAlgRank(myRank, ranks, layoutInfo.myAlgRank));

    if (layoutInfo.reuseCclBuffer) {
        CollectEmptySlots(ranks, tempAlgParams.ranksForInputData, layoutInfo.rankSize, layoutInfo.emptySlots);
        CHK_PRT_RET(layoutInfo.emptySlots.size() < layoutInfo.rankSize,
            HCCL_ERROR("[InitMeshRsLayoutInfo] emptySlots size[%zu] is less than rankSize[%u].",
                layoutInfo.emptySlots.size(), layoutInfo.rankSize),
            HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

u64 GetMeshRsInputOffset(const TemplateDataParams &tempAlgParams, size_t idx)
{
    return tempAlgParams.dataOffset + tempAlgParams.sliceOffset +
           static_cast<u64>(idx) * tempAlgParams.dataStride;
}

u64 GetMeshRsFinalCclOffset(const TemplateDataParams &tempAlgParams, const MeshRsLayoutInfo &layoutInfo,
                            size_t idx, u32 rank)
{
    if (layoutInfo.reuseCclBuffer) {
        return tempAlgParams.sliceOffset + static_cast<u64>(rank) * tempAlgParams.scratchStride;
    }
    const u64 groupIdx = static_cast<u64>(idx) / layoutInfo.rankSize;
    const u64 slot = groupIdx * layoutInfo.rankSize + layoutInfo.myAlgRank;
    return tempAlgParams.sliceOffset + slot * tempAlgParams.scratchStride;
}

u64 GetMeshRsTempCclOffset(const TemplateDataParams &tempAlgParams, const MeshRsLayoutInfo &layoutInfo,
                           size_t idx, u32 algRank)
{
    if (layoutInfo.reuseCclBuffer) {
        return tempAlgParams.sliceOffset + static_cast<u64>(layoutInfo.emptySlots[algRank]) *
            tempAlgParams.scratchStride;
    }
    const u64 groupIdx = static_cast<u64>(idx) / layoutInfo.rankSize;
    const u64 slot = groupIdx * layoutInfo.rankSize + algRank;
    return tempAlgParams.sliceOffset + slot * tempAlgParams.scratchStride;
}

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
    const u64 stride = tempAlgParams.scratchStride;
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank, ranks, myAlgRank));
    std::vector<u32> ranksForInputData = tempAlgParams.ranksForInputData;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    const u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    // 尾块语义与 PreCopy/PostCopy 对齐：tailCount 是均分后的零头，尾块 = 整块 + 零头
    const u64 tailSize = (tempAlgParams.tailCount == 0) ? (sliceSize) : (sliceSize + tempAlgParams.tailCount * dataTypeSize);
    const u32 tailRankId = ranks[rankSize - 1];
    // outputBufferType==OUTPUT 时 rx 直接落 output，不经本地 ccl 中转，无需 PostCopy；
    // outputBufferType==HCCL_BUFFER 时保留原逻辑：rx 落本地 ccl，由 PostCopy 搬到 output。
    const bool directToOutput = (tempAlgParams.outputBufferType == BufferType::OUTPUT);
    HCCL_INFO("[RunMeshAllGather] myAlgRank=%u, rankSize=%u, dataTypeSize=%u, sliceSize=%lu, directToOutput=%d",
              myAlgRank, rankSize, dataTypeSize, sliceSize, static_cast<int>(directToOutput));
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
        // tx：本端 ccl[myRank slots] -> 对端 ccl（Read 模式下 tx 由对端主动读取，这里保持描述符一致）
        MeshSlicePair txSlicePair{tempAlgParams.cclBufferPtr, nullptr, txSrcSlicesAll, txDstSlicesAll};
        AddRankDataSlices(sliceInfo, ranksForInputData, txSlicePair);
        // rx：从对端 ccl 读到本地。directToOutput 时直接落 outputBufferPtr（按 output 布局），
        // 否则落 cclBufferPtr（按 ccl 布局，后续由 PostCopy 搬到 output）。
        if (directToOutput) {
            MeshSlicePair rxSlicePair{nullptr, tempAlgParams.outputBufferPtr, rxSrcSlicesAll, rxDstSlicesAll};
            AddRankDataSlicesToOutput(sliceInfo, connectedInputRanks, rxSlicePair);
        } else {
            MeshSlicePair rxSlicePair{nullptr, tempAlgParams.cclBufferPtr, rxSrcSlicesAll, rxDstSlicesAll};
            AddRankDataSlices(sliceInfo, connectedInputRanks, rxSlicePair);
        }
        txRxSlicesLists.emplace_back(SlicesList(txSrcSlicesAll, txDstSlicesAll),
                                     SlicesList(rxSrcSlicesAll, rxDstSlicesAll), connectedRank, connectedRank);
        HCCL_INFO("[RunMeshAllGather] Build TxRxSlicesList: dataType=%d, sliceSize=%lu, stride=%lu, "
                  "connectedRank=%u, txRankNum=%zu, rxRankNum=%zu, txRxSlicesListNum=%zu, directToOutput=%d",
                  static_cast<int>(tempAlgParams.dataType), sliceSize, stride, connectedRank,
                  ranksForInputData.size(), connectedInputRanks.size(), txRxSlicesLists.size(),
                  static_cast<int>(directToOutput));
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
    const size_t inputSize = tempAlgParams.ranksForInputData.size();
    ranksForOutputData.clear();
    if (rankSize <= 1) {
        ranksForOutputData = {myRank};
        HCCL_INFO("[RunMeshScatter] no send/recv needed, ranksForOutputDataNum=%zu", ranksForOutputData.size());
        return HCCL_SUCCESS;
    }

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank, ranks, myAlgRank));
    u32 rootAlgRank = 0;
    HcclResult rootRet = GetAlgRank(tempAlgParams.root, ranks, rootAlgRank);
    bool rootInRanks = (rootRet == HCCL_SUCCESS);
    HCCL_INFO("[RunMeshScatter] myAlgRank=%u, rootInRanks=%d, rankSize=%u, inputSize=%zu",
        myAlgRank, static_cast<int>(rootInRanks), rankSize, inputSize);

    // ranksForOutputData 取 ranksForInputData 中 idx % rankSize == myAlgRank 的 rank。
    // 单层场景退化为 idx == myAlgRank；Parallel 场景保留其他层对应 rank 的数据归属。
    for (size_t idx = 0; idx < inputSize; ++idx) {
        if (static_cast<u32>(idx) % rankSize == myAlgRank) {
            ranksForOutputData.emplace_back(tempAlgParams.ranksForInputData[idx]);
        }
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    const u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    const u64 tailSize = (tempAlgParams.tailCount == 0) ? (sliceSize) : (sliceSize + tempAlgParams.tailCount * dataTypeSize);
    const u32 tailRankId = ranks[rankSize - 1];
    const MeshSliceInfo sliceInfo{tempAlgParams, sliceSize, tailSize, tempAlgParams.scratchStride, tailRankId};
    // root 不在 ranks_ 内时，只输出 ranksForOutputData，不构造通信描述符
    if (!rootInRanks) {
        HCCL_INFO("[RunMeshScatter] root not in ranks, skip communication, outputRanksNum=%zu",
            ranksForOutputData.size());
        return HCCL_SUCCESS;
    }
    // root 按 ranksForInputData 布局，把每个非 root rank 该收的 slot 发给它。
    // Mesh 公式：idx % rankSize == algRank 的 rank 收 slot idx。
    if (myRank == tempAlgParams.root) {
        for (u32 algRank = 0; algRank < rankSize; ++algRank) {
            const u32 remoteRank = ranks[algRank];
            if (remoteRank == myRank) {
                continue;
            }
            std::vector<u32> txRankIds;
            for (size_t idx = 0; idx < inputSize; ++idx) {
                if (static_cast<u32>(idx) % rankSize == algRank) {
                    txRankIds.emplace_back(tempAlgParams.ranksForInputData[idx]);
                }
            }
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            MeshSlicePair txSlicePair{tempAlgParams.cclBufferPtr, nullptr, txSrcSlices, txDstSlices};
            AddRankDataSlices(sliceInfo, txRankIds, txSlicePair);
            txRxSlicesLists.emplace_back(SlicesList(txSrcSlices, txDstSlices), SlicesList({}, {}), remoteRank, remoteRank);
        }
        return HCCL_SUCCESS;
    }
    // 非 root 只收不发：接收自己该收的 slots
    std::vector<u32> rxRankIds;
    for (size_t idx = 0; idx < inputSize; ++idx) {
        if (static_cast<u32>(idx) % rankSize == myAlgRank) {
            rxRankIds.emplace_back(tempAlgParams.ranksForInputData[idx]);
        }
    }
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    MeshSlicePair rxSlicePair{nullptr, tempAlgParams.cclBufferPtr, rxSrcSlices, rxDstSlices};
    AddRankDataSlices(sliceInfo, rxRankIds, rxSlicePair);
    txRxSlicesLists.emplace_back(SlicesList({}, {}), SlicesList(rxSrcSlices, rxDstSlices),
                                 tempAlgParams.root, tempAlgParams.root);
    return HCCL_SUCCESS;
}

HcclResult RunMeshReduceScatter(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks, u32 myRank,
                                std::vector<u32> &ranksForOutputData, std::vector<TxRxSlicesList> &txRxSlicesLists)
{
    txRxSlicesLists.clear();
    CHK_RET(CheckInputDataRanks(tempAlgParams, "RunMeshReduceScatter"));

    const u32 rankSize = static_cast<u32>(ranks.size());
    ranksForOutputData.clear();
    if (rankSize <= 1) {
        ranksForOutputData = tempAlgParams.ranksForInputData;
        HCCL_INFO("[RunMeshReduceScatter] no send/recv needed, ranksForOutputDataNum=%zu",
                  ranksForOutputData.size());
        return HCCL_SUCCESS;
    }

    MeshRsLayoutInfo layoutInfo;
    CHK_RET(InitMeshRsLayoutInfo(tempAlgParams, ranks, myRank, layoutInfo));
    const u32 myAlgRank = layoutInfo.myAlgRank;

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    const u64 sliceSize = tempAlgParams.sliceCount * dataTypeSize;
    const u64 tailSize = (tempAlgParams.tailCount == 0) ? (sliceSize) :
                         (sliceSize + tempAlgParams.tailCount * dataTypeSize);
    const u32 tailRankId = ranks[rankSize - 1];
    const size_t inputSize = tempAlgParams.ranksForInputData.size();
    HCCL_INFO("[RunMeshReduceScatter] myAlgRank=%u, rankSize=%u, dataTypeSize=%u, sliceSize=%lu, tailSize=%lu",
              myAlgRank, rankSize, dataTypeSize, sliceSize, tailSize);

    // 1. 计算 ranksForOutputData：所有 idx % rankSize == myAlgRank 的 ranksForInputData[idx]。
    for (size_t idx = 0; idx < inputSize; ++idx) {
        if (static_cast<u32>(idx) % rankSize == myAlgRank) {
            ranksForOutputData.emplace_back(tempAlgParams.ranksForInputData[idx]);
        }
    }

    // 2. 为每个对端 algRank c 构造一个 TxRxSlicesList。
    //    - reuseCclBuffer 场景（inputBufferType==HCCL_BUFFER）：ccl buffer 覆盖更大域，
    //      有效 slot = ccl[rank]（按 rank 值，NHR 写入），有空位可用。
    //      tx/rx 目标用空位 emptySlots[algRank]，避免覆盖有效数据；
    //      LocalReduce 从空位 reduce 到归约目标 slot ccl[rank]。
    //    - INPUT 场景（inputBufferType==INPUT）：ccl buffer 仅覆盖本 subComm 域，无空位。
    //      tx/rx 目标按 algRank 维度 (g*rankSize + algRank) 寻址，与 PreCopy 的 slot 布局一致。

    // reuseCclBuffer 场景：收集空位 slot（不在 ranksForInputData 中的 rank）。

    // 计算 rx 目标 slot 的 ccl 偏移：
    //   reuseCclBuffer：按 rank 值寻址 ccl[rank]；
    //   INPUT：按 algRank 维度寻址 ccl[g*rankSize + srcAlgRank]。
    // 计算对端空位 slot 的 ccl 偏移（仅 tx 目标和 rx 源需要，二者 offset 一致）：
    //   reuseCclBuffer：空位 emptySlots[myAlgRank]（tx）或 emptySlots[connectedAlgRank]（rx）；
    //   INPUT：algRank 维度 (g*rankSize + srcAlgRank)。

    for (u32 connectedAlgRank = 0; connectedAlgRank < rankSize; ++connectedAlgRank) {
        if (connectedAlgRank == myAlgRank) {
            continue;
        }
        const u32 connectedRank = ranks[connectedAlgRank];
        std::vector<DataSlice> txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices;
        for (size_t idx = 0; idx < inputSize; ++idx) {
            const u32 algRankOfIdx = static_cast<u32>(idx) % rankSize;
            // isTx：本卡发对端归约集合切片（idx % rankSize == connectedAlgRank）给对端；
            // isRx：本卡收本卡归约集合切片（idx % rankSize == myAlgRank）。
            const bool isTx = (algRankOfIdx == connectedAlgRank);
            const bool isRx = (algRankOfIdx == myAlgRank);
            if (!isTx && !isRx) {
                continue;
            }
            const u32 rank = tempAlgParams.ranksForInputData[idx];
            const u64 curSliceSize = (rank == tailRankId) ? tailSize : sliceSize;
            if (curSliceSize == 0) {
                continue;
            }
            const u64 sliceCount = curSliceSize / dataTypeSize;
            // tx 目标 = 对端空位 emptySlots[myAlgRank]；rx 目标 = 本卡空位 emptySlots[connectedAlgRank]。
            // rx 数据源 offset 与目标 offset 一致（对端 tx 目标 slot 即本卡 rx 源 slot），
            // bufferPtr=nullptr 表示远端内存。
            // INPUT 场景：tx 数据源来自 user input（按 idx 寻址）。
            if (isTx) {
                void *srcBufPtr = layoutInfo.reuseCclBuffer ? tempAlgParams.cclBufferPtr : tempAlgParams.inputBufferPtr;
                const u64 srcOff = layoutInfo.reuseCclBuffer ?
                    GetMeshRsFinalCclOffset(tempAlgParams, layoutInfo, idx, rank) :
                    GetMeshRsInputOffset(tempAlgParams, idx);
                txSrcSlices.emplace_back(srcBufPtr, srcOff, curSliceSize, sliceCount);
                txDstSlices.emplace_back(nullptr,
                    GetMeshRsTempCclOffset(tempAlgParams, layoutInfo, idx, myAlgRank), curSliceSize, sliceCount);
            } else {
                const u64 off = GetMeshRsTempCclOffset(tempAlgParams, layoutInfo, idx, connectedAlgRank);
                rxSrcSlices.emplace_back(nullptr, off, curSliceSize, sliceCount);
                rxDstSlices.emplace_back(tempAlgParams.cclBufferPtr, off, curSliceSize, sliceCount);
            }
        }
        if (txSrcSlices.empty() && rxDstSlices.empty()) {
            continue;
        }
        txRxSlicesLists.emplace_back(SlicesList(txSrcSlices, txDstSlices),
                                     SlicesList(rxSrcSlices, rxDstSlices), connectedRank, connectedRank);
        HCCL_INFO("[RunMeshReduceScatter] Build TxRxSlicesList: connectedRank=%u, connectedAlgRank=%u, "
                  "txSliceNum=%zu, rxSliceNum=%zu, txRxSlicesListNum=%zu, reuseCcl=%d",
                  connectedRank, connectedAlgRank, txSrcSlices.size(), rxDstSlices.size(),
                  txRxSlicesLists.size(), static_cast<int>(layoutInfo.reuseCclBuffer));
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
