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

namespace {

// Direct-call primitives receive a flat rank list instead of the old template's
// cached subCommRanks_. Keep the conversion local so every planner path can use
// the same "algorithm rank" index in offset formulas.
HcclResult GetRankIndex(const std::vector<u32> &ranks, u32 rank, u32 &rankIdx)
{
    const u32 rankSize = static_cast<u32>(ranks.size());
    for (u32 i = 0; i < rankSize; ++i) {
        if (ranks[i] == rank) {
            rankIdx = i;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("[MeshAllGatherPlanner] rank[%u] is not in ranks.", rank);
    return HCCL_E_PARA;
}

MeshAllGatherSendRecvMode ResolveMeshAllGatherSendRecvMode(const TemplateDataParams &tempAlgParams,
                                                           const TemplateResource &templateResource,
                                                           const MeshAllGatherPrimitiveOptions &options)
{
    // Z-axis old code switches read/write from buffer types. Other variants will
    // be moved here as they are ported, keeping SendRecv selection out of the
    // execution loop.
    if (options.sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR) {
        bool dmaRead = (tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER &&
                        tempAlgParams.buffInfo.outBuffType != BufferType::HCCL_BUFFER);
        return dmaRead ? MeshAllGatherSendRecvMode::DMA_READ : MeshAllGatherSendRecvMode::BATCH_WRITE;
    }

    // Ordinary AICPU Mesh AllGather uses SendRecvRead; the current extracted common split kept
    // the old IsPcieProtocol branch. Keep that behavior until each variant is wired explicitly.
    return IsPcieProtocol(templateResource.channels) ?
        MeshAllGatherSendRecvMode::DMA_READ : MeshAllGatherSendRecvMode::BATCH_WRITE;
}

HcclResult BuildMeshAllGatherCommonChannelSplitPlan(const TemplateDataParams &tempAlgParams,
                                                    const TemplateResource &templateResource,
                                                    const std::vector<u32> &ranks, u32 myRank,
                                                    MeshAllGatherSlicePlan &plan)
{
    // This mirrors the currently extracted RunMeshAllGather behavior: split one
    // peer's slice across all channels in the peer's port group, then emit one
    // task per peer/channel. It is intentionally separate from NORMAL_FIXED
    // because old ordinary Mesh AllGather used channel[0] and did not do this.
    const u32 rankSize = static_cast<u32>(ranks.size());
    u32 myRankIdx = 0;
    CHK_RET(GetRankIndex(ranks, myRank, myRankIdx));

    HcclDataType dataType = tempAlgParams.dataType;
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    u64 sliceSize = tempAlgParams.sliceSize;
    u64 tailSize = tempAlgParams.tailSize;
    u64 base = tempAlgParams.buffInfo.hcclBuffBaseOff;

    const std::vector<ChannelInfo> &portGroup = templateResource.channels.begin()->second;
    u32 channelsPerRank = static_cast<u32>(portGroup.size());
    std::vector<u64> ec, sizeOut, elemOffset;
    CHK_RET(CalcDataSplitByPortGroupCommon(sliceSize / dataTypeSize, dataTypeSize, portGroup, ec, sizeOut, elemOffset, channelsPerRank));
    std::vector<u64> ecT, sizeTail, elemOffsetTail;
    if (tailSize > 0) {
        CHK_RET(CalcDataSplitByPortGroupCommon(tailSize / dataTypeSize, dataTypeSize, portGroup, ecT, sizeTail, elemOffsetTail, channelsPerRank));
    }

    u32 t = 0;
    for (u32 i = 1; i < rankSize; ++i) {
        u32 peerIdx = (myRankIdx + i) % rankSize;
        bool peerTail = (peerIdx == rankSize - 1 && tailSize > 0);
        for (u32 ch = 0; ch < channelsPerRank; ++ch) {
            const ChannelInfo &link = templateResource.channels.at(ranks[peerIdx])[ch];
            u64 mySz = sizeOut[ch];
            u64 myOff = base + sliceSize * myRankIdx + elemOffset[ch];
            u64 peerSz = peerTail ? sizeTail[ch] : sizeOut[ch];
            u64 peerOff = base + sliceSize * peerIdx + (peerTail ? elemOffsetTail[ch] : elemOffset[ch]);

            MeshAllGatherPeerChannelPlan task;
            task.peerRank = ranks[peerIdx];
            task.peerAlgRank = peerIdx;
            task.channelIdx = ch;
            task.threadIdx = t++;
            task.link = &link;
            // These four slices are the executable boundary of the planner:
            // txSrc/txDst describe the local data this rank exposes to peer;
            // rxSrc/rxDst describe where this rank reads peer data from and stores it.
            task.txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, myOff, mySz, mySz / dataTypeSize);
            task.txDstSlices.emplace_back(link.remoteCclMem.addr, myOff, mySz, mySz / dataTypeSize);
            task.rxSrcSlices.emplace_back(link.remoteCclMem.addr, peerOff, peerSz, peerSz / dataTypeSize);
            task.rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, peerOff, peerSz, peerSz / dataTypeSize);
            plan.tasks.emplace_back(task);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BuildMeshAllGatherNormalFixedPlan(const TemplateDataParams &tempAlgParams,
                                             const TemplateResource &templateResource,
                                             const std::vector<u32> &ranks, u32 myRank,
                                             MeshAllGatherSlicePlan &plan)
{
    // TODO: port old InsTempAllGatherMesh1D::RunAllGatherMesh exactly:
    // 1. use one channel per peer: channels.at(peerRank)[0]
    // 2. build one DataSlice per repeat
    // 3. preserve outputPtr/remoteCclMem/remoteOutputGraphMode/symmetric-memory address selection
    // 4. preserve tail handling: connectedAlgRank == rankSize - 1
    // The current common split path intentionally remains separate because it is not equivalent
    // to ordinary Mesh AllGather when channelsPerRank > 1.
    (void)tempAlgParams;
    (void)templateResource;
    (void)ranks;
    (void)myRank;
    (void)plan;
    return HCCL_E_NOT_SUPPORT;
}

HcclResult BuildMeshAllGatherVariableCountPlan(const TemplateDataParams &tempAlgParams,
                                               const TemplateResource &templateResource,
                                               const std::vector<u32> &ranks, u32 myRank,
                                               MeshAllGatherSlicePlan &plan)
{
    // TODO: port old InsTempAllGatherVMesh1D::RunAllGatherVMesh:
    // allRankSliceSize/allRankDispls/allRankProcessedDataCount decide per-rank offset/size.
    (void)tempAlgParams;
    (void)templateResource;
    (void)ranks;
    (void)myRank;
    (void)plan;
    return HCCL_E_NOT_SUPPORT;
}

HcclResult BuildMeshAllGatherOmniPipeStepPlan(const TemplateDataParams &tempAlgParams,
                                              const TemplateResource &templateResource,
                                              const std::vector<u32> &ranks, u32 myRank,
                                              MeshAllGatherSlicePlan &plan)
{
    // TODO: port old InsTempAllGatherOmniPipeMesh1D::RunAllGatherMesh:
    // stepSliceInfo.{stepSliceSize,stepCount,stepInputSliceStride,stepOutputSliceStride,
    // inputOmniPipeSliceStride,outputOmniPipeSliceStride} decide each step slice.
    (void)tempAlgParams;
    (void)templateResource;
    (void)ranks;
    (void)myRank;
    (void)plan;
    return HCCL_E_NOT_SUPPORT;
}

HcclResult BuildMeshAllGatherZAxisDetourPlan(const TemplateDataParams &tempAlgParams,
                                             const TemplateResource &templateResource,
                                             const std::vector<u32> &ranks, u32 myRank,
                                             const ZAxisDetourConfig &zAxis,
                                             MeshAllGatherSlicePlan &plan)
{
    // TODO: port old InsTempAllGatherMesh1D1DZAxisDetour:
    // 1. resource planner must preserve level0/level1 channel boundary in zAxis
    // 2. call CalcDataSplitByPortGroupZAxisDetour(totalCount, dataTypeSize, peerChannels, ...)
    // 3. build DataSlice with elemOffset/sizeOut per channel
    // 4. use the same dmaRead tail rule as CalcSliceSizeForChannel
    (void)tempAlgParams;
    (void)templateResource;
    (void)ranks;
    (void)myRank;
    (void)zAxis;
    (void)plan;
    return HCCL_E_NOT_SUPPORT;
}

HcclResult BuildMeshAllGatherMeshChunkPlan(const TemplateDataParams &tempAlgParams,
                                           const TemplateResource &templateResource,
                                           const std::vector<u32> &ranks, u32 myRank,
                                           MeshAllGatherSlicePlan &plan)
{
    // TODO: meshchunk is an AllReduce two-shot variant, not an ordinary AllGather template.
    // Add RankSliceInfo/chunk metadata to PrimitiveOptions before wiring this mode.
    (void)tempAlgParams;
    (void)templateResource;
    (void)ranks;
    (void)myRank;
    (void)plan;
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace

HcclResult BuildMeshAllGatherSlicePlan(const TemplateDataParams &tempAlgParams,
                                       const TemplateResource &templateResource,
                                       const std::vector<u32> &ranks, u32 myRank,
                                       const MeshAllGatherPrimitiveOptions &options,
                                       MeshAllGatherSlicePlan &plan)
{
    // Planner dispatch is the only place that should branch on algorithm
    // variants. RunMeshAllGather below only consumes the resulting DataSlice
    // tasks, which keeps the communication trunk independent from layout rules.
    plan.tasks.clear();
    if (ranks.size() <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }

    plan.sendRecvMode = ResolveMeshAllGatherSendRecvMode(tempAlgParams, templateResource, options);
    switch (options.sliceMode) {
        case MeshAllGatherSliceMode::NORMAL_FIXED:
            return BuildMeshAllGatherNormalFixedPlan(tempAlgParams, templateResource, ranks, myRank, plan);
        case MeshAllGatherSliceMode::VARIABLE_COUNT:
            return BuildMeshAllGatherVariableCountPlan(tempAlgParams, templateResource, ranks, myRank, plan);
        case MeshAllGatherSliceMode::OMNIPIPE_STEP:
            return BuildMeshAllGatherOmniPipeStepPlan(tempAlgParams, templateResource, ranks, myRank, plan);
        case MeshAllGatherSliceMode::COMMON_CHANNEL_SPLIT:
            return BuildMeshAllGatherCommonChannelSplitPlan(tempAlgParams, templateResource, ranks, myRank, plan);
        case MeshAllGatherSliceMode::Z_AXIS_DETOUR:
            CHK_PRT_RET(!options.hasZAxisDetourConfig,
                        HCCL_ERROR("[MeshAllGatherPlanner] missing z-axis detour config."), HCCL_E_PARA);
            return BuildMeshAllGatherZAxisDetourPlan(tempAlgParams, templateResource, ranks, myRank,
                                                     options.zAxis, plan);
        case MeshAllGatherSliceMode::MESH_CHUNK:
            return BuildMeshAllGatherMeshChunkPlan(tempAlgParams, templateResource, ranks, myRank, plan);
        default:
            HCCL_ERROR("[MeshAllGatherPlanner] unsupported slice mode.");
            return HCCL_E_PARA;
    }
}

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            EngineType engineType, const std::vector<u32> &ranks, u32 myRank,
                            const MeshAllGatherPrimitiveOptions &options)
{
    (void)engineType;
    // Workflow:
    // 1. Convert params/resource/options into a variant-specific slice plan.
    // 2. Execute every planned peer/channel task with a uniform SendRecvInfo.
    // This is the target shape for removing old template class state.
    MeshAllGatherSlicePlan plan;
    CHK_RET(BuildMeshAllGatherSlicePlan(tempAlgParams, templateResource, ranks, myRank, options, plan));

    HcclDataType dataType = tempAlgParams.dataType;
    for (const auto &task : plan.tasks) {
        CHK_PRT_RET(task.link == nullptr || task.threadIdx >= templateResource.threads.size(),
                    HCCL_ERROR("[RunMeshAllGather] invalid slice plan task."), HCCL_E_PARA);
        SendRecvInfo info{{*task.link, *task.link},
                          {{task.txSrcSlices, task.txDstSlices}, {task.rxSrcSlices, task.rxDstSlices}},
                          dataType};
        CHK_RET(plan.sendRecvMode == MeshAllGatherSendRecvMode::DMA_READ ?
                    SendRecvRead(info, templateResource.threads[task.threadIdx]) :
                    SendRecvBatchWrite(info, templateResource.threads[task.threadIdx]));
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            EngineType engineType, const std::vector<u32> &ranks, u32 myRank)
{
    MeshAllGatherPrimitiveOptions options;
    // Preserve the current extracted primitive behavior. Future callers should pass an explicit
    // mode from TemplateDesc.variant instead of relying on this compatibility overload.
    options.sliceMode = MeshAllGatherSliceMode::COMMON_CHANNEL_SPLIT;
    return RunMeshAllGather(tempAlgParams, templateResource, engineType, ranks, myRank, options);
}

HcclResult RunMeshReduceScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
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
    u64 sliceSize = tempAlgParams.sliceSize;
    u64 tailSize = tempAlgParams.tailSize;
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u64 base = tempAlgParams.buffInfo.hcclBuffBaseOff;

    const std::vector<ChannelInfo> &portGroup = templateResource.channels.begin()->second;
    u32 channelsPerRank = static_cast<u32>(portGroup.size());
    std::vector<u64> ec, sizeOut, elemOffset;
    CHK_RET(CalcDataSplitByPortGroupCommon(sliceSize / dataTypeSize, dataTypeSize, portGroup, ec, sizeOut, elemOffset, channelsPerRank));
    std::vector<u64> ecT, sizeTail, elemOffsetTail;
    if (tailSize > 0) {
        CHK_RET(CalcDataSplitByPortGroupCommon(tailSize / dataTypeSize, dataTypeSize, portGroup, ecT, sizeTail, elemOffsetTail, channelsPerRank));
    }

    u32 t = 0;
    for (u32 i = 1; i < rankSize; ++i) {
        u32 peerIdx = (myRankIdx + i) % rankSize;
        bool peerTail = (peerIdx == rankSize - 1 && tailSize > 0);
        for (u32 ch = 0; ch < channelsPerRank; ++ch) {
            const ChannelInfo &link = templateResource.channels.at(ranks[peerIdx])[ch];
            u64 peerSz = peerTail ? sizeTail[ch] : sizeOut[ch];
            u64 peerOff = base + sliceSize * peerIdx + (peerTail ? elemOffsetTail[ch] : elemOffset[ch]);
            u64 mySz = sizeOut[ch];
            u64 myOff = base + sliceSize * myRankIdx + elemOffset[ch];
            std::vector<DataSlice> txSrc{{tempAlgParams.buffInfo.inputPtr, peerOff, peerSz, peerSz / dataTypeSize}};
            std::vector<DataSlice> txDst{{link.remoteCclMem.addr, myOff, mySz, mySz / dataTypeSize}};
            std::vector<DataSlice> rxSrc{{link.remoteCclMem.addr, myOff, mySz, mySz / dataTypeSize}};
            std::vector<DataSlice> rxDst{{tempAlgParams.buffInfo.hcclBuff.addr, myOff, mySz, mySz / dataTypeSize}};
            SendRecvReduceInfo info{{link, link}, {{txSrc, txDst}, {rxSrc, rxDst}}, dataType, {}};
            CHK_RET(isDmaRead ? SendRecvReadReduce(info, templateResource.threads[t]) :
                                SendRecvBatchWriteReduce(info, templateResource.threads[t]));
            t++;
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
