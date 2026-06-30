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

const char *MeshAllGatherSendRecvModeName(MeshAllGatherSendRecvMode mode)
{
    switch (mode) {
        case MeshAllGatherSendRecvMode::DMA_READ:
            return "DMA_READ";
        case MeshAllGatherSendRecvMode::WRITE:
            return "WRITE";
        case MeshAllGatherSendRecvMode::BATCH_WRITE:
            return "BATCH_WRITE";
        default:
            return "UNKNOWN";
    }
}

void DumpMeshAllGatherDataSlices(const char *tag, const std::vector<DataSlice> &slices)
{
    for (u32 sliceIdx = 0; sliceIdx < slices.size(); ++sliceIdx) {
        const DataSlice &slice = slices[sliceIdx];
        HCCL_DEBUG("[RunMeshAllGather][PlanDump][%s] sliceIdx[%u], base[%p], offset[%llu], "
                   "size[%llu], count[%llu].",
                   tag, sliceIdx, slice.addr_, static_cast<unsigned long long>(slice.offset_),
                   static_cast<unsigned long long>(slice.size_), static_cast<unsigned long long>(slice.count_));
    }
}

void DumpMeshAllGatherSlicePlan(const MeshAllGatherSlicePlan &plan)
{
    HCCL_DEBUG("[RunMeshAllGather][PlanDump] sendRecvMode[%s], taskNum[%u].",
               MeshAllGatherSendRecvModeName(plan.sendRecvMode), static_cast<u32>(plan.tasks.size()));
    for (u32 taskIdx = 0; taskIdx < plan.tasks.size(); ++taskIdx) {
        const MeshAllGatherPeerChannelPlan &task = plan.tasks[taskIdx];
        HCCL_DEBUG("[RunMeshAllGather][PlanDump] taskIdx[%u], threadIdx[%u], connectedRank[%u], "
                   "connectedAlgRank[%u], channelIdx[%u], linkRemote[%p].",
                   taskIdx, task.threadIdx, task.connectedRank, task.connectedAlgRank, task.channelIdx,
                   task.linkRemote);
        DumpMeshAllGatherDataSlices("txSrc", task.txSrcSlices);
        DumpMeshAllGatherDataSlices("txDst", task.txDstSlices);
        DumpMeshAllGatherDataSlices("rxSrc", task.rxSrcSlices);
        DumpMeshAllGatherDataSlices("rxDst", task.rxDstSlices);
    }
}

}  // namespace

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            EngineType engineType, const std::vector<u32> &ranks, u32 myRank,
                            const MeshAllGatherPrimitiveOptions &options)
{
    (void)engineType;
    // plan 是 planner 和执行层之间的边界：上面负责算清楚 DataSlice，下面只负责发起通信。
    MeshAllGatherSlicePlan plan;
    CHK_RET(BuildMeshAllGatherSlicePlan(tempAlgParams, templateResource, ranks, myRank, options, plan));
    DumpMeshAllGatherSlicePlan(plan);

    const HcclDataType dataType = tempAlgParams.dataType;
    for (const auto &task : plan.tasks) {
        CHK_PRT_RET(task.linkRemote == nullptr || task.threadIdx >= templateResource.threads.size(),
                    HCCL_ERROR("[RunMeshAllGather] invalid slice plan task."), HCCL_E_PARA);
        SendRecvInfo sendRecvInfo{{*task.linkRemote, *task.linkRemote},
                                  {{task.txSrcSlices, task.txDstSlices}, {task.rxSrcSlices, task.rxDstSlices}},
                                  dataType};
        if (plan.sendRecvMode == MeshAllGatherSendRecvMode::DMA_READ) {
            CHK_RET(SendRecvRead(sendRecvInfo, templateResource.threads[task.threadIdx]));
        } else if (plan.sendRecvMode == MeshAllGatherSendRecvMode::WRITE) {
            CHK_RET(SendRecvWrite(sendRecvInfo, templateResource.threads[task.threadIdx]));
        } else {
            CHK_RET(SendRecvBatchWrite(sendRecvInfo, templateResource.threads[task.threadIdx]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            EngineType engineType, const std::vector<u32> &ranks, u32 myRank)
{
    MeshAllGatherPrimitiveOptions options;
    // 保留当前已抽取 primitive 的行为。
    // 后续调用者应从 TemplateDesc.variant 显式传入 mode，而不是依赖这个兼容重载。
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
