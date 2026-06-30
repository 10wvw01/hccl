/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 */

#include "mesh_allgather_planner.h"

namespace ops_hccl {

namespace {

struct MeshAllGatherChannelSlice {
    u32 channelIdx{0};
    const ChannelInfo *linkRemote{nullptr};
    u64 elemOffset{0};
    u64 txSliceSize{0};
    u64 rxSliceSize{0};
    u64 txSliceCount{0};
    u64 rxSliceCount{0};
};

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

HcclResult CheckRankVectorSize(const std::vector<u64> &values, u32 rankIdx, const char *name)
{
    CHK_PRT_RET(values.size() <= rankIdx,
                HCCL_ERROR("[MeshAllGatherPlanner] invalid %s size[%u], rankIdx[%u].",
                           name, static_cast<u32>(values.size()), rankIdx),
                HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult CheckStepRankSize(const StepSliceInfo &stepSliceInfo, u32 rankIdx)
{
    CHK_PRT_RET(stepSliceInfo.stepCount.size() <= rankIdx ||
                    stepSliceInfo.stepSliceSize.size() <= rankIdx ||
                    stepSliceInfo.stepInputSliceStride.size() <= rankIdx ||
                    stepSliceInfo.stepOutputSliceStride.size() <= rankIdx ||
                    stepSliceInfo.inputOmniPipeSliceStride.size() <= rankIdx ||
                    stepSliceInfo.outputOmniPipeSliceStride.size() <= rankIdx,
                HCCL_ERROR("[MeshAllGatherPlanner] invalid omnipipe rank slice info, rankIdx[%u].", rankIdx),
                HCCL_E_PARA);

    const u32 stepNum = static_cast<u32>(stepSliceInfo.inputOmniPipeSliceStride[rankIdx].size());
    CHK_PRT_RET(stepSliceInfo.stepCount[rankIdx].size() < stepNum ||
                    stepSliceInfo.stepSliceSize[rankIdx].size() < stepNum ||
                    stepSliceInfo.outputOmniPipeSliceStride[rankIdx].size() < stepNum,
                HCCL_ERROR("[MeshAllGatherPlanner] invalid omnipipe step slice info, rankIdx[%u].", rankIdx),
                HCCL_E_PARA);
    return HCCL_SUCCESS;
}

MeshAllGatherSendRecvMode ResolveMeshAllGatherSendRecvMode(const TemplateDataParams &tempAlgParams,
                                                           const TemplateResource &templateResource,
                                                           const MeshAllGatherPrimitiveOptions &options)
{
    if (options.sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR) {
        const bool dmaRead = (tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER &&
                              tempAlgParams.buffInfo.outBuffType != BufferType::HCCL_BUFFER);
        return dmaRead ? MeshAllGatherSendRecvMode::DMA_READ : MeshAllGatherSendRecvMode::WRITE;
    }

    if (options.sliceMode == MeshAllGatherSliceMode::VARIABLE_COUNT ||
        options.sliceMode == MeshAllGatherSliceMode::OMNIPIPE_STEP) {
        return MeshAllGatherSendRecvMode::WRITE;
    }

    if (options.sliceMode == MeshAllGatherSliceMode::NORMAL_FIXED) {
        return MeshAllGatherSendRecvMode::DMA_READ;
    }

    return IsPcieProtocol(templateResource.channels) ?
        MeshAllGatherSendRecvMode::DMA_READ : MeshAllGatherSendRecvMode::BATCH_WRITE;
}

HcclResult PushSingleChannelSlice(const std::vector<ChannelInfo> &channels, u64 txSliceSize, u64 rxSliceSize,
                                  u64 txSliceCount, u64 rxSliceCount,
                                  std::vector<MeshAllGatherChannelSlice> &channelSlices)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[MeshAllGatherPlanner] empty channel."), HCCL_E_PARA);
    channelSlices.push_back({0, &channels[0], 0, txSliceSize, rxSliceSize, txSliceCount, rxSliceCount});
    return HCCL_SUCCESS;
}

HcclResult ResolveMeshAllGatherChannelSlices(const TemplateDataParams &tempAlgParams,
                                             const TemplateResource &templateResource,
                                             const MeshAllGatherPrimitiveOptions &options,
                                             u32 myAlgRank, u32 connectedRank, u32 connectedAlgRank,
                                             u32 rankSize, u32 dataTypeSize,
                                             MeshAllGatherSendRecvMode sendRecvMode,
                                             std::vector<MeshAllGatherChannelSlice> &channelSlices)
{
    CHK_PRT_RET(templateResource.channels.count(connectedRank) == 0 ||
                    templateResource.channels.at(connectedRank).empty(),
                HCCL_ERROR("[MeshAllGatherPlanner] connectedRank[%u] has no channel.", connectedRank),
                HCCL_E_PARA);

    const std::vector<ChannelInfo> &channels = templateResource.channels.at(connectedRank);
    const bool zAxisWriteTail =
        (options.sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR &&
         sendRecvMode == MeshAllGatherSendRecvMode::WRITE &&
         myAlgRank == rankSize - 1 && tempAlgParams.tailSize > 0);
    const bool connectedRankHasTail = (options.sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR &&
        sendRecvMode == MeshAllGatherSendRecvMode::WRITE) ?
        zAxisWriteTail : (connectedAlgRank == rankSize - 1 && tempAlgParams.tailSize > 0);
    const u64 sliceSize = connectedRankHasTail ? tempAlgParams.tailSize : tempAlgParams.sliceSize;

    if (options.sliceMode == MeshAllGatherSliceMode::NORMAL_FIXED) {
        return PushSingleChannelSlice(channels, sliceSize, sliceSize, sliceSize / dataTypeSize,
                                      sliceSize / dataTypeSize, channelSlices);
    }

    if (options.sliceMode == MeshAllGatherSliceMode::VARIABLE_COUNT) {
        // all_gather_v 仍复用 Mesh 外层 peer/channel 遍历，只把本端和对端的 rank slice size 改成变长输入。
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankSliceSize, myAlgRank, "allRankSliceSize"));
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankSliceSize, connectedAlgRank, "allRankSliceSize"));
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankDispls, myAlgRank, "allRankDispls"));
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankDispls, connectedAlgRank, "allRankDispls"));
        return PushSingleChannelSlice(channels, tempAlgParams.allRankSliceSize[myAlgRank],
                                      tempAlgParams.allRankSliceSize[connectedAlgRank],
                                      tempAlgParams.count, tempAlgParams.count, channelSlices);
    }

    if (options.sliceMode == MeshAllGatherSliceMode::OMNIPIPE_STEP) {
        // OmniPipe Mesh 仍是一个 peer 对应一个 channel task，差异是 task 内会追加多组 step DataSlice。
        CHK_RET(CheckStepRankSize(tempAlgParams.stepSliceInfo, myAlgRank));
        CHK_RET(CheckStepRankSize(tempAlgParams.stepSliceInfo, connectedAlgRank));
        const u32 stepNum = static_cast<u32>(
            tempAlgParams.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size());
        CHK_PRT_RET(tempAlgParams.stepSliceInfo.stepSliceSize[connectedAlgRank].size() < stepNum ||
                        tempAlgParams.stepSliceInfo.outputOmniPipeSliceStride[connectedAlgRank].size() < stepNum,
                    HCCL_ERROR("[MeshAllGatherPlanner] omnipipe peer step size mismatch."), HCCL_E_PARA);
        return PushSingleChannelSlice(channels, 0, 0, 0, 0, channelSlices);
    }

    if (options.sliceMode == MeshAllGatherSliceMode::COMMON_CHANNEL_SPLIT) {
        const u32 channelsPerRank = static_cast<u32>(channels.size());
        std::vector<u64> ec;
        std::vector<u64> sizeOut;
        std::vector<u64> elemOffset;
        CHK_RET(CalcDataSplitByPortGroupCommon(sliceSize / dataTypeSize, dataTypeSize, channels,
                                                ec, sizeOut, elemOffset, channelsPerRank));
        for (u32 channelIdx = 0; channelIdx < channelsPerRank; ++channelIdx) {
            channelSlices.push_back({channelIdx, &channels[channelIdx], elemOffset[channelIdx],
                                     sizeOut[channelIdx], sizeOut[channelIdx], ec[channelIdx], ec[channelIdx]});
        }
        return HCCL_SUCCESS;
    }

    if (options.sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR) {
        std::vector<u64> ec;
        std::vector<u64> sizeOut;
        std::vector<u64> elemOffset;
        CHK_RET(CalcDataSplitByPortGroupZAxisDetour(sliceSize / dataTypeSize, dataTypeSize, channels,
                                                    ec, sizeOut, elemOffset,
                                                    options.zAxis.level0ChannelNumPerRank,
                                                    options.zAxis.level1ChannelNumPerRank,
                                                    static_cast<float>(options.zAxis.level0DataRatio)));
        for (u32 channelIdx = 0; channelIdx < channels.size(); ++channelIdx) {
            channelSlices.push_back({channelIdx, &channels[channelIdx], elemOffset[channelIdx],
                                     sizeOut[channelIdx], sizeOut[channelIdx], ec[channelIdx], ec[channelIdx]});
        }
        return HCCL_SUCCESS;
    }

    return HCCL_E_NOT_SUPPORT;
}

void AppendCclBufferSlices(const TemplateDataParams &tempAlgParams, u64 txOutOffset, u64 rxOutOffset,
                           const MeshAllGatherChannelSlice &channelSlice,
                           MeshAllGatherPeerChannelPlan &task)
{
    task.txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr,
                                  txOutOffset, channelSlice.txSliceSize, channelSlice.txSliceCount);
    task.txDstSlices.emplace_back(channelSlice.linkRemote->remoteCclMem.addr,
                                  txOutOffset, channelSlice.txSliceSize, channelSlice.txSliceCount);
    task.rxSrcSlices.emplace_back(channelSlice.linkRemote->remoteCclMem.addr,
                                  rxOutOffset, channelSlice.rxSliceSize, channelSlice.rxSliceCount);
    task.rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr,
                                  rxOutOffset, channelSlice.rxSliceSize, channelSlice.rxSliceCount);
}

void AppendOutputRemoteSlices(const TemplateDataParams &tempAlgParams, u64 txOutOffset, u64 rxOutOffset,
                              u64 txDstOffset, u64 rxSrcOffset,
                              const MeshAllGatherChannelSlice &channelSlice,
                              MeshAllGatherPeerChannelPlan &task)
{
    void *txDstPtr = (!tempAlgParams.enableRemoteMemAccess) ?
        channelSlice.linkRemote->remoteCclMem.addr : channelSlice.linkRemote->remoteOutputGraphMode.addr;
    void *rxSrcPtr = (!tempAlgParams.enableRemoteMemAccess) ?
        channelSlice.linkRemote->remoteCclMem.addr : channelSlice.linkRemote->remoteOutputGraphMode.addr;

    task.txSrcSlices.emplace_back(tempAlgParams.buffInfo.outputPtr,
                                  txOutOffset, channelSlice.txSliceSize, channelSlice.txSliceCount);
    task.txDstSlices.emplace_back(txDstPtr, txDstOffset, channelSlice.txSliceSize, channelSlice.txSliceCount);
    task.rxSrcSlices.emplace_back(rxSrcPtr, rxSrcOffset, channelSlice.rxSliceSize, channelSlice.rxSliceCount);
    task.rxDstSlices.emplace_back(tempAlgParams.buffInfo.outputPtr,
                                  rxOutOffset, channelSlice.rxSliceSize, channelSlice.rxSliceCount);
}

void AppendOmniPipeStepSlices(const TemplateDataParams &tempAlgParams, u32 myAlgRank, u32 connectedAlgRank,
                              const MeshAllGatherChannelSlice &channelSlice,
                              MeshAllGatherPeerChannelPlan &task)
{
    const StepSliceInfo &stepSliceInfo = tempAlgParams.stepSliceInfo;
    const u32 stepNum = static_cast<u32>(stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size());
    for (u32 rpt = 0; rpt < stepNum; ++rpt) {
        const u64 txBaseOff = tempAlgParams.buffInfo.inBuffBaseOff +
            stepSliceInfo.inputOmniPipeSliceStride[myAlgRank][rpt];
        const u64 rxBaseOff = tempAlgParams.buffInfo.outBuffBaseOff +
            stepSliceInfo.outputOmniPipeSliceStride[connectedAlgRank][rpt];
        const u64 txOffset = stepSliceInfo.stepInputSliceStride[myAlgRank] + txBaseOff;
        const u64 rxOffset = stepSliceInfo.stepOutputSliceStride[connectedAlgRank] + rxBaseOff;

        task.txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr,
                                      txOffset, stepSliceInfo.stepSliceSize[myAlgRank][rpt],
                                      stepSliceInfo.stepCount[myAlgRank][rpt]);
        task.txDstSlices.emplace_back(channelSlice.linkRemote->remoteCclMem.addr,
                                      txOffset, stepSliceInfo.stepSliceSize[myAlgRank][rpt],
                                      stepSliceInfo.stepCount[myAlgRank][rpt]);
        task.rxSrcSlices.emplace_back(channelSlice.linkRemote->remoteCclMem.addr,
                                      rxOffset, stepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                                      stepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);
        task.rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr,
                                      rxOffset, stepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                                      stepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);
    }
}

void AppendMeshAllGatherDataSlices(const TemplateDataParams &tempAlgParams,
                                   MeshAllGatherSliceMode sliceMode,
                                   u32 myAlgRank, u32 connectedAlgRank, u32 rankSize,
                                   const MeshAllGatherChannelSlice &channelSlice,
                                   MeshAllGatherPeerChannelPlan &task)
{
    if (sliceMode == MeshAllGatherSliceMode::COMMON_CHANNEL_SPLIT) {
        const u64 txOutOffset = tempAlgParams.buffInfo.hcclBuffBaseOff +
            tempAlgParams.sliceSize * myAlgRank + channelSlice.elemOffset;
        const u64 rxOutOffset = tempAlgParams.buffInfo.hcclBuffBaseOff +
            tempAlgParams.sliceSize * connectedAlgRank + channelSlice.elemOffset;
        AppendCclBufferSlices(tempAlgParams, txOutOffset, rxOutOffset, channelSlice, task);
        return;
    }

    if (sliceMode == MeshAllGatherSliceMode::OMNIPIPE_STEP) {
        // 旧 OmniPipe Mesh 在同一次 SendRecvWrite 中携带多段 step slice，这里不拆外层 task。
        AppendOmniPipeStepSlices(tempAlgParams, myAlgRank, connectedAlgRank, channelSlice, task);
        return;
    }

    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
        const bool variableCount = (sliceMode == MeshAllGatherSliceMode::VARIABLE_COUNT);
        // all_gather_v 只改变 rank 级 output displ 和 rank slice size，其余远端地址选择沿用普通路径。
        const u64 scratchRepeatStride = variableCount ?
            tempAlgParams.sliceSize * DATATYPE_SIZE_TABLE[tempAlgParams.dataType] : tempAlgParams.sliceSize * rankSize;
        u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
        if (sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR &&
            tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
            scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams.inputRepeatStride;
        }

        const u64 txOutOffset = variableCount ?
            tempAlgParams.allRankDispls[myAlgRank] * DATATYPE_SIZE_TABLE[tempAlgParams.dataType] + outBaseOff :
            tempAlgParams.outputSliceStride * myAlgRank + outBaseOff + channelSlice.elemOffset;
        const u64 rxOutOffset = variableCount ?
            tempAlgParams.allRankDispls[connectedAlgRank] * DATATYPE_SIZE_TABLE[tempAlgParams.dataType] + outBaseOff :
            tempAlgParams.outputSliceStride * connectedAlgRank + outBaseOff + channelSlice.elemOffset;
        const u64 txScratchOffset = scratchBase + tempAlgParams.sliceSize * myAlgRank + channelSlice.elemOffset;
        const u64 rxScratchOffset = scratchBase +
            ((sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR) ? tempAlgParams.inputSliceStride :
                                                                    tempAlgParams.sliceSize) * connectedAlgRank +
            channelSlice.elemOffset;
        const u64 txDstOffset = (!tempAlgParams.enableRemoteMemAccess) ? txScratchOffset : txOutOffset;
        const u64 rxSrcOffset = (!tempAlgParams.enableRemoteMemAccess) ? rxScratchOffset : rxOutOffset;
        AppendOutputRemoteSlices(tempAlgParams, txOutOffset, rxOutOffset, txDstOffset, rxSrcOffset,
                                 channelSlice, task);
    }
}

}  // namespace

HcclResult BuildMeshAllGatherSlicePlan(const TemplateDataParams &tempAlgParams,
                                       const TemplateResource &templateResource,
                                       const std::vector<u32> &ranks, u32 myRank,
                                       const MeshAllGatherPrimitiveOptions &options,
                                       MeshAllGatherSlicePlan &plan)
{
    plan.tasks.clear();
    if (ranks.size() <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }

    if (options.sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR && !options.hasZAxisDetourConfig) {
        HCCL_ERROR("[MeshAllGatherPlanner] missing z-axis detour config.");
        return HCCL_E_PARA;
    }

    plan.sendRecvMode = ResolveMeshAllGatherSendRecvMode(tempAlgParams, templateResource, options);

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
    CHK_PRT_RET(!rankFound, HCCL_ERROR("[MeshAllGatherPlanner] rank[%u] is not in ranks.", myRank), HCCL_E_PARA);

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    u32 threadIdx = 0;
    for (u32 i = 1; i < rankSize; ++i) {
        const u32 connectedAlgRank = (options.sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR) ?
            (i <= myAlgRank ? i - 1 : i) : (myAlgRank + i) % rankSize;
        const u32 connectedRank = ranks[connectedAlgRank];

        std::vector<MeshAllGatherChannelSlice> channelSlices;
        CHK_RET(ResolveMeshAllGatherChannelSlices(tempAlgParams, templateResource, options, myAlgRank, connectedRank,
                                                  connectedAlgRank, rankSize, dataTypeSize, plan.sendRecvMode,
                                                  channelSlices));
        for (const auto &channelSlice : channelSlices) {
            CHK_PRT_RET(channelSlice.linkRemote == nullptr,
                        HCCL_ERROR("[MeshAllGatherPlanner] invalid channel slice."), HCCL_E_PARA);
            MeshAllGatherPeerChannelPlan task;
            task.connectedRank = connectedRank;
            task.connectedAlgRank = connectedAlgRank;
            task.channelIdx = channelSlice.channelIdx;
            task.threadIdx = threadIdx++;
            task.linkRemote = channelSlice.linkRemote;
            AppendMeshAllGatherDataSlices(tempAlgParams, options.sliceMode, myAlgRank, connectedAlgRank,
                                          rankSize, channelSlice, task);
            plan.tasks.emplace_back(task);
        }
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
