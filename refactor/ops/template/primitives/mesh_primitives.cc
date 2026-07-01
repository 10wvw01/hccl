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

enum class MeshAllGatherSendRecvMode {
    DMA_READ,
    WRITE,
    BATCH_WRITE,
};

struct MeshAllGatherChannelSlice {
    u32 channelIdx{0};
    const ChannelInfo *linkRemote{nullptr};
    u64 elemOffset{0};
    u64 txSliceSize{0};
    u64 rxSliceSize{0};
    u64 txSliceCount{0};
    u64 rxSliceCount{0};
};

struct MeshAllGatherPeerChannelPlan {
    u32 connectedRank{0};
    u32 connectedAlgRank{0};
    u32 channelIdx{0};
    u32 threadIdx{0};
    const ChannelInfo *linkRemote{nullptr};
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
};

struct MeshAllGatherSlicePlan {
    MeshAllGatherSendRecvMode sendRecvMode{MeshAllGatherSendRecvMode::DMA_READ};
    std::vector<MeshAllGatherPeerChannelPlan> tasks;
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
                HCCL_ERROR("[MeshAllGatherPlan] invalid %s size[%u], rankIdx[%u].",
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
                HCCL_ERROR("[MeshAllGatherPlan] invalid omnipipe rank slice info, rankIdx[%u].", rankIdx),
                HCCL_E_PARA);

    const u32 stepNum = static_cast<u32>(stepSliceInfo.inputOmniPipeSliceStride[rankIdx].size());
    CHK_PRT_RET(stepSliceInfo.stepCount[rankIdx].size() < stepNum ||
                    stepSliceInfo.stepSliceSize[rankIdx].size() < stepNum ||
                    stepSliceInfo.outputOmniPipeSliceStride[rankIdx].size() < stepNum,
                HCCL_ERROR("[MeshAllGatherPlan] invalid omnipipe step slice info, rankIdx[%u].", rankIdx),
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
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[MeshAllGatherPlan] empty channel."), HCCL_E_PARA);
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
                HCCL_ERROR("[MeshAllGatherPlan] connectedRank[%u] has no channel.", connectedRank),
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
        // all_gather_v 只替换本端/对端 rank 的 size，peer/channel 主循环继续复用普通 Mesh。
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankSliceSize, myAlgRank, "allRankSliceSize"));
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankSliceSize, connectedAlgRank, "allRankSliceSize"));
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankDispls, myAlgRank, "allRankDispls"));
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankDispls, connectedAlgRank, "allRankDispls"));
        return PushSingleChannelSlice(channels, tempAlgParams.allRankSliceSize[myAlgRank],
                                      tempAlgParams.allRankSliceSize[connectedAlgRank],
                                      tempAlgParams.count, tempAlgParams.count, channelSlices);
    }

    if (options.sliceMode == MeshAllGatherSliceMode::OMNIPIPE_STEP) {
        // OmniPipe 的差异在 task 内部追加多组 step slice，而不是拆出一套外层执行循环。
        CHK_RET(CheckStepRankSize(tempAlgParams.stepSliceInfo, myAlgRank));
        CHK_RET(CheckStepRankSize(tempAlgParams.stepSliceInfo, connectedAlgRank));
        const u32 stepNum = static_cast<u32>(
            tempAlgParams.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size());
        CHK_PRT_RET(tempAlgParams.stepSliceInfo.stepSliceSize[connectedAlgRank].size() < stepNum ||
                        tempAlgParams.stepSliceInfo.outputOmniPipeSliceStride[connectedAlgRank].size() < stepNum,
                    HCCL_ERROR("[MeshAllGatherPlan] omnipipe peer step size mismatch."), HCCL_E_PARA);
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

void AppendOutputRemoteSlices(const TemplateDataParams &tempAlgParams, u64 txOutOffset,
                              u64 txDstOffset, u64 rxSrcOffset, u64 rxDstOffset,
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
                                  rxDstOffset, channelSlice.rxSliceSize, channelSlice.rxSliceCount);
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
                                   u32 repeatBegin, u32 repeatEnd, MeshAllGatherPeerChannelPlan &task)
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
        AppendOmniPipeStepSlices(tempAlgParams, myAlgRank, connectedAlgRank, channelSlice, task);
        return;
    }

    for (u32 rpt = repeatBegin; rpt < repeatEnd; ++rpt) {
        const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
        const bool variableCount = (sliceMode == MeshAllGatherSliceMode::VARIABLE_COUNT);
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
        // all_gather_v 旧模板里 rxSrc 使用 output offset，rxDst 使用 scratch/remote-read offset。
        // 普通 Mesh AllGather 旧模板则相反；这里仅对 VARIABLE_COUNT 保持旧 all_gather_v 的摆法。
        const u64 rxSrcSliceOffset = variableCount ? rxOutOffset : rxSrcOffset;
        const u64 rxDstSliceOffset = variableCount ? rxSrcOffset : rxOutOffset;
        AppendOutputRemoteSlices(tempAlgParams, txOutOffset, txDstOffset, rxSrcSliceOffset, rxDstSliceOffset,
                                 channelSlice, task);
    }
}

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
        HCCL_ERROR("[MeshAllGatherPlan] missing z-axis detour config.");
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
    CHK_PRT_RET(!rankFound, HCCL_ERROR("[MeshAllGatherPlan] rank[%u] is not in ranks.", myRank), HCCL_E_PARA);

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    const bool variableCount = (options.sliceMode == MeshAllGatherSliceMode::VARIABLE_COUNT);
    const u32 repeatTaskNum = variableCount ? tempAlgParams.repeatNum : 1;
    for (u32 repeatTaskIdx = 0; repeatTaskIdx < repeatTaskNum; ++repeatTaskIdx) {
        u32 threadIdx = 0;
        const u32 repeatBegin = variableCount ? repeatTaskIdx : 0;
        const u32 repeatEnd = variableCount ? repeatTaskIdx + 1 : tempAlgParams.repeatNum;
        for (u32 i = 1; i < rankSize; ++i) {
            const u32 connectedAlgRank = (options.sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR) ?
                (i <= myAlgRank ? i - 1 : i) : (myAlgRank + i) % rankSize;
            const u32 connectedRank = ranks[connectedAlgRank];

            std::vector<MeshAllGatherChannelSlice> channelSlices;
            CHK_RET(ResolveMeshAllGatherChannelSlices(tempAlgParams, templateResource, options, myAlgRank,
                                                      connectedRank, connectedAlgRank, rankSize, dataTypeSize,
                                                      plan.sendRecvMode, channelSlices));
            for (const auto &channelSlice : channelSlices) {
                CHK_PRT_RET(channelSlice.linkRemote == nullptr,
                            HCCL_ERROR("[MeshAllGatherPlan] invalid channel slice."), HCCL_E_PARA);
                MeshAllGatherPeerChannelPlan task;
                task.connectedRank = connectedRank;
                task.connectedAlgRank = connectedAlgRank;
                task.channelIdx = channelSlice.channelIdx;
                task.threadIdx = threadIdx++;
                task.linkRemote = channelSlice.linkRemote;
                AppendMeshAllGatherDataSlices(tempAlgParams, options.sliceMode, myAlgRank, connectedAlgRank,
                                              rankSize, channelSlice, repeatBegin, repeatEnd, task);
                plan.tasks.emplace_back(task);
            }
        }
    }
    return HCCL_SUCCESS;
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
    const HcclDataType dataType = tempAlgParams.dataType;
    // 这里假设上层 executor/resource planning 已经申请好 threads/channels/notify/sync。
    // 当前 primitive 只消费 plan 并校验 threadIdx 边界，不负责补齐资源规划。
    for (const auto &task : plan.tasks) {
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
