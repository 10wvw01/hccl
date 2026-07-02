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

enum class MeshSendRecvMode {
    DMA_READ,
    WRITE,
    BATCH_WRITE,
};

enum class MeshPrimitiveOp {
    ALL_GATHER,
    REDUCE_SCATTER,
};

struct MeshChannelSlice {
    u32 channelIdx{0};
    const ChannelInfo *linkRemote{nullptr};
    u64 elemOffset{0};
    u64 txSliceSize{0};
    u64 rxSliceSize{0};
    u64 txSliceCount{0};
    u64 rxSliceCount{0};
};

struct MeshTransferSlices {
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
};

struct MeshSliceBuildInfo {
    MeshPrimitiveOp opType{MeshPrimitiveOp::ALL_GATHER};
    MeshTransferSliceMode sliceMode{MeshTransferSliceMode::NORMAL_FIXED};
    u32 myAlgRank{0};
    u32 connectedAlgRank{0};
    u32 rankSize{0};
    u32 repeatBegin{0};
    u32 repeatEnd{0};
    const MeshChannelSlice *channelSlice{nullptr};
    const ChannelInfo *linkRemote{nullptr};
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

HcclResult PushSingleChannelSlice(const std::vector<ChannelInfo> &channels, u64 txSliceSize, u64 rxSliceSize,
                                  u64 txSliceCount, u64 rxSliceCount,
                                  std::vector<MeshChannelSlice> &channelSlices)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[MeshAllGatherPlan] empty channel."), HCCL_E_PARA);
    channelSlices.push_back({0, &channels[0], 0, txSliceSize, rxSliceSize, txSliceCount, rxSliceCount});
    return HCCL_SUCCESS;
}

HcclResult ResolveMeshChannelSlices(const TemplateDataParams &tempAlgParams,
                                    const TemplateResource &templateResource,
                                    const MeshPrimitiveOptions &options,
                                    MeshPrimitiveOp opType,
                                    u32 myAlgRank, u32 connectedRank, u32 connectedAlgRank,
                                    u32 rankSize, u32 dataTypeSize,
                                    MeshSendRecvMode sendRecvMode,
                                    std::vector<MeshChannelSlice> &channelSlices)
{
    CHK_PRT_RET(templateResource.channels.count(connectedRank) == 0 ||
                    templateResource.channels.at(connectedRank).empty(),
                HCCL_ERROR("[MeshChannelSlices] connectedRank[%u] has no channel.", connectedRank),
                HCCL_E_PARA);

    const std::vector<ChannelInfo> &channels = templateResource.channels.at(connectedRank);
    CHK_PRT_RET(options.sliceMode == MeshTransferSliceMode::MESH_CHUNK,
                HCCL_ERROR("[MeshChannelSlices] MeshChunk is not a normal mesh channel slice mode."),
                HCCL_E_NOT_SUPPORT);

    // ---- AllGather-only modes (early return) ----
    if (options.sliceMode == MeshTransferSliceMode::NORMAL_FIXED) {
        CHK_PRT_RET(opType != MeshPrimitiveOp::ALL_GATHER,
                    HCCL_ERROR("[MeshChannelSlices] NORMAL_FIXED is only supported by AllGather."), HCCL_E_NOT_SUPPORT);
        const bool hasTail = (connectedAlgRank == rankSize - 1 && tempAlgParams.tailSize > 0);
        const u64 sliceSize = hasTail ? tempAlgParams.tailSize : tempAlgParams.sliceSize;
        return PushSingleChannelSlice(channels, sliceSize, sliceSize, sliceSize / dataTypeSize,
                                      sliceSize / dataTypeSize, channelSlices);
    }

    if (options.sliceMode == MeshTransferSliceMode::VARIABLE_COUNT) {
        CHK_PRT_RET(opType != MeshPrimitiveOp::ALL_GATHER,
                    HCCL_ERROR("[MeshChannelSlices] VARIABLE_COUNT is only supported by AllGather."), HCCL_E_NOT_SUPPORT);
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankSliceSize, myAlgRank, "allRankSliceSize"));
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankSliceSize, connectedAlgRank, "allRankSliceSize"));
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankDispls, myAlgRank, "allRankDispls"));
        CHK_RET(CheckRankVectorSize(tempAlgParams.allRankDispls, connectedAlgRank, "allRankDispls"));
        return PushSingleChannelSlice(channels, tempAlgParams.allRankSliceSize[myAlgRank],
                                      tempAlgParams.allRankSliceSize[connectedAlgRank],
                                      tempAlgParams.count, tempAlgParams.count, channelSlices);
    }

    // ---- OmniPipe (shared) ----
    if (options.sliceMode == MeshTransferSliceMode::OMNIPIPE_STEP) {
        CHK_RET(CheckStepRankSize(tempAlgParams.stepSliceInfo, myAlgRank));
        CHK_RET(CheckStepRankSize(tempAlgParams.stepSliceInfo, connectedAlgRank));
        const u32 stepNum = static_cast<u32>(
            tempAlgParams.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size());
        CHK_PRT_RET(tempAlgParams.stepSliceInfo.stepCount[connectedAlgRank].size() < stepNum ||
                        tempAlgParams.stepSliceInfo.stepSliceSize[connectedAlgRank].size() < stepNum ||
                        tempAlgParams.stepSliceInfo.inputOmniPipeSliceStride[connectedAlgRank].size() < stepNum ||
                        tempAlgParams.stepSliceInfo.outputOmniPipeSliceStride[connectedAlgRank].size() < stepNum,
                    HCCL_ERROR("[MeshChannelSlices] omnipipe peer step size mismatch."), HCCL_E_PARA);
        return PushSingleChannelSlice(channels, 0, 0, 0, 0, channelSlices);
    }

    // ---- Tail-aware slice size (shared: COMMON_CHANNEL_SPLIT / Z_AXIS_DETOUR) ----
    // Z-axis WRITE mode uses myAlgRank for tail; all other paths use connectedAlgRank.
    // ReduceScatter never passes WRITE, so it always takes the standard fallback.
    const bool zAxisWriteTail = (options.sliceMode == MeshTransferSliceMode::Z_AXIS_DETOUR &&
                                 sendRecvMode == MeshSendRecvMode::WRITE &&
                                 myAlgRank == rankSize - 1 && tempAlgParams.tailSize > 0);
    const bool connectedRankHasTail = (options.sliceMode == MeshTransferSliceMode::Z_AXIS_DETOUR &&
        sendRecvMode == MeshSendRecvMode::WRITE) ?
        zAxisWriteTail : (connectedAlgRank == rankSize - 1 && tempAlgParams.tailSize > 0);
    const u64 sliceSize = connectedRankHasTail ? tempAlgParams.tailSize : tempAlgParams.sliceSize;

    // ---- Channel split (shared) ----
    std::vector<u64> ec;
    std::vector<u64> sizeOut;
    std::vector<u64> elemOffset;
    if (options.sliceMode == MeshTransferSliceMode::Z_AXIS_DETOUR) {
        CHK_RET(CalcDataSplitByPortGroupZAxisDetour(sliceSize / dataTypeSize, dataTypeSize, channels,
                                                    ec, sizeOut, elemOffset,
                                                    options.zAxis.level0ChannelNumPerRank,
                                                    options.zAxis.level1ChannelNumPerRank,
                                                    static_cast<float>(options.zAxis.level0DataRatio)));
    } else {
        const u32 channelsPerRank = static_cast<u32>(channels.size());
        CHK_RET(CalcDataSplitByPortGroupCommon(sliceSize / dataTypeSize, dataTypeSize, channels,
                                                ec, sizeOut, elemOffset, channelsPerRank));
    }

    for (u32 channelIdx = 0; channelIdx < channels.size(); ++channelIdx) {
        channelSlices.push_back({channelIdx, &channels[channelIdx], elemOffset[channelIdx],
                                 sizeOut[channelIdx], sizeOut[channelIdx], ec[channelIdx], ec[channelIdx]});
    }
    return HCCL_SUCCESS;
}

HcclResult BuildMeshTransferSlices(const TemplateDataParams &tempAlgParams,
                                   const MeshSliceBuildInfo &sliceBuildInfo,
                                   MeshTransferSlices &slices)
{
    slices = MeshTransferSlices{};

    if (sliceBuildInfo.sliceMode == MeshTransferSliceMode::MESH_CHUNK) {
        HCCL_ERROR("[MeshTransferSlices] MeshChunk has chunk scheduling and sync semantics outside this helper.");
        return HCCL_E_NOT_SUPPORT;
    }

    // ---- COMMON_CHANNEL_SPLIT -------------------------------------------------
    if (sliceBuildInfo.sliceMode == MeshTransferSliceMode::COMMON_CHANNEL_SPLIT) {
        CHK_PRT_RET(sliceBuildInfo.channelSlice == nullptr,
                    HCCL_ERROR("[MeshTransferSlices] invalid build info."), HCCL_E_PARA);
        const MeshChannelSlice &cs = *sliceBuildInfo.channelSlice;
        if (sliceBuildInfo.opType == MeshPrimitiveOp::ALL_GATHER) {
            const u64 txOff = tempAlgParams.buffInfo.hcclBuffBaseOff +
                tempAlgParams.sliceSize * sliceBuildInfo.myAlgRank + cs.elemOffset;
            const u64 rxOff = tempAlgParams.buffInfo.hcclBuffBaseOff +
                tempAlgParams.sliceSize * sliceBuildInfo.connectedAlgRank + cs.elemOffset;
            slices.txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr,
                                            txOff, cs.txSliceSize, cs.txSliceCount);
            slices.txDstSlices.emplace_back(cs.linkRemote->remoteCclMem.addr,
                                            txOff, cs.txSliceSize, cs.txSliceCount);
            slices.rxSrcSlices.emplace_back(cs.linkRemote->remoteCclMem.addr,
                                            rxOff, cs.rxSliceSize, cs.rxSliceCount);
            slices.rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr,
                                            rxOff, cs.rxSliceSize, cs.rxSliceCount);
            return HCCL_SUCCESS;
        }
        // ReduceScatter common split: fall through to the RS repeat loop below.
    }

    // ---- OMNIPIPE_STEP --------------------------------------------------------
    if (sliceBuildInfo.sliceMode == MeshTransferSliceMode::OMNIPIPE_STEP) {
        CHK_PRT_RET(sliceBuildInfo.channelSlice == nullptr,
                    HCCL_ERROR("[MeshTransferSlices] invalid build info."), HCCL_E_PARA);
        const MeshChannelSlice &cs = *sliceBuildInfo.channelSlice;
        const StepSliceInfo &step = tempAlgParams.stepSliceInfo;
        const u32 myAlgRank = sliceBuildInfo.myAlgRank;
        const u32 connectedAlgRank = sliceBuildInfo.connectedAlgRank;
        const u32 stepNum = static_cast<u32>(step.inputOmniPipeSliceStride[myAlgRank].size());

        if (sliceBuildInfo.opType == MeshPrimitiveOp::ALL_GATHER) {
            for (u32 s = 0; s < stepNum; ++s) {
                const u64 txOff = tempAlgParams.buffInfo.inBuffBaseOff +
                    step.inputOmniPipeSliceStride[myAlgRank][s] + step.stepInputSliceStride[myAlgRank];
                const u64 rxOff = tempAlgParams.buffInfo.outBuffBaseOff +
                    step.outputOmniPipeSliceStride[connectedAlgRank][s] +
                    step.stepOutputSliceStride[connectedAlgRank];
                slices.txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, txOff,
                                                step.stepSliceSize[myAlgRank][s],
                                                step.stepCount[myAlgRank][s]);
                slices.txDstSlices.emplace_back(cs.linkRemote->remoteCclMem.addr, txOff,
                                                step.stepSliceSize[myAlgRank][s],
                                                step.stepCount[myAlgRank][s]);
                slices.rxSrcSlices.emplace_back(cs.linkRemote->remoteCclMem.addr, rxOff,
                                                step.stepSliceSize[connectedAlgRank][s],
                                                step.stepSliceSize[connectedAlgRank][s]);
                slices.rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, rxOff,
                                                step.stepSliceSize[connectedAlgRank][s],
                                                step.stepSliceSize[connectedAlgRank][s]);
            }
        } else {
            for (u32 s = 0; s < stepNum; ++s) {
                const u64 txSrcOff = tempAlgParams.buffInfo.inBuffBaseOff +
                    step.stepInputSliceStride[connectedAlgRank] +
                    step.inputOmniPipeSliceStride[connectedAlgRank][s];
                const u64 txDstOff = tempAlgParams.buffInfo.hcclBuffBaseOff +
                    step.stepOutputSliceStride[myAlgRank] +
                    step.outputOmniPipeSliceStride[myAlgRank][s];
                const u64 rxSrcOff = tempAlgParams.buffInfo.inBuffBaseOff +
                    step.stepInputSliceStride[myAlgRank] +
                    step.inputOmniPipeSliceStride[myAlgRank][s];
                const u64 rxDstOff = tempAlgParams.buffInfo.hcclBuffBaseOff +
                    step.stepOutputSliceStride[connectedAlgRank] +
                    step.outputOmniPipeSliceStride[connectedAlgRank][s];
                slices.txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, txSrcOff,
                                                step.stepSliceSize[connectedAlgRank][s],
                                                step.stepCount[connectedAlgRank][s]);
                slices.txDstSlices.emplace_back(sliceBuildInfo.linkRemote->remoteCclMem.addr, txDstOff,
                                                step.stepSliceSize[connectedAlgRank][s],
                                                step.stepCount[connectedAlgRank][s]);
                slices.rxSrcSlices.emplace_back(sliceBuildInfo.linkRemote->remoteCclMem.addr, rxSrcOff,
                                                step.stepSliceSize[myAlgRank][s],
                                                step.stepCount[myAlgRank][s]);
                slices.rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, rxDstOff,
                                                step.stepSliceSize[myAlgRank][s],
                                                step.stepCount[myAlgRank][s]);
            }
        }
        return HCCL_SUCCESS;
    }

    // ---- Repeat-based modes ---------------------------------------------------
    // Covers: RS COMMON_CHANNEL_SPLIT (fell through), RS Z_AXIS_DETOUR,
    //         AG NORMAL_FIXED / VARIABLE_COUNT / Z_AXIS_DETOUR
    CHK_PRT_RET(sliceBuildInfo.channelSlice == nullptr,
                HCCL_ERROR("[MeshTransferSlices] invalid build info."), HCCL_E_PARA);
    const MeshChannelSlice &cs = *sliceBuildInfo.channelSlice;

    if (sliceBuildInfo.opType == MeshPrimitiveOp::REDUCE_SCATTER) {
        CHK_PRT_RET(sliceBuildInfo.linkRemote == nullptr,
                    HCCL_ERROR("[MeshTransferSlices] invalid reduce scatter build info."), HCCL_E_PARA);
        const bool connectedRankHasTail =
            (sliceBuildInfo.connectedAlgRank == sliceBuildInfo.rankSize - 1 && tempAlgParams.tailSize > 0);
        const u64 outputSliceStride = connectedRankHasTail ? tempAlgParams.tailSize : tempAlgParams.sliceSize;
        for (u32 rpt = sliceBuildInfo.repeatBegin; rpt < sliceBuildInfo.repeatEnd; ++rpt) {
            const u64 repeatInBase = tempAlgParams.buffInfo.inBuffBaseOff + rpt * tempAlgParams.inputRepeatStride;
            const u64 repeatOutBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
            const u64 rxSrcOffset = repeatInBase +
                sliceBuildInfo.myAlgRank * tempAlgParams.inputSliceStride + cs.elemOffset;
            const u64 rxDstOffset = repeatOutBase +
                sliceBuildInfo.connectedAlgRank * outputSliceStride + cs.elemOffset;
            const u64 txSrcOffset = repeatInBase +
                sliceBuildInfo.connectedAlgRank * tempAlgParams.inputSliceStride + cs.elemOffset;
            const u64 txDstOffset = repeatOutBase +
                sliceBuildInfo.myAlgRank * outputSliceStride + cs.elemOffset;

            slices.txSrcSlices.emplace_back(tempAlgParams.buffInfo.inputPtr, txSrcOffset,
                                            cs.txSliceSize, cs.txSliceCount);
            slices.txDstSlices.emplace_back(sliceBuildInfo.linkRemote->remoteCclMem.addr, txDstOffset,
                                            cs.txSliceSize, cs.txSliceCount);
            slices.rxSrcSlices.emplace_back(sliceBuildInfo.linkRemote->remoteCclMem.addr, rxSrcOffset,
                                            cs.rxSliceSize, cs.rxSliceCount);
            slices.rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, rxDstOffset,
                                            cs.rxSliceSize, cs.rxSliceCount);
        }
        return HCCL_SUCCESS;
    }

    // ALL_GATHER repeat loop
    const bool variableCount = (sliceBuildInfo.sliceMode == MeshTransferSliceMode::VARIABLE_COUNT);
    for (u32 rpt = sliceBuildInfo.repeatBegin; rpt < sliceBuildInfo.repeatEnd; ++rpt) {
        const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
        const u64 scratchRepeatStride = variableCount ?
            tempAlgParams.sliceSize * DATATYPE_SIZE_TABLE[tempAlgParams.dataType] :
            tempAlgParams.sliceSize * sliceBuildInfo.rankSize;
        u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
        if (sliceBuildInfo.sliceMode == MeshTransferSliceMode::Z_AXIS_DETOUR &&
            tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
            scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams.inputRepeatStride;
        }

        const u64 txOutOffset = variableCount ?
            tempAlgParams.allRankDispls[sliceBuildInfo.myAlgRank] * DATATYPE_SIZE_TABLE[tempAlgParams.dataType] +
                outBaseOff :
            tempAlgParams.outputSliceStride * sliceBuildInfo.myAlgRank + outBaseOff + cs.elemOffset;
        const u64 rxOutOffset = variableCount ?
            tempAlgParams.allRankDispls[sliceBuildInfo.connectedAlgRank] *
                DATATYPE_SIZE_TABLE[tempAlgParams.dataType] + outBaseOff :
            tempAlgParams.outputSliceStride * sliceBuildInfo.connectedAlgRank + outBaseOff + cs.elemOffset;
        const u64 txScratchOffset = scratchBase +
            tempAlgParams.sliceSize * sliceBuildInfo.myAlgRank + cs.elemOffset;
        const u64 rxScratchOffset = scratchBase +
            ((sliceBuildInfo.sliceMode == MeshTransferSliceMode::Z_AXIS_DETOUR) ? tempAlgParams.inputSliceStride :
                                                                                   tempAlgParams.sliceSize) *
                sliceBuildInfo.connectedAlgRank +
            cs.elemOffset;
        const u64 txDstOffset = (!tempAlgParams.enableRemoteMemAccess) ? txScratchOffset : txOutOffset;
        const u64 rxSrcOffset = (!tempAlgParams.enableRemoteMemAccess) ? rxScratchOffset : rxOutOffset;
        // all_gather_v swaps rxSrc/rxDst offset convention vs normal AllGather.
        const u64 rxSrcSliceOffset = variableCount ? rxOutOffset : rxSrcOffset;
        const u64 rxDstSliceOffset = variableCount ? rxSrcOffset : rxOutOffset;

        void *txDstPtr = (!tempAlgParams.enableRemoteMemAccess) ?
            cs.linkRemote->remoteCclMem.addr : cs.linkRemote->remoteOutputGraphMode.addr;
        void *rxSrcPtr = (!tempAlgParams.enableRemoteMemAccess) ?
            cs.linkRemote->remoteCclMem.addr : cs.linkRemote->remoteOutputGraphMode.addr;

        slices.txSrcSlices.emplace_back(tempAlgParams.buffInfo.outputPtr,
                                        txOutOffset, cs.txSliceSize, cs.txSliceCount);
        slices.txDstSlices.emplace_back(txDstPtr, txDstOffset, cs.txSliceSize, cs.txSliceCount);
        slices.rxSrcSlices.emplace_back(rxSrcPtr, rxSrcSliceOffset, cs.rxSliceSize, cs.rxSliceCount);
        slices.rxDstSlices.emplace_back(tempAlgParams.buffInfo.outputPtr,
                                        rxDstSliceOffset, cs.rxSliceSize, cs.rxSliceCount);
    }
    return HCCL_SUCCESS;
}

}  // namespace

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            EngineType engineType, const std::vector<u32> &ranks, u32 myRank,
                            const MeshPrimitiveOptions &options)
{
    (void)engineType;
    if (ranks.size() <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }

    if (options.sliceMode == MeshTransferSliceMode::Z_AXIS_DETOUR && !options.hasZAxisDetourConfig) {
        HCCL_ERROR("[RunMeshAllGather] missing z-axis detour config.");
        return HCCL_E_PARA;
    }

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
    CHK_PRT_RET(!rankFound, HCCL_ERROR("[RunMeshAllGather] rank[%u] is not in ranks.", myRank), HCCL_E_PARA);

    const HcclDataType dataType = tempAlgParams.dataType;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    MeshSendRecvMode sendRecvMode = MeshSendRecvMode::BATCH_WRITE;
    if (options.sliceMode == MeshTransferSliceMode::Z_AXIS_DETOUR) {
        const bool dmaRead = (tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER &&
                              tempAlgParams.buffInfo.outBuffType != BufferType::HCCL_BUFFER);
        sendRecvMode = dmaRead ? MeshSendRecvMode::DMA_READ : MeshSendRecvMode::WRITE;
    } else if (options.sliceMode == MeshTransferSliceMode::VARIABLE_COUNT ||
               options.sliceMode == MeshTransferSliceMode::OMNIPIPE_STEP) {
        sendRecvMode = MeshSendRecvMode::WRITE;
    } else if (options.sliceMode == MeshTransferSliceMode::NORMAL_FIXED) {
        sendRecvMode = MeshSendRecvMode::DMA_READ;
    } else {
        sendRecvMode = IsPcieProtocol(templateResource.channels) ?
            MeshSendRecvMode::DMA_READ : MeshSendRecvMode::BATCH_WRITE;
    }
    const bool variableCount = (options.sliceMode == MeshTransferSliceMode::VARIABLE_COUNT);
    const u32 repeatTaskNum = variableCount ? tempAlgParams.repeatNum : 1;
    for (u32 repeatTaskIdx = 0; repeatTaskIdx < repeatTaskNum; ++repeatTaskIdx) {
        u32 threadIdx = 0;
        const u32 repeatBegin = variableCount ? repeatTaskIdx : 0;
        const u32 repeatEnd = variableCount ? repeatTaskIdx + 1 : tempAlgParams.repeatNum;
        for (u32 i = 1; i < rankSize; ++i) {
            const u32 connectedAlgRank = (options.sliceMode == MeshTransferSliceMode::Z_AXIS_DETOUR) ?
                (i <= myAlgRank ? i - 1 : i) : (myAlgRank + i) % rankSize;
            const u32 connectedRank = ranks[connectedAlgRank];

            std::vector<MeshChannelSlice> channelSlices;
            CHK_RET(ResolveMeshChannelSlices(tempAlgParams, templateResource, options,
                                             MeshPrimitiveOp::ALL_GATHER, myAlgRank,
                                             connectedRank, connectedAlgRank, rankSize, dataTypeSize,
                                             sendRecvMode, channelSlices));
            for (const auto &channelSlice : channelSlices) {
                CHK_PRT_RET(channelSlice.linkRemote == nullptr || threadIdx >= templateResource.threads.size(),
                            HCCL_ERROR("[RunMeshAllGather] invalid transfer slice task."), HCCL_E_PARA);
                MeshSliceBuildInfo sliceBuildInfo;
                sliceBuildInfo.opType = MeshPrimitiveOp::ALL_GATHER;
                sliceBuildInfo.sliceMode = options.sliceMode;
                sliceBuildInfo.myAlgRank = myAlgRank;
                sliceBuildInfo.connectedAlgRank = connectedAlgRank;
                sliceBuildInfo.rankSize = rankSize;
                sliceBuildInfo.repeatBegin = repeatBegin;
                sliceBuildInfo.repeatEnd = repeatEnd;
                sliceBuildInfo.channelSlice = &channelSlice;

                MeshTransferSlices slices;
                CHK_RET(BuildMeshTransferSlices(tempAlgParams, sliceBuildInfo, slices));
                SendRecvInfo sendRecvInfo{{*channelSlice.linkRemote, *channelSlice.linkRemote},
                                          {{slices.txSrcSlices, slices.txDstSlices},
                                           {slices.rxSrcSlices, slices.rxDstSlices}},
                                          dataType};
                if (sendRecvMode == MeshSendRecvMode::DMA_READ) {
                    CHK_RET(SendRecvRead(sendRecvInfo, templateResource.threads[threadIdx]));
                } else if (sendRecvMode == MeshSendRecvMode::WRITE) {
                    CHK_RET(SendRecvWrite(sendRecvInfo, templateResource.threads[threadIdx]));
                } else {
                    CHK_RET(SendRecvBatchWrite(sendRecvInfo, templateResource.threads[threadIdx]));
                }
                ++threadIdx;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            EngineType engineType, const std::vector<u32> &ranks, u32 myRank)
{
    MeshPrimitiveOptions options;
    // 保留当前已抽取 primitive 的行为。
    // 后续调用者应从 TemplateDesc.variant 显式传入 mode，而不是依赖这个兼容重载。
    options.sliceMode = MeshTransferSliceMode::COMMON_CHANNEL_SPLIT;
    return RunMeshAllGather(tempAlgParams, templateResource, engineType, ranks, myRank, options);
}

HcclResult RunMeshReduceScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                                EngineType engineType, const std::vector<u32> &ranks, u32 myRank,
                                const MeshPrimitiveOptions &options)
{
    (void)engineType;
    if (options.sliceMode == MeshTransferSliceMode::MESH_CHUNK) {
        HCCL_ERROR("[RunMeshReduceScatter] MeshChunk needs chunk scheduling, PreCopy/PostCopy and sync outside this primitive.");
        return HCCL_E_NOT_SUPPORT;
    }
    if (options.sliceMode == MeshTransferSliceMode::NORMAL_FIXED ||
        options.sliceMode == MeshTransferSliceMode::VARIABLE_COUNT) {
        HCCL_ERROR("[RunMeshReduceScatter] unsupported slice mode for mesh reduce scatter.");
        return HCCL_E_NOT_SUPPORT;
    }
    if (options.sliceMode == MeshTransferSliceMode::Z_AXIS_DETOUR && !options.hasZAxisDetourConfig) {
        HCCL_ERROR("[RunMeshReduceScatter] missing z-axis detour config.");
        return HCCL_E_PARA;
    }

    const u32 rankSize = static_cast<u32>(ranks.size());
    if (rankSize <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }
    u32 myAlgRank = 0;
    bool rankFound = false;
    for (u32 i = 0; i < rankSize; ++i) {
        if (ranks[i] == myRank) {
            myAlgRank = i;
            rankFound = true;
            break;
        }
    }
    CHK_PRT_RET(!rankFound, HCCL_ERROR("[RunMeshReduceScatter] rank[%u] is not in ranks.", myRank), HCCL_E_PARA);

    const HcclDataType dataType = tempAlgParams.dataType;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];

    u32 threadIdx = 0;
    for (u32 i = 1; i < rankSize; ++i) {
        const u32 connectedAlgRank = (myAlgRank + i) % rankSize;
        const u32 connectedRank = ranks[connectedAlgRank];

        std::vector<MeshChannelSlice> channelSlices;
        // RS never uses WRITE sendRecvMode, so the Z-axis tail logic in
        // ResolveMeshChannelSlices falls through to the standard connectedAlgRank path.
        CHK_RET(ResolveMeshChannelSlices(tempAlgParams, templateResource, options,
                                         MeshPrimitiveOp::REDUCE_SCATTER, myAlgRank,
                                         connectedRank, connectedAlgRank, rankSize, dataTypeSize,
                                         MeshSendRecvMode::BATCH_WRITE, channelSlices));
        for (const auto &channelSlice : channelSlices) {
            CHK_PRT_RET(channelSlice.linkRemote == nullptr || threadIdx >= templateResource.threads.size(),
                        HCCL_ERROR("[RunMeshReduceScatter] invalid transfer slice task."), HCCL_E_PARA);
            MeshSliceBuildInfo sliceBuildInfo;
            sliceBuildInfo.opType = MeshPrimitiveOp::REDUCE_SCATTER;
            sliceBuildInfo.sliceMode = options.sliceMode;
            sliceBuildInfo.myAlgRank = myAlgRank;
            sliceBuildInfo.connectedAlgRank = connectedAlgRank;
            sliceBuildInfo.rankSize = rankSize;
            sliceBuildInfo.repeatBegin = 0;
            sliceBuildInfo.repeatEnd = tempAlgParams.repeatNum;
            sliceBuildInfo.channelSlice = &channelSlice;
            sliceBuildInfo.linkRemote = channelSlice.linkRemote;

            MeshTransferSlices slices;
            CHK_RET(BuildMeshTransferSlices(tempAlgParams, sliceBuildInfo, slices));
            SendRecvInfo sendRecvInfo{{*channelSlice.linkRemote, *channelSlice.linkRemote},
                                      {{slices.txSrcSlices, slices.txDstSlices},
                                       {slices.rxSrcSlices, slices.rxDstSlices}},
                                      dataType};
            if (options.sliceMode == MeshTransferSliceMode::OMNIPIPE_STEP) {
                CHK_RET(SendRecvWrite(sendRecvInfo, templateResource.threads[threadIdx]));
            } else {
                CHK_RET(SendRecvBatchWrite(sendRecvInfo, templateResource.threads[threadIdx]));
            }
            ++threadIdx;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshReduceScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                                EngineType engineType, const std::vector<u32> &ranks, u32 myRank)
{
    MeshPrimitiveOptions options;
    options.sliceMode = MeshTransferSliceMode::COMMON_CHANNEL_SPLIT;
    return RunMeshReduceScatter(tempAlgParams, templateResource, engineType, ranks, myRank, options);
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
