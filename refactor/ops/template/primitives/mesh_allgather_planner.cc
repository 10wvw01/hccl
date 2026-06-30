/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 */

#include "mesh_allgather_planner.h"

namespace ops_hccl {

namespace {

struct MeshAllGatherChannelSlice {
    // MeshAllGatherChannelSlice 是 planner 内部的中间结果：
    // 表示一个 connectedRank 的数据落到某个 channel 上时，需要额外叠加的 offset 和 size/count。
    u32 channelIdx{0};
    const ChannelInfo *linkRemote{nullptr};
    // elemOffset 对应 channel split 后相对该 rank slice 起点的字节偏移。
    u64 elemOffset{0};
    // tx/rx 分开是为了兼容 all_gather_v、OmniPipe 等发送和接收大小可能不同的变体。
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

MeshAllGatherSendRecvMode ResolveMeshAllGatherSendRecvMode(const TemplateDataParams &tempAlgParams,
                                                           const TemplateResource &templateResource,
                                                           const MeshAllGatherPrimitiveOptions &options)
{
    // Z-axis 旧代码会根据 buffer 类型切换 read/write 模式。
    // 其他变体在迁移时也应把 SendRecv 模式选择收敛到这里，避免执行循环里夹杂变体判断。
    if (options.sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR) {
        bool dmaRead = (tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER &&
                        tempAlgParams.buffInfo.outBuffType != BufferType::HCCL_BUFFER);
        return dmaRead ? MeshAllGatherSendRecvMode::DMA_READ : MeshAllGatherSendRecvMode::WRITE;
    }

    if (options.sliceMode == MeshAllGatherSliceMode::NORMAL_FIXED) {
        return MeshAllGatherSendRecvMode::DMA_READ;
    }

    // 普通 AICPU Mesh AllGather 使用 SendRecvRead；当前已抽取的 common split 保留了旧的 IsPcieProtocol 分支。
    // 在各变体显式接入之前，兼容入口继续保留这个行为。
    return IsPcieProtocol(templateResource.channels) ?
        MeshAllGatherSendRecvMode::DMA_READ : MeshAllGatherSendRecvMode::BATCH_WRITE;
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
        channelSlices.push_back({0, &channels[0], 0, sliceSize, sliceSize,
                                 sliceSize / dataTypeSize, sliceSize / dataTypeSize});
        return HCCL_SUCCESS;
    }

    if (options.sliceMode == MeshAllGatherSliceMode::COMMON_CHANNEL_SPLIT) {
        const u32 channelsPerRank = static_cast<u32>(channels.size());
        // ec/sizeOut/elemOffset 是 CalcDataSplitByPortGroupCommon 的旧输出语义：
        // 每个 channel 上的元素数、字节数、相对 offset。
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

    if (options.sliceMode == MeshAllGatherSliceMode::VARIABLE_COUNT) {
        // 架构接入点：all_gather_v 不需要复制外层 Mesh 遍历，只需要在这里生成“变长 channel piece”。
        return HCCL_E_NOT_SUPPORT;
    }

    if (options.sliceMode == MeshAllGatherSliceMode::OMNIPIPE_STEP) {
        // 架构接入点：OmniPipe 的差异不在 peer 遍历，而在一个 peer 内有多段 step slice。
        return HCCL_E_NOT_SUPPORT;
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

    if (options.sliceMode == MeshAllGatherSliceMode::MESH_CHUNK) {
        // 架构接入点：meshchunk 需要 PrimitiveOptions 额外携带 RankSliceInfo/chunk 元数据。
        return HCCL_E_NOT_SUPPORT;
    }

    (void)options;
    return HCCL_E_NOT_SUPPORT;
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
        task.txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr,
                                      txOutOffset, channelSlice.txSliceSize, channelSlice.txSliceCount);
        task.txDstSlices.emplace_back(channelSlice.linkRemote->remoteCclMem.addr,
                                      txOutOffset, channelSlice.txSliceSize, channelSlice.txSliceCount);
        task.rxSrcSlices.emplace_back(channelSlice.linkRemote->remoteCclMem.addr,
                                      rxOutOffset, channelSlice.rxSliceSize, channelSlice.rxSliceCount);
        task.rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr,
                                      rxOutOffset, channelSlice.rxSliceSize, channelSlice.rxSliceCount);
        return;
    }

    if (sliceMode == MeshAllGatherSliceMode::VARIABLE_COUNT) {
        // TODO: 使用 allRankDispls/allRankProcessedDataCount 计算变长 txOutOffset/rxOutOffset。
        return;
    }

    if (sliceMode == MeshAllGatherSliceMode::OMNIPIPE_STEP) {
        // TODO: 遍历 stepSliceInfo 中当前 myAlgRank/connectedAlgRank 的每个 rpt 生成多组 DataSlice。
        return;
    }

    if (sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR) {
        for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
            const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
            const u64 scratchRepeatStride = tempAlgParams.sliceSize * rankSize;
            u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
            if (tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
                scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams.inputRepeatStride;
            }
            const u64 txOutOffset =
                tempAlgParams.outputSliceStride * myAlgRank + outBaseOff + channelSlice.elemOffset;
            const u64 txScratchOffset =
                scratchBase + tempAlgParams.sliceSize * myAlgRank + channelSlice.elemOffset;
            const u64 txDstOffset = (!tempAlgParams.enableRemoteMemAccess) ? txScratchOffset : txOutOffset;
            const u64 rxOutOffset =
                tempAlgParams.outputSliceStride * connectedAlgRank + outBaseOff + channelSlice.elemOffset;
            const u64 rxScratchOffset =
                scratchBase + tempAlgParams.inputSliceStride * connectedAlgRank + channelSlice.elemOffset;
            const u64 rxSrcOffset = (!tempAlgParams.enableRemoteMemAccess) ? rxScratchOffset : rxOutOffset;
            void *txDstPtr = (!tempAlgParams.enableRemoteMemAccess) ?
                channelSlice.linkRemote->remoteCclMem.addr : channelSlice.linkRemote->remoteOutputGraphMode.addr;
            void *rxSrcPtr = (!tempAlgParams.enableRemoteMemAccess) ?
                channelSlice.linkRemote->remoteCclMem.addr : channelSlice.linkRemote->remoteOutputGraphMode.addr;

            task.txSrcSlices.emplace_back(tempAlgParams.buffInfo.outputPtr,
                                          txOutOffset, channelSlice.txSliceSize, channelSlice.txSliceCount);
            task.txDstSlices.emplace_back(txDstPtr, txDstOffset, channelSlice.txSliceSize,
                                          channelSlice.txSliceCount);
            task.rxDstSlices.emplace_back(tempAlgParams.buffInfo.outputPtr,
                                          rxOutOffset, channelSlice.rxSliceSize, channelSlice.rxSliceCount);
            task.rxSrcSlices.emplace_back(rxSrcPtr, rxSrcOffset, channelSlice.rxSliceSize,
                                          channelSlice.rxSliceCount);
        }
        return;
    }

    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * rankSize;
        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
        const u64 txOutOffset = tempAlgParams.outputSliceStride * myAlgRank + outBaseOff + channelSlice.elemOffset;
        const u64 rxOutOffset = tempAlgParams.outputSliceStride * connectedAlgRank + outBaseOff +
            channelSlice.elemOffset;
        const u64 txScratchOffset = scratchBase + tempAlgParams.sliceSize * myAlgRank + channelSlice.elemOffset;
        const u64 rxScratchOffset = scratchBase + tempAlgParams.sliceSize * connectedAlgRank +
            channelSlice.elemOffset;
        const u64 txDstOffset = (!tempAlgParams.enableRemoteMemAccess) ? txScratchOffset : txOutOffset;
        const u64 rxSrcOffset = (!tempAlgParams.enableRemoteMemAccess) ? rxScratchOffset : rxOutOffset;
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
}

}  // namespace

HcclResult BuildMeshAllGatherSlicePlan(const TemplateDataParams &tempAlgParams,
                                       const TemplateResource &templateResource,
                                       const std::vector<u32> &ranks, u32 myRank,
                                       const MeshAllGatherPrimitiveOptions &options,
                                       MeshAllGatherSlicePlan &plan)
{
    // 公共主干只负责：找 myAlgRank、按 Mesh 顺序遍历 connectedRank、生成 task。
    // 变体差异只放在 channel piece 计算和 DataSlice 填充里，避免退化成“一个变体一个 template”。
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
        // 普通 Mesh 旧代码按 myAlgRank 后继环形遍历；Z-axis 旧代码按 subCommRanks_ 原始顺序跳过自己。
        const u32 connectedAlgRank = (options.sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR) ?
            (i <= myAlgRank ? i - 1 : i) : (myAlgRank + i) % rankSize;
        const u32 connectedRank = ranks[connectedAlgRank];
        // channelSlices 把“一个 connectedRank”进一步拆成一个或多个 channel 任务；
        // 普通 Mesh 只有一项，common split/Z-axis 会有多项。
        std::vector<MeshAllGatherChannelSlice> channelSlices;
        CHK_RET(ResolveMeshAllGatherChannelSlices(tempAlgParams, templateResource, options, myAlgRank, connectedRank,
                                                  connectedAlgRank, rankSize, dataTypeSize, plan.sendRecvMode,
                                                  channelSlices));
        for (const auto &channelSlice : channelSlices) {
            CHK_PRT_RET(channelSlice.linkRemote == nullptr,
                        HCCL_ERROR("[MeshAllGatherPlanner] invalid channel slice."), HCCL_E_PARA);
            // task 是执行层可以直接消费的 SendRecvInfo 原料；这里之后不再重新计算 offset。
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
