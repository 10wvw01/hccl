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
    const bool connectedRankHasTail =
        (connectedAlgRank == rankSize - 1 && tempAlgParams.tailSize > 0) || zAxisWriteTail;
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
        // 旧逻辑输入来自 tempAlgParams.allRankSliceSize/allRankDispls/allRankProcessedDataCount：
        // 1. channel 仍可先按普通 Mesh 使用 channels[0]；如果后续也要 channel split，可在这里再套
        //    CalcDataSplitByPortGroupCommon。
        // 2. txSliceSize/txSliceCount 使用 allRankSliceSize[myAlgRank] 和本 rank count。
        // 3. rxSliceSize/rxSliceCount 使用 allRankSliceSize[connectedAlgRank] 和 connectedRank count。
        // 4. AppendMeshAllGatherDataSlices 需要用 allRankDispls 或 allRankProcessedDataCount 计算
        //    txOutOffset/rxOutOffset，而不是固定 sliceSize * rank。
        // 这个 mode 可以继续复用主干的 connectedRank、threadIdx 和 SendRecvInfo 执行。
        return HCCL_E_NOT_SUPPORT;
    }

    if (options.sliceMode == MeshAllGatherSliceMode::OMNIPIPE_STEP) {
        // 架构接入点：OmniPipe 的差异不在 peer 遍历，而在一个 peer 内有多段 step slice。
        // 旧逻辑输入来自 tempAlgParams.stepSliceInfo：
        // 1. channel piece 可先返回 channels[0]，保持每个 connectedRank 一个 task。
        // 2. AppendMeshAllGatherDataSlices 按
        //    inputOmniPipeSliceStride/outputOmniPipeSliceStride 找每个 rpt 的 base。
        // 3. tx/rx offset 分别叠加 stepInputSliceStride[myAlgRank] 和
        //    stepOutputSliceStride[connectedAlgRank]。
        // 4. 每个 rpt 使用 stepSliceSize/stepCount 生成一组 DataSlice。
        // 这样 OmniPipe 只替换“repeat slice 生成规则”，不需要替换 Mesh 主干。
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
        // 架构接入点：meshchunk 来自 AllReduce two-shot 的 AllGather 阶段，不是普通 AllGather template。
        // 它需要 PrimitiveOptions 额外携带 RankSliceInfo/chunk 元数据：
        // 1. Resolve 阶段根据 chunk 所属 rank、chunk offset、chunk size 生成 channel piece。
        // 2. Append 阶段按 chunk 的全局/局部 offset 填 txOutOffset/rxOutOffset。
        // 3. 如果 meshchunk 仍按 Mesh 顺序和 peer 通信，外层主干可复用；如果它有特殊 peer 顺序，
        //    才需要把“peer traversal policy”也抽成一个小策略。
        // 这个 mode 用来验证当前架构还缺一个 meshchunk metadata 输入，而不是缺一套新 template。
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
        // TODO: 使用 allRankDispls/allRankProcessedDataCount 计算变长 txOutOffset/rxOutOffset，
        // 再按 channelSlice.txSliceSize/rxSliceSize 分别生成 send/recv DataSlice。
        // 这个分支只替换 offset/size 公式，仍复用外层 task 和执行循环。
        return;
    }

    if (sliceMode == MeshAllGatherSliceMode::OMNIPIPE_STEP) {
        // TODO: 遍历 stepSliceInfo 中当前 myAlgRank/connectedAlgRank 的每个 rpt，
        // 按 stepInputSliceStride/stepOutputSliceStride 和 OmniPipe slice stride 生成多组 DataSlice。
        // 它对应旧 OmniPipe 中“一个 peer 内多个 step slice”的逻辑。
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
