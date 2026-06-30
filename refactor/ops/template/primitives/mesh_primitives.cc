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

// 直接函数调用的 primitive 收到的是扁平 ranks 列表，不再持有旧 template 缓存的 subCommRanks_。
// 这里统一完成 rank 到算法内 rank index 的转换，保证各 planner 分支使用同一套 offset 下标语义。
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
    // Z-axis 旧代码会根据 buffer 类型切换 read/write 模式。
    // 其他变体在迁移时也应把 SendRecv 模式选择收敛到这里，避免执行循环里夹杂变体判断。
    if (options.sliceMode == MeshAllGatherSliceMode::Z_AXIS_DETOUR) {
        bool dmaRead = (tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER &&
                        tempAlgParams.buffInfo.outBuffType != BufferType::HCCL_BUFFER);
        return dmaRead ? MeshAllGatherSendRecvMode::DMA_READ : MeshAllGatherSendRecvMode::BATCH_WRITE;
    }

    // 普通 AICPU Mesh AllGather 使用 SendRecvRead；当前已抽取的 common split 保留了旧的 IsPcieProtocol 分支。
    // 在各变体显式接入之前，兼容入口继续保留这个行为。
    return IsPcieProtocol(templateResource.channels) ?
        MeshAllGatherSendRecvMode::DMA_READ : MeshAllGatherSendRecvMode::BATCH_WRITE;
}

HcclResult BuildMeshAllGatherCommonChannelSplitPlan(const TemplateDataParams &tempAlgParams,
                                                    const TemplateResource &templateResource,
                                                    const std::vector<u32> &ranks, u32 myRank,
                                                    MeshAllGatherSlicePlan &plan)
{
    // 这里复刻当前已抽取 RunMeshAllGather 的行为：把一个 peer 的 slice 切到该 peer port group 的所有
    // channel 上，然后每个 peer/channel 生成一个 task。
    // 它必须和 NORMAL_FIXED 分开，因为旧普通 Mesh AllGather 只用 channel[0]，没有这层 channel 再切片。
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
            // 这四组 slice 是 planner 输出给执行层的边界：
            // txSrc/txDst 描述本 rank 暴露给 peer 的本地数据；
            // rxSrc/rxDst 描述本 rank 从哪里读取 peer 数据，以及最终写到哪里。
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
    // TODO: 精确迁移旧 InsTempAllGatherMesh1D::RunAllGatherMesh：
    // 1. 每个 peer 只使用一个 channel：channels.at(peerRank)[0]
    // 2. 每个 repeat 构造一个 DataSlice
    // 3. 保留 outputPtr/remoteCclMem/remoteOutputGraphMode/symmetric-memory 的地址选择逻辑
    // 4. 保留 tail 处理：connectedAlgRank == rankSize - 1
    // 当前 common split 路径故意单独保留，因为 channelsPerRank > 1 时它与普通 Mesh AllGather 不等价。
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
    // TODO: 迁移旧 InsTempAllGatherVMesh1D::RunAllGatherVMesh：
    // allRankSliceSize/allRankDispls/allRankProcessedDataCount 决定每个 rank 的 offset/size。
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
    // TODO: 迁移旧 InsTempAllGatherOmniPipeMesh1D::RunAllGatherMesh：
    // stepSliceInfo.{stepSliceSize,stepCount,stepInputSliceStride,stepOutputSliceStride,
    // inputOmniPipeSliceStride,outputOmniPipeSliceStride} 决定每个 step 的 slice。
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
    // TODO: 迁移旧 InsTempAllGatherMesh1D1DZAxisDetour：
    // 1. resource planner 必须把 level0/level1 channel 边界保存在 zAxis 中
    // 2. 调用 CalcDataSplitByPortGroupZAxisDetour(totalCount, dataTypeSize, peerChannels, ...)
    // 3. 基于每个 channel 的 elemOffset/sizeOut 构造 DataSlice
    // 4. 使用与 CalcSliceSizeForChannel 一致的 dmaRead tail 规则
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
    // TODO: meshchunk 是 AllReduce two-shot 里的变体，不是普通 AllGather template。
    // 接入这个 mode 之前，需要先把 RankSliceInfo/chunk 元数据补进 PrimitiveOptions。
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
    // planner dispatch 是唯一应该按算法变体分支的地方。
    // 下面的 RunMeshAllGather 只消费生成后的 DataSlice task，让通信主干不依赖具体布局规则。
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
    // 工作流：
    // 1. 把 params/resource/options 转换成变体专属的 slice plan。
    // 2. 用统一的 SendRecvInfo 执行每个已规划的 peer/channel task。
    // 这是去掉旧 template 类成员状态后的目标形态。
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
