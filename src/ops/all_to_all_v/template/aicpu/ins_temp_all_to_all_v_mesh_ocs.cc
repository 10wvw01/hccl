/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <string>
#include <vector>

#include "aicpu/ins_temp_all_to_all_v_mesh_ocs.h"
#include "alltoall_ocs_selector_helper.h"

namespace ops_hccl {

InsTempAlltoAllVMeshOcs::InsTempAlltoAllVMeshOcs(
    const OpParam& param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempAlltoAllVMeshOcs::~InsTempAlltoAllVMeshOcs()
{
}
void InsTempAlltoAllVMeshOcs::GetOcsGroupNumInfoFromTopo(const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (topoInfo != nullptr) {
        numGroups_ = topoInfo->ocsGroupNum;
    }
    return;
}

u64 InsTempAlltoAllVMeshOcs::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;

    /* 计算对齐后并发度，即 CCL Buffer 槽位数 */
    concurrentSendRecvNum_ = CalcOcsAdjustedConcurrent(templateRankSize_, numGroups_);
    if (concurrentSendRecvNum_ == 0) {
        concurrentSendRecvNum_ = 1;
    }
    HCCL_INFO("[InsTempAlltoAllVMeshOcs][CalcScratchMultiple] myRank=%u, templateRankSize_[%u] numGroups_[%u] "
        "concurrentSendRecvNum_[%u]", myRank_, templateRankSize_, numGroups_, concurrentSendRecvNum_);
    return concurrentSendRecvNum_;
}

HcclResult InsTempAlltoAllVMeshOcs::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(numGroups_ <= 1,
        HCCL_ERROR("[InsTempAlltoAllVMeshOcs][CalcRes] myRank=%u, invalid numGroups_=%u", myRank_, numGroups_), HCCL_E_INTERNAL);
    /* ---- 步骤 1：通道申请 ---- */
    std::vector<HcclChannelDesc> level0Channels;

    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        /* CLOS 拓扑：使用 CalcChannelRequestMesh1DWithPriorityTopo 申请通道，
         * 然后过滤出 UBC_CTP 协议的通道（高速链路，优先用于 OCS 跨 Server 通信） */
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs,
            CommTopo::COMM_TOPO_1DMESH));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_DEBUG("[InsTempAlltoAllVMeshOcs][CalcRes] Get Channel Success!");
    } else {
        /* 非 CLOS 拓扑：使用标准 Mesh1D 通道申请 */
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    /* 将 Level0 通道列表加入资源请求 */
    resourceRequest.channels.push_back(level0Channels);
    /* ---- 步骤 2：计算 OCS 并发度和从线程数 ---- */
    /* 计算每 rank 的 channel 数（可能因链路过滤而少于申请数） */
    channelsPerRank_ = CalcChannelsPerRank(level0Channels);

    /*  确保并发度是 numGroups_ 的整数倍，避免组间负载不均 */
    u32 adjustedConcurrent = CalcOcsAdjustedConcurrent(templateRankSize_, numGroups_);

    /* 从线程数 = 并发度 × 每 rank 的 channel 数
     * 每路通信的每个 channel 由独立从线程执行，实现多通道并行 */
    resourceRequest.slaveThreadNum = adjustedConcurrent * channelsPerRank_;

    HCCL_INFO("[InsTempAlltoAllVMeshOcs][CalcRes] level0Topo[%u] level0PcieMix[%u] templateRankSize_[%u] "
              "numGroups_[%u] adjustedConcurrent[%u] level0ChannelNum[%zu] channelsPerRank[%u] slaveThreadNum[%u]",
              static_cast<u32>(topoInfo->level0Topo), static_cast<u32>(topoInfo->level0PcieMix),
              templateRankSize_, numGroups_, adjustedConcurrent,
              level0Channels.size(), channelsPerRank_, resourceRequest.slaveThreadNum);

    /* ---- 步骤 4：notify 配置 ----
     * 每个从线程需要 1 个 notify（用于 PreSync/PostSync）
     * 主线程需要 slaveThreadNum 个 notify（用于接收各从线程的完成通知） */
    for (u32 index = 0; index < resourceRequest.slaveThreadNum; index++) {
        resourceRequest.notifyNumPerThread.push_back(channelsPerRank_);
    }
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMeshOcs::KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
    TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAlltoAllVMeshOcs][KernelRun] myRank=%u Run Start", myRank_);

    /* 保存线程总数（主线程 + 从线程） */
    threadNum_ = templateResource.threads.size();

    /* 保存数据类型信息，用于后续偏移/大小计算 */
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = SIZE_TABLE[dataType_];

    /* ---- 前置校验 ---- */
    /* 并发度不能为 0 */
    CHK_PRT_RET(concurrentSendRecvNum_ == 0,
        HCCL_ERROR("[InsTempAlltoAllVMeshOcs][RunLimitedConcurrencyOCS] myRank=%u, concurrentSendRecvNum_ is zero",
            myRank_),
        HCCL_E_INTERNAL);
    /* numGroups_ 不能为 0，且 rankSize 必须能被 numGroups_ 整除 */
    CHK_PRT_RET(numGroups_ <= 1 || templateRankSize_ % numGroups_ != 0,
        HCCL_ERROR("[InsTempAlltoAllVMeshOcs][RunLimitedConcurrencyOCS] myRank=%u, invalid numGroups_=%u rankSize=%u",
            myRank_, numGroups_, templateRankSize_),
        HCCL_E_INTERNAL);
    /* 并发度必须能被 numGroups_ 整除（保证每组并发 plane 数相同） */
    CHK_PRT_RET(concurrentSendRecvNum_ % numGroups_ != 0,
        HCCL_ERROR("[InsTempAlltoAllVMeshOcs][RunLimitedConcurrencyOCS] myRank=%u, concurrentSendRecvNum_=%u "
            "cannot be divided by numGroups_=%u", myRank_, concurrentSendRecvNum_, numGroups_),
        HCCL_E_INTERNAL);

    /* 查找本 rank 在 subCommRanks_[0] 中的逻辑下标
     * 逻辑下标 = plane 编号（同一 group 内的位置），用于环步进计算
     * 例如：subCommRanks_[0] = {0,1,2,...,31}，myRank_=5 → myAlgRank=5 */
    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = static_cast<u32>(std::distance(subCommRanks_[0].begin(), iter));
    } else {
        /* 本 rank 不在子通信域中，属于配置错误 */
        HCCL_ERROR("[InsTempAlltoAllVMeshOcs][KernelRun] myRank=%u not found in subCommRanks_", myRank_);
        return HCCL_E_INTERNAL;
    }
    channelsPerRank_ = CalcChannelsPerRank(templateResource.channels);

    /* 调用 OCS 有限并发算法主入口 */
    CHK_RET(RunLimitedConcurrencyOcs(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank));

    HCCL_INFO("[InsTempAlltoAllVMeshOcs][KernelRun] myRank=%u Run End", myRank_);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMeshOcs::RunLimitedConcurrencyOcs(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams, u32 myAlgRank)
{
    HCCL_INFO("[InsTempAlltoAllVMeshOcs][RunLimitedConcurrencyOCS] myRank=%u rankSize=%u concurrent=%u numGroups=%u",
        myRank_, templateRankSize_, concurrentSendRecvNum_, numGroups_);

    /* ---- 步骤 1：LocalCopy（自环数据）----
     * 本 rank 发给自己的数据不需要走网络，直接从 input 拷贝到 output */
    DataSlice localSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
        tempAlgParams.sdispls[myAlgRank] * dataTypeSize_,
        tempAlgParams.sendCounts[myAlgRank] * dataTypeSize_, tempAlgParams.sendCounts[myAlgRank]);
    DataSlice localDstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.rdispls[myAlgRank] * dataTypeSize_,
        tempAlgParams.recvCounts[myAlgRank] * dataTypeSize_, tempAlgParams.recvCounts[myAlgRank]);
    if (tempAlgParams.sendCounts[myAlgRank] > 0) {
        /* 使用主线程（threads[0]）执行 LocalCopy */
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], localSrcSlice, localDstSlice)));
    }

    /* ---- 步骤 2：计算轮次参数 ---- */
    /* numPlanes = 每个 group 内的 rank 数 */
    u32 numPlanes = templateRankSize_ / numGroups_;
    /* concurrentPlanesPerRound = 每轮每 group 并发的 plane 数 */
    u32 concurrentPlanesPerRound = concurrentSendRecvNum_ / numGroups_;
    /* commRounds = 总通信轮数 = ceil(numPlanes / concurrentPlanesPerRound) */
    u32 commRounds = (numPlanes + concurrentPlanesPerRound - 1) / concurrentPlanesPerRound;

    HCCL_INFO("[InsTempAlltoAllVMeshOcs][RunLimitedConcurrencyOCS] myRank=%u numPlanes=%u "
        "concurrentPlanesPerRound=%u commRounds=%u",
        myRank_, numPlanes, concurrentPlanesPerRound, commRounds);

    /* 预分配通信对端数组，避免每轮重复分配 */
    std::vector<u32> sendToRanks;
    std::vector<u32> recvFromRanks;
    /* 从线程列表：threads[1..]（不含主线程 threads[0]） */
    std::vector<ThreadHandle> subThreads;
    if (threadNum_ > 1) {
        subThreads.assign(threads.begin() + 1, threads.end());
    }

    /* ---- 步骤 3：多轮通信循环 ---- */
    for (u32 roundIdx = 0; roundIdx < commRounds; roundIdx++) {
        /* 3a. 计算本轮的发送目标 rank 列表和接收源 rank 列表 */
        CHK_RET(CalcCommRankSetforOneRoundOcs(roundIdx, concurrentPlanesPerRound, sendToRanks, recvFromRanks));

        /* 日志：打印本轮通信对端（便于调试） */
        std::string sendRanksStr;
        std::string recvRanksStr;
        for (size_t i = 0; i < sendToRanks.size(); ++i) {
            if (i > 0) {
                sendRanksStr.push_back(',');
                recvRanksStr.push_back(',');
            }
            sendRanksStr += std::to_string(sendToRanks[i]);
            recvRanksStr += std::to_string(recvFromRanks[i]);
        }
        HCCL_INFO("[InsTempAlltoAllVMeshOcs][RunLimitedConcurrencyOCS] myRank=%u round=%u/%u sendRanks[%s] recvRanks[%s]",      
            myRank_, roundIdx + 1, commRounds,
            sendRanksStr.empty() ? "-" : sendRanksStr.c_str(),
            recvRanksStr.empty() ? "-" : recvRanksStr.c_str());

        /* 3b. 主线程唤醒所有从线程（PreSync）
         * 多线程时才需要同步；单线程时跳过 */
        if (threadNum_ > 1) {
            GetNotifyIdxMainToSub(notifyIdxMainToSub_);
            CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
        }

        /* 3c. 遍历通信对端，按 channel 分配从线程执行 SendRecv
         * queIdx 从 1 开始（0 是主线程） */
        u32 queIdx = 1;
        for (size_t idx = 0; idx < sendToRanks.size(); idx++) {
            u32 sendRank = sendToRanks[idx];
            u32 recvRank = recvFromRanks[idx];

            /* 跳过自环（sendRank == myAlgRank && recvRank == myAlgRank）
             * 自环数据已在步骤 1 中由 LocalCopy 处理 */
            if (sendRank == myAlgRank && recvRank == myAlgRank) {
                continue;
            }

            /* 判断是否需要发送/接收：
             * - sendCounts[sendRank] > 0 且 sendRank 不是自己 → 需要发送
             * - recvCounts[recvRank] > 0 且 recvRank 不是自己 → 需要接收 */
            const bool doSend = tempAlgParams.sendCounts[sendRank] > 0 && sendRank != myAlgRank;
            const bool doRecv = tempAlgParams.recvCounts[recvRank] > 0 && recvRank != myAlgRank;
            if (!doSend && !doRecv) {
                continue;
            }

            /* 查找发送/接收方向的 channel 列表 */
            const std::vector<ChannelInfo>* sendChannelVec = nullptr;
            const std::vector<ChannelInfo>* recvChannelVec = nullptr;

            if (doSend) {
                auto itSend = channels.find(sendRank);
                CHK_PRT_RET(itSend == channels.end() || itSend->second.empty(),
                    HCCL_ERROR("[InsTempAlltoAllVMeshOcs][RunLimitedConcurrencyOCS] myRank=%u, sendRankPhysic=%u "
                        "not found in channels", myRank_, sendRank),
                    HCCL_E_INTERNAL);
                sendChannelVec = &itSend->second;
                /* 按 channel（Port Group）分片发送数据 */
                CHK_RET(CalcDataSplitByPortGroup(tempAlgParams.sendCounts[sendRank], dataTypeSize_, *sendChannelVec,
                    sendCountsSplit_, sendSizeSplit_, sendOffsetSplit_));
            }
            if (doRecv) {
                auto itRecv = channels.find(recvRank);
                CHK_PRT_RET(itRecv == channels.end() || itRecv->second.empty(),
                    HCCL_ERROR("[InsTempAlltoAllVMeshOcs][RunLimitedConcurrencyOCS] myRank=%u, recvRankPhysic=%u "
                        "not found in channels", myRank_, recvRank),
                    HCCL_E_INTERNAL);
                recvChannelVec = &itRecv->second;
                /* 按 channel（Port Group）分片接收数据 */
                CHK_RET(CalcDataSplitByPortGroup(tempAlgParams.recvCounts[recvRank], dataTypeSize_, *recvChannelVec,
                    recvCountsSplit_, recvSizeSplit_, recvOffsetSplit_));
            }

            /* 确定 channel 数量：send 和 recv 的 channel 数必须一致（同时发收时） */
            u32 channelCount = 0;
            if (sendChannelVec != nullptr && recvChannelVec != nullptr) {
                CHK_PRT_RET(sendChannelVec->size() != recvChannelVec->size(),
                    HCCL_ERROR("[InsTempAlltoAllVMeshOcs][RunLimitedConcurrencyOCS] myRank=%u send/recv channel size "
                        "mismatch send[%zu] recv[%zu]", myRank_, sendChannelVec->size(), recvChannelVec->size()),
                    HCCL_E_INTERNAL);
                channelCount = static_cast<u32>(sendChannelVec->size());
            } else if (sendChannelVec != nullptr) {
                channelCount = static_cast<u32>(sendChannelVec->size());
            } else {
                channelCount = static_cast<u32>(recvChannelVec->size());
            }
            /* 计算 CCL Buffer 槽位索引 */
            u32 dstIdx = CalcSendDstIdxOcs(sendRank, concurrentPlanesPerRound);
            u32 srcIdx = CalcRecvSrcIdxOcs(recvRank, concurrentPlanesPerRound);

            /* 空通道占位（当 doSend=false 或 doRecv=false 时使用） */
            static const std::vector<ChannelInfo> kEmptyChannels;
            const std::vector<ChannelInfo> &sendChannels = sendChannelVec != nullptr ? *sendChannelVec : kEmptyChannels;
            const std::vector<ChannelInfo> &recvChannels = recvChannelVec != nullptr ? *recvChannelVec : kEmptyChannels;

            /* 按 channel 分配从线程执行数据搬移
             * 每个 channel 由一个独立从线程处理，实现多通道并行 */
            for (u32 channelId = 0; channelId < channelCount; channelId++) {
                /* 校验线程索引不越界 */
                CHK_PRT_RET(queIdx >= threads.size(),
                    HCCL_ERROR("[InsTempAlltoAllVMeshOcs][RunLimitedConcurrencyOCS] myRank=%u queIdx[%u] >= threads[%zu]",
                        myRank_, queIdx, threads.size()),
                    HCCL_E_INTERNAL);
                /* 执行单通道数据路径：SendWrite/RecvWrite + PostCopy */
                CHK_RET(RunSendRecvByChannelOcs(tempAlgParams, sendChannels, recvChannels,
                    sendRank, recvRank, dstIdx, srcIdx, threads[queIdx], channelId, doSend, doRecv));
                queIdx++;
            }
        }

        /* 3d. 主线程等待所有从线程完成（PostSync）
         * 确保本轮所有 SendRecv 完成后，再开始下一轮（避免 CCL 槽位冲突） */
        if (threadNum_ > 1) {
            GetNotifyIdxSubToMain(notifyIdxSubToMain_);
            CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMeshOcs::CalcCommRankSetforOneRoundOcs(u32 roundIdx, u32 concurrentPlanesPerRound,
    std::vector<u32> &sendToRanks, std::vector<u32> &recvFromRanks)
{
    sendToRanks.clear();
    recvFromRanks.clear();

    /* ---- 计算 plane 维度参数 ---- */
    /* numPlanes = 每个 group 内的 rank 数 = templateRankSize_ / numGroups_ */
    u32 numPlanes = templateRankSize_ / numGroups_;
    /* localRank = 本 rank 在 plane 维度的位置（同一 group 内的编号） */
    u32 localRank = myRank_ % numPlanes;

    HCCL_DEBUG("[InsTempAlltoAllVMeshOcs][CalcCommRankSetforOneRoundOcs] myRank=%u localRank=%u numGroups=%u "
        "numPlanes=%u concurrentPlanesPerRound=%u roundIdx=%u",
        myRank_, localRank, numGroups_, numPlanes, concurrentPlanesPerRound, roundIdx);

    /* ---- 遍历本轮的每个并发 plane 和每个 group，生成通信对端 ---- */
    for (u32 planeIdx = 0; planeIdx < concurrentPlanesPerRound; planeIdx++) {
        /* 发送目标 plane：从 localRank+1 开始，顺时针步进
         * roundIdx * concurrentPlanesPerRound 跳过前几轮已处理的 plane
         * planeIdx 在本轮内的偏移 */
        u32 sendPlane = (localRank + 1 + roundIdx * concurrentPlanesPerRound + planeIdx) % numPlanes;

        /* 接收源 plane：从 localRank-1 开始，逆时针步进
         * 与 send 方向对称，保证 send/recv 配对正确 */
        u32 recvPlane = (localRank + numPlanes - 1 - roundIdx * concurrentPlanesPerRound - planeIdx) % numPlanes;

        /* 每个 plane × 每个 group 产生一对通信对端
         * 逻辑 rank = groupId × numPlanes + planeIdx */
        for (u32 groupId = 0; groupId < numGroups_; groupId++) {
            sendToRanks.push_back(groupId * numPlanes + sendPlane);
            recvFromRanks.push_back(groupId * numPlanes + recvPlane);
        }
    }
    return HCCL_SUCCESS;
}

void InsTempAlltoAllVMeshOcs::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    if (threadNum_ <= 1) {
        return;
    }
    u32 slaveThreadNum = threadNum_ - 1;
    /* 所有从线程使用 notify index 0(主线程 notify index) */
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

u32 InsTempAlltoAllVMeshOcs::CalcSendDstIdxOcs(u32 dstRank, u32 concurrentPlanesPerRound) const
{
    u32 numPlanes = templateRankSize_ / numGroups_;
    /* 对端的 plane 编号 */
    u32 peerPlane = dstRank % numPlanes;
    /* 本端的 plane 编号 */
    u32 myPlane = myRank_ % numPlanes;
    /* 本端所在的 group 编号（发送时数据来自本 group，槽位与 groupId 关联） */
    u32 groupIdSrc = myRank_ / numPlanes;

    /* 顺时针距离：从 myPlane 到 peerPlane 需要顺时针走几步 */
    u32 clockwiseDis = (peerPlane - myPlane + numPlanes) % numPlanes;
    /* 距离 1 对应第 0 个并发 plane（planeIdx=0），距离 2 对应第 1 个，以此类推
     * +concurrentPlanesPerRound 取模防止负数 */
    u32 planeIdx = (clockwiseDis - 1 + concurrentPlanesPerRound) % concurrentPlanesPerRound;

    /* 全局槽位 = planeIdx × numGroups_ + groupId
     * 同一 plane 偏移下，不同 group 的槽位相邻排列 */
    return planeIdx * numGroups_ + groupIdSrc;
}

u32 InsTempAlltoAllVMeshOcs::CalcRecvSrcIdxOcs(u32 srcRank, u32 concurrentPlanesPerRound) const
{
    u32 numPlanes = templateRankSize_ / numGroups_;
    /* 源端的 plane 编号 */
    u32 peerPlane = srcRank % numPlanes;
    /* 本端的 plane 编号 */
    u32 myPlane = myRank_ % numPlanes;
    /* 源端的 group 编号（接收时数据来自源 group，槽位与源 groupId 关联） */
    u32 groupIdSrc = srcRank / numPlanes;

    /* 顺时针距离：从 peerPlane 到 myPlane 需要顺时针走几步
     * 与 Send 方向相反：Recv 是对方发给我，距离从对方视角看是到我的顺时针距离 */
    u32 clockwiseDis = (myPlane - peerPlane + numPlanes) % numPlanes;
    u32 planeIdx = (clockwiseDis - 1 + concurrentPlanesPerRound) % concurrentPlanesPerRound;

    return planeIdx * numGroups_ + groupIdSrc;
}

void InsTempAlltoAllVMeshOcs::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = threadNum_ - 1;
    /* 从线程 i 使用 notify index i（逐个确认，确保所有从线程完成后才进入下一轮） */
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
    return;
}

HcclResult InsTempAlltoAllVMeshOcs::RunSendRecvByChannelOcs(const TemplateDataParams &tempAlgParams,
    const std::vector<ChannelInfo> &sendChannels, const std::vector<ChannelInfo> &recvChannels,
    u32 sendRank, u32 recvRank, u32 dstIdx, u32 srcIdx, const ThreadHandle &thread, u32 channelId,
    bool doSend, bool doRecv) const
{
    /* 构造发送/接收的源/目标数据切片 */
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    if (doSend) {
        /* 发送源：用户输入 buffer，偏移 = sdispls[sendRank] × dataTypeSize + channelOffset */
        void* remoteCclBuffAddr = sendChannels[channelId].remoteCclMem.addr;
        DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.sdispls[sendRank] * dataTypeSize_ + sendOffsetSplit_[channelId],
            sendSizeSplit_[channelId], sendCountsSplit_[channelId]);
        /* 发送目标：对端 CCL Buffer 的 dstIdx 槽位，偏移 = dstIdx × stride + baseOff + channelOffset */
        DataSlice txDstSlice = DataSlice(remoteCclBuffAddr,
            static_cast<u64>(dstIdx) * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff +
            sendOffsetSplit_[channelId], sendSizeSplit_[channelId], sendCountsSplit_[channelId]);
        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);
    }

    if (doRecv) {
        /* 接收源：占位（RecvWrite 不需要指定源地址，由链路写入） */
        DataSlice rxSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, 0, 0, 0);
        /* 接收目标：本端 CCL Buffer 的 srcIdx 槽位，偏移 = srcIdx × stride + baseOff + channelOffset
         * 注意：必须写入 CCL Buffer，不能直写 output（否则 PostCopy 读脏数据） */
        DataSlice rxDstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
            static_cast<u64>(srcIdx) * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff +
            recvOffsetSplit_[channelId], recvSizeSplit_[channelId], recvCountsSplit_[channelId]);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);
    }

    /* ---- 执行数据传输 ----
     * 优先使用 SendRecvWrite（同一线程同时发收，减少同步开销）
     * 仅发或仅收时使用 SendWrite 或 RecvWrite */
    if (doSend && doRecv) {
        /* 同时发送和接收：使用 SendRecvWrite，一次提交双向操作 */
        SendRecvInfo sendRecvInfo{{sendChannels[channelId], recvChannels[channelId]},
            {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_};
        CHK_PRT_RET(SendRecvWrite(sendRecvInfo, thread),
            HCCL_ERROR("[InsTempAlltoAllVMeshOcs][RunSendRecvByChannelOcs] myRank=%u SendRecvWrite failed", myRank_),
            HCCL_E_INTERNAL);
    } else if (doSend) {
        /* 仅发送 */
        DataInfo sendInfo{sendChannels[channelId], {txSrcSlices, txDstSlices}, dataType_};
        CHK_PRT_RET(SendWrite(sendInfo, thread),
            HCCL_ERROR("[InsTempAlltoAllVMeshOcs][RunSendRecvByChannelOcs] myRank=%u SendWrite failed", myRank_),
            HCCL_E_INTERNAL);
    } else if (doRecv) {
        /* 仅接收 */
        DataInfo recvInfo{recvChannels[channelId], {rxSrcSlices, rxDstSlices}, dataType_};
        CHK_PRT_RET(RecvWrite(recvInfo, thread),
            HCCL_ERROR("[InsTempAlltoAllVMeshOcs][RunSendRecvByChannelOcs] myRank=%u RecvWrite failed", myRank_),
            HCCL_E_INTERNAL);
    }

    /* ---- recv 后执行 PostCopy ----
     * 将 CCL Buffer 槽位中的数据搬到用户输出 buffer
     * 注：RecvWrite/LocalCopy 在 batch mode 下位于同一 thread 的 task 队列中，
     * 硬件按 FIFO 顺序执行，因此 LocalCopy 开始时 RecvWrite 已完成数据写入 */
    if (doRecv && recvSizeSplit_[channelId] > 0) {
        CHK_RET(PostCopyOcs(tempAlgParams, thread, srcIdx, recvRank,
            recvSizeSplit_[channelId], recvCountsSplit_[channelId], recvOffsetSplit_[channelId]));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMeshOcs::PostCopyOcs(const TemplateDataParams &tempAlgParams, const ThreadHandle &thread,
    u32 srcIdx, u32 recvRank, u64 recvSize, u64 recvCount, u64 recvOffset) const
{
    /* 源：CCL Buffer 的 srcIdx 槽位，偏移 = srcIdx × inputSliceStride + baseOff + channelOffset */
    DataSlice localCopySrcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
        static_cast<u64>(srcIdx) * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff + recvOffset,
        recvSize, recvCount);
    /* 目标：用户输出 buffer，偏移 = rdispls[recvRank] × dataTypeSize + channelOffset */
    DataSlice localCopyDstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.rdispls[recvRank] * dataTypeSize_ + recvOffset,
        recvSize, recvCount);
    CHK_RET(static_cast<HcclResult>(LocalCopy(thread, localCopySrcSlice, localCopyDstSlice)));
    return HCCL_SUCCESS;
}

}