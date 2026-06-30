/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_all_to_all_v_mesh_1D.h"

namespace ops_hccl {
namespace {
constexpr u32 ALLTOALL_DETOUR_RANK_SIZE = 8;
constexpr u32 ALLTOALL_DETOUR_SRC_RANK = 0;
constexpr u32 ALLTOALL_DETOUR_RELAY_RANK = 3;
constexpr u32 ALLTOALL_DETOUR_DST_BEGIN = 4;
constexpr u32 ALLTOALL_DETOUR_DST_END = 7;
}

InsTempAlltoAllVMesh1D::InsTempAlltoAllVMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks), opType_(param.opType)
{
}

InsTempAlltoAllVMesh1D::~InsTempAlltoAllVMesh1D()
{
}

HcclResult InsTempAlltoAllVMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    u32 threadNum = templateRankSize_;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // usrIn和cclBuffer大小相同
    return IsAlltoAllDetourCandidate() ? 2 : 1;
}

HcclResult InsTempAlltoAllVMesh1D::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = SIZE_TABLE[dataType_];
    cclBufferCountPerRank_ = tempAlgParams.outputSliceStride / dataTypeSize_;
    HCCL_INFO("[InsTempAlltoAllVMesh1D] Run Start");

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunALLtoALL(templateResource.channels, templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    if (IsAlltoAllDetourEnabled()) {
        u32 myAlgRank = 0;
        auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
        if (iter != subCommRanks_[0].end()) {
            myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
        } else {
            HCCL_ERROR("[InsTempAlltoAllVMesh1D][KernelRun] subCommRanks_ or myRank_ is error.");
            return HCCL_E_INTERNAL;
        }
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
        CHK_RET(RunDetourForward(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank));
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempAlltoAllVMesh1D] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunALLtoALL(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunALLtoALL] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }

    if (IsAlltoAllDetourEnabled()) {
        HCCL_INFO("[InsTempAlltoAllVMesh1D][RunALLtoALL] enable alltoall detour, myAlgRank[%u], rankSize[%u].",
            myAlgRank, templateRankSize_);
        CHK_RET(RunDetourPreStage(channels, threads, tempAlgParams, myAlgRank));
    }

    for (u32 queIdx = 0; queIdx < threadNum_; queIdx++) {
        if (queIdx == myAlgRank) {
            // local copy
            DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
                tempAlgParams.sdispls[myAlgRank] * dataTypeSize_,
                tempAlgParams.sendCounts[myAlgRank] * dataTypeSize_, tempAlgParams.sendCounts[myAlgRank]);
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff,
                tempAlgParams.sendCounts[myAlgRank] * dataTypeSize_, tempAlgParams.sendCounts[myAlgRank]);

            if (tempAlgParams.sendCounts[myAlgRank] > 0) {
                CHK_RET(static_cast<HcclResult>(LocalCopy(threads[queIdx], srcSlice, dstSlice)));
            }
            continue;
        }

        u32 nextRank = queIdx; // 逻辑rank
        u32 remoteRank = subCommRanks_[0][nextRank]; // 物理rank

        const ChannelInfo &linkSend = channels.at(remoteRank)[0]; // 发给哪个rank
        const ChannelInfo &linkRecv = channels.at(remoteRank)[0]; // 收哪个rank的数据
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        u64 sendCount = ShouldSkipDirectSend(myAlgRank, nextRank) ? 0 : tempAlgParams.sendCounts[nextRank];
        u64 recvCount = ShouldSkipDirectRecv(myAlgRank, nextRank) ? 0 : tempAlgParams.recvCounts[nextRank];
        void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;
        // repeatNum为1，所以这里不考虑重复场景
        DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, tempAlgParams.sdispls[nextRank] * dataTypeSize_,
            sendCount * dataTypeSize_, sendCount);
        DataSlice txDstSlice = DataSlice(remoteCclBuffAddr,
            myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff,
            sendCount * dataTypeSize_, sendCount);

        DataSlice rxSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.rdispls[myAlgRank] * dataTypeSize_,
            recvCount * dataTypeSize_, recvCount);
        DataSlice rxDstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
            nextRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff,
            recvCount * dataTypeSize_, recvCount);

        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);

        // 不用SendRecvWrite接口里面，因为recv 0 也会去等
        DataInfo sendInfo{linkSend, {txSrcSlices, txDstSlices}};
        DataInfo recvInfo{linkRecv, {rxSrcSlices, rxDstSlices}};
        SendRecvInfo sendRecvInfo{{linkSend, linkRecv},
                             {{txSrcSlices, txDstSlices},{rxSrcSlices, rxDstSlices}}};
        if (sendCount > 0 && recvCount > 0) {
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[queIdx]),
                HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvInfo failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else { // 其中一个或者两个为0
            if (sendCount > 0) {
                CHK_PRT_RET(SendWrite(sendInfo, threads[queIdx]),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL sendInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
            if (recvCount > 0) {
                CHK_PRT_RET(RecvWrite(recvInfo, threads[queIdx]),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL recvInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

bool InsTempAlltoAllVMesh1D::IsAlltoAllDetourCandidate() const
{
    return opType_ == HcclCMDType::HCCL_CMD_ALLTOALL && templateRankSize_ == ALLTOALL_DETOUR_RANK_SIZE;
}

bool InsTempAlltoAllVMesh1D::IsAlltoAllDetourEnabled() const
{
    return IsAlltoAllDetourCandidate() && threadNum_ > 1;
}

bool InsTempAlltoAllVMesh1D::IsAlltoAllDetourDstRank(const u32 algRank) const
{
    return algRank >= ALLTOALL_DETOUR_DST_BEGIN && algRank <= ALLTOALL_DETOUR_DST_END;
}

bool InsTempAlltoAllVMesh1D::ShouldSkipDirectSend(const u32 myAlgRank, const u32 remoteAlgRank) const
{
    return IsAlltoAllDetourEnabled() && myAlgRank == ALLTOALL_DETOUR_SRC_RANK &&
        IsAlltoAllDetourDstRank(remoteAlgRank);
}

bool InsTempAlltoAllVMesh1D::ShouldSkipDirectRecv(const u32 myAlgRank, const u32 remoteAlgRank) const
{
    return IsAlltoAllDetourEnabled() && IsAlltoAllDetourDstRank(myAlgRank) &&
        remoteAlgRank == ALLTOALL_DETOUR_SRC_RANK;
}

u32 InsTempAlltoAllVMesh1D::CalcDetourScratchRank(const u32 dstAlgRank) const
{
    return templateRankSize_ + dstAlgRank - ALLTOALL_DETOUR_DST_BEGIN;
}

HcclResult InsTempAlltoAllVMesh1D::RunDetourPreStage(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams,
    const u32 myAlgRank) const
{
    if (myAlgRank != ALLTOALL_DETOUR_SRC_RANK && myAlgRank != ALLTOALL_DETOUR_RELAY_RANK) {
        return HCCL_SUCCESS;
    }

    u32 peerAlgRank = myAlgRank == ALLTOALL_DETOUR_SRC_RANK ?
        ALLTOALL_DETOUR_RELAY_RANK : ALLTOALL_DETOUR_SRC_RANK;
    u32 peerRank = subCommRanks_[0][peerAlgRank];
    auto channelIter = channels.find(peerRank);
    if (channelIter == channels.end() || channelIter->second.empty()) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunDetourPreStage] peerRank[%u] does not exist in channels map.",
            peerRank);
        return HCCL_E_PARA;
    }
    const ChannelInfo &channel = channelIter->second[0];
    const ThreadHandle &thread = threads[peerAlgRank];

    for (u32 dstAlgRank = ALLTOALL_DETOUR_DST_BEGIN; dstAlgRank <= ALLTOALL_DETOUR_DST_END; dstAlgRank++) {
        u64 detourCount = tempAlgParams.sendCounts[dstAlgRank];
        if (detourCount == 0) {
            continue;
        }

        u64 detourSize = detourCount * dataTypeSize_;
        u64 scratchOffset = CalcDetourScratchRank(dstAlgRank) * cclBufferCountPerRank_ * dataTypeSize_ +
            tempAlgParams.buffInfo.hcclBuffBaseOff;
        std::vector<DataSlice> srcSlices;
        std::vector<DataSlice> dstSlices;
        if (myAlgRank == ALLTOALL_DETOUR_SRC_RANK) {
            srcSlices.push_back(DataSlice(tempAlgParams.buffInfo.inputPtr,
                tempAlgParams.sdispls[dstAlgRank] * dataTypeSize_, detourSize, detourCount));
            dstSlices.push_back(DataSlice(channel.remoteCclMem.addr, scratchOffset, detourSize, detourCount));
            DataInfo sendInfo{channel, {srcSlices, dstSlices}};
            CHK_PRT_RET(SendWrite(sendInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunDetourPreStage] SendWrite failed."),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            srcSlices.push_back(DataSlice(tempAlgParams.buffInfo.inputPtr, 0, detourSize, detourCount));
            dstSlices.push_back(DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, scratchOffset, detourSize, detourCount));
            DataInfo recvInfo{channel, {srcSlices, dstSlices}};
            CHK_PRT_RET(RecvWrite(recvInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunDetourPreStage] RecvWrite failed."),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunDetourForward(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams,
    const u32 myAlgRank) const
{
    if (myAlgRank != ALLTOALL_DETOUR_RELAY_RANK && !IsAlltoAllDetourDstRank(myAlgRank)) {
        return HCCL_SUCCESS;
    }

    if (myAlgRank == ALLTOALL_DETOUR_RELAY_RANK) {
        for (u32 dstAlgRank = ALLTOALL_DETOUR_DST_BEGIN; dstAlgRank <= ALLTOALL_DETOUR_DST_END; dstAlgRank++) {
            u64 detourCount = tempAlgParams.sendCounts[dstAlgRank];
            if (detourCount == 0) {
                continue;
            }

            u32 dstRank = subCommRanks_[0][dstAlgRank];
            auto channelIter = channels.find(dstRank);
            if (channelIter == channels.end() || channelIter->second.empty()) {
                HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunDetourForward] dstRank[%u] does not exist in channels map.",
                    dstRank);
                return HCCL_E_PARA;
            }
            const ChannelInfo &channel = channelIter->second[0];
            u64 detourSize = detourCount * dataTypeSize_;
            u64 scratchOffset = CalcDetourScratchRank(dstAlgRank) * cclBufferCountPerRank_ * dataTypeSize_ +
                tempAlgParams.buffInfo.hcclBuffBaseOff;
            std::vector<DataSlice> srcSlices;
            std::vector<DataSlice> dstSlices;
            srcSlices.push_back(DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, scratchOffset,
                detourSize, detourCount));
            dstSlices.push_back(DataSlice(channel.remoteCclMem.addr,
                ALLTOALL_DETOUR_SRC_RANK * cclBufferCountPerRank_ * dataTypeSize_ +
                tempAlgParams.buffInfo.hcclBuffBaseOff, detourSize, detourCount));
            DataInfo sendInfo{channel, {srcSlices, dstSlices}};
            CHK_PRT_RET(SendWrite(sendInfo, threads[dstAlgRank]),
                HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunDetourForward] SendWrite failed."),
                HcclResult::HCCL_E_INTERNAL);
        }
        return HCCL_SUCCESS;
    }

    u64 detourCount = tempAlgParams.recvCounts[ALLTOALL_DETOUR_SRC_RANK];
    if (detourCount == 0) {
        return HCCL_SUCCESS;
    }

    u32 relayRank = subCommRanks_[0][ALLTOALL_DETOUR_RELAY_RANK];
    auto channelIter = channels.find(relayRank);
    if (channelIter == channels.end() || channelIter->second.empty()) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunDetourForward] relayRank[%u] does not exist in channels map.",
            relayRank);
        return HCCL_E_PARA;
    }
    const ChannelInfo &channel = channelIter->second[0];
    u64 detourSize = detourCount * dataTypeSize_;
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;
    srcSlices.push_back(DataSlice(tempAlgParams.buffInfo.inputPtr, 0, detourSize, detourCount));
    dstSlices.push_back(DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
        ALLTOALL_DETOUR_SRC_RANK * cclBufferCountPerRank_ * dataTypeSize_ +
        tempAlgParams.buffInfo.hcclBuffBaseOff, detourSize, detourCount));
    DataInfo recvInfo{channel, {srcSlices, dstSlices}};
    CHK_PRT_RET(RecvWrite(recvInfo, threads[ALLTOALL_DETOUR_RELAY_RANK]),
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunDetourForward] RecvWrite failed."),
        HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PostCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads) const
{
    // ccl buffer的数据搬运到usrout
    for (u32 queIdx = 0; queIdx < threadNum_; queIdx++) {
        // local copy
        u32 myAlgRank = queIdx;
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
            myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff,
            tempAlgParams.recvCounts[myAlgRank] * dataTypeSize_, tempAlgParams.recvCounts[myAlgRank]);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
            tempAlgParams.rdispls[myAlgRank] * dataTypeSize_,
            tempAlgParams.recvCounts[myAlgRank] * dataTypeSize_, tempAlgParams.recvCounts[myAlgRank]);
        if (tempAlgParams.recvCounts[myAlgRank] > 0) {
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads[queIdx], srcSlice, dstSlice)));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAlltoAllVMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = templateRankSize_;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAlltoAllVMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}
} // namespace Hccl
