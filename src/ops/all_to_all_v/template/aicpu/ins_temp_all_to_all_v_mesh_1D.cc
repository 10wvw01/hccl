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

HcclResult InsTempAlltoAllVMesh1D::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
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
    return IsAlltoAllDetourCandidate() ? 2 : 1;
}

HcclResult InsTempAlltoAllVMesh1D::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAlltoAllVMesh1D][KernelRun] Run Start");
    threadNum_ = templateResource.threads.size();
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = SIZE_TABLE[dataType_];
    cclBufferCountPerRank_ = tempAlgParams.outputSliceStride / dataTypeSize_;
    isDmaRead_ = IsPcieProtocol(templateResource.channels);
    HCCL_DEBUG("[InsTempAlltoAllVMesh1D][KernelRun] Use Dma Read[%d]", isDmaRead_);

    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][KernelRun] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }

    if (isDmaRead_) {
        if (threadNum_ > 1) {
            std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
            GetNotifyIdxMainToSub(notifyIdxMainToSub_);
            CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
        }
        if (IsAlltoAllDetourEnabled()) {
            CHK_RET(RunDetourPreStage(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank));
        }
        CHK_RET(RunALLtoALL(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank));
        if (threadNum_ > 1) {
            std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
            GetNotifyIdxSubToMain(notifyIdxSubToMain_);
            CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        }
        if (IsAlltoAllDetourEnabled()) {
            if (threadNum_ > 1) {
                std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                    templateResource.threads.end());
                GetNotifyIdxMainToSub(notifyIdxMainToSub_);
                CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
            }
            CHK_RET(RunDetourForward(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank));
            if (threadNum_ > 1) {
                std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                    templateResource.threads.end());
                GetNotifyIdxSubToMain(notifyIdxSubToMain_);
                CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
            }
        }
    } else {
        if (threadNum_ > 1) {
            std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
            GetNotifyIdxMainToSub(notifyIdxMainToSub_);
            CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
        }
        if (IsAlltoAllDetourEnabled()) {
            CHK_RET(RunDetourPreStage(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank));
        }
        CHK_RET(RunALLtoALL(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank));
        if (threadNum_ > 1) {
            std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
            GetNotifyIdxSubToMain(notifyIdxSubToMain_);
            CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        }

        if (IsAlltoAllDetourEnabled()) {
            if (threadNum_ > 1) {
                std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                    templateResource.threads.end());
                GetNotifyIdxMainToSub(notifyIdxMainToSub_);
                CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
            }
            CHK_RET(RunDetourForward(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank));
            if (threadNum_ > 1) {
                std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                    templateResource.threads.end());
                GetNotifyIdxSubToMain(notifyIdxSubToMain_);
                CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
            }
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
    }

    HCCL_INFO("[InsTempAlltoAllVMesh1D][KernelRun] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunALLtoALL(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams,
    const u32 myAlgRank)
{
    for (u32 queIdx = 0; queIdx < threadNum_; queIdx++) {
        if (queIdx == myAlgRank) {
            CHK_RET(LocalCopyForMyRank(tempAlgParams, threads[queIdx], myAlgRank));
            continue;
        }

        u32 nextRank = queIdx; // 逻辑rank
        u32 remoteRank = subCommRanks_[0][nextRank]; // 物理rank
        const ChannelInfo &linkSend = channels.at(remoteRank)[0];
        const ChannelInfo &linkRecv = channels.at(remoteRank)[0];
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        u64 sendCount = ShouldSkipDirectSend(myAlgRank, nextRank) ? 0 : tempAlgParams.sendCounts[nextRank];
        u64 recvCount = ShouldSkipDirectRecv(myAlgRank, nextRank) ? 0 : tempAlgParams.recvCounts[nextRank];
        if (isDmaRead_ && sendCount > 0) {
            CHK_RET(PreCopyForRemoteRank(tempAlgParams, threads[queIdx], nextRank, sendCount));
        }

        void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;
        DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.sdispls[nextRank] * dataTypeSize_, sendCount * dataTypeSize_, sendCount);
        DataSlice txDstSlice = DataSlice(remoteCclBuffAddr,
            CalcCclBuffOffset(myAlgRank, tempAlgParams), sendCount * dataTypeSize_, sendCount);

        void *rxDstAddr = isDmaRead_ ? tempAlgParams.buffInfo.outputPtr : tempAlgParams.buffInfo.hcclBuff.addr;
        u64 rxDstOffset = isDmaRead_ ? tempAlgParams.rdispls[nextRank] * dataTypeSize_ :
            CalcCclBuffOffset(nextRank, tempAlgParams);
        DataSlice rxSrcSlice = DataSlice(remoteCclBuffAddr,
            CalcCclBuffOffset(myAlgRank, tempAlgParams), recvCount * dataTypeSize_, recvCount);
        DataSlice rxDstSlice = DataSlice(rxDstAddr, rxDstOffset, recvCount * dataTypeSize_, recvCount);

        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);

        DataInfo sendInfo{linkSend, {txSrcSlices, txDstSlices}};
        DataInfo recvInfo{linkRecv, {rxSrcSlices, rxDstSlices}};
        SendRecvInfo sendRecvInfo{{linkSend, linkRecv},
                             {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
        CHK_RET(RunSendRecv(sendRecvInfo, sendInfo, recvInfo, threads[queIdx], sendCount, recvCount));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::LocalCopyForMyRank(
    const TemplateDataParams &tempAlgParams, const ThreadHandle &thread, const u32 myAlgRank) const
{
    void *dstAddr = isDmaRead_ ? tempAlgParams.buffInfo.outputPtr : tempAlgParams.buffInfo.hcclBuff.addr;
    u64 dstOffset = isDmaRead_ ? tempAlgParams.rdispls[myAlgRank] * dataTypeSize_ :
        CalcCclBuffOffset(myAlgRank, tempAlgParams);
    DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
        tempAlgParams.sdispls[myAlgRank] * dataTypeSize_,
        tempAlgParams.sendCounts[myAlgRank] * dataTypeSize_, tempAlgParams.sendCounts[myAlgRank]);
    DataSlice dstSlice = DataSlice(dstAddr, dstOffset,
        tempAlgParams.sendCounts[myAlgRank] * dataTypeSize_, tempAlgParams.sendCounts[myAlgRank]);

    if (tempAlgParams.sendCounts[myAlgRank] > 0) {
        CHK_RET(static_cast<HcclResult>(LocalCopy(thread, srcSlice, dstSlice)));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PreCopyForRemoteRank(const TemplateDataParams &tempAlgParams,
    const ThreadHandle &thread, const u32 remoteAlgRank, const u64 sendCount) const
{
    DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
        tempAlgParams.sdispls[remoteAlgRank] * dataTypeSize_, sendCount * dataTypeSize_, sendCount);
    DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
        CalcCclBuffOffset(remoteAlgRank, tempAlgParams), sendCount * dataTypeSize_, sendCount);
    CHK_RET(static_cast<HcclResult>(LocalCopy(thread, srcSlice, dstSlice)));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunSendRecv(const SendRecvInfo &sendRecvInfo,
    const DataInfo &sendInfo, const DataInfo &recvInfo, const ThreadHandle& thread,
    const u64 sendCount, const u64 recvCount) const
{
    if (isDmaRead_) {
        if (sendCount > 0 && recvCount > 0) {
            CHK_PRT_RET(SendRecvRead(sendRecvInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvRead failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            if (sendCount > 0) {
                CHK_PRT_RET(SendRead(sendInfo, thread),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRead failed"),
                    HcclResult::HCCL_E_INTERNAL);
            } else if (recvCount > 0) {
                CHK_PRT_RET(RecvRead(recvInfo, thread),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL RecvRead failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    } else {
        if (sendCount > 0 && recvCount > 0) {
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvWrite failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            if (sendCount > 0) {
                CHK_PRT_RET(SendWrite(sendInfo, thread),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendWrite failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
            if (recvCount > 0) {
                CHK_PRT_RET(RecvWrite(recvInfo, thread),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL RecvWrite failed"),
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

u64 InsTempAlltoAllVMesh1D::CalcCclBuffOffset(const u32 algRank, const TemplateDataParams &tempAlgParams) const
{
    return algRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;
}

u32 InsTempAlltoAllVMesh1D::CalcDetourScratchRank(const u32 dstAlgRank) const
{
    return templateRankSize_ + dstAlgRank - ALLTOALL_DETOUR_DST_BEGIN;
}

bool InsTempAlltoAllVMesh1D::IsPcieProtocol(const std::map<u32, std::vector<ChannelInfo>> &channels) const
{
    for (auto iter = channels.begin(); iter != channels.end(); iter++) {
        for (const auto &channel : iter->second) {
            if (channel.protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
                return true;
            }
        }
    }
    return false;
}

HcclResult InsTempAlltoAllVMesh1D::RunDetourPreStage(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams,
    const u32 myAlgRank) const
{
    if (!IsAlltoAllDetourEnabled()) {
        return HCCL_SUCCESS;
    }
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

        if (myAlgRank == ALLTOALL_DETOUR_SRC_RANK && isDmaRead_) {
            CHK_RET(PreCopyForRemoteRank(tempAlgParams, thread, dstAlgRank, detourCount));
        }

        u64 detourSize = detourCount * dataTypeSize_;
        std::vector<DataSlice> srcSlices;
        std::vector<DataSlice> dstSlices;
        if (myAlgRank == ALLTOALL_DETOUR_SRC_RANK) {
            void *srcAddr = isDmaRead_ ? tempAlgParams.buffInfo.hcclBuff.addr : tempAlgParams.buffInfo.inputPtr;
            u64 srcOffset = isDmaRead_ ? CalcCclBuffOffset(dstAlgRank, tempAlgParams) :
                tempAlgParams.sdispls[dstAlgRank] * dataTypeSize_;
            srcSlices.push_back(DataSlice(srcAddr, srcOffset, detourSize, detourCount));
            dstSlices.push_back(DataSlice(channel.remoteCclMem.addr,
                CalcCclBuffOffset(CalcDetourScratchRank(dstAlgRank), tempAlgParams), detourSize, detourCount));
            DataInfo sendInfo{channel, {srcSlices, dstSlices}};
            HcclResult ret = isDmaRead_ ? SendRead(sendInfo, thread) : SendWrite(sendInfo, thread);
            CHK_PRT_RET(ret, HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunDetourPreStage] Send failed."),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            void *dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
            srcSlices.push_back(DataSlice(channel.remoteCclMem.addr,
                CalcCclBuffOffset(dstAlgRank, tempAlgParams), detourSize, detourCount));
            dstSlices.push_back(DataSlice(dstAddr,
                CalcCclBuffOffset(CalcDetourScratchRank(dstAlgRank), tempAlgParams), detourSize, detourCount));
            DataInfo recvInfo{channel, {srcSlices, dstSlices}};
            HcclResult ret = isDmaRead_ ? RecvRead(recvInfo, thread) : RecvWrite(recvInfo, thread);
            CHK_PRT_RET(ret, HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunDetourPreStage] Recv failed."),
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
    if (!IsAlltoAllDetourEnabled()) {
        return HCCL_SUCCESS;
    }
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
            std::vector<DataSlice> srcSlices;
            std::vector<DataSlice> dstSlices;
            srcSlices.push_back(DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                CalcCclBuffOffset(CalcDetourScratchRank(dstAlgRank), tempAlgParams), detourSize, detourCount));
            dstSlices.push_back(DataSlice(channel.remoteCclMem.addr,
                CalcCclBuffOffset(ALLTOALL_DETOUR_SRC_RANK, tempAlgParams), detourSize, detourCount));
            DataInfo sendInfo{channel, {srcSlices, dstSlices}};
            HcclResult ret = isDmaRead_ ? SendRead(sendInfo, threads[dstAlgRank]) :
                SendWrite(sendInfo, threads[dstAlgRank]);
            CHK_PRT_RET(ret, HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunDetourForward] Send failed."),
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
    void *dstAddr = isDmaRead_ ? tempAlgParams.buffInfo.outputPtr : tempAlgParams.buffInfo.hcclBuff.addr;
    u64 dstOffset = isDmaRead_ ? tempAlgParams.rdispls[ALLTOALL_DETOUR_SRC_RANK] * dataTypeSize_ :
        CalcCclBuffOffset(ALLTOALL_DETOUR_SRC_RANK, tempAlgParams);
    srcSlices.push_back(DataSlice(channel.remoteCclMem.addr,
        CalcCclBuffOffset(CalcDetourScratchRank(myAlgRank), tempAlgParams), detourSize, detourCount));
    dstSlices.push_back(DataSlice(dstAddr, dstOffset, detourSize, detourCount));
    DataInfo recvInfo{channel, {srcSlices, dstSlices}};
    HcclResult ret = isDmaRead_ ? RecvRead(recvInfo, threads[ALLTOALL_DETOUR_RELAY_RANK]) :
        RecvWrite(recvInfo, threads[ALLTOALL_DETOUR_RELAY_RANK]);
    CHK_PRT_RET(ret, HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunDetourForward] Recv failed."),
        HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PostCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads) const
{
    // ccl buffer的数据搬运到usrout
    for (u32 queIdx = 0; queIdx < threadNum_; queIdx++) {
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
            CalcCclBuffOffset(queIdx, tempAlgParams),
            tempAlgParams.recvCounts[queIdx] * dataTypeSize_, tempAlgParams.recvCounts[queIdx]);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
            tempAlgParams.rdispls[queIdx] * dataTypeSize_,
            tempAlgParams.recvCounts[queIdx] * dataTypeSize_, tempAlgParams.recvCounts[queIdx]);
        if (tempAlgParams.recvCounts[queIdx] > 0) {
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
