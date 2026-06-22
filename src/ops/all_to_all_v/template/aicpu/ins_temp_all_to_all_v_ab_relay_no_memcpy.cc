/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_all_to_all_v_ab_relay_no_memcpy.h"
#include <algorithm>

namespace ops_hccl {
namespace {
bool HasPeerData(const std::vector<u64> &counts, u32 rank)
{
    return rank < counts.size() && counts[rank] > 0;
}
}

InsTempAlltoAllVABRelayNoMemcpy::InsTempAlltoAllVABRelayNoMemcpy(
    const OpParam &param, u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

HcclResult InsTempAlltoAllVABRelayNoMemcpy::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVABRelayNoMemcpy::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}

void InsTempAlltoAllVABRelayNoMemcpy::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    for (u32 i = 1; i < threadNum_; ++i) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAlltoAllVABRelayNoMemcpy::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    for (u32 i = 1; i < threadNum_; ++i) {
        notifyIdxSubToMain.push_back(i - 1);
    }
}

HcclResult InsTempAlltoAllVABRelayNoMemcpy::KernelRun(
    const OpParam &param, const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
{
    threadNum_ = templateResource.threads.size();
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataType_ >= HCCL_DATA_TYPE_RESERVED || dataTypeSize_ == 0,
                HCCL_ERROR("[A2AV_AB_RELAY][Template] invalid datatype[%d].", static_cast<int>(dataType_)),
                HcclResult::HCCL_E_INTERNAL);
    CHK_RET(CheckCommonParams(tempAlgParams, templateResource));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdx;
        GetNotifyIdxMainToSub(notifyIdx);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdx));
    }
    if (phase_ == A2AVABRelayPhase::PREROUTE_TO_RELAY) {
        CHK_RET(RunPrerouteToRelay(tempAlgParams, templateResource));
    } else {
        CHK_RET(RunRelayToOutput(tempAlgParams, templateResource));
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdx;
        GetNotifyIdxSubToMain(notifyIdx);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVABRelayNoMemcpy::CheckCommonParams(
    const TemplateDataParams &params, const TemplateResource &resource) const
{
    CHK_PRT_RET(rankSize_ == 0 || meshSize_ == 0 || rankSize_ % meshSize_ != 0 || groupNum_ == 0,
                HCCL_ERROR("[A2AV_AB_RELAY][Template] invalid relay dims. rank=%u rankSize=%u mesh=%u group=%u",
                           myRank_, rankSize_, meshSize_, groupNum_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(params.sendCounts.size() < rankSize_ || params.recvCounts.size() < rankSize_ ||
                    params.sdispls.size() < rankSize_ || params.rdispls.size() < rankSize_,
                HCCL_ERROR("[A2AV_AB_RELAY][Template] invalid A2AV vectors. rank=%u send=%zu recv=%zu "
                           "sdispl=%zu rdispl=%zu rankSize=%u",
                           myRank_, params.sendCounts.size(), params.recvCounts.size(), params.sdispls.size(),
                           params.rdispls.size(), rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(phase_ == A2AVABRelayPhase::RELAY_TO_OUTPUT &&
                    params.remoteRdispls.size() < rankSize_,
                HCCL_ERROR("[A2AV_AB_RELAY][Template] missing remote rdispls. rank=%u remoteRdispls=%zu rankSize=%u",
                           myRank_, params.remoteRdispls.size(), rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(resource.threads.empty(),
                HCCL_ERROR("[A2AV_AB_RELAY][Template] empty thread resource. rank=%u", myRank_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(slotStride_ == 0,
                HCCL_ERROR("[A2AV_AB_RELAY][Template] zero slot stride. rank=%u", myRank_),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVABRelayNoMemcpy::CheckSliceRange(
    const char *tag, u32 peerRank, u64 srcOffset, u64 dstOffset, u64 byteSize, u64 srcLimit, u64 dstLimit) const
{
    CHK_PRT_RET(srcOffset + byteSize > srcLimit || dstOffset + byteSize > dstLimit,
                HCCL_ERROR("[%s] slice out of range. rank=%u peer=%u srcOff=%llu dstOff=%llu "
                           "size=%llu srcLimit=%llu dstLimit=%llu",
                           tag, myRank_, peerRank, srcOffset, dstOffset, byteSize, srcLimit, dstLimit),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVABRelayNoMemcpy::CalcRelaySlotOffset(u32 srcRank, u32 dstRank) const
{
    // Use intra-group relative srcRank and inter-group dstRank to keep scratch
    // address space O(rankSize) instead of O(rankSize^2). Only same-group src
    // ranks ever write/read relay slots, so (srcLocal * groupNum + dstGroup)
    // uniquely identifies each (src, dst) pair with max index rankSize-1.
    u32 srcLocal = srcRank % meshSize_;
    u32 dstGroup = meshSize_ == 0 ? 0 : dstRank / meshSize_;
    return (static_cast<u64>(srcLocal) * groupNum_ + dstGroup) * slotStride_;
}

void InsTempAlltoAllVABRelayNoMemcpy::CalcChannelSplit(
    u64 count, const std::vector<ChannelInfo> &channels, u32 channelIdx, u64 &splitCount, u64 &splitOffsetCount) const
{
    u64 totalWeight = 0;
    u64 prefixWeight = 0;
    for (u32 idx = 0; idx < channels.size(); ++idx) {
        u64 weight = std::max(1u, channels[idx].portGroupSize);
        if (idx < channelIdx) {
            prefixWeight += weight;
        }
        totalWeight += weight;
    }
    if (totalWeight == 0 || channelIdx >= channels.size()) {
        splitCount = 0;
        splitOffsetCount = 0;
        return;
    }
    u64 start = count * prefixWeight / totalWeight;
    u64 end = count * (prefixWeight + std::max(1u, channels[channelIdx].portGroupSize)) / totalWeight;
    splitOffsetCount = start;
    splitCount = end - start;
}

HcclResult InsTempAlltoAllVABRelayNoMemcpy::RunPeerSendRecv(
    const ChannelInfo &channel, const std::vector<DataSlice> &txSrcSlices,
    const std::vector<DataSlice> &txDstSlices, const ThreadHandle &thread) const
{
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    std::vector<DataSlice> safeTxSrcSlices = txSrcSlices;
    std::vector<DataSlice> safeTxDstSlices = txDstSlices;
    if (safeTxSrcSlices.empty()) {
        safeTxSrcSlices.emplace_back(nullptr, 0, 0, 0);
        safeTxDstSlices.emplace_back(nullptr, 0, 0, 0);
    }
    SendRecvInfo sendRecvInfo{{channel, channel}, {{safeTxSrcSlices, safeTxDstSlices}, {rxSrcSlices, rxDstSlices}},
                              dataType_};
    CHK_PRT_RET(SendRecvWrite(sendRecvInfo, thread),
                HCCL_ERROR("[A2AV_AB_RELAY][Template] SendRecvWrite failed. rank=%u peer=%u slices=%zu",
                           myRank_, channel.remoteRank, txSrcSlices.size()),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVABRelayNoMemcpy::BuildPrerouteSlices(
    u32 peerRank, const ChannelInfo &channel, const TemplateDataParams &params,
    std::vector<DataSlice> &txSrcSlices, std::vector<DataSlice> &txDstSlices) const
{
    txSrcSlices.clear();
    txDstSlices.clear();
    u32 myGroup = myRank_ / meshSize_;
    u32 myLocal = myRank_ % meshSize_;
    u32 peerLocal = peerRank % meshSize_;
    if (myGroup != peerRank / meshSize_ || myLocal == peerLocal) {
        return HCCL_SUCCESS;
    }
    for (u32 dstGroup = 0; dstGroup < groupNum_; ++dstGroup) {
        u32 dstRank = dstGroup * meshSize_ + peerLocal;
        if (!HasPeerData(params.sendCounts, dstRank)) {
            continue;
        }
        u64 count = params.sendCounts[dstRank];
        u64 byteSize = count * dataTypeSize_;
        u64 srcOffset = params.sdispls[dstRank] * dataTypeSize_;
        u64 dstOffset = CalcRelaySlotOffset(myRank_, dstRank);
        CHK_RET(CheckSliceRange("A2AV_AB_RELAY_PREROUTE", peerRank, srcOffset, dstOffset, byteSize,
                                params.buffInfo.inputSize, channel.remoteCclMem.size));
        txSrcSlices.emplace_back(params.buffInfo.inputPtr, srcOffset, byteSize, count);
        txDstSlices.emplace_back(channel.remoteCclMem.addr, dstOffset, byteSize, count);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVABRelayNoMemcpy::BuildRelaySlices(
    u32 peerRank, const std::vector<ChannelInfo> &channels, u32 channelIdx, const TemplateDataParams &params,
    std::vector<DataSlice> &txSrcSlices, std::vector<DataSlice> &txDstSlices) const
{
    txSrcSlices.clear();
    txDstSlices.clear();
    u32 myLocal = myRank_ % meshSize_;
    u32 peerLocal = peerRank % meshSize_;
    if (myLocal != peerLocal || myRank_ / meshSize_ == peerRank / meshSize_) {
        return HCCL_SUCCESS;
    }
    const ChannelInfo &channel = channels[channelIdx];
    for (u32 srcLocal = 0; srcLocal < meshSize_; ++srcLocal) {
        u32 srcRank = (myRank_ / meshSize_) * meshSize_ + srcLocal;
        u32 dstRank = peerRank;
        u64 count = 0;
        u64 srcOffset = 0;
        if (srcRank == myRank_) {
            count = params.sendCounts[dstRank];
            if (count == 0) {
                continue;
            }
            srcOffset = params.sdispls[dstRank] * dataTypeSize_;
        } else {
            u64 srcTailCount = 0;
            if (channel.remoteAlltoAllVRecvCounts.size() > srcRank) {
                srcTailCount = channel.remoteAlltoAllVRecvCounts[srcRank] > params.alltoAllVABThreshold ?
                    channel.remoteAlltoAllVRecvCounts[srcRank] - params.alltoAllVABThreshold : 0;
            }
            count = srcTailCount;
            if (count == 0) {
                continue;
            }
            srcOffset = CalcRelaySlotOffset(srcRank, dstRank);
        }
        u64 splitCount = 0;
        u64 splitOffsetCount = 0;
        CalcChannelSplit(count, channels, channelIdx, splitCount, splitOffsetCount);
        if (splitCount == 0) {
            continue;
        }
        u64 byteSize = splitCount * dataTypeSize_;
        u64 splitByteOffset = splitOffsetCount * dataTypeSize_;
        void *srcPtr = srcRank == myRank_ ? params.buffInfo.inputPtr : params.buffInfo.hcclBuff.addr;
        u64 srcLimit = srcRank == myRank_ ? params.buffInfo.inputSize : params.buffInfo.hcclBuff.size;
        CHK_PRT_RET(channel.remoteAlltoAllVRdispls.size() <= srcRank ||
                        channel.remoteAlltoAllVRecvCounts.size() <= srcRank,
                    HCCL_ERROR("[A2AV_AB_RELAY][Template] missing remote dst info. rank=%u peer=%u "
                               "srcRank=%u remoteRdispls=%zu remoteRecvCounts=%zu",
                               myRank_, peerRank, srcRank, channel.remoteAlltoAllVRdispls.size(),
                               channel.remoteAlltoAllVRecvCounts.size()),
                    HcclResult::HCCL_E_INTERNAL);
        u64 dstOffset = (channel.remoteAlltoAllVRdispls[srcRank] +
                         std::min(params.alltoAllVABThreshold, channel.remoteAlltoAllVRecvCounts[srcRank])) *
                        dataTypeSize_ + splitByteOffset;
        CHK_RET(CheckSliceRange("A2AV_AB_RELAY_OUTPUT", peerRank, srcOffset + splitByteOffset, dstOffset, byteSize,
                                srcLimit, channel.remoteOutputGraphMode.size));
        txSrcSlices.emplace_back(srcPtr, srcOffset + splitByteOffset, byteSize, splitCount);
        txDstSlices.emplace_back(channel.remoteOutputGraphMode.addr, dstOffset, byteSize, splitCount);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVABRelayNoMemcpy::RunPrerouteToRelay(
    const TemplateDataParams &params, const TemplateResource &resource)
{
    u32 threadIdx = 1;
    for (const auto &item : resource.channels) {
        u32 peerRank = item.first;
        if (item.second.empty() || peerRank == myRank_) {
            continue;
        }
        if (peerRank / meshSize_ != myRank_ / meshSize_) {
            continue;
        }
        const ChannelInfo &channel = item.second[0];
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        CHK_RET(BuildPrerouteSlices(peerRank, channel, params, txSrcSlices, txDstSlices));
        ThreadHandle thread = resource.threads[std::min(threadIdx, static_cast<u32>(resource.threads.size() - 1))];
        HCCL_WARNING("[A2AV_AB_RELAY][A_PREROUTE] rank=%u peer=%u slices=%zu threadIdx=%u",
                     myRank_, peerRank, txSrcSlices.size(), std::min(threadIdx, static_cast<u32>(resource.threads.size() - 1)));
        CHK_RET(RunPeerSendRecv(channel, txSrcSlices, txDstSlices, thread));
        ++threadIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVABRelayNoMemcpy::RunRelayToOutput(
    const TemplateDataParams &params, const TemplateResource &resource)
{
    u32 myGroup = myRank_ / meshSize_;
    for (u32 srcLocal = 0; srcLocal < meshSize_; ++srcLocal) {
        u32 srcRank = myGroup * meshSize_ + srcLocal;
        if (!HasPeerData(params.recvCounts, srcRank)) {
            continue;
        }
        u64 count = params.recvCounts[srcRank];
        u64 byteSize = count * dataTypeSize_;
        void *srcPtr = srcRank == myRank_ ? params.buffInfo.inputPtr : params.buffInfo.hcclBuff.addr;
        u64 srcOffset = srcRank == myRank_ ? params.sdispls[myRank_] * dataTypeSize_ :
            CalcRelaySlotOffset(srcRank, myRank_);
        u64 srcLimit = srcRank == myRank_ ? params.buffInfo.inputSize : params.buffInfo.hcclBuff.size;
        u64 dstOffset = params.rdispls[srcRank] * dataTypeSize_;
        CHK_RET(CheckSliceRange("A2AV_AB_RELAY_LOCAL_OUTPUT", myRank_, srcOffset, dstOffset, byteSize,
                                srcLimit, params.buffInfo.outputSize));
        DataSlice srcSlice(srcPtr, srcOffset, byteSize, count);
        DataSlice dstSlice(params.buffInfo.outputPtr, dstOffset, byteSize, count);
        CHK_RET(static_cast<HcclResult>(LocalCopy(resource.threads[0], srcSlice, dstSlice)));
        HCCL_WARNING("[A2AV_AB_RELAY][B_LOCAL] rank=%u srcRank=%u count=%llu srcOff=%llu dstOff=%llu",
                     myRank_, srcRank, count, srcOffset, dstOffset);
    }
    u32 threadIdx = 1;
    for (const auto &item : resource.channels) {
        u32 peerRank = item.first;
        if (item.second.empty() || peerRank == myRank_) {
            continue;
        }
        const std::vector<ChannelInfo> &channels = item.second;
        for (u32 channelIdx = 0; channelIdx < channels.size(); ++channelIdx) {
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            CHK_RET(BuildRelaySlices(peerRank, channels, channelIdx, params, txSrcSlices, txDstSlices));
            ThreadHandle thread = resource.threads[std::min(threadIdx, static_cast<u32>(resource.threads.size() - 1))];
            HCCL_WARNING("[A2AV_AB_RELAY][B_RELAY] rank=%u peer=%u channel=%u/%zu slices=%zu threadIdx=%u",
                         myRank_, peerRank, channelIdx, channels.size(), txSrcSlices.size(),
                         std::min(threadIdx, static_cast<u32>(resource.threads.size() - 1)));
            CHK_RET(RunPeerSendRecv(channels[channelIdx], txSrcSlices, txDstSlices, thread));
            ++threadIdx;
        }
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
