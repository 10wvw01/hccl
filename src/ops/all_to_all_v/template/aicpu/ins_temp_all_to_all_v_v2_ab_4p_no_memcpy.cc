/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#include "aicpu/ins_temp_all_to_all_v_v2_ab_4p_no_memcpy.h"
#include <algorithm>

namespace ops_hccl {
namespace {
bool HasCount(const std::vector<u64> &counts, u32 rank)
{
    return rank < counts.size() && counts[rank] > 0;
}
}

InsTempAlltoAllVV2AB4PNoMemcpy::InsTempAlltoAllVV2AB4PNoMemcpy(
    const OpParam &param, u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVV2AB4PNoMemcpy::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}

void InsTempAlltoAllVV2AB4PNoMemcpy::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    for (u32 i = 1; i < threadNum_; ++i) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAlltoAllVV2AB4PNoMemcpy::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    for (u32 i = 1; i < threadNum_; ++i) {
        notifyIdxSubToMain.push_back(i - 1);
    }
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::KernelRun(
    const OpParam &param, const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
{
    threadNum_ = templateResource.threads.size();
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataType_ >= HCCL_DATA_TYPE_RESERVED || dataTypeSize_ == 0,
                HCCL_ERROR("[A2AV_V2_AB_4P][Template] invalid datatype[%d].", static_cast<int>(dataType_)),
                HcclResult::HCCL_E_INTERNAL);
    CHK_RET(CheckParams(tempAlgParams, templateResource));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdx;
        GetNotifyIdxMainToSub(notifyIdx);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdx));
    }
    HcclResult ret = HCCL_SUCCESS;
    if (phase_ == A2AVV2AB4PPhase::STAGE0_TO_RELAY) {
        ret = RunStage0ToRelay(tempAlgParams, templateResource);
    } else if (phase_ == A2AVV2AB4PPhase::STAGE1_TO_OUTPUT) {
        ret = RunStage1ToOutput(tempAlgParams, templateResource);
    } else if (phase_ == A2AVV2AB4PPhase::B_SPRAY_TO_OUTPUT) {
        ret = RunBSprayToOutput(tempAlgParams, templateResource);
    } else {
        ret = RunBColorToOutput(tempAlgParams, templateResource);
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdx;
        GetNotifyIdxSubToMain(notifyIdx);
        HcclResult postRet = PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdx);
        if (ret == HCCL_SUCCESS) {
            ret = postRet;
        }
    }
    return ret;
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::CheckParams(
    const TemplateDataParams &params, const TemplateResource &resource) const
{
    CHK_PRT_RET(rankSize_ == 0 || meshSize_ == 0 || rankSize_ % meshSize_ != 0 || groupNum_ == 0,
                HCCL_ERROR("[A2AV_V2_AB_4P][Template] invalid dims rank=%u rankSize=%u mesh=%u group=%u",
                           myRank_, rankSize_, meshSize_, groupNum_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(params.sendCounts.size() < rankSize_ || params.recvCounts.size() < rankSize_ ||
                    params.sdispls.size() < rankSize_ || params.rdispls.size() < rankSize_,
                HCCL_ERROR("[A2AV_V2_AB_4P][Template] invalid alltoallv vectors rank=%u send=%zu recv=%zu "
                           "sdispl=%zu rdispl=%zu rankSize=%u", myRank_, params.sendCounts.size(),
                           params.recvCounts.size(), params.sdispls.size(), params.rdispls.size(), rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(resource.threads.empty(),
                HCCL_ERROR("[A2AV_V2_AB_4P][Template] empty thread resource rank=%u", myRank_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(slotStride_ == 0,
                HCCL_ERROR("[A2AV_V2_AB_4P][Template] zero slot stride rank=%u", myRank_),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::CheckSliceRange(const char *tag, u32 peerRank,
    u64 srcOffset, u64 dstOffset, u64 byteSize, u64 srcLimit, u64 dstLimit) const
{
    CHK_PRT_RET(srcOffset + byteSize > srcLimit || dstOffset + byteSize > dstLimit,
                HCCL_ERROR("[A2AV_V2_AB_4P][Range][%s] rank=%u peer=%u srcOff=%llu dstOff=%llu "
                           "size=%llu srcLimit=%llu dstLimit=%llu", tag, myRank_, peerRank, srcOffset,
                           dstOffset, byteSize, srcLimit, dstLimit),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::RunPeerSendRecv(const ChannelInfo &channel,
    const std::vector<DataSlice> &txSrcSlices, const std::vector<DataSlice> &txDstSlices,
    const ThreadHandle &thread) const
{
    std::vector<DataSlice> safeTxSrcSlices = txSrcSlices;
    std::vector<DataSlice> safeTxDstSlices = txDstSlices;
    if (safeTxSrcSlices.empty()) {
        safeTxSrcSlices.emplace_back(nullptr, 0, 0, 0);
        safeTxDstSlices.emplace_back(nullptr, 0, 0, 0);
    }
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    SendRecvInfo sendRecvInfo{{channel, channel}, {{safeTxSrcSlices, safeTxDstSlices}, {rxSrcSlices, rxDstSlices}},
                              dataType_};
    CHK_RET(SendRecvWrite(sendRecvInfo, thread));
    return HCCL_SUCCESS;
}

void InsTempAlltoAllVV2AB4PNoMemcpy::SplitPairCount(u64 count, u64 &part0, u64 &part1) const
{
    long double scaled = static_cast<long double>(count) * static_cast<long double>(pathSplitRatio_);
    part0 = static_cast<u64>(scaled);
    if (part0 > count) {
        part0 = count;
    }
    part1 = count - part0;
}

void InsTempAlltoAllVV2AB4PNoMemcpy::SplitABPairCount(u64 count, u64 &aCount, u64 &bCount) const
{
    long double scaled = static_cast<long double>(count) * static_cast<long double>(abRatio_);
    aCount = static_cast<u64>(scaled);
    if (aCount > count) {
        aCount = count;
    }
    bCount = count - aCount;
}

void InsTempAlltoAllVV2AB4PNoMemcpy::GetSplitParts(u32 srcRank, u32 dstRank, u64 count,
    std::vector<SplitPart> &parts) const
{
    parts.clear();
    if (count == 0 || srcRank == dstRank) {
        return;
    }
    u64 part0 = 0;
    u64 part1 = 0;
    u64 aCount = 0;
    u64 bCount = 0;
    SplitABPairCount(count, aCount, bCount);
    (void)bCount;
    SplitPairCount(aCount, part0, part1);
    const u32 srcGroup = srcRank / meshSize_;
    const u32 dstGroup = dstRank / meshSize_;
    const u32 srcLocal = srcRank % meshSize_;
    const u32 dstLocal = dstRank % meshSize_;
    u32 relay0 = dstGroup == srcGroup ? dstRank : static_cast<u32>(dstGroup * meshSize_ + srcLocal);
    u32 relay1 = dstGroup == srcGroup ? srcRank : static_cast<u32>(srcGroup * meshSize_ + dstLocal);
    parts.push_back({relay0, part0, 0});
    parts.push_back({relay1, part1, part0});
}

u64 InsTempAlltoAllVV2AB4PNoMemcpy::CalcRelaySlotOffset(u32 srcRank, u32 dstRank, u32 partIdx) const
{
    u64 slotIdx = (static_cast<u64>(srcRank) * rankSize_ + dstRank) * 2 + partIdx;
    return slotIdx * slotStride_;
}

u64 GetExactRelaySlotOffset(const TemplateDataParams &params, u32 rankSize, u32 srcRank, u32 dstRank, u32 partIdx,
    u64 fallbackOffset)
{
    u64 slotIdx = (static_cast<u64>(srcRank) * rankSize + dstRank) * 2 + partIdx;
    if (slotIdx < params.alltoAllVV2SlotOffsets.size()) {
        return params.alltoAllVV2SlotOffsets[slotIdx];
    }
    return fallbackOffset;
}

bool InsTempAlltoAllVV2AB4PNoMemcpy::IsV2Peer(u32 peerRank) const
{
    return meshSize_ != 0 && peerRank < rankSize_ && peerRank / meshSize_ != myRank_ / meshSize_;
}

u32 InsTempAlltoAllVV2AB4PNoMemcpy::SelectChannelIdx(
    u32 peerRank, const std::vector<ChannelInfo> &channels) const
{
    if (channels.empty() || !IsV2Peer(peerRank) || channels.size() == 1) {
        return 0;
    }

    const u32 myGroup = myRank_ / meshSize_;
    const u32 peerGroup = peerRank / meshSize_;
    if (groupNum_ == 4 && channels.size() >= 3) {
        static constexpr u32 K4_EDGE_COLOR[4][4] = {
            {0, 0, 1, 2},
            {0, 0, 2, 1},
            {1, 2, 0, 0},
            {2, 1, 0, 0},
        };
        return K4_EDGE_COLOR[myGroup][peerGroup];
    }

    const u32 myLocal = myRank_ % meshSize_;
    const u32 peerLocal = peerRank % meshSize_;
    const u32 lowGroup = std::min(myGroup, peerGroup);
    const u32 highGroup = std::max(myGroup, peerGroup);
    const u32 lowLocal = std::min(myLocal, peerLocal);
    const u32 highLocal = std::max(myLocal, peerLocal);
    const u32 hash = lowGroup * groupNum_ + highGroup + lowLocal * meshSize_ + highLocal;
    return hash % static_cast<u32>(channels.size());
}

u32 InsTempAlltoAllVV2AB4PNoMemcpy::SelectBChannelIdx(const std::vector<ChannelInfo> &channels) const
{
    if (channels.size() >= 4) {
        return 3;
    }
    return channels.empty() ? 0 : static_cast<u32>(channels.size() - 1);
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::BuildBSlices(u32 dstRank, const ChannelInfo &channel,
    const TemplateDataParams &params, std::vector<DataSlice> &txSrcSlices,
    std::vector<DataSlice> &txDstSlices) const
{
    txSrcSlices.clear();
    txDstSlices.clear();
    if (dstRank == myRank_ || !HasCount(params.sendCounts, dstRank)) {
        return HCCL_SUCCESS;
    }
    u64 aCount = 0;
    u64 bCount = 0;
    SplitABPairCount(params.sendCounts[dstRank], aCount, bCount);
    if (bCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(channel.remoteAlltoAllVRdispls.size() <= myRank_,
                HCCL_ERROR("[A2AV_V2_AB_4P][BuildBSlices] missing remote rdispls. rank=%u dst=%u size=%zu",
                           myRank_, dstRank, channel.remoteAlltoAllVRdispls.size()),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(channel.remoteOutputGraphMode.addr == nullptr,
                HCCL_ERROR("[A2AV_V2_AB_4P][BuildBSlices] null remote output. rank=%u dst=%u",
                           myRank_, dstRank),
                HcclResult::HCCL_E_INTERNAL);
    u64 byteSize = bCount * dataTypeSize_;
    u64 srcOffset = (params.sdispls[dstRank] + aCount) * dataTypeSize_;
    u64 dstOffset = (channel.remoteAlltoAllVRdispls[myRank_] + aCount) * dataTypeSize_;
    CHK_RET(CheckSliceRange("B_TO_OUTPUT", dstRank, srcOffset, dstOffset, byteSize,
                            params.buffInfo.inputSize, channel.remoteOutputGraphMode.size));
    txSrcSlices.emplace_back(params.buffInfo.inputPtr, srcOffset, byteSize, bCount);
    txDstSlices.emplace_back(channel.remoteOutputGraphMode.addr, dstOffset, byteSize, bCount);
    return HCCL_SUCCESS;
}

u32 InsTempAlltoAllVV2AB4PNoMemcpy::GetCompleteGraphEdgeColorNum(u32 vertexNum) const
{
    return vertexNum % 2 == 0 ? vertexNum - 1 : vertexNum;
}

u32 InsTempAlltoAllVV2AB4PNoMemcpy::GetCompleteGraphEdgeColor(u32 rankA, u32 rankB, u32 vertexNum) const
{
    if (rankA == rankB || vertexNum <= 1) {
        return 0;
    }
    if (vertexNum % 2 == 1) {
        u32 colorNum = GetCompleteGraphEdgeColorNum(vertexNum);
        for (u32 color = 0; color < colorNum; ++color) {
            for (u32 v = 0; v < vertexNum; ++v) {
                u32 peer = (color + vertexNum - v) % vertexNum;
                if (v < peer && ((v == rankA && peer == rankB) || (v == rankB && peer == rankA))) {
                    return color;
                }
            }
        }
        return 0;
    }
    std::vector<u32> vertices;
    vertices.reserve(vertexNum - 1);
    for (u32 rank = 0; rank + 1 < vertexNum; ++rank) {
        vertices.push_back(rank);
    }
    const u32 fixedRank = vertexNum - 1;
    for (u32 color = 0; color + 1 < vertexNum; ++color) {
        if ((rankA == fixedRank && rankB == vertices[0]) ||
            (rankB == fixedRank && rankA == vertices[0])) {
            return color;
        }
        for (u32 i = 1; i < vertexNum / 2; ++i) {
            u32 a = vertices[i];
            u32 b = vertices[vertexNum - 1 - i];
            if ((rankA == a && rankB == b) || (rankA == b && rankB == a)) {
                return color;
            }
        }
        u32 last = vertices.back();
        for (u32 i = static_cast<u32>(vertices.size() - 1); i > 0; --i) {
            vertices[i] = vertices[i - 1];
        }
        vertices[0] = last;
    }
    return 0;
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::RunBSprayToOutput(
    const TemplateDataParams &params, const TemplateResource &resource) const
{
    u32 threadIdx = 1;
    for (const auto &item : resource.channels) {
        u32 dstRank = item.first;
        if (dstRank == myRank_ || item.second.empty()) {
            continue;
        }
        u32 channelIdx = SelectBChannelIdx(item.second);
        const ChannelInfo &channel = item.second[channelIdx];
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        CHK_RET(BuildBSlices(dstRank, channel, params, txSrcSlices, txDstSlices));
        u32 runThreadIdx = std::min(threadIdx, static_cast<u32>(resource.threads.size() - 1));
        u64 totalBytes = 0;
        for (const auto &slice : txSrcSlices) {
            totalBytes += slice.size_;
        }
        HCCL_WARNING("[A2AV_V2_AB_4P][B_SPRAY] rank=%u peer=%u links=%zu selected=%u slices=%zu bytes=%llu "
                     "threadIdx=%u",
                     myRank_, dstRank, item.second.size(), channelIdx, txSrcSlices.size(), totalBytes,
                     runThreadIdx);
        CHK_RET(RunPeerSendRecv(channel, txSrcSlices, txDstSlices, resource.threads[runThreadIdx]));
        ++threadIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::RunBColorToOutput(
    const TemplateDataParams &params, const TemplateResource &resource) const
{
    u32 colorNum = GetCompleteGraphEdgeColorNum(rankSize_);
    for (u32 color = 0; color < colorNum; ++color) {
        for (const auto &item : resource.channels) {
            u32 dstRank = item.first;
            if (dstRank == myRank_ || item.second.empty() ||
                GetCompleteGraphEdgeColor(myRank_, dstRank, rankSize_) != color) {
                continue;
            }
            u32 channelIdx = SelectBChannelIdx(item.second);
            const ChannelInfo &channel = item.second[channelIdx];
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            CHK_RET(BuildBSlices(dstRank, channel, params, txSrcSlices, txDstSlices));
            u64 totalBytes = 0;
            for (const auto &slice : txSrcSlices) {
                totalBytes += slice.size_;
            }
            HCCL_WARNING("[A2AV_V2_AB_4P][B_COLOR] rank=%u color=%u peer=%u links=%zu selected=%u "
                         "slices=%zu bytes=%llu",
                         myRank_, color, dstRank, item.second.size(), channelIdx, txSrcSlices.size(), totalBytes);
            CHK_RET(RunPeerSendRecv(channel, txSrcSlices, txDstSlices, resource.threads[1]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::BuildStage0Slices(u32 relayRank, const ChannelInfo &channel,
    const TemplateDataParams &params, std::vector<DataSlice> &txSrcSlices,
    std::vector<DataSlice> &txDstSlices) const
{
    txSrcSlices.clear();
    txDstSlices.clear();
    for (u32 dstRank = 0; dstRank < rankSize_; ++dstRank) {
        if (dstRank == myRank_ || !HasCount(params.sendCounts, dstRank)) {
            continue;
        }
        std::vector<SplitPart> parts;
        GetSplitParts(myRank_, dstRank, params.sendCounts[dstRank], parts);
        for (u32 partIdx = 0; partIdx < parts.size(); ++partIdx) {
            const SplitPart &part = parts[partIdx];
            if (part.relayRank != relayRank || part.count == 0) {
                continue;
            }
            if (part.relayRank == myRank_) {
                continue;
            }
            u64 byteSize = part.count * dataTypeSize_;
            u64 srcOffset = (params.sdispls[dstRank] + part.offsetCount) * dataTypeSize_;
            void *dstPtr = channel.remoteCclMem.addr;
            u64 dstLimit = channel.remoteCclMem.size;
            u64 dstOffset = GetExactRelaySlotOffset(params, rankSize_, myRank_, dstRank, partIdx,
                CalcRelaySlotOffset(myRank_, dstRank, partIdx));
            const char *rangeTag = "STAGE0_TO_RELAY";
            if (part.relayRank == dstRank) {
                CHK_PRT_RET(channel.remoteAlltoAllVRdispls.size() <= myRank_,
                            HCCL_ERROR("[A2AV_V2_AB_4P][Stage0DirectOutput] missing remote rdispls. "
                                       "rank=%u dst=%u rdispls=%zu need=%u",
                                       myRank_, dstRank, channel.remoteAlltoAllVRdispls.size(), myRank_),
                            HcclResult::HCCL_E_INTERNAL);
                CHK_PRT_RET(channel.remoteOutputGraphMode.addr == nullptr,
                            HCCL_ERROR("[A2AV_V2_AB_4P][Stage0DirectOutput] null remote output. "
                                       "rank=%u dst=%u", myRank_, dstRank),
                            HcclResult::HCCL_E_INTERNAL);
                dstPtr = channel.remoteOutputGraphMode.addr;
                dstLimit = channel.remoteOutputGraphMode.size;
                dstOffset = (channel.remoteAlltoAllVRdispls[myRank_] + part.offsetCount) * dataTypeSize_;
                rangeTag = "STAGE0_TO_OUTPUT";
            } else {
                CHK_PRT_RET(channel.remoteCclMem.addr == nullptr,
                            HCCL_ERROR("[A2AV_V2_AB_4P][Stage0Relay] null remote ccl. rank=%u relay=%u "
                                       "dst=%u part=%u", myRank_, relayRank, dstRank, partIdx),
                            HcclResult::HCCL_E_INTERNAL);
            }
            CHK_RET(CheckSliceRange(rangeTag, relayRank, srcOffset, dstOffset, byteSize,
                                    params.buffInfo.inputSize, dstLimit));
            txSrcSlices.emplace_back(params.buffInfo.inputPtr, srcOffset, byteSize, part.count);
            txDstSlices.emplace_back(dstPtr, dstOffset, byteSize, part.count);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::BuildStage1Slices(u32 finalDst, const ChannelInfo &channel,
    const TemplateDataParams &params, std::vector<DataSlice> &txSrcSlices,
    std::vector<DataSlice> &txDstSlices) const
{
    txSrcSlices.clear();
    txDstSlices.clear();
    for (u32 srcRank = 0; srcRank < rankSize_; ++srcRank) {
        if (srcRank == finalDst || channel.remoteAlltoAllVRecvCounts.size() <= srcRank ||
            channel.remoteAlltoAllVRdispls.size() <= srcRank) {
            continue;
        }
        u64 count = channel.remoteAlltoAllVRecvCounts[srcRank];
        if (count == 0) {
            continue;
        }
        CHK_PRT_RET(channel.remoteOutputGraphMode.addr == nullptr,
                    HCCL_ERROR("[A2AV_V2_AB_4P][Stage1Output] null remote output. rank=%u dst=%u",
                               myRank_, finalDst),
                    HcclResult::HCCL_E_INTERNAL);
        std::vector<SplitPart> parts;
        GetSplitParts(srcRank, finalDst, count, parts);
        for (u32 partIdx = 0; partIdx < parts.size(); ++partIdx) {
            const SplitPart &part = parts[partIdx];
            if (part.relayRank != myRank_ || part.count == 0) {
                continue;
            }
            u64 byteSize = part.count * dataTypeSize_;
            if (part.relayRank == finalDst) {
                continue;
            }
            void *srcPtr = params.buffInfo.hcclBuff.addr;
            u64 srcLimit = params.buffInfo.hcclBuff.size;
            u64 srcOffset = GetExactRelaySlotOffset(params, rankSize_, srcRank, finalDst, partIdx,
                CalcRelaySlotOffset(srcRank, finalDst, partIdx));
            const char *rangeTag = "STAGE1_TO_OUTPUT";
            if (srcRank == myRank_) {
                srcPtr = params.buffInfo.inputPtr;
                srcLimit = params.buffInfo.inputSize;
                srcOffset = (params.sdispls[finalDst] + part.offsetCount) * dataTypeSize_;
                rangeTag = "STAGE1_INPUT_TO_OUTPUT";
            } else {
                CHK_PRT_RET(params.buffInfo.hcclBuff.addr == nullptr,
                            HCCL_ERROR("[A2AV_V2_AB_4P][Stage1Relay] null local ccl. rank=%u src=%u "
                                       "dst=%u part=%u", myRank_, srcRank, finalDst, partIdx),
                            HcclResult::HCCL_E_INTERNAL);
            }
            u64 dstOffset = (channel.remoteAlltoAllVRdispls[srcRank] + part.offsetCount) * dataTypeSize_;
            CHK_RET(CheckSliceRange(rangeTag, finalDst, srcOffset, dstOffset, byteSize,
                                    srcLimit, channel.remoteOutputGraphMode.size));
            txSrcSlices.emplace_back(srcPtr, srcOffset, byteSize, part.count);
            txDstSlices.emplace_back(channel.remoteOutputGraphMode.addr, dstOffset, byteSize, part.count);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::RunStage0ToRelay(
    const TemplateDataParams &params, const TemplateResource &resource) const
{
    u32 threadIdx = 1;
    for (const auto &item : resource.channels) {
        u32 relayRank = item.first;
        if (relayRank == myRank_ || item.second.empty()) {
            continue;
        }
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        u32 channelIdx = SelectChannelIdx(relayRank, item.second);
        const ChannelInfo &channel = item.second[channelIdx];
        CHK_RET(BuildStage0Slices(relayRank, channel, params, txSrcSlices, txDstSlices));
        u32 runThreadIdx = std::min(threadIdx, static_cast<u32>(resource.threads.size() - 1));
        ThreadHandle thread = resource.threads[runThreadIdx];
        u64 totalBytes = 0;
        for (const auto &slice : txSrcSlices) {
            totalBytes += slice.size_;
        }
        HCCL_WARNING("[A2AV_V2_AB_4P][SelectChannel] phase=stage0 rank=%u peer=%u links=%zu "
                     "selected=%u portGroupSize=%u slices=%zu bytes=%llu threadIdx=%u",
                     myRank_, relayRank, item.second.size(), channelIdx, channel.portGroupSize,
                     txSrcSlices.size(), totalBytes, runThreadIdx);
        CHK_RET(RunPeerSendRecv(channel, txSrcSlices, txDstSlices, thread));
        ++threadIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2AB4PNoMemcpy::RunStage1ToOutput(
    const TemplateDataParams &params, const TemplateResource &resource) const
{
    u32 threadIdx = 1;
    for (const auto &item : resource.channels) {
        u32 finalDst = item.first;
        if (finalDst == myRank_ || item.second.empty()) {
            continue;
        }
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        u32 channelIdx = SelectChannelIdx(finalDst, item.second);
        const ChannelInfo &channel = item.second[channelIdx];
        CHK_RET(BuildStage1Slices(finalDst, channel, params, txSrcSlices, txDstSlices));
        u32 runThreadIdx = std::min(threadIdx, static_cast<u32>(resource.threads.size() - 1));
        ThreadHandle thread = resource.threads[runThreadIdx];
        u64 totalBytes = 0;
        for (const auto &slice : txSrcSlices) {
            totalBytes += slice.size_;
        }
        HCCL_WARNING("[A2AV_V2_AB_4P][SelectChannel] phase=stage1 rank=%u peer=%u links=%zu "
                     "selected=%u portGroupSize=%u slices=%zu bytes=%llu threadIdx=%u",
                     myRank_, finalDst, item.second.size(), channelIdx, channel.portGroupSize,
                     txSrcSlices.size(), totalBytes, runThreadIdx);
        CHK_RET(RunPeerSendRecv(channel, txSrcSlices, txDstSlices, thread));
        ++threadIdx;
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
