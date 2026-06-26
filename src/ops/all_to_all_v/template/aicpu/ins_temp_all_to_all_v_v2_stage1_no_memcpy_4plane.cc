/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#include "aicpu/ins_temp_all_to_all_v_v2_stage1_no_memcpy_4plane.h"
#include <algorithm>
#include <map>

namespace ops_hccl {
namespace {
bool HasCount(const std::vector<u64> &counts, u32 rank)
{
    return rank < counts.size() && counts[rank] > 0;
}

u64 SumSliceBytes(const std::vector<DataSlice> &slices)
{
    u64 bytes = 0;
    for (const auto &slice : slices) {
        bytes += slice.size_;
    }
    return bytes;
}

struct PeerSendPlan4Plane {
    u32 peerRank{0};
    u32 mainChannelIdx{0};
    u32 extraChannelIdx{0};
    const ChannelInfo *mainChannel{nullptr};
    const ChannelInfo *extraChannel{nullptr};
    std::vector<DataSlice> mainSrcSlices;
    std::vector<DataSlice> mainDstSlices;
    std::vector<DataSlice> extraSrcSlices;
    std::vector<DataSlice> extraDstSlices;
};

u32 GetK4EdgeColor(u32 a, u32 b)
{
    static constexpr u32 K4_EDGE_COLOR[4][4] = {
        {0, 0, 1, 2},
        {0, 0, 2, 1},
        {1, 2, 0, 0},
        {2, 1, 0, 0},
    };
    return K4_EDGE_COLOR[a][b];
}

std::vector<std::vector<u32>> BuildExtraPeerRounds(u32 myRank, u32 rankSize, u32 meshSize,
    const std::map<u32, PeerSendPlan4Plane> &plans)
{
    constexpr u32 UBX_4X4_RANK_SIZE = 16;
    constexpr u32 UBX_4X4_MESH_SIZE = 4;
    constexpr u32 UBX_4X4_ROUND_NUM = 6;

    if (rankSize == UBX_4X4_RANK_SIZE && meshSize == UBX_4X4_MESH_SIZE) {
        std::vector<std::vector<u32>> rounds(UBX_4X4_ROUND_NUM);
        u32 myGroup = myRank / meshSize;
        u32 myLocal = myRank % meshSize;
        for (const auto &item : plans) {
            u32 peer = item.first;
            u32 peerGroup = peer / meshSize;
            u32 peerLocal = peer % meshSize;
            if (peerGroup == myGroup && peerLocal != myLocal) {
                rounds[GetK4EdgeColor(myLocal, peerLocal)].push_back(peer);
            } else if (peerLocal == myLocal && peerGroup != myGroup) {
                rounds[3 + GetK4EdgeColor(myGroup, peerGroup)].push_back(peer);
            } else {
                HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][ExtraRounds] rank=%u peer=%u not rook-edge, skip extra.",
                             myRank, peer);
            }
        }
        return rounds;
    }

    std::vector<std::vector<u32>> rounds;
    for (u32 left = 0; left < rankSize; ++left) {
        for (u32 right = left + 1; right < rankSize; ++right) {
            if (left != myRank && right != myRank) {
                continue;
            }
            u32 peer = left == myRank ? right : left;
            if (plans.find(peer) != plans.end()) {
                rounds.push_back({peer});
            }
        }
    }
    return rounds;
}
}

InsTempAlltoAllVV2Stage1NoMemcpy4Plane::InsTempAlltoAllVV2Stage1NoMemcpy4Plane(
    const OpParam &param, u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVV2Stage1NoMemcpy4Plane::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}

void InsTempAlltoAllVV2Stage1NoMemcpy4Plane::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    for (u32 i = 1; i < threadNum_; ++i) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAlltoAllVV2Stage1NoMemcpy4Plane::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    for (u32 i = 1; i < threadNum_; ++i) {
        notifyIdxSubToMain.push_back(i - 1);
    }
}

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::KernelRun(
    const OpParam &param, const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
{
    threadNum_ = templateResource.threads.size();
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataType_ >= HCCL_DATA_TYPE_RESERVED || dataTypeSize_ == 0,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Template] invalid datatype[%d].", static_cast<int>(dataType_)),
                HcclResult::HCCL_E_INTERNAL);
    CHK_RET(CheckParams(tempAlgParams, templateResource));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdx;
        GetNotifyIdxMainToSub(notifyIdx);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdx));
    }
    HcclResult ret = phase_ == A2AVV2Stage1NoMemcpy4PlanePhase::STAGE0_TO_RELAY ?
        RunStage0ToRelay(tempAlgParams, templateResource) : RunStage1ToOutput(tempAlgParams, templateResource);
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

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::CheckParams(
    const TemplateDataParams &params, const TemplateResource &resource) const
{
    CHK_PRT_RET(rankSize_ == 0 || meshSize_ == 0 || rankSize_ % meshSize_ != 0 || groupNum_ == 0,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Template] invalid dims rank=%u rankSize=%u mesh=%u group=%u",
                           myRank_, rankSize_, meshSize_, groupNum_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(params.sendCounts.size() < rankSize_ || params.recvCounts.size() < rankSize_ ||
                    params.sdispls.size() < rankSize_ || params.rdispls.size() < rankSize_,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Template] invalid alltoallv vectors rank=%u send=%zu recv=%zu "
                           "sdispl=%zu rdispl=%zu rankSize=%u", myRank_, params.sendCounts.size(),
                           params.recvCounts.size(), params.sdispls.size(), params.rdispls.size(), rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(resource.threads.empty(),
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Template] empty thread resource rank=%u", myRank_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(slotStride_ == 0,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Template] zero slot stride rank=%u", myRank_),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::CheckSliceRange(const char *tag, u32 peerRank,
    u64 srcOffset, u64 dstOffset, u64 byteSize, u64 srcLimit, u64 dstLimit) const
{
    CHK_PRT_RET(srcOffset + byteSize > srcLimit || dstOffset + byteSize > dstLimit,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Range][%s] rank=%u peer=%u srcOff=%llu dstOff=%llu "
                           "size=%llu srcLimit=%llu dstLimit=%llu", tag, myRank_, peerRank, srcOffset,
                           dstOffset, byteSize, srcLimit, dstLimit),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::RunPeerSendRecv(const ChannelInfo &channel,
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

void InsTempAlltoAllVV2Stage1NoMemcpy4Plane::SplitPairCount(u64 count, u64 &part0, u64 &part1) const
{
    long double scaled = static_cast<long double>(count) * static_cast<long double>(splitRatio_);
    part0 = static_cast<u64>(scaled);
    if (part0 > count) {
        part0 = count;
    }
    part1 = count - part0;
}

void InsTempAlltoAllVV2Stage1NoMemcpy4Plane::GetSplitParts(u32 srcRank, u32 dstRank, u64 count,
    std::vector<SplitPart> &parts) const
{
    parts.clear();
    if (count == 0 || srcRank == dstRank) {
        return;
    }
    u64 part0 = 0;
    u64 part1 = 0;
    SplitPairCount(count, part0, part1);
    const u32 srcGroup = srcRank / meshSize_;
    const u32 dstGroup = dstRank / meshSize_;
    const u32 srcLocal = srcRank % meshSize_;
    const u32 dstLocal = dstRank % meshSize_;
    u32 relay0 = dstGroup == srcGroup ? dstRank : static_cast<u32>(dstGroup * meshSize_ + srcLocal);
    u32 relay1 = dstGroup == srcGroup ? srcRank : static_cast<u32>(srcGroup * meshSize_ + dstLocal);
    parts.push_back({relay0, part0, 0});
    parts.push_back({relay1, part1, part0});
}

u64 InsTempAlltoAllVV2Stage1NoMemcpy4Plane::CalcRelaySlotOffset(u32 srcRank, u32 dstRank, u32 partIdx) const
{
    u64 slotIdx = (static_cast<u64>(srcRank) * rankSize_ + dstRank) * 2 + partIdx;
    return slotIdx * slotStride_;
}

static u64 GetExactRelaySlotOffset(const TemplateDataParams &params, u32 rankSize, u32 srcRank, u32 dstRank,
    u32 partIdx, u64 fallbackOffset)
{
    u64 slotIdx = (static_cast<u64>(srcRank) * rankSize + dstRank) * 2 + partIdx;
    if (slotIdx < params.alltoAllVV2SlotOffsets.size()) {
        return params.alltoAllVV2SlotOffsets[slotIdx];
    }
    return fallbackOffset;
}

bool InsTempAlltoAllVV2Stage1NoMemcpy4Plane::IsV2Peer(u32 peerRank) const
{
    return meshSize_ != 0 && peerRank < rankSize_ && peerRank / meshSize_ != myRank_ / meshSize_;
}

u32 InsTempAlltoAllVV2Stage1NoMemcpy4Plane::SelectChannelIdx(
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

u32 InsTempAlltoAllVV2Stage1NoMemcpy4Plane::SelectExtraChannelIdx(const std::vector<ChannelInfo> &channels) const
{
    if (channels.size() >= 5) {
        return 4;
    }
    if (channels.size() >= 4) {
        return 3;
    }
    return 0;
}

u64 InsTempAlltoAllVV2Stage1NoMemcpy4Plane::CalcExtraQuotaBytes(u64 totalStageBytes, u64 activePeerNum) const
{
    constexpr u64 TOTAL_LANE_NUM = 7;
    if (activePeerNum == 0) {
        return 0;
    }
    return totalStageBytes / TOTAL_LANE_NUM / activePeerNum;
}

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::SplitSlicesForExtraLane(
    const std::vector<DataSlice> &inSrcSlices, const std::vector<DataSlice> &inDstSlices, u64 extraQuotaBytes,
    std::vector<DataSlice> &mainSrcSlices, std::vector<DataSlice> &mainDstSlices,
    std::vector<DataSlice> &extraSrcSlices, std::vector<DataSlice> &extraDstSlices) const
{
    mainSrcSlices.clear();
    mainDstSlices.clear();
    extraSrcSlices.clear();
    extraDstSlices.clear();
    CHK_PRT_RET(inSrcSlices.size() != inDstSlices.size(),
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][SplitExtra] slice size mismatch rank=%u src=%zu dst=%zu",
                           myRank_, inSrcSlices.size(), inDstSlices.size()),
                HcclResult::HCCL_E_INTERNAL);
    u64 remainExtraBytes = extraQuotaBytes;
    for (size_t i = 0; i < inSrcSlices.size(); ++i) {
        const DataSlice &src = inSrcSlices[i];
        const DataSlice &dst = inDstSlices[i];
        u64 extraBytes = std::min(remainExtraBytes, src.size_);
        extraBytes = extraBytes / dataTypeSize_ * dataTypeSize_;
        if (extraBytes > 0) {
            u64 extraCount = extraBytes / dataTypeSize_;
            extraSrcSlices.emplace_back(src.addr_, src.offset_, extraBytes, extraCount);
            extraDstSlices.emplace_back(dst.addr_, dst.offset_, extraBytes, extraCount);
            remainExtraBytes -= extraBytes;
        }
        u64 mainBytes = src.size_ - extraBytes;
        if (mainBytes > 0) {
            u64 mainCount = mainBytes / dataTypeSize_;
            mainSrcSlices.emplace_back(src.addr_, src.offset_ + extraBytes, mainBytes, mainCount);
            mainDstSlices.emplace_back(dst.addr_, dst.offset_ + extraBytes, mainBytes, mainCount);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::RebindExtraDstSlices(
    const ChannelInfo &mainChannel, const ChannelInfo &extraChannel, std::vector<DataSlice> &extraDstSlices) const
{
    for (auto &slice : extraDstSlices) {
        if (slice.addr_ == mainChannel.remoteOutputGraphMode.addr) {
            CHK_PRT_RET(extraChannel.remoteOutputGraphMode.addr == nullptr,
                        HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][RebindExtra] null extra output rank=%u",
                                   myRank_),
                        HcclResult::HCCL_E_INTERNAL);
            slice.addr_ = extraChannel.remoteOutputGraphMode.addr;
        } else if (slice.addr_ == mainChannel.remoteCclMem.addr) {
            CHK_PRT_RET(extraChannel.remoteCclMem.addr == nullptr,
                        HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][RebindExtra] null extra ccl rank=%u",
                                   myRank_),
                        HcclResult::HCCL_E_INTERNAL);
            slice.addr_ = extraChannel.remoteCclMem.addr;
        } else {
            HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][RebindExtra] unknown dst addr rank=%u addr=%p",
                       myRank_, slice.addr_);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::CopySelfToOutput(
    const TemplateDataParams &params, const ThreadHandle &thread) const
{
    if (!HasCount(params.sendCounts, myRank_)) {
        return HCCL_SUCCESS;
    }
    u64 count = params.sendCounts[myRank_];
    u64 byteSize = count * dataTypeSize_;
    u64 srcOffset = params.sdispls[myRank_] * dataTypeSize_;
    u64 dstOffset = params.rdispls[myRank_] * dataTypeSize_;
    CHK_RET(CheckSliceRange("SELF_COPY", myRank_, srcOffset, dstOffset, byteSize,
                            params.buffInfo.inputSize, params.buffInfo.outputSize));
    DataSlice srcSlice(params.buffInfo.inputPtr, srcOffset, byteSize, count);
    DataSlice dstSlice(params.buffInfo.outputPtr, dstOffset, byteSize, count);
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][SelfCopy] rank=%u count=%llu srcOff=%llu dstOff=%llu",
                 myRank_, count, srcOffset, dstOffset);
    CHK_RET(static_cast<HcclResult>(LocalCopy(thread, srcSlice, dstSlice)));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::BuildStage0Slices(u32 relayRank, const ChannelInfo &channel,
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
                            HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Stage0DirectOutput] missing remote rdispls. "
                                       "rank=%u dst=%u rdispls=%zu need=%u",
                                       myRank_, dstRank, channel.remoteAlltoAllVRdispls.size(), myRank_),
                            HcclResult::HCCL_E_INTERNAL);
                CHK_PRT_RET(channel.remoteOutputGraphMode.addr == nullptr,
                            HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Stage0DirectOutput] null remote output. "
                                       "rank=%u dst=%u", myRank_, dstRank),
                            HcclResult::HCCL_E_INTERNAL);
                dstPtr = channel.remoteOutputGraphMode.addr;
                dstLimit = channel.remoteOutputGraphMode.size;
                dstOffset = (channel.remoteAlltoAllVRdispls[myRank_] + part.offsetCount) * dataTypeSize_;
                rangeTag = "STAGE0_TO_OUTPUT";
            } else {
                CHK_PRT_RET(channel.remoteCclMem.addr == nullptr,
                            HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Stage0Relay] null remote ccl. rank=%u relay=%u "
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

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::BuildStage1Slices(u32 finalDst, const ChannelInfo &channel,
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
                    HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Stage1Output] null remote output. rank=%u dst=%u",
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
                            HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Stage1Relay] null local ccl. rank=%u src=%u "
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

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::RunStage0ToRelay(
    const TemplateDataParams &params, const TemplateResource &resource) const
{
    CHK_RET(CopySelfToOutput(params, resource.threads[0]));
    std::map<u32, PeerSendPlan4Plane> plans;
    u64 totalStageBytes = 0;
    u64 activePeerNum = 0;
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
        ++activePeerNum;
        totalStageBytes += SumSliceBytes(txSrcSlices);
        PeerSendPlan4Plane plan;
        plan.peerRank = relayRank;
        plan.mainChannelIdx = channelIdx;
        plan.extraChannelIdx = SelectExtraChannelIdx(item.second);
        plan.mainChannel = &channel;
        if (item.second.size() >= 4) {
            plan.extraChannel = &item.second[plan.extraChannelIdx];
        }
        plan.mainSrcSlices = std::move(txSrcSlices);
        plan.mainDstSlices = std::move(txDstSlices);
        plans.emplace(relayRank, std::move(plan));
    }

    u64 extraQuotaBytes = CalcExtraQuotaBytes(totalStageBytes, activePeerNum);
    for (auto &item : plans) {
        PeerSendPlan4Plane &plan = item.second;
        if (plan.extraChannel == nullptr || extraQuotaBytes == 0) {
            continue;
        }
        std::vector<DataSlice> mainSrcSlices;
        std::vector<DataSlice> mainDstSlices;
        std::vector<DataSlice> extraSrcSlices;
        std::vector<DataSlice> extraDstSlices;
        CHK_RET(SplitSlicesForExtraLane(plan.mainSrcSlices, plan.mainDstSlices, extraQuotaBytes, mainSrcSlices,
                                        mainDstSlices, extraSrcSlices, extraDstSlices));
        CHK_RET(RebindExtraDstSlices(*plan.mainChannel, *plan.extraChannel, extraDstSlices));
        plan.mainSrcSlices = std::move(mainSrcSlices);
        plan.mainDstSlices = std::move(mainDstSlices);
        plan.extraSrcSlices = std::move(extraSrcSlices);
        plan.extraDstSlices = std::move(extraDstSlices);
    }

    u32 extraThreadIdx = static_cast<u32>(resource.threads.size() - 1);
    ThreadHandle extraThread = resource.threads[extraThreadIdx];
    auto rounds = BuildExtraPeerRounds(myRank_, rankSize_, meshSize_, plans);
    for (u32 roundIdx = 0; roundIdx < rounds.size(); ++roundIdx) {
        for (u32 peerRank : rounds[roundIdx]) {
            const PeerSendPlan4Plane &plan = plans.at(peerRank);
            if (plan.extraChannel == nullptr) {
                continue;
            }
            HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][SelectChannel] phase=stage0-extra rank=%u round=%u peer=%u "
                         "links=%zu selected=%u portGroupSize=%u slices=%zu bytes=%llu threadIdx=%u quota=%llu",
                         myRank_, roundIdx, plan.peerRank, resource.channels.at(plan.peerRank).size(),
                         plan.extraChannelIdx, plan.extraChannel->portGroupSize, plan.extraSrcSlices.size(),
                         SumSliceBytes(plan.extraSrcSlices), extraThreadIdx, extraQuotaBytes);
            CHK_RET(RunPeerSendRecv(*plan.extraChannel, plan.extraSrcSlices, plan.extraDstSlices, extraThread));
        }
    }

    u32 threadIdx = 1;
    for (const auto &item : plans) {
        const PeerSendPlan4Plane &plan = item.second;
        u32 runThreadIdx = std::min(threadIdx, static_cast<u32>(resource.threads.size() - 1));
        ThreadHandle thread = resource.threads[runThreadIdx];
        HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][SelectChannel] phase=stage0-main rank=%u peer=%u links=%zu "
                     "selected=%u portGroupSize=%u slices=%zu bytes=%llu threadIdx=%u",
                     myRank_, plan.peerRank, resource.channels.at(plan.peerRank).size(), plan.mainChannelIdx,
                     plan.mainChannel->portGroupSize, plan.mainSrcSlices.size(), SumSliceBytes(plan.mainSrcSlices),
                     runThreadIdx);
        CHK_RET(RunPeerSendRecv(*plan.mainChannel, plan.mainSrcSlices, plan.mainDstSlices, thread));
        ++threadIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVV2Stage1NoMemcpy4Plane::RunStage1ToOutput(
    const TemplateDataParams &params, const TemplateResource &resource) const
{
    std::map<u32, PeerSendPlan4Plane> plans;
    u64 totalStageBytes = 0;
    u64 activePeerNum = 0;
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
        ++activePeerNum;
        totalStageBytes += SumSliceBytes(txSrcSlices);
        PeerSendPlan4Plane plan;
        plan.peerRank = finalDst;
        plan.mainChannelIdx = channelIdx;
        plan.extraChannelIdx = SelectExtraChannelIdx(item.second);
        plan.mainChannel = &channel;
        if (item.second.size() >= 4) {
            plan.extraChannel = &item.second[plan.extraChannelIdx];
        }
        plan.mainSrcSlices = std::move(txSrcSlices);
        plan.mainDstSlices = std::move(txDstSlices);
        plans.emplace(finalDst, std::move(plan));
    }

    u64 extraQuotaBytes = CalcExtraQuotaBytes(totalStageBytes, activePeerNum);
    for (auto &item : plans) {
        PeerSendPlan4Plane &plan = item.second;
        if (plan.extraChannel == nullptr || extraQuotaBytes == 0) {
            continue;
        }
        std::vector<DataSlice> mainSrcSlices;
        std::vector<DataSlice> mainDstSlices;
        std::vector<DataSlice> extraSrcSlices;
        std::vector<DataSlice> extraDstSlices;
        CHK_RET(SplitSlicesForExtraLane(plan.mainSrcSlices, plan.mainDstSlices, extraQuotaBytes, mainSrcSlices,
                                        mainDstSlices, extraSrcSlices, extraDstSlices));
        CHK_RET(RebindExtraDstSlices(*plan.mainChannel, *plan.extraChannel, extraDstSlices));
        plan.mainSrcSlices = std::move(mainSrcSlices);
        plan.mainDstSlices = std::move(mainDstSlices);
        plan.extraSrcSlices = std::move(extraSrcSlices);
        plan.extraDstSlices = std::move(extraDstSlices);
    }

    u32 extraThreadIdx = static_cast<u32>(resource.threads.size() - 1);
    ThreadHandle extraThread = resource.threads[extraThreadIdx];
    auto rounds = BuildExtraPeerRounds(myRank_, rankSize_, meshSize_, plans);
    for (u32 roundIdx = 0; roundIdx < rounds.size(); ++roundIdx) {
        for (u32 peerRank : rounds[roundIdx]) {
            const PeerSendPlan4Plane &plan = plans.at(peerRank);
            if (plan.extraChannel == nullptr) {
                continue;
            }
            HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][SelectChannel] phase=stage1-extra rank=%u round=%u peer=%u "
                         "links=%zu selected=%u portGroupSize=%u slices=%zu bytes=%llu threadIdx=%u quota=%llu",
                         myRank_, roundIdx, plan.peerRank, resource.channels.at(plan.peerRank).size(),
                         plan.extraChannelIdx, plan.extraChannel->portGroupSize, plan.extraSrcSlices.size(),
                         SumSliceBytes(plan.extraSrcSlices), extraThreadIdx, extraQuotaBytes);
            CHK_RET(RunPeerSendRecv(*plan.extraChannel, plan.extraSrcSlices, plan.extraDstSlices, extraThread));
        }
    }

    u32 threadIdx = 1;
    for (const auto &item : plans) {
        const PeerSendPlan4Plane &plan = item.second;
        u32 runThreadIdx = std::min(threadIdx, static_cast<u32>(resource.threads.size() - 1));
        ThreadHandle thread = resource.threads[runThreadIdx];
        HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][SelectChannel] phase=stage1-main rank=%u peer=%u links=%zu "
                     "selected=%u portGroupSize=%u slices=%zu bytes=%llu threadIdx=%u",
                     myRank_, plan.peerRank, resource.channels.at(plan.peerRank).size(), plan.mainChannelIdx,
                     plan.mainChannel->portGroupSize, plan.mainSrcSlices.size(), SumSliceBytes(plan.mainSrcSlices),
                     runThreadIdx);
        CHK_RET(RunPeerSendRecv(*plan.mainChannel, plan.mainSrcSlices, plan.mainDstSlices, thread));
        ++threadIdx;
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
