/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#include "ins_v2_alltoallv_parallel_v2_stage1_no_memcpy_4plane_executor.h"
#include <algorithm>
#include <cstring>
#include "ins_temp_all_to_all_v_v2_stage1_no_memcpy_4plane.h"
#include "template_utils.h"
#include "topo_match_alltoall_pod_direct.h"
#include "topo_match_ubx_v2.h"

namespace ops_hccl {
namespace {
constexpr u32 TEMPLATE_NUM = 2;
constexpr double DEFAULT_SPLIT_RATIO = 0.5;
constexpr double MIN_SPLIT_RATIO = 0.0;
constexpr double MAX_SPLIT_RATIO = 1.0;

bool IsA2AVV2Stage1NoMemcpy4PlaneAlg(const OpParam &param)
{
    return std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpy4Plane") == 0 ||
           std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpy4PlanePodUbxV2") == 0 ||
           std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpy4PlanePodDirect") == 0;
}

bool IsSingleLayerUbx4x2(const TopoInfoWithNetLayerDetails *topoInfo)
{
    return topoInfo != nullptr && topoInfo->topoLevelNums == 1 && topoInfo->userRankSize == 8 &&
           !topoInfo->level0PcieMix &&
           (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS || topoInfo->level0Topo == Level0Shape::CLOS);
}

void BuildSingleLayerUbx4x2Hierarchy(u32 userRank, std::vector<std::vector<u32>> &intraHierarchyInfo,
                                     std::vector<std::vector<u32>> &interHierarchyInfo)
{
    constexpr u32 UBX_4X2_MESH_SIZE = 4;
    constexpr u32 UBX_4X2_RANK_SIZE = 8;
    intraHierarchyInfo.clear();
    interHierarchyInfo.clear();
    std::vector<u32> meshRanks;
    u32 meshBaseRank = userRank / UBX_4X2_MESH_SIZE * UBX_4X2_MESH_SIZE;
    for (u32 rank = meshBaseRank; rank < meshBaseRank + UBX_4X2_MESH_SIZE; ++rank) {
        meshRanks.push_back(rank);
    }
    intraHierarchyInfo = {meshRanks};
    std::vector<u32> closRanks;
    closRanks.push_back(userRank);
    for (u32 rank = 0; rank < UBX_4X2_RANK_SIZE; ++rank) {
        if (rank / UBX_4X2_MESH_SIZE != userRank / UBX_4X2_MESH_SIZE) {
            closRanks.push_back(rank);
        }
    }
    interHierarchyInfo = {closRanks};
}

u64 AlignUp(u64 value, u64 align)
{
    return align == 0 ? value : ((value + align - 1) / align) * align;
}

void SplitV2PairCount(u64 count, double splitRatio, u64 &part0, u64 &part1)
{
    long double scaled = static_cast<long double>(count) * static_cast<long double>(splitRatio);
    part0 = static_cast<u64>(scaled);
    if (part0 > count) {
        part0 = count;
    }
    part1 = count - part0;
}
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
u64 InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    u64 count = 1;
    for (const auto &ranks : vTopo) {
        count *= ranks.size();
    }
    return count;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::BuildHierarchyInfo(
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    CHK_PTR_NULL(topoInfo);
    intraHierarchyInfo_.clear();
    interHierarchyInfo_.clear();
    CHK_PRT_RET(algHierarchyInfo.infos.empty() || algHierarchyInfo.infos[0].empty(),
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][BuildHierarchyInfo] invalid hierarchy."),
                HcclResult::HCCL_E_INTERNAL);
    if (IsSingleLayerUbx4x2(topoInfo)) {
        BuildSingleLayerUbx4x2Hierarchy(topoInfo->userRank, intraHierarchyInfo_, interHierarchyInfo_);
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        CHK_PRT_RET(algHierarchyInfo.infos[0].size() < 2 || algHierarchyInfo.infos[0][0].empty(),
                    HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][BuildHierarchyInfo] invalid MESH_1D_CLOS hierarchy."),
                    HcclResult::HCCL_E_INTERNAL);
        intraHierarchyInfo_ = {algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        closRanks.push_back(topoInfo->userRank);
        u32 meshSize = algHierarchyInfo.infos[0][0].size();
        for (auto rank : algHierarchyInfo.infos[0][1]) {
            if (rank / meshSize != topoInfo->userRank / meshSize) {
                closRanks.push_back(rank);
            }
        }
        interHierarchyInfo_ = {closRanks};
    } else {
        for (const auto &level0Ranks : algHierarchyInfo.infos[0]) {
            if (std::find(level0Ranks.begin(), level0Ranks.end(), topoInfo->userRank) != level0Ranks.end()) {
                intraHierarchyInfo_ = {level0Ranks};
                break;
            }
        }
        CHK_PRT_RET(intraHierarchyInfo_.empty() || algHierarchyInfo.infos.size() < 2 ||
                        algHierarchyInfo.infos[1].empty(),
                    HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][BuildHierarchyInfo] invalid direct hierarchy."),
                    HcclResult::HCCL_E_INTERNAL);
        interHierarchyInfo_ = algHierarchyInfo.infos[1];
    }
    meshSize_ = GetRankSize(intraHierarchyInfo_);
    rankSize_ = topoInfo->userRankSize;
    CHK_PRT_RET(meshSize_ == 0 || rankSize_ % meshSize_ != 0,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][BuildHierarchyInfo] invalid mesh/rank. rankSize=%llu mesh=%llu",
                           rankSize_, meshSize_),
                HcclResult::HCCL_E_INTERNAL);
    groupNum_ = rankSize_ / meshSize_;
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][BuildHierarchyInfo] rank=%u rankSize=%llu mesh=%llu group=%llu",
                 topoInfo->userRank, rankSize_, meshSize_, groupNum_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    resourceRequest = AlgResourceRequest{};
    CHK_PRT_RET(param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV || !IsA2AVV2Stage1NoMemcpy4PlaneAlg(param),
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][CalcRes] unsupported op/alg. opType=%d alg=%s",
                           param.opType, param.algName),
                HcclResult::HCCL_E_NOT_SUPPORT);
    myRank_ = topoInfo->userRank;
    CHK_RET(BuildHierarchyInfo(topoInfo, algHierarchyInfo));

    std::vector<HcclChannelDesc> meshChannels;
    std::vector<HcclChannelDesc> closChannels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, intraHierarchyInfo_, meshChannels));
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, interHierarchyInfo_, closChannels));
    CHK_PRT_RET(meshChannels.empty() || closChannels.empty(),
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][CalcRes] empty channel request. mesh=%zu clos=%zu",
                           meshChannels.size(), closChannels.size()),
                HcclResult::HCCL_E_INTERNAL);

    u32 maxPeerNum = 0;
    if (meshSize_ > 0 && groupNum_ > 0) {
        maxPeerNum = static_cast<u32>((meshSize_ - 1) + (groupNum_ - 1));
    }
    stage0Meta_.slaveThreadNum = maxPeerNum + 1;
    stage0Meta_.notifyNumOnMainThread = stage0Meta_.slaveThreadNum;
    stage0Meta_.notifyNumPerThread.assign(stage0Meta_.slaveThreadNum, 1);
    stage1Meta_ = stage0Meta_;

    resourceRequest.notifyNumOnMainThread = TEMPLATE_NUM;
    resourceRequest.slaveThreadNum = stage0Meta_.slaveThreadNum + stage1Meta_.slaveThreadNum + TEMPLATE_NUM;
    resourceRequest.notifyNumPerThread.emplace_back(stage0Meta_.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              stage0Meta_.notifyNumPerThread.begin(),
                                              stage0Meta_.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(stage1Meta_.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              stage1Meta_.notifyNumPerThread.begin(),
                                              stage1Meta_.notifyNumPerThread.end());
    resourceRequest.channels.emplace_back(meshChannels);
    resourceRequest.channels.emplace_back(closChannels);
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][CalcRes] rank=%u stage0Slave=%u stage1Slave=%u totalSlave=%u "
                 "meshChannels=%zu closChannels=%zu maxPeer=%u",
                 myRank_, stage0Meta_.slaveThreadNum, stage1Meta_.slaveThreadNum,
                 resourceRequest.slaveThreadNum, meshChannels.size(), closChannels.size(), maxPeerNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::RestoreChannelMaps(
    const AlgResourceCtxSerializable &resCtx)
{
    remoteRankToChannelInfo_.clear();
    intraLinkMap_.clear();
    interLinkMap_.clear();
    allLinkMap_.clear();
    remoteMaxSendCountsWithoutSelf_.assign(resCtx.topoInfo.userRankSize, 0);
    remoteCountInfoValid_.assign(resCtx.topoInfo.userRankSize, false);
    remoteRankToChannelInfo_.resize(resCtx.channels.size());
    for (u32 level = 0; level < resCtx.channels.size(); ++level) {
        for (const auto &channel : resCtx.channels[level]) {
            remoteRankToChannelInfo_[level][channel.remoteRank].push_back(channel);
        }
    }
    CHK_PRT_RET(remoteRankToChannelInfo_.size() < 2,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][RestoreChannelMaps] expected 2 channel levels, got %zu.",
                           remoteRankToChannelInfo_.size()),
                HcclResult::HCCL_E_INTERNAL);
    intraLinkMap_ = remoteRankToChannelInfo_[0];
    interLinkMap_ = remoteRankToChannelInfo_[1];
    allLinkMap_ = intraLinkMap_;
    for (const auto &item : interLinkMap_) {
        if (allLinkMap_.count(item.first) == 0) {
            allLinkMap_[item.first] = item.second;
        }
    }
    for (const auto &item : allLinkMap_) {
        if (item.second.empty()) {
            continue;
        }
        const ChannelInfo &channel = item.second[0];
        CHK_PRT_RET(!channel.hasRemoteAlltoAllVInfo,
                    HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][RestoreChannelMaps] missing remote a2av info. rank=%u peer=%u",
                               myRank_, item.first),
                    HcclResult::HCCL_E_INTERNAL);
        remoteMaxSendCountsWithoutSelf_[item.first] = channel.remoteAlltoAllVMaxSendCountWithoutSelf;
        remoteCountInfoValid_[item.first] = true;
        HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][RestoreChannelMaps] rank=%u peer=%u links=%zu remoteMax=%llu",
                     myRank_, item.first, item.second.size(), channel.remoteAlltoAllVMaxSendCountWithoutSelf);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::BuildRuntimeMetas()
{
    u32 stage0PeerNum = static_cast<u32>(stage0LinkMap_.size());
    u32 stage1PeerNum = static_cast<u32>(stage1LinkMap_.size());
    u32 maxPeerNum = meshSize_ > 0 && groupNum_ > 0 ? static_cast<u32>((meshSize_ - 1) + (groupNum_ - 1)) : 0;
    CHK_PRT_RET(stage0PeerNum > maxPeerNum || stage1PeerNum > maxPeerNum,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][BuildRuntimeMetas] active peers exceed resource upper bound. "
                           "rank=%u stage0=%u stage1=%u max=%u", myRank_, stage0PeerNum, stage1PeerNum,
                           maxPeerNum),
                HcclResult::HCCL_E_INTERNAL);
    stage0Meta_.slaveThreadNum = stage0PeerNum + 1;
    stage0Meta_.notifyNumOnMainThread = stage0Meta_.slaveThreadNum;
    stage0Meta_.notifyNumPerThread.assign(stage0Meta_.slaveThreadNum, 1);
    stage1Meta_.slaveThreadNum = stage1PeerNum + 1;
    stage1Meta_.notifyNumOnMainThread = stage1Meta_.slaveThreadNum;
    stage1Meta_.notifyNumPerThread.assign(stage1Meta_.slaveThreadNum, 1);
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][BuildRuntimeMetas] rank=%u stage0Peers=%u stage1Peers=%u "
                 "stage0Slave=%u stage1Slave=%u",
                 myRank_, stage0PeerNum, stage1PeerNum, stage0Meta_.slaveThreadNum,
                 stage1Meta_.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
bool InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::IsSameGroup(u32 rankA, u32 rankB) const
{
    return meshSize_ != 0 && rankA / meshSize_ == rankB / meshSize_;
}

template <typename AlgTopoMatch>
double InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::GetSplitRatio(const OpParam &param) const
{
    double ratio = param.opConfig.multipleDimensionSplitRatio;
    if (ratio < MIN_SPLIT_RATIO || ratio > MAX_SPLIT_RATIO) {
        HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][Ratio] invalid ratio[%f], use default[%f].",
                     ratio, DEFAULT_SPLIT_RATIO);
        return DEFAULT_SPLIT_RATIO;
    }
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][Ratio] use ratio[%f] from opConfig.", ratio);
    return ratio;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::AddStageLink(
    u32 peerRank, std::map<u32, std::vector<ChannelInfo>> &stageLinkMap) const
{
    if (peerRank == myRank_) {
        return HCCL_SUCCESS;
    }
    auto &srcMap = IsSameGroup(myRank_, peerRank) ? intraLinkMap_ : interLinkMap_;
    auto it = srcMap.find(peerRank);
    if (it != srcMap.end() && !it->second.empty()) {
        stageLinkMap[peerRank] = it->second;
        return HCCL_SUCCESS;
    }
    auto fallback = allLinkMap_.find(peerRank);
    if (fallback != allLinkMap_.end() && !fallback->second.empty()) {
        stageLinkMap[peerRank] = fallback->second;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][AddStageLink] missing channel. rank=%u peer=%u", myRank_, peerRank);
    return HcclResult::HCCL_E_INTERNAL;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::BuildStageLinkMaps(
    const TemplateDataParams &params)
{
    stage0LinkMap_.clear();
    stage1LinkMap_.clear();

    auto addStage0Relay = [this](u32 srcRank, u32 dstRank, u64 count) -> HcclResult {
        if (count == 0 || srcRank == dstRank) {
            return HCCL_SUCCESS;
        }
        const u32 srcGroup = srcRank / meshSize_;
        const u32 dstGroup = dstRank / meshSize_;
        const u32 srcLocal = srcRank % meshSize_;
        const u32 dstLocal = dstRank % meshSize_;
        u64 part0 = 0;
        u64 part1 = 0;
        SplitV2PairCount(count, splitRatio_, part0, part1);
        if (srcGroup == dstGroup) {
            if (part0 > 0 && (srcRank == myRank_ || dstRank == myRank_)) {
                CHK_RET(AddStageLink(srcRank == myRank_ ? dstRank : srcRank, stage0LinkMap_));
            }
            return HCCL_SUCCESS;
        }
        if (part0 > 0) {
            u32 relay0 = static_cast<u32>(dstGroup * meshSize_ + srcLocal);
            if (srcRank == myRank_ || relay0 == myRank_) {
                CHK_RET(AddStageLink(srcRank == myRank_ ? relay0 : srcRank, stage0LinkMap_));
            }
        }
        if (part1 > 0) {
            u32 relay1 = static_cast<u32>(srcGroup * meshSize_ + dstLocal);
            if (srcRank == myRank_ || relay1 == myRank_) {
                CHK_RET(AddStageLink(srcRank == myRank_ ? relay1 : srcRank, stage0LinkMap_));
            }
        }
        return HCCL_SUCCESS;
    };

    auto addStage1Final = [this](u32 srcRank, u32 finalDst, u64 count) -> HcclResult {
        if (count == 0 || srcRank == finalDst) {
            return HCCL_SUCCESS;
        }
        const u32 srcGroup = srcRank / meshSize_;
        const u32 dstGroup = finalDst / meshSize_;
        const u32 srcLocal = srcRank % meshSize_;
        const u32 dstLocal = finalDst % meshSize_;
        u64 part0 = 0;
        u64 part1 = 0;
        SplitV2PairCount(count, splitRatio_, part0, part1);
        if (srcGroup == dstGroup) {
            if (part1 > 0 && (srcRank == myRank_ || finalDst == myRank_)) {
                CHK_RET(AddStageLink(srcRank == myRank_ ? finalDst : srcRank, stage1LinkMap_));
            }
            return HCCL_SUCCESS;
        }
        if (part0 > 0) {
            u32 relay0 = static_cast<u32>(dstGroup * meshSize_ + srcLocal);
            if (relay0 == myRank_ || finalDst == myRank_) {
                CHK_RET(AddStageLink(relay0 == myRank_ ? finalDst : relay0, stage1LinkMap_));
            }
        }
        if (part1 > 0) {
            u32 relay1 = static_cast<u32>(srcGroup * meshSize_ + dstLocal);
            if (relay1 == myRank_ || finalDst == myRank_) {
                CHK_RET(AddStageLink(relay1 == myRank_ ? finalDst : relay1, stage1LinkMap_));
            }
        }
        return HCCL_SUCCESS;
    };

    for (u32 dstRank = 0; dstRank < rankSize_; ++dstRank) {
        u64 count = 0;
        CHK_RET(GetPairCountFromRecvMatrix(params, myRank_, dstRank, count));
        CHK_RET(addStage0Relay(myRank_, dstRank, count));
        CHK_RET(addStage1Final(myRank_, dstRank, count));
    }
    for (u32 dstRank = 0; dstRank < rankSize_; ++dstRank) {
        u64 count = 0;
        CHK_RET(GetPairCountFromRecvMatrix(params, dstRank, myRank_, count));
        CHK_RET(addStage0Relay(dstRank, myRank_, count));
        CHK_RET(addStage1Final(dstRank, myRank_, count));
    }

    for (u32 srcRank = 0; srcRank < rankSize_; ++srcRank) {
        if (srcRank == myRank_) {
            continue;
        }
        for (u32 finalDst = 0; finalDst < rankSize_; ++finalDst) {
            u64 count = 0;
            CHK_RET(GetPairCountFromRecvMatrix(params, srcRank, finalDst, count));
            CHK_RET(addStage1Final(srcRank, finalDst, count));
        }
    }
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][BuildStageLinkMaps] rank=%u stage0Peers=%zu stage1Peers=%zu",
                 myRank_, stage0LinkMap_.size(), stage1LinkMap_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::PrepareTemplateResources(
    const AlgResourceCtxSerializable &resCtx)
{
    threads_ = resCtx.threads;
    u32 stage0ThreadNum = stage0Meta_.slaveThreadNum + 1;
    u32 stage1ThreadNum = stage1Meta_.slaveThreadNum + 1;
    u32 required = 1 + stage0ThreadNum + stage1ThreadNum;
    CHK_PRT_RET(threads_.size() < required,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][PrepareTemplateResources] threads[%zu] < required[%u].",
                           threads_.size(), required),
                HcclResult::HCCL_E_INTERNAL);
    mainThread_ = threads_[0];
    auto it = threads_.begin() + 1;
    stage0Threads_.assign(it, it + stage0ThreadNum);
    it += stage0ThreadNum;
    stage1Threads_.assign(it, it + stage1ThreadNum);
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][PrepareTemplateResources] rank=%u totalThreads=%zu required=%u "
                 "stage0=%u stage1=%u",
                 myRank_, threads_.size(), required, stage0ThreadNum, stage1ThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::BuildBaseParams(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, TemplateDataParams &params)
{
    const u64 *sendCounts = reinterpret_cast<const u64 *>(param.all2AllVDataDes.sendCounts);
    const u64 *recvCounts = reinterpret_cast<const u64 *>(param.all2AllVDataDes.recvCounts);
    const u64 *sdispls = reinterpret_cast<const u64 *>(param.all2AllVDataDes.sdispls);
    const u64 *rdispls = reinterpret_cast<const u64 *>(param.all2AllVDataDes.rdispls);
    CHK_PTR_NULL(sendCounts);
    CHK_PTR_NULL(recvCounts);
    CHK_PTR_NULL(sdispls);
    CHK_PTR_NULL(rdispls);
    params.enableRemoteMemAccess = true;
    params.buffInfo.inputPtr = param.inputPtr;
    params.buffInfo.inBuffType = BufferType::INPUT;
    params.buffInfo.inputSize = param.inputSize * dataTypeSize_;
    params.buffInfo.hcclBuff = resCtx.cclMem;
    params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    params.buffInfo.hcclBuffSize = resCtx.cclMem.size;
    params.buffInfo.outputPtr = param.outputPtr;
    params.buffInfo.outBuffType = BufferType::OUTPUT;
    params.buffInfo.outputSize = param.outputSize * dataTypeSize_;
    params.sendCounts.assign(rankSize_, 0);
    params.recvCounts.assign(rankSize_, 0);
    params.sdispls.assign(rankSize_, 0);
    params.rdispls.assign(rankSize_, 0);
    params.remoteRdispls.assign(rankSize_, 0);
    params.remoteRecvCounts.assign(rankSize_, 0);
    u64 total = 0;
    for (u64 i = 0; i < rankSize_; ++i) {
        params.sendCounts[i] = sendCounts[i];
        params.recvCounts[i] = recvCounts[i];
        params.sdispls[i] = sdispls[i];
        params.rdispls[i] = rdispls[i];
        total += sendCounts[i];
    }
    params.count = total;
    params.sliceSize = total * dataTypeSize_;
    params.repeatNum = 1;
    for (u64 i = 0; i < rankSize_; ++i) {
        if (i == myRank_) {
            continue;
        }
        CHK_PRT_RET(i >= remoteCountInfoValid_.size() || !remoteCountInfoValid_[i],
                    HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][BaseParam] missing remote count info. rank=%u peer=%llu",
                               myRank_, i),
                    HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::GetPairCountFromRecvMatrix(
    const TemplateDataParams &params, u32 srcRank, u32 dstRank, u64 &count) const
{
    count = 0;
    if (srcRank == dstRank) {
        return HCCL_SUCCESS;
    }
    if (dstRank == myRank_) {
        CHK_PRT_RET(srcRank >= params.recvCounts.size(),
                    HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][PairCount] invalid local src. rank=%u src=%u",
                               myRank_, srcRank),
                    HcclResult::HCCL_E_INTERNAL);
        count = params.recvCounts[srcRank];
        return HCCL_SUCCESS;
    }
    auto it = allLinkMap_.find(dstRank);
    CHK_PRT_RET(it == allLinkMap_.end() || it->second.empty() ||
                    it->second[0].remoteAlltoAllVRecvCounts.size() <= srcRank,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][PairCount] missing remote recv count. "
                           "rank=%u src=%u dst=%u", myRank_, srcRank, dstRank),
                HcclResult::HCCL_E_INTERNAL);
    count = it->second[0].remoteAlltoAllVRecvCounts[srcRank];
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::GetGlobalMaxSendCount(
    u64 &globalMaxSend) const
{
    globalMaxSend = 0;
    for (u64 count : remoteMaxSendCountsWithoutSelf_) {
        globalMaxSend = std::max(globalMaxSend, count);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::BuildExactSlotOffsets(
    TemplateDataParams &params)
{
    const u64 slotNum = rankSize_ * rankSize_ * 2;
    params.alltoAllVV2SlotOffsets.assign(slotNum, 0);
    exactSlotBytes_ = 0;
    u64 directPartNum = 0;
    u64 scratchPartNum = 0;
    auto slotIndex = [this](u64 srcRank, u64 dstRank, u64 partIdx) {
        return (srcRank * rankSize_ + dstRank) * 2 + partIdx;
    };
    for (u64 srcRank = 0; srcRank < rankSize_; ++srcRank) {
        for (u64 dstRank = 0; dstRank < rankSize_; ++dstRank) {
            if (srcRank == dstRank) {
                continue;
            }
            u64 count = 0;
            CHK_RET(GetPairCountFromRecvMatrix(params, static_cast<u32>(srcRank), static_cast<u32>(dstRank),
                                               count));
            u64 part0 = 0;
            u64 part1 = 0;
            SplitV2PairCount(count, splitRatio_, part0, part1);
            u64 parts[2] = {part0, part1};
            u64 srcGroup = srcRank / meshSize_;
            u64 dstGroup = dstRank / meshSize_;
            u64 srcLocal = srcRank % meshSize_;
            u64 dstLocal = dstRank % meshSize_;
            u64 relays[2] = {
                srcGroup == dstGroup ? dstRank : dstGroup * meshSize_ + srcLocal,
                srcGroup == dstGroup ? srcRank : srcGroup * meshSize_ + dstLocal
            };
            for (u64 partIdx = 0; partIdx < 2; ++partIdx) {
                u64 bytes = parts[partIdx] * dataTypeSize_;
                if (bytes == 0) {
                    continue;
                }
                if (relays[partIdx] == srcRank || relays[partIdx] == dstRank) {
                    ++directPartNum;
                    continue;
                }
                u64 idx = slotIndex(srcRank, dstRank, partIdx);
                params.alltoAllVV2SlotOffsets[idx] = exactSlotBytes_;
                exactSlotBytes_ += AlignUp(bytes, HCCL_MIN_SLICE_ALIGN);
                ++scratchPartNum;
            }
        }
    }
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][ExactSlot] rank=%u compactBytes=%llu slotNum=%llu rankSize=%llu "
                 "scratchParts=%llu directParts=%llu",
                 myRank_, exactSlotBytes_, slotNum, rankSize_, scratchPartNum, directPartNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::CheckScratch(
    const AlgResourceCtxSerializable &resCtx) const
{
    CHK_PRT_RET(exactSlotBytes_ > resCtx.cclMem.size,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Scratch] insufficient ccl buffer. rank=%u required=%llu "
                           "cclSize=%llu slotStride=%llu rankSize=%llu",
                           myRank_, exactSlotBytes_, resCtx.cclMem.size, slotStride_, rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::RunStage0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const TemplateDataParams &params)
{
    InsTempAlltoAllVV2Stage1NoMemcpy4Plane temp(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    temp.SetV2Stage1NoMemcpyInfo(A2AVV2Stage1NoMemcpy4PlanePhase::STAGE0_TO_RELAY,
                                 static_cast<u32>(rankSize_), static_cast<u32>(meshSize_), slotStride_,
                                 splitRatio_);
    TemplateResource resource;
    resource.channels = stage0LinkMap_;
    resource.threads = stage0Threads_;
    resource.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][Stage0][PRE] rank=%u count=%llu slotStride=%llu",
                 myRank_, params.count, slotStride_);
    CHK_RET(PreSyncInterThreads(mainThread_, {stage0Threads_[0]}, {stage0Meta_.notifyNumOnMainThread}));
    CHK_RET(temp.KernelRun(param, params, resource));
    CHK_RET(PostSyncInterThreads(mainThread_, {stage0Threads_[0]}, {0}));
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][Stage0][POST] rank=%u", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::RunStage1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const TemplateDataParams &params)
{
    InsTempAlltoAllVV2Stage1NoMemcpy4Plane temp(param, resCtx.topoInfo.userRank, interHierarchyInfo_);
    temp.SetV2Stage1NoMemcpyInfo(A2AVV2Stage1NoMemcpy4PlanePhase::STAGE1_TO_OUTPUT,
                                 static_cast<u32>(rankSize_), static_cast<u32>(meshSize_), slotStride_,
                                 splitRatio_);
    TemplateResource resource;
    resource.channels = stage1LinkMap_;
    resource.threads = stage1Threads_;
    resource.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][Stage1][PRE] rank=%u count=%llu slotStride=%llu",
                 myRank_, params.count, slotStride_);
    CHK_RET(PreSyncInterThreads(mainThread_, {stage1Threads_[0]}, {stage1Meta_.notifyNumOnMainThread}));
    CHK_RET(temp.KernelRun(param, params, resource));
    CHK_RET(PostSyncInterThreads(mainThread_, {stage1Threads_[0]}, {1}));
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][Stage1][POST] rank=%u", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor<AlgTopoMatch>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][Orchestrate] start alg=%s", param.algName);
    CHK_PRT_RET(param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV || !IsA2AVV2Stage1NoMemcpy4PlaneAlg(param),
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Orchestrate] unsupported op/alg. opType=%d alg=%s",
                           param.opType, param.algName),
                HcclResult::HCCL_E_NOT_SUPPORT);
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataType_ >= HCCL_DATA_TYPE_RESERVED || dataTypeSize_ == 0,
                HCCL_ERROR("[A2AV_V2_STAGE1_4P_NM][Orchestrate] invalid dataType=%d", static_cast<int>(dataType_)),
                HcclResult::HCCL_E_INTERNAL);
    splitRatio_ = GetSplitRatio(param);
    CHK_RET(BuildHierarchyInfo(&resCtx.topoInfo, resCtx.algHierarchyInfo));
    CHK_RET(RestoreChannelMaps(resCtx));
    TemplateDataParams params;
    CHK_RET(BuildBaseParams(param, resCtx, params));
    dataCount_ = params.count;
    CHK_RET(BuildStageLinkMaps(params));
    CHK_RET(BuildRuntimeMetas());
    CHK_RET(PrepareTemplateResources(resCtx));
    u64 globalMaxSend = 0;
    CHK_RET(GetGlobalMaxSendCount(globalMaxSend));
    for (u64 count : params.sendCounts) {
        globalMaxSend = std::max(globalMaxSend, count);
    }
    u64 maxPart0 = 0;
    u64 maxPart1 = 0;
    SplitV2PairCount(globalMaxSend, splitRatio_, maxPart0, maxPart1);
    u64 maxPart = std::max(maxPart0, maxPart1);
    slotStride_ = AlignUp(maxPart * dataTypeSize_, HCCL_MIN_SLICE_ALIGN);
    if (slotStride_ == 0) {
        slotStride_ = HCCL_MIN_SLICE_ALIGN;
    }
    CHK_RET(BuildExactSlotOffsets(params));
    CHK_RET(CheckScratch(resCtx));
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][Orchestrate] rank=%u total=%llu globalMax=%llu splitRatio=%f "
                 "fallbackSlotStride=%llu exactSlotBytes=%llu",
                 myRank_, dataCount_, globalMaxSend, splitRatio_, slotStride_, exactSlotBytes_);
    CHK_RET(RunStage0(param, resCtx, params));
    CHK_RET(RunStage1(param, resCtx, params));
    HCCL_WARNING("[A2AV_V2_STAGE1_4P_NM][Orchestrate] end.");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpy4Plane,
                          InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor,
                          TopoMatchUBX_V2);

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpy4PlanePodUbxV2,
                          InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor,
                          TopoMatchUBX_V2);

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpy4PlanePodDirect,
                          InsV2AlltoAllVParallelV2Stage1NoMemcpy4PlaneExecutor,
                          TopoMatchAlltoAllPodDirect);

} // namespace ops_hccl
