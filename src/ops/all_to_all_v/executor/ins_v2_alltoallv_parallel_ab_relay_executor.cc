/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_alltoallv_parallel_ab_relay_executor.h"
#include <algorithm>
#include <cstring>
#include "ins_temp_all_to_all_v_ab_relay_no_memcpy.h"
#include "ins_temp_alltoall_mesh_2d_v3_no_memcpy.h"
#include "ins_temp_alltoall_mesh_clos_v3_no_memcpy.h"
#include "template_utils.h"
#include "topo_match_ubx_v2.h"
#include "topo_match_alltoall_pod_direct.h"

namespace ops_hccl {
namespace {
constexpr double DEFAULT_AB_RATIO = 0.8;
constexpr double MIN_AB_RATIO = 0.0;
constexpr double MAX_AB_RATIO = 1.0;
constexpr u32 A_TEMPLATE_NUM = 2;
constexpr u32 RELAY_TEMPLATE_NUM = 2;
constexpr u32 MIN_TEMPLATE_THREAD_NUM = 1;

bool IsA2AVABRelayAlg(const OpParam &param)
{
    return std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV3ABRelayNoMemcpy") == 0 ||
           std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV3ABRelayNoMemcpyPodUbxV2") == 0 ||
           std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV3ABRelayNoMemcpyPodDirect") == 0;
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

std::vector<ChannelInfo> SelectChannelsByPortGroupSize(const std::vector<ChannelInfo> &channels,
                                                       bool preferAggregate)
{
    std::vector<ChannelInfo> selected;
    for (const auto &channel : channels) {
        bool isAggregate = channel.portGroupSize > 1;
        if (isAggregate == preferAggregate) {
            selected.push_back(channel);
        }
    }
    return selected.empty() ? channels : selected;
}

u64 AlignUp(u64 value, u64 align)
{
    return align == 0 ? value : ((value + align - 1) / align) * align;
}
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
u64 InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    u64 count = 1;
    for (const auto &ranks : vTopo) {
        count *= ranks.size();
    }
    return count;
}

template <typename AlgTopoMatch>
double InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::GetABRatio(const OpParam &param) const
{
    double ratio = param.opConfig.multipleDimensionSplitRatio;
    if (ratio < MIN_AB_RATIO || ratio > MAX_AB_RATIO) {
        HCCL_WARNING("[A2AV_AB_RELAY][Ratio] invalid ratio[%f], use default[%f].", ratio, DEFAULT_AB_RATIO);
        return DEFAULT_AB_RATIO;
    }
    HCCL_WARNING("[A2AV_AB_RELAY][Ratio] use ratio[%f] from opConfig.", ratio);
    return ratio;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::BuildHierarchyInfo(
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    CHK_PTR_NULL(topoInfo);
    intraHierarchyInfo_.clear();
    interHierarchyInfo_.clear();
    CHK_PRT_RET(algHierarchyInfo.infos.empty() || algHierarchyInfo.infos[0].empty(),
                HCCL_ERROR("[A2AV_AB_RELAY][BuildHierarchyInfo] invalid hierarchy."),
                HcclResult::HCCL_E_INTERNAL);
    if (IsSingleLayerUbx4x2(topoInfo)) {
        BuildSingleLayerUbx4x2Hierarchy(topoInfo->userRank, intraHierarchyInfo_, interHierarchyInfo_);
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        CHK_PRT_RET(algHierarchyInfo.infos[0].size() < 2 || algHierarchyInfo.infos[0][0].empty(),
                    HCCL_ERROR("[A2AV_AB_RELAY][BuildHierarchyInfo] invalid MESH_1D_CLOS hierarchy."),
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
                    HCCL_ERROR("[A2AV_AB_RELAY][BuildHierarchyInfo] invalid direct hierarchy."),
                    HcclResult::HCCL_E_INTERNAL);
        interHierarchyInfo_ = algHierarchyInfo.infos[1];
    }
    meshSize_ = GetRankSize(intraHierarchyInfo_);
    rankSize_ = topoInfo->userRankSize;
    CHK_PRT_RET(meshSize_ == 0 || rankSize_ % meshSize_ != 0,
                HCCL_ERROR("[A2AV_AB_RELAY][BuildHierarchyInfo] invalid mesh/rank. rankSize=%llu mesh=%llu",
                           rankSize_, meshSize_),
                HcclResult::HCCL_E_INTERNAL);
    groupNum_ = rankSize_ / meshSize_;
    rankSizeLevel1_ = GetRankSize(interHierarchyInfo_);
    HCCL_WARNING("[A2AV_AB_RELAY][BuildHierarchyInfo] rank=%u rankSize=%llu mesh=%llu group=%llu inter=%llu",
                 topoInfo->userRank, rankSize_, meshSize_, groupNum_, rankSizeLevel1_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    resourceRequest = AlgResourceRequest{};
    CHK_PRT_RET(param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV || !IsA2AVABRelayAlg(param),
                HCCL_ERROR("[A2AV_AB_RELAY][CalcRes] unsupported op/alg. opType=%d alg=%s",
                           param.opType, param.algName),
                HcclResult::HCCL_E_NOT_SUPPORT);
    myRank_ = topoInfo->userRank;
    CHK_RET(BuildHierarchyInfo(topoInfo, algHierarchyInfo));

    InsTempAlltoAllMesh2DV3NoMemcpy aIntraTemp(param, topoInfo->userRank, intraHierarchyInfo_);
    InsTempAlltoAllMeshClosV3NoMemcpy aInterTemp(param, topoInfo->userRank, interHierarchyInfo_);
    aIntraTemp.SetMeshDimensions(rankSize_, myRank_, meshSize_, groupNum_);
    aInterTemp.SetMeshDimensions(rankSize_, myRank_, meshSize_, groupNum_);
    AlgResourceRequest aIntraReq;
    AlgResourceRequest aInterReq;
    CHK_RET(aIntraTemp.CalcRes(comm, param, topoInfo, aIntraReq));
    CHK_RET(aInterTemp.CalcRes(comm, param, topoInfo, aInterReq));
    CHK_PRT_RET(aIntraReq.channels.empty() || aInterReq.channels.empty(),
                HCCL_ERROR("[A2AV_AB_RELAY][CalcRes] empty A channel request."),
                HcclResult::HCCL_E_INTERNAL);
    aIntraMeta_ = {aIntraReq.slaveThreadNum, aIntraReq.notifyNumOnMainThread, aIntraReq.notifyNumPerThread};
    aInterMeta_ = {aInterReq.slaveThreadNum, aInterReq.notifyNumOnMainThread, aInterReq.notifyNumPerThread};

    u32 closLinks = std::max(1u, CalcChannelsPerRank(aInterReq.channels[0]));
    u32 relayPeerNum = groupNum_ > 0 ? static_cast<u32>(groupNum_ - 1) : 0;
    u32 meshPeerNum = meshSize_ > 0 ? static_cast<u32>(meshSize_ - 1) : 0;
    bRelayMeta_.slaveThreadNum = std::max(1u, std::max(meshPeerNum, relayPeerNum * closLinks));
    bRelayMeta_.notifyNumOnMainThread = bRelayMeta_.slaveThreadNum;
    bRelayMeta_.notifyNumPerThread.assign(bRelayMeta_.slaveThreadNum, 1);

    resourceRequest.notifyNumOnMainThread = A_TEMPLATE_NUM + RELAY_TEMPLATE_NUM;
    resourceRequest.slaveThreadNum = aIntraMeta_.slaveThreadNum + aInterMeta_.slaveThreadNum +
                                     bRelayMeta_.slaveThreadNum + A_TEMPLATE_NUM + RELAY_TEMPLATE_NUM;
    resourceRequest.notifyNumPerThread.emplace_back(aIntraMeta_.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              aIntraMeta_.notifyNumPerThread.begin(),
                                              aIntraMeta_.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(aInterMeta_.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              aInterMeta_.notifyNumPerThread.begin(),
                                              aInterMeta_.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(bRelayMeta_.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              bRelayMeta_.notifyNumPerThread.begin(),
                                              bRelayMeta_.notifyNumPerThread.end());
    resourceRequest.channels.emplace_back(aIntraReq.channels[0]);
    resourceRequest.channels.emplace_back(aInterReq.channels[0]);
    HCCL_WARNING("[A2AV_AB_RELAY][CalcRes] rank=%u aIntraSlave=%u aInterSlave=%u bRelaySlave=%u "
                 "meshPeer=%u relayPeer=%u closLinks=%u totalSlave=%u channelGroups=%zu",
                 myRank_, aIntraMeta_.slaveThreadNum, aInterMeta_.slaveThreadNum, bRelayMeta_.slaveThreadNum,
                 meshPeerNum, relayPeerNum, closLinks, resourceRequest.slaveThreadNum,
                 resourceRequest.channels.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::RestoreChannelMaps(
    const AlgResourceCtxSerializable &resCtx)
{
    remoteRankToChannelInfo_.clear();
    intraLinkMap_.clear();
    interLinkMap_.clear();
    aInterLinkMap_.clear();
    bRelayLinkMap_.clear();
    remoteTotalSendCountsWithoutSelf_.assign(resCtx.topoInfo.userRankSize, 0);
    remoteMaxSendCountsWithoutSelf_.assign(resCtx.topoInfo.userRankSize, 0);
    remoteTotalSendCountsValid_.assign(resCtx.topoInfo.userRankSize, false);
    remoteRankToChannelInfo_.resize(resCtx.channels.size());
    for (u32 level = 0; level < resCtx.channels.size(); ++level) {
        for (const auto &channel : resCtx.channels[level]) {
            remoteRankToChannelInfo_[level][channel.remoteRank].push_back(channel);
        }
    }
    CHK_PRT_RET(remoteRankToChannelInfo_.size() < 2,
                HCCL_ERROR("[A2AV_AB_RELAY][RestoreChannelMaps] expected 2 channel levels, got %zu.",
                           remoteRankToChannelInfo_.size()),
                HcclResult::HCCL_E_INTERNAL);
    intraLinkMap_ = remoteRankToChannelInfo_[0];
    interLinkMap_ = remoteRankToChannelInfo_[1];
    if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix &&
        !interLinkMap_.empty()) {
        std::map<u32, std::vector<ChannelInfo>> mergedInterMap;
        for (auto rank : interHierarchyInfo_[0]) {
            if (intraLinkMap_.count(rank) != 0) {
                mergedInterMap[rank] = intraLinkMap_[rank];
            } else if (interLinkMap_.count(rank) != 0) {
                mergedInterMap[rank] = interLinkMap_[rank];
            }
        }
        interLinkMap_ = mergedInterMap;
    }
    for (const auto &item : interLinkMap_) {
        aInterLinkMap_[item.first] = SelectChannelsByPortGroupSize(item.second, false);
    }
    u32 myLocal = myRank_ % meshSize_;
    for (u32 group = 0; group < groupNum_; ++group) {
        u32 peer = group * meshSize_ + myLocal;
        if (peer == myRank_) {
            continue;
        }
        auto it = interLinkMap_.find(peer);
        CHK_PRT_RET(it == interLinkMap_.end() || it->second.empty(),
                    HCCL_ERROR("[A2AV_AB_RELAY][RestoreChannelMaps] missing relay clos link. rank=%u peer=%u",
                               myRank_, peer),
                    HcclResult::HCCL_E_INTERNAL);
        bRelayLinkMap_[peer] = SelectChannelsByPortGroupSize(it->second, false);
        HCCL_WARNING("[A2AV_AB_RELAY][RestoreChannelMaps] rank=%u bPeer=%u rawLinks=%zu runLinks=%zu",
                     myRank_, peer, it->second.size(), bRelayLinkMap_[peer].size());
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::BuildRuntimeMetas()
{
    u32 aIntraRankSize = intraHierarchyInfo_.empty() ? 1 : static_cast<u32>(intraHierarchyInfo_[0].size());
    u32 aIntraThreadNum = std::max(MIN_TEMPLATE_THREAD_NUM, aIntraRankSize - 1);
    aIntraMeta_.slaveThreadNum = aIntraThreadNum > 0 ? aIntraThreadNum - 1 : 0;
    aIntraMeta_.notifyNumOnMainThread = aIntraMeta_.slaveThreadNum;
    aIntraMeta_.notifyNumPerThread.assign(aIntraMeta_.slaveThreadNum, 1);
    u32 aInterThreadNum = std::max(MIN_TEMPLATE_THREAD_NUM, static_cast<u32>(CalcChannelsPerRank(aInterLinkMap_)));
    aInterMeta_.slaveThreadNum = aInterThreadNum > 0 ? aInterThreadNum - 1 : 0;
    aInterMeta_.notifyNumOnMainThread = aInterMeta_.slaveThreadNum;
    aInterMeta_.notifyNumPerThread.assign(aInterMeta_.slaveThreadNum, 1);
    u32 bRelayThreadNum = 1;
    for (const auto &item : bRelayLinkMap_) {
        bRelayThreadNum += item.second.size();
    }
    u32 meshPrerouteThreadNum = intraLinkMap_.size() + 1;
    bRelayThreadNum = std::max(bRelayThreadNum, meshPrerouteThreadNum);
    bRelayMeta_.slaveThreadNum = bRelayThreadNum > 0 ? bRelayThreadNum - 1 : 0;
    bRelayMeta_.notifyNumOnMainThread = bRelayMeta_.slaveThreadNum;
    bRelayMeta_.notifyNumPerThread.assign(bRelayMeta_.slaveThreadNum, 1);
    HCCL_WARNING("[A2AV_AB_RELAY][BuildRuntimeMetas] rank=%u aIntraSlave=%u aInterSlave=%u "
                 "bRelaySlave=%u bRelayPeers=%zu",
                 myRank_, aIntraMeta_.slaveThreadNum, aInterMeta_.slaveThreadNum,
                 bRelayMeta_.slaveThreadNum, bRelayLinkMap_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::PrepareTemplateResources(
    const AlgResourceCtxSerializable &resCtx)
{
    threads_ = resCtx.threads;
    u32 aIntraThreadNum = aIntraMeta_.slaveThreadNum + 1;
    u32 aInterThreadNum = aInterMeta_.slaveThreadNum + 1;
    u32 bRelayThreadNum = bRelayMeta_.slaveThreadNum + 1;
    u32 required = 1 + aIntraThreadNum + aInterThreadNum + bRelayThreadNum;
    CHK_PRT_RET(threads_.size() < required,
                HCCL_ERROR("[A2AV_AB_RELAY][PrepareTemplateResources] threads[%zu] < required[%u].",
                           threads_.size(), required),
                HcclResult::HCCL_E_INTERNAL);
    mainThread_ = threads_[0];
    auto it = threads_.begin() + 1;
    aIntraThreads_.assign(it, it + aIntraThreadNum);
    it += aIntraThreadNum;
    aInterThreads_.assign(it, it + aInterThreadNum);
    it += aInterThreadNum;
    bRelayThreads_.assign(it, it + bRelayThreadNum);
    HCCL_WARNING("[A2AV_AB_RELAY][PrepareTemplateResources] rank=%u totalThreads=%zu required=%u "
                 "aIntra=%u aInter=%u bRelay=%u",
                 myRank_, threads_.size(), required, aIntraThreadNum, aInterThreadNum, bRelayThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::BuildBaseParams(
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
    params.buffInfo.inBuffBaseOff = 0;
    params.buffInfo.hcclBuff = resCtx.cclMem;
    params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    params.buffInfo.hcclBuffSize = resCtx.cclMem.size;
    params.buffInfo.hcclBuffBaseOff = 0;
    params.buffInfo.outputPtr = param.outputPtr;
    params.buffInfo.outBuffType = BufferType::OUTPUT;
    params.buffInfo.outputSize = param.outputSize * dataTypeSize_;
    params.buffInfo.outBuffBaseOff = 0;
    params.inputSliceStride = 0;
    params.outputSliceStride = 0;
    params.sliceSize = dataSize_;
    params.count = dataCount_;
    params.repeatNum = 1;
    params.sendCounts.assign(rankSize_, 0);
    params.recvCounts.assign(rankSize_, 0);
    params.sdispls.assign(rankSize_, 0);
    params.rdispls.assign(rankSize_, 0);
    params.remoteRdispls.assign(rankSize_, 0);
    params.remoteRecvCounts.assign(rankSize_, 0);
    for (u64 i = 0; i < rankSize_; ++i) {
        params.sendCounts[i] = sendCounts[i];
        params.recvCounts[i] = recvCounts[i];
        params.sdispls[i] = sdispls[i];
        params.rdispls[i] = rdispls[i];
    }
    auto fillRemoteInfo = [this, &params](const std::map<u32, std::vector<ChannelInfo>> &linkMap) {
        for (const auto &item : linkMap) {
            if (item.second.empty()) {
                continue;
            }
            const u32 remoteRank = item.first;
            const ChannelInfo &channel = item.second[0];
            if (remoteRank < params.remoteRdispls.size() && channel.hasRemoteAlltoAllVInfo) {
                params.remoteRdispls[remoteRank] = channel.remoteAlltoAllVRdisplForLocalRank;
                params.remoteRecvCounts[remoteRank] = channel.remoteAlltoAllVRecvCountForLocalRank;
                if (remoteRank < remoteTotalSendCountsWithoutSelf_.size()) {
                    remoteTotalSendCountsWithoutSelf_[remoteRank] =
                        channel.remoteAlltoAllVTotalSendCountWithoutSelf;
                    remoteMaxSendCountsWithoutSelf_[remoteRank] =
                        channel.remoteAlltoAllVMaxSendCountWithoutSelf;
                    remoteTotalSendCountsValid_[remoteRank] = true;
                }
            }
        }
    };
    fillRemoteInfo(intraLinkMap_);
    fillRemoteInfo(interLinkMap_);
    fillRemoteInfo(aInterLinkMap_);
    fillRemoteInfo(bRelayLinkMap_);
    for (u64 i = 0; i < rankSize_; ++i) {
        if (i == myRank_) {
            continue;
        }
        CHK_PRT_RET(i >= remoteTotalSendCountsValid_.size() || !remoteTotalSendCountsValid_[i],
                    HCCL_ERROR("[A2AV_AB_RELAY][BaseParam] missing remote count info. rank=%u peer=%llu",
                               myRank_, i),
                    HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::GetGlobalMaxSendCount(u64 &globalMaxSend) const
{
    globalMaxSend = 0;
    for (u64 count : remoteMaxSendCountsWithoutSelf_) {
        globalMaxSend = std::max(globalMaxSend, count);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::SplitABParams(
    const TemplateDataParams &baseParams, double ratio, TemplateDataParams &aParams, TemplateDataParams &bParams) const
{
    aParams = baseParams;
    bParams = baseParams;
    u64 globalMaxSend = 0;
    CHK_RET(GetGlobalMaxSendCount(globalMaxSend));
    for (u64 count : baseParams.sendCounts) {
        globalMaxSend = std::max(globalMaxSend, count);
    }
    u64 threshold = static_cast<u64>(static_cast<double>(globalMaxSend) * ratio);
    u64 aTotal = 0;
    u64 bTotal = 0;
    for (u64 i = 0; i < rankSize_; ++i) {
        aParams.sendCounts[i] = std::min(threshold, baseParams.sendCounts[i]);
        aParams.recvCounts[i] = std::min(threshold, baseParams.recvCounts[i]);
        bParams.sendCounts[i] = baseParams.sendCounts[i] - aParams.sendCounts[i];
        bParams.recvCounts[i] = baseParams.recvCounts[i] - aParams.recvCounts[i];
        bParams.sdispls[i] = baseParams.sdispls[i] + aParams.sendCounts[i];
        bParams.rdispls[i] = baseParams.rdispls[i] + aParams.recvCounts[i];
        if (i < aParams.remoteRecvCounts.size()) {
            aParams.remoteRecvCounts[i] = aParams.sendCounts[i];
            bParams.remoteRecvCounts[i] = bParams.sendCounts[i];
        }
        if (i < bParams.remoteRdispls.size()) {
            bParams.remoteRdispls[i] = baseParams.remoteRdispls[i] + aParams.sendCounts[i];
        }
        aTotal += aParams.sendCounts[i];
        bTotal += bParams.sendCounts[i];
        if (i != myRank_ && (baseParams.sendCounts[i] > 0 || baseParams.recvCounts[i] > 0)) {
            HCCL_WARNING("[A2AV_AB_RELAY][SplitPeer] rank=%u peer=%llu baseSend=%llu aSend=%llu "
                         "bSend=%llu baseRecv=%llu aRecv=%llu bRecv=%llu", myRank_, i,
                         baseParams.sendCounts[i], aParams.sendCounts[i], bParams.sendCounts[i],
                         baseParams.recvCounts[i], aParams.recvCounts[i], bParams.recvCounts[i]);
        }
    }
    aParams.count = aTotal;
    aParams.sliceSize = aTotal * dataTypeSize_;
    aParams.alltoAllVABThreshold = threshold;
    aParams.alltoAllVABMeshSize = meshSize_;
    bParams.count = bTotal;
    bParams.sliceSize = bTotal * dataTypeSize_;
    bParams.alltoAllVABThreshold = threshold;
    bParams.alltoAllVABMeshSize = meshSize_;
    HCCL_WARNING("[A2AV_AB_RELAY][Split] rank=%u ratio=%f globalMax=%llu threshold=%llu aCount=%llu bCount=%llu",
                 myRank_, ratio, globalMaxSend, threshold, aTotal, bTotal);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::CheckRelayScratch(
    const TemplateDataParams &bParams, const AlgResourceCtxSerializable &resCtx) const
{
    u64 maxBCount = 0;
    for (u64 count : bParams.sendCounts) {
        maxBCount = std::max(maxBCount, count);
    }
    for (u64 count : bParams.recvCounts) {
        maxBCount = std::max(maxBCount, count);
    }
    // Relay scratch only holds same-group src ranks' B data for inter-group
    // dst peers: meshSize_ * groupNum_ slots == rankSize_ slots, O(N).
    u64 required = rankSize_ * slotStride_;
    CHK_PRT_RET(maxBCount > 0 && required > resCtx.cclMem.size,
                HCCL_ERROR("[A2AV_AB_RELAY][Scratch] insufficient ccl buffer. rank=%u required=%llu "
                           "cclSize=%llu slotStride=%llu maxBCount=%llu rankSize=%llu",
                           myRank_, required, resCtx.cclMem.size, slotStride_, maxBCount, rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::RunAOriginalAndPreroute(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const TemplateDataParams &aParams,
    const TemplateDataParams &bParams)
{
    bool hasA = aParams.count > 0;
    bool hasB = bParams.count > 0;
    if (!hasA && !hasB) {
        HCCL_WARNING("[A2AV_AB_RELAY][RunAOriginalAndPreroute] skip empty A/B.");
        return HCCL_SUCCESS;
    }
    InsTempAlltoAllMesh2DV3NoMemcpy aIntraTemp(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    InsTempAlltoAllMeshClosV3NoMemcpy aInterTemp(param, resCtx.topoInfo.userRank, interHierarchyInfo_);
    aIntraTemp.SetMeshDimensions(rankSize_, myRank_, meshSize_, groupNum_);
    aInterTemp.SetMeshDimensions(rankSize_, myRank_, meshSize_, groupNum_);
    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS || param.engine == CommEngine::COMM_ENGINE_AIV) {
        aInterTemp.SetchannelsPerRank(aInterLinkMap_);
    }
    TemplateResource aIntraRes;
    aIntraRes.channels = intraLinkMap_;
    aIntraRes.threads = aIntraThreads_;
    aIntraRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    TemplateResource aInterRes;
    aInterRes.channels = aInterLinkMap_;
    aInterRes.threads = aInterThreads_;
    aInterRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    InsTempAlltoAllVABRelayNoMemcpy relayTemp(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    relayTemp.SetRelayInfo(A2AVABRelayPhase::PREROUTE_TO_RELAY, static_cast<u32>(rankSize_),
                           static_cast<u32>(meshSize_), slotStride_);
    TemplateResource relayRes;
    relayRes.channels = intraLinkMap_;
    relayRes.threads = bRelayThreads_;
    relayRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    TemplateDataParams aIntraParams = aParams;
    if (myRank_ < aIntraParams.sendCounts.size()) {
        u64 selfCount = aIntraParams.sendCounts[myRank_];
        aIntraParams.count = aIntraParams.count >= selfCount ? aIntraParams.count - selfCount : 0;
        aIntraParams.sliceSize = aIntraParams.count * dataTypeSize_;
        aIntraParams.sendCounts[myRank_] = 0;
        aIntraParams.recvCounts[myRank_] = 0;
    }
    std::vector<ThreadHandle> stageMains;
    std::vector<u32> mainNotifyIdx;
    if (hasA) {
        stageMains.push_back(aInterThreads_[0]);
        mainNotifyIdx.push_back(aInterMeta_.notifyNumOnMainThread);
        stageMains.push_back(aIntraThreads_[0]);
        mainNotifyIdx.push_back(aIntraMeta_.notifyNumOnMainThread);
    }
    if (hasB) {
        stageMains.push_back(bRelayThreads_[0]);
        mainNotifyIdx.push_back(bRelayMeta_.notifyNumOnMainThread);
    }
    CHK_RET(PreSyncInterThreads(mainThread_, stageMains, mainNotifyIdx));
    HCCL_WARNING("[A2AV_AB_RELAY][Stage1] rank=%u submit order: A_CLOS -> A_MESH -> B_PREROUTE. "
                 "hasA=%d hasB=%d", myRank_, hasA, hasB);
    if (hasA) {
        HCCL_WARNING("[A2AV_AB_RELAY][Stage1][A_CLOS_SUBMIT] rank=%u count=%llu", myRank_, aParams.count);
        CHK_RET(aInterTemp.KernelRun(param, aParams, aInterRes));
        HCCL_WARNING("[A2AV_AB_RELAY][Stage1][A_MESH_SUBMIT] rank=%u count=%llu", myRank_, aIntraParams.count);
        CHK_RET(aIntraTemp.KernelRun(param, aIntraParams, aIntraRes));
    }
    if (hasB) {
        HCCL_WARNING("[A2AV_AB_RELAY][Stage1][B_PREROUTE_SUBMIT] rank=%u count=%llu", myRank_, bParams.count);
        CHK_RET(relayTemp.KernelRun(param, bParams, relayRes));
    }
    std::vector<u32> postNotifyIdx;
    postNotifyIdx.reserve(stageMains.size());
    for (u32 i = 0; i < stageMains.size(); ++i) {
        postNotifyIdx.push_back(i);
    }
    CHK_RET(PostSyncInterThreads(mainThread_, stageMains, postNotifyIdx));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::RunBRelay(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const TemplateDataParams &bParams)
{
    if (bParams.count == 0) {
        HCCL_WARNING("[A2AV_AB_RELAY][RunBRelay] skip empty B.");
        return HCCL_SUCCESS;
    }
    InsTempAlltoAllVABRelayNoMemcpy relayTemp(param, resCtx.topoInfo.userRank, {std::vector<u32>{}});
    relayTemp.SetRelayInfo(A2AVABRelayPhase::RELAY_TO_OUTPUT, static_cast<u32>(rankSize_),
                           static_cast<u32>(meshSize_), slotStride_);
    TemplateResource relayRes;
    relayRes.channels = bRelayLinkMap_;
    relayRes.threads = bRelayThreads_;
    relayRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    CHK_RET(PreSyncInterThreads(mainThread_, {bRelayThreads_[0]}, {bRelayMeta_.notifyNumOnMainThread}));
    CHK_RET(relayTemp.KernelRun(param, bParams, relayRes));
    CHK_RET(PostSyncInterThreads(mainThread_, {bRelayThreads_[0]}, {3}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABRelayExecutor<AlgTopoMatch>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_WARNING("[A2AV_AB_RELAY][Orchestrate] start alg=%s", param.algName);
    CHK_PRT_RET(param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV || !IsA2AVABRelayAlg(param),
                HCCL_ERROR("[A2AV_AB_RELAY][Orchestrate] unsupported op/alg. opType=%d alg=%s",
                           param.opType, param.algName),
                HcclResult::HCCL_E_NOT_SUPPORT);
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataType_ >= HCCL_DATA_TYPE_RESERVED || dataTypeSize_ == 0,
                HCCL_ERROR("[A2AV_AB_RELAY][Orchestrate] invalid dataType=%d", static_cast<int>(dataType_)),
                HcclResult::HCCL_E_INTERNAL);
    const u64 *sendCounts = reinterpret_cast<const u64 *>(param.all2AllVDataDes.sendCounts);
    CHK_PTR_NULL(sendCounts);
    dataCount_ = 0;
    for (u64 i = 0; i < rankSize_; ++i) {
        dataCount_ += sendCounts[i];
    }
    dataSize_ = dataCount_ * dataTypeSize_;
    CHK_RET(BuildHierarchyInfo(&resCtx.topoInfo, resCtx.algHierarchyInfo));
    CHK_RET(RestoreChannelMaps(resCtx));
    CHK_RET(BuildRuntimeMetas());
    CHK_RET(PrepareTemplateResources(resCtx));
    TemplateDataParams baseParams;
    CHK_RET(BuildBaseParams(param, resCtx, baseParams));
    TemplateDataParams aParams;
    TemplateDataParams bParams;
    CHK_RET(SplitABParams(baseParams, GetABRatio(param), aParams, bParams));
    u64 globalMaxSend = 0;
    CHK_RET(GetGlobalMaxSendCount(globalMaxSend));
    for (u64 count : baseParams.sendCounts) {
        globalMaxSend = std::max(globalMaxSend, count);
    }
    u64 maxBCount = globalMaxSend > bParams.alltoAllVABThreshold ?
        globalMaxSend - bParams.alltoAllVABThreshold : 0;
    slotStride_ = AlignUp(maxBCount * dataTypeSize_, HCCL_MIN_SLICE_ALIGN);
    if (slotStride_ == 0) {
        slotStride_ = HCCL_MIN_SLICE_ALIGN;
    }
    CHK_RET(CheckRelayScratch(bParams, resCtx));
    CHK_RET(RunAOriginalAndPreroute(param, resCtx, aParams, bParams));
    CHK_RET(RunBRelay(param, resCtx, bParams));
    HCCL_WARNING("[A2AV_AB_RELAY][Orchestrate] end.");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV3ABRelayNoMemcpy,
                          InsV2AlltoAllVParallelABRelayExecutor,
                          TopoMatchUBX_V2);

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV3ABRelayNoMemcpyPodUbxV2,
                          InsV2AlltoAllVParallelABRelayExecutor,
                          TopoMatchUBX_V2);

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV3ABRelayNoMemcpyPodDirect,
                          InsV2AlltoAllVParallelABRelayExecutor,
                          TopoMatchAlltoAllPodDirect);

} // namespace ops_hccl
