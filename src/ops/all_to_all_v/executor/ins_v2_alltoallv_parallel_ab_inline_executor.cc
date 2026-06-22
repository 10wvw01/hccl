/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_alltoallv_parallel_ab_inline_executor.h"
#include <algorithm>
#include <cstring>
#include "ins_temp_all_to_all_v_ab_inline_no_memcpy.h"
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
constexpr u32 TEMPLATE_NUM = 3;
constexpr u32 A_CLOS_LINK_NUM = 3;
constexpr u32 B_CLOS_LINK_IDX = 3;

bool IsA2AVABInlineAlg(const OpParam &param)
{
    return std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV3ABInlineNoMemcpy") == 0 ||
           std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV3ABInlineNoMemcpyPodUbxV2") == 0 ||
           std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV3ABInlineNoMemcpyPodDirect") == 0;
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
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
u64 InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    u64 count = 1;
    for (const auto &ranks : vTopo) {
        count *= ranks.size();
    }
    return count;
}

template <typename AlgTopoMatch>
double InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::GetABRatio(const OpParam &param) const
{
    double ratio = param.opConfig.multipleDimensionSplitRatio;
    if (ratio < MIN_AB_RATIO || ratio > MAX_AB_RATIO) {
        HCCL_WARNING("[A2AV_AB_INLINE][Ratio] invalid ratio[%f], use default[%f].", ratio, DEFAULT_AB_RATIO);
        return DEFAULT_AB_RATIO;
    }
    HCCL_WARNING("[A2AV_AB_INLINE][Ratio] use ratio[%f] from opConfig.", ratio);
    return ratio;
}

template <typename AlgTopoMatch>
bool InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::IsIntraPeer(u32 peerRank) const
{
    return meshSize_ != 0 && peerRank / meshSize_ == myRank_ / meshSize_;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::BuildHierarchyInfo(
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    CHK_PTR_NULL(topoInfo);
    intraHierarchyInfo_.clear();
    interHierarchyInfo_.clear();
    CHK_PRT_RET(algHierarchyInfo.infos.empty() || algHierarchyInfo.infos[0].empty(),
                HCCL_ERROR("[A2AV_AB_INLINE][BuildHierarchyInfo] invalid hierarchy."),
                HcclResult::HCCL_E_INTERNAL);
    if (IsSingleLayerUbx4x2(topoInfo)) {
        BuildSingleLayerUbx4x2Hierarchy(topoInfo->userRank, intraHierarchyInfo_, interHierarchyInfo_);
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        CHK_PRT_RET(algHierarchyInfo.infos[0].size() < 2 || algHierarchyInfo.infos[0][0].empty(),
                    HCCL_ERROR("[A2AV_AB_INLINE][BuildHierarchyInfo] invalid MESH_1D_CLOS hierarchy."),
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
                    HCCL_ERROR("[A2AV_AB_INLINE][BuildHierarchyInfo] invalid direct hierarchy."),
                    HcclResult::HCCL_E_INTERNAL);
        interHierarchyInfo_ = algHierarchyInfo.infos[1];
    }
    meshSize_ = GetRankSize(intraHierarchyInfo_);
    rankSize_ = topoInfo->userRankSize;
    CHK_PRT_RET(meshSize_ == 0 || rankSize_ % meshSize_ != 0,
                HCCL_ERROR("[A2AV_AB_INLINE][BuildHierarchyInfo] invalid mesh/rank. rankSize=%llu mesh=%llu",
                           rankSize_, meshSize_),
                HcclResult::HCCL_E_INTERNAL);
    groupNum_ = rankSize_ / meshSize_;
    HCCL_WARNING("[A2AV_AB_INLINE][BuildHierarchyInfo] rank=%u rankSize=%llu mesh=%llu group=%llu",
                 topoInfo->userRank, rankSize_, meshSize_, groupNum_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    resourceRequest = AlgResourceRequest{};
    CHK_PRT_RET(param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV || !IsA2AVABInlineAlg(param),
                HCCL_ERROR("[A2AV_AB_INLINE][CalcRes] unsupported op/alg. opType=%d alg=%s",
                           param.opType, param.algName),
                HcclResult::HCCL_E_NOT_SUPPORT);
    myRank_ = topoInfo->userRank;
    CHK_RET(BuildHierarchyInfo(topoInfo, algHierarchyInfo));
    InsTempAlltoAllMesh2DV3NoMemcpy meshTemp(param, topoInfo->userRank, intraHierarchyInfo_);
    InsTempAlltoAllMeshClosV3NoMemcpy closTemp(param, topoInfo->userRank, interHierarchyInfo_);
    meshTemp.SetMeshDimensions(rankSize_, myRank_, meshSize_, groupNum_);
    closTemp.SetMeshDimensions(rankSize_, myRank_, meshSize_, groupNum_);
    AlgResourceRequest meshReq;
    AlgResourceRequest closReq;
    CHK_RET(meshTemp.CalcRes(comm, param, topoInfo, meshReq));
    CHK_RET(closTemp.CalcRes(comm, param, topoInfo, closReq));
    CHK_PRT_RET(meshReq.channels.empty() || closReq.channels.empty(),
                HCCL_ERROR("[A2AV_AB_INLINE][CalcRes] empty channel request."),
                HcclResult::HCCL_E_INTERNAL);
    meshMeta_ = {meshReq.slaveThreadNum, meshReq.notifyNumOnMainThread, meshReq.notifyNumPerThread};
    aClosMeta_.slaveThreadNum = A_CLOS_LINK_NUM > 0 ? A_CLOS_LINK_NUM - 1 : 0;
    aClosMeta_.notifyNumOnMainThread = aClosMeta_.slaveThreadNum;
    aClosMeta_.notifyNumPerThread.assign(aClosMeta_.slaveThreadNum, 1);
    u32 closPeerNum = rankSize_ > meshSize_ ? static_cast<u32>(rankSize_ - meshSize_) : 0;
    bClosMeta_.slaveThreadNum = closPeerNum;
    bClosMeta_.notifyNumOnMainThread = bClosMeta_.slaveThreadNum;
    bClosMeta_.notifyNumPerThread.assign(bClosMeta_.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = TEMPLATE_NUM;
    resourceRequest.slaveThreadNum = meshMeta_.slaveThreadNum + aClosMeta_.slaveThreadNum +
                                     bClosMeta_.slaveThreadNum + TEMPLATE_NUM;
    resourceRequest.notifyNumPerThread.emplace_back(meshMeta_.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              meshMeta_.notifyNumPerThread.begin(), meshMeta_.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(aClosMeta_.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              aClosMeta_.notifyNumPerThread.begin(), aClosMeta_.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(bClosMeta_.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              bClosMeta_.notifyNumPerThread.begin(), bClosMeta_.notifyNumPerThread.end());
    resourceRequest.channels.emplace_back(meshReq.channels[0]);
    resourceRequest.channels.emplace_back(closReq.channels[0]);
    HCCL_WARNING("[A2AV_AB_INLINE][CalcRes] rank=%u meshSlave=%u aClosSlave=%u bClosSlave=%u totalSlave=%u",
                 myRank_, meshMeta_.slaveThreadNum, aClosMeta_.slaveThreadNum, bClosMeta_.slaveThreadNum,
                 resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::RestoreChannelMaps(
    const AlgResourceCtxSerializable &resCtx)
{
    remoteRankToChannelInfo_.clear();
    intraLinkMap_.clear();
    interLinkMap_.clear();
    aClosLinkMap_.clear();
    bClosLinkMap_.clear();
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
                HCCL_ERROR("[A2AV_AB_INLINE][RestoreChannelMaps] expected 2 channel levels, got %zu.",
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
    for (u32 peer = 0; peer < rankSize_; ++peer) {
        if (peer == myRank_ || IsIntraPeer(peer)) {
            continue;
        }
        auto it = interLinkMap_.find(peer);
        CHK_PRT_RET(it == interLinkMap_.end() || it->second.size() <= B_CLOS_LINK_IDX,
                    HCCL_ERROR("[A2AV_AB_INLINE][RestoreChannelMaps] insufficient clos links. rank=%u peer=%u "
                               "links=%zu need=%u",
                               myRank_, peer, it == interLinkMap_.end() ? 0 : it->second.size(), B_CLOS_LINK_IDX + 1),
                    HcclResult::HCCL_E_INTERNAL);
        aClosLinkMap_[peer].assign(it->second.begin(), it->second.begin() + A_CLOS_LINK_NUM);
        bClosLinkMap_[peer].push_back(it->second[B_CLOS_LINK_IDX]);
        HCCL_WARNING("[A2AV_AB_INLINE][RestoreChannelMaps] rank=%u peer=%u rawLinks=%zu aLinks=%zu bLinks=%zu",
                     myRank_, peer, it->second.size(), aClosLinkMap_[peer].size(), bClosLinkMap_[peer].size());
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::BuildRuntimeMetas()
{
    u32 meshThreadNum = intraLinkMap_.size() > 0 ? static_cast<u32>(intraLinkMap_.size()) : 1;
    meshMeta_.slaveThreadNum = meshThreadNum > 0 ? meshThreadNum - 1 : 0;
    meshMeta_.notifyNumOnMainThread = meshMeta_.slaveThreadNum;
    meshMeta_.notifyNumPerThread.assign(meshMeta_.slaveThreadNum, 1);
    u32 aClosThreadNum = A_CLOS_LINK_NUM;
    aClosMeta_.slaveThreadNum = aClosThreadNum > 0 ? aClosThreadNum - 1 : 0;
    aClosMeta_.notifyNumOnMainThread = aClosMeta_.slaveThreadNum;
    aClosMeta_.notifyNumPerThread.assign(aClosMeta_.slaveThreadNum, 1);
    u32 bClosThreadNum = bClosLinkMap_.size() + 1;
    bClosMeta_.slaveThreadNum = bClosThreadNum > 0 ? bClosThreadNum - 1 : 0;
    bClosMeta_.notifyNumOnMainThread = bClosMeta_.slaveThreadNum;
    bClosMeta_.notifyNumPerThread.assign(bClosMeta_.slaveThreadNum, 1);
    HCCL_WARNING("[A2AV_AB_INLINE][BuildRuntimeMetas] rank=%u meshSlave=%u aClosSlave=%u bClosSlave=%u "
                 "bPeers=%zu",
                 myRank_, meshMeta_.slaveThreadNum, aClosMeta_.slaveThreadNum, bClosMeta_.slaveThreadNum,
                 bClosLinkMap_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::PrepareTemplateResources(
    const AlgResourceCtxSerializable &resCtx)
{
    threads_ = resCtx.threads;
    u32 meshThreadNum = meshMeta_.slaveThreadNum + 1;
    u32 aClosThreadNum = aClosMeta_.slaveThreadNum + 1;
    u32 bClosThreadNum = bClosMeta_.slaveThreadNum + 1;
    u32 required = 1 + meshThreadNum + aClosThreadNum + bClosThreadNum;
    CHK_PRT_RET(threads_.size() < required,
                HCCL_ERROR("[A2AV_AB_INLINE][PrepareTemplateResources] threads[%zu] < required[%u].",
                           threads_.size(), required),
                HcclResult::HCCL_E_INTERNAL);
    mainThread_ = threads_[0];
    auto it = threads_.begin() + 1;
    meshThreads_.assign(it, it + meshThreadNum);
    it += meshThreadNum;
    aClosThreads_.assign(it, it + aClosThreadNum);
    it += aClosThreadNum;
    bClosThreads_.assign(it, it + bClosThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::BuildBaseParams(
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
            u32 remoteRank = item.first;
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
    fillRemoteInfo(aClosLinkMap_);
    fillRemoteInfo(bClosLinkMap_);
    for (u64 i = 0; i < rankSize_; ++i) {
        if (i == myRank_) {
            continue;
        }
        CHK_PRT_RET(i >= remoteTotalSendCountsValid_.size() || !remoteTotalSendCountsValid_[i],
                    HCCL_ERROR("[A2AV_AB_INLINE][BaseParam] missing remote count info. rank=%u peer=%llu",
                               myRank_, i),
                    HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::GetGlobalMaxSendCount(u64 &globalMaxSend) const
{
    globalMaxSend = 0;
    for (u64 count : remoteMaxSendCountsWithoutSelf_) {
        globalMaxSend = std::max(globalMaxSend, count);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::SplitParams(
    const TemplateDataParams &baseParams, double ratio, TemplateDataParams &meshParams,
    TemplateDataParams &aClosParams, TemplateDataParams &bClosParams) const
{
    meshParams = baseParams;
    aClosParams = baseParams;
    bClosParams = baseParams;
    u64 globalMaxSend = 0;
    CHK_RET(GetGlobalMaxSendCount(globalMaxSend));
    for (u64 count : baseParams.sendCounts) {
        globalMaxSend = std::max(globalMaxSend, count);
    }
    u64 threshold = static_cast<u64>(static_cast<double>(globalMaxSend) * ratio);
    u64 meshTotal = 0;
    u64 aTotal = 0;
    u64 bTotal = 0;
    for (u64 i = 0; i < rankSize_; ++i) {
        bool isIntra = i == myRank_ || IsIntraPeer(static_cast<u32>(i));
        if (isIntra) {
            aClosParams.sendCounts[i] = 0;
            aClosParams.recvCounts[i] = 0;
            bClosParams.sendCounts[i] = 0;
            bClosParams.recvCounts[i] = 0;
            aClosParams.remoteRecvCounts[i] = 0;
            bClosParams.remoteRecvCounts[i] = 0;
            meshTotal += meshParams.sendCounts[i];
            continue;
        }
        meshParams.sendCounts[i] = 0;
        meshParams.recvCounts[i] = 0;
        meshParams.remoteRecvCounts[i] = 0;
        aClosParams.sendCounts[i] = std::min(threshold, baseParams.sendCounts[i]);
        aClosParams.recvCounts[i] = std::min(threshold, baseParams.recvCounts[i]);
        bClosParams.sendCounts[i] = baseParams.sendCounts[i] - aClosParams.sendCounts[i];
        bClosParams.recvCounts[i] = baseParams.recvCounts[i] - aClosParams.recvCounts[i];
        bClosParams.sdispls[i] = baseParams.sdispls[i] + aClosParams.sendCounts[i];
        bClosParams.rdispls[i] = baseParams.rdispls[i] + aClosParams.recvCounts[i];
        aClosParams.remoteRecvCounts[i] = aClosParams.sendCounts[i];
        bClosParams.remoteRecvCounts[i] = bClosParams.sendCounts[i];
        bClosParams.remoteRdispls[i] = baseParams.remoteRdispls[i] + aClosParams.sendCounts[i];
        aTotal += aClosParams.sendCounts[i];
        bTotal += bClosParams.sendCounts[i];
    }
    meshParams.count = meshTotal;
    meshParams.sliceSize = meshTotal * dataTypeSize_;
    aClosParams.count = aTotal;
    aClosParams.sliceSize = aTotal * dataTypeSize_;
    bClosParams.count = bTotal;
    bClosParams.sliceSize = bTotal * dataTypeSize_;
    HCCL_WARNING("[A2AV_AB_INLINE][Split] rank=%u ratio=%f threshold=%llu mesh=%llu aClos=%llu bClos=%llu",
                 myRank_, ratio, threshold, meshTotal, aTotal, bTotal);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::RunInlineTemplates(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const TemplateDataParams &meshParams,
    const TemplateDataParams &aClosParams, const TemplateDataParams &bClosParams)
{
    InsTempAlltoAllMesh2DV3NoMemcpy meshTemp(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    InsTempAlltoAllMeshClosV3NoMemcpy aClosTemp(param, resCtx.topoInfo.userRank, interHierarchyInfo_);
    InsTempAlltoAllVABInlineNoMemcpy bClosTemp(param, resCtx.topoInfo.userRank, interHierarchyInfo_);
    meshTemp.SetMeshDimensions(rankSize_, myRank_, meshSize_, groupNum_);
    aClosTemp.SetMeshDimensions(rankSize_, myRank_, meshSize_, groupNum_);
    CHK_PRT_RET(param.engine != CommEngine::COMM_ENGINE_AICPU_TS,
                HCCL_ERROR("[A2AV_AB_INLINE][RunInlineTemplates] only AICPU_TS engine is supported. "
                           "engine=%d", static_cast<int>(param.engine)),
                HcclResult::HCCL_E_NOT_SUPPORT);
    {
        aClosTemp.SetchannelsPerRank(aClosLinkMap_);
    }
    TemplateResource meshRes;
    meshRes.channels = intraLinkMap_;
    meshRes.threads = meshThreads_;
    meshRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    TemplateResource aClosRes;
    aClosRes.channels = aClosLinkMap_;
    aClosRes.threads = aClosThreads_;
    aClosRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    TemplateResource bClosRes;
    bClosRes.channels = bClosLinkMap_;
    bClosRes.threads = bClosThreads_;
    bClosRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    CHK_RET(PreSyncInterThreads(mainThread_, {meshThreads_[0], aClosThreads_[0], bClosThreads_[0]},
                                {meshMeta_.notifyNumOnMainThread, aClosMeta_.notifyNumOnMainThread,
                                 bClosMeta_.notifyNumOnMainThread}));
    HcclResult ret = meshTemp.KernelRun(param, meshParams, meshRes);
    HcclResult aRet = aClosTemp.KernelRun(param, aClosParams, aClosRes);
    HcclResult bRet = bClosTemp.KernelRun(param, bClosParams, bClosRes);
    HcclResult postRet = PostSyncInterThreads(mainThread_, {meshThreads_[0], aClosThreads_[0], bClosThreads_[0]},
                                              {0, 1, 2});
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    if (aRet != HCCL_SUCCESS) {
        return aRet;
    }
    if (bRet != HCCL_SUCCESS) {
        return bRet;
    }
    return postRet;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABInlineExecutor<AlgTopoMatch>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_WARNING("[A2AV_AB_INLINE][Orchestrate] start alg=%s", param.algName);
    CHK_PRT_RET(param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV || !IsA2AVABInlineAlg(param),
                HCCL_ERROR("[A2AV_AB_INLINE][Orchestrate] unsupported op/alg. opType=%d alg=%s",
                           param.opType, param.algName),
                HcclResult::HCCL_E_NOT_SUPPORT);
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataType_ >= HCCL_DATA_TYPE_RESERVED || dataTypeSize_ == 0,
                HCCL_ERROR("[A2AV_AB_INLINE][Orchestrate] invalid dataType=%d", static_cast<int>(dataType_)),
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
    TemplateDataParams meshParams;
    TemplateDataParams aClosParams;
    TemplateDataParams bClosParams;
    CHK_RET(SplitParams(baseParams, GetABRatio(param), meshParams, aClosParams, bClosParams));
    CHK_RET(RunInlineTemplates(param, resCtx, meshParams, aClosParams, bClosParams));
    HCCL_WARNING("[A2AV_AB_INLINE][Orchestrate] end.");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV3ABInlineNoMemcpy,
                          InsV2AlltoAllVParallelABInlineExecutor,
                          TopoMatchUBX_V2);

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV3ABInlineNoMemcpyPodUbxV2,
                          InsV2AlltoAllVParallelABInlineExecutor,
                          TopoMatchUBX_V2);

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV3ABInlineNoMemcpyPodDirect,
                          InsV2AlltoAllVParallelABInlineExecutor,
                          TopoMatchAlltoAllPodDirect);

} // namespace ops_hccl
