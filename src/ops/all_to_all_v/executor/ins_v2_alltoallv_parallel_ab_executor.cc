/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_alltoallv_parallel_ab_executor.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include "ins_temp_all_to_all_v_mesh_1D.h"
#include "ins_temp_alltoall_mesh_2d_v3_no_memcpy.h"
#include "ins_temp_alltoall_mesh_clos_v3_no_memcpy.h"
#include "topo_match_ubx_v2.h"
#include "topo_match_alltoall_pod_direct.h"

namespace ops_hccl {
namespace {
constexpr double DEFAULT_AB_RATIO = 0.8;
constexpr double MIN_AB_RATIO = 0.0;
constexpr double MAX_AB_RATIO = 1.0;
constexpr u32 A_TEMPLATE_NUM = 2;
constexpr u32 B_TEMPLATE_NUM = 1;

bool IsA2AVABAlg(const OpParam &param)
{
    return std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV3ABNoMemcpy") == 0 ||
           std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV3ABNoMemcpyPodUbxV2") == 0 ||
           std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV3ABNoMemcpyPodDirect") == 0;
}

bool IsSingleLayerUbx4x2(const TopoInfoWithNetLayerDetails *topoInfo)
{
    return topoInfo != nullptr && topoInfo->topoLevelNums == 1 && topoInfo->userRankSize == 8 &&
           !topoInfo->level0PcieMix &&
           (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS ||
            topoInfo->level0Topo == Level0Shape::CLOS);
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
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
uint64_t InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    uint64_t count = 1;
    for (const auto &ranks : vTopo) {
        count *= ranks.size();
    }
    return count;
}

template <typename AlgTopoMatch>
double InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::GetABRatio() const
{
    const char *env = std::getenv("HCCL_A2AV_AB_RATIO");
    if (env == nullptr) {
        return DEFAULT_AB_RATIO;
    }
    char *end = nullptr;
    double ratio = std::strtod(env, &end);
    if (end == env || ratio < MIN_AB_RATIO || ratio > MAX_AB_RATIO) {
        HCCL_WARNING("[A2AV_AB][Ratio] invalid HCCL_A2AV_AB_RATIO[%s], use default[%f].",
                     env, DEFAULT_AB_RATIO);
        return DEFAULT_AB_RATIO;
    }
    return ratio;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::BuildHierarchyInfo(
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    CHK_PTR_NULL(topoInfo);
    intraHierarchyInfo_.clear();
    interHierarchyInfo_.clear();
    bCalcHierarchyInfo_.clear();
    bRunHierarchyInfo_.clear();
    CHK_PRT_RET(algHierarchyInfo.infos.empty() || algHierarchyInfo.infos[0].empty(),
                HCCL_ERROR("[A2AV_AB][BuildHierarchyInfo] invalid algHierarchyInfo."),
                HcclResult::HCCL_E_INTERNAL);

    if (IsSingleLayerUbx4x2(topoInfo)) {
        BuildSingleLayerUbx4x2Hierarchy(topoInfo->userRank, intraHierarchyInfo_, interHierarchyInfo_);
        HCCL_WARNING("[A2AV_AB][BuildHierarchyInfo] Single-layer UBX 4x2 normalized. "
                     "rank=%u level0Topo=%d intra=%zu inter=%zu infos0=%zu infos1=%zu",
                     topoInfo->userRank, static_cast<int>(topoInfo->level0Topo),
                     intraHierarchyInfo_[0].size(), interHierarchyInfo_[0].size(),
                     algHierarchyInfo.infos.empty() ? 0 : algHierarchyInfo.infos[0].size(),
                     algHierarchyInfo.infos.size() > 1 ? algHierarchyInfo.infos[1].size() : 0);
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        CHK_PRT_RET(algHierarchyInfo.infos.size() < 1 || algHierarchyInfo.infos[0].size() < 2 ||
                        algHierarchyInfo.infos[0][0].empty() || algHierarchyInfo.infos[0][1].empty(),
                    HCCL_ERROR("[A2AV_AB][BuildHierarchyInfo] invalid MESH_1D_CLOS hierarchy. "
                               "infos.size=%zu infos[0].size=%zu",
                               algHierarchyInfo.infos.size(), algHierarchyInfo.infos.empty() ? 0 : algHierarchyInfo.infos[0].size()),
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
        if (intraHierarchyInfo_.empty()) {
            intraHierarchyInfo_ = algHierarchyInfo.infos[0];
        }
        CHK_PRT_RET(algHierarchyInfo.infos.size() < 2 || algHierarchyInfo.infos[1].empty(),
                    HCCL_ERROR("[A2AV_AB][BuildHierarchyInfo] invalid direct hierarchy. infos.size=%zu",
                               algHierarchyInfo.infos.size()),
                    HcclResult::HCCL_E_INTERNAL);
        interHierarchyInfo_ = algHierarchyInfo.infos[1];
    }
    std::vector<u32> fullRanks;
    fullRanks.reserve(topoInfo->userRankSize);
    for (u32 rank = 0; rank < topoInfo->userRankSize; ++rank) {
        fullRanks.push_back(rank);
    }

    if (IsSingleLayerUbx4x2(topoInfo)) {
        bCalcHierarchyInfo_ = {fullRanks};
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        CHK_PRT_RET(algHierarchyInfo.infos[0].size() != 2,
                    HCCL_ERROR("[A2AV_AB][BuildHierarchyInfo] MESH_1D_CLOS B calc hierarchy requires 2 level0 groups."),
                    HcclResult::HCCL_E_INTERNAL);
        if (topoInfo->topoLevelNums == 1) {
            bCalcHierarchyInfo_ = {fullRanks};
        } else {
            CHK_PRT_RET(algHierarchyInfo.infos.size() < 2 || algHierarchyInfo.infos[1].empty(),
                        HCCL_ERROR("[A2AV_AB][BuildHierarchyInfo] invalid B hierarchy for MESH_1D_CLOS."),
                        HcclResult::HCCL_E_INTERNAL);
            bCalcHierarchyInfo_.push_back(algHierarchyInfo.infos[0][1]);
            bCalcHierarchyInfo_.push_back(fullRanks);
        }
    } else {
        bCalcHierarchyInfo_ = {fullRanks};
    }

    bRunHierarchyInfo_ = {fullRanks};
    rankSizeLevel0_ = GetRankSize(intraHierarchyInfo_);
    rankSizeLevel1_ = GetRankSize(interHierarchyInfo_);
    rankSize_ = topoInfo->userRankSize;
    HCCL_WARNING("[A2AV_AB][BuildHierarchyInfo] rank=%u rankSize=%llu intra=%llu inter=%llu "
                 "bCalcDims=%zu bRunDims=%zu bRunSize=%zu",
                 topoInfo->userRank, rankSize_, rankSizeLevel0_, rankSizeLevel1_,
                 bCalcHierarchyInfo_.size(), bRunHierarchyInfo_.size(), bRunHierarchyInfo_[0].size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    resourceRequest = AlgResourceRequest{};
    CHK_PRT_RET(param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV || !IsA2AVABAlg(param),
                HCCL_ERROR("[A2AV_AB][CalcRes] only AllToAllV AB no-memcpy alg is supported. opType=%d alg=%s",
                           param.opType, param.algName),
                HcclResult::HCCL_E_NOT_SUPPORT);
    myRank_ = topoInfo->userRank;
    CHK_RET(BuildHierarchyInfo(topoInfo, algHierarchyInfo));

    InsTempAlltoAllMesh2DV3NoMemcpy aIntraTemp(param, topoInfo->userRank, intraHierarchyInfo_);
    InsTempAlltoAllMeshClosV3NoMemcpy aInterTemp(param, topoInfo->userRank, interHierarchyInfo_);
    InsTempAlltoAllVMesh1D bTemp(param, topoInfo->userRank, bCalcHierarchyInfo_);
    aIntraTemp.SetMeshDimensions(rankSize_, myRank_, rankSizeLevel0_, rankSizeLevel1_);
    aInterTemp.SetMeshDimensions(rankSize_, myRank_, rankSizeLevel0_, rankSizeLevel1_);

    AlgResourceRequest aIntraReq;
    AlgResourceRequest aInterReq;
    AlgResourceRequest bReq;
    CHK_RET(aIntraTemp.CalcRes(comm, param, topoInfo, aIntraReq));
    CHK_RET(aInterTemp.CalcRes(comm, param, topoInfo, aInterReq));
    CHK_RET(bTemp.CalcRes(comm, param, topoInfo, bReq));

    CHK_PRT_RET(aIntraReq.channels.empty() || aInterReq.channels.empty() || bReq.channels.empty(),
                HCCL_ERROR("[A2AV_AB][CalcRes] empty channel request. aIntra=%zu aInter=%zu b=%zu",
                           aIntraReq.channels.size(), aInterReq.channels.size(), bReq.channels.size()),
                HcclResult::HCCL_E_INTERNAL);

    aIntraMeta_ = {aIntraReq.slaveThreadNum, aIntraReq.notifyNumOnMainThread, aIntraReq.notifyNumPerThread};
    aInterMeta_ = {aInterReq.slaveThreadNum, aInterReq.notifyNumOnMainThread, aInterReq.notifyNumPerThread};
    bMeta_ = {bReq.slaveThreadNum, bReq.notifyNumOnMainThread, bReq.notifyNumPerThread};

    resourceRequest.notifyNumOnMainThread = A_TEMPLATE_NUM + B_TEMPLATE_NUM;
    resourceRequest.slaveThreadNum = aIntraReq.slaveThreadNum + aInterReq.slaveThreadNum + bReq.slaveThreadNum +
                                     A_TEMPLATE_NUM + B_TEMPLATE_NUM;
    resourceRequest.notifyNumPerThread.emplace_back(aIntraReq.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              aIntraReq.notifyNumPerThread.begin(),
                                              aIntraReq.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(aInterReq.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              aInterReq.notifyNumPerThread.begin(),
                                              aInterReq.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(bReq.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              bReq.notifyNumPerThread.begin(),
                                              bReq.notifyNumPerThread.end());

    resourceRequest.channels.emplace_back(aIntraReq.channels[0]);
    resourceRequest.channels.emplace_back(aInterReq.channels[0]);
    resourceRequest.channels.emplace_back(bReq.channels[0]);
    HCCL_WARNING("[A2AV_AB][CalcRes] rank=%u aIntraSlave=%u aInterSlave=%u bSlave=%u "
                 "aIntraNotifyMain=%u aInterNotifyMain=%u bNotifyMain=%u totalSlave=%u "
                 "notifyMain=%u channelGroups=%zu",
                 topoInfo->userRank, aIntraReq.slaveThreadNum, aInterReq.slaveThreadNum,
                 bReq.slaveThreadNum, aIntraReq.notifyNumOnMainThread,
                 aInterReq.notifyNumOnMainThread, bReq.notifyNumOnMainThread,
                 resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread,
                 resourceRequest.channels.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::RestoreChannelMaps(
    const AlgResourceCtxSerializable &resCtx)
{
    remoteRankToChannelInfo_.clear();
    intraLinkMap_.clear();
    interLinkMap_.clear();
    fullLinkMap_.clear();
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    CHK_PRT_RET(remoteRankToChannelInfo_.size() < 3,
                HCCL_ERROR("[A2AV_AB][RestoreChannelMaps] expected 3 channel levels, got %zu.",
                           remoteRankToChannelInfo_.size()),
                HcclResult::HCCL_E_INTERNAL);
    intraLinkMap_ = remoteRankToChannelInfo_[0];
    interLinkMap_ = remoteRankToChannelInfo_[1];
    fullLinkMap_ = remoteRankToChannelInfo_[2];

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
    HCCL_WARNING("[A2AV_AB][RestoreChannelMaps] rank=%u channelLevels=%zu intra=%zu inter=%zu full=%zu",
                 resCtx.topoInfo.userRank, remoteRankToChannelInfo_.size(), intraLinkMap_.size(),
                 interLinkMap_.size(), fullLinkMap_.size());
    for (u32 rank = 0; rank < resCtx.topoInfo.userRankSize; ++rank) {
        if (rank == resCtx.topoInfo.userRank) {
            continue;
        }
        CHK_PRT_RET(fullLinkMap_.count(rank) == 0 || fullLinkMap_[rank].empty(),
                    HCCL_ERROR("[A2AV_AB][RestoreChannelMaps] B full channel missing. myRank=%u peer=%u "
                               "fullMapSize=%zu",
                               resCtx.topoInfo.userRank, rank, fullLinkMap_.size()),
                    HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::PrepareTemplateResources(
    const AlgResourceCtxSerializable &resCtx)
{
    threads_ = resCtx.threads;
    aIntraThreads_.clear();
    aInterThreads_.clear();
    bThreads_.clear();
    templateMainThreads_.clear();
    syncNotifyOnTemplates_.clear();
    syncNotifyOnMain_.clear();
    CHK_PRT_RET(threads_.size() < 4,
                HCCL_ERROR("[A2AV_AB][PrepareTemplateResources] insufficient threads[%zu].", threads_.size()),
                HcclResult::HCCL_E_INTERNAL);
    u32 aIntraThreadNum = aIntraMeta_.slaveThreadNum + 1;
    u32 aInterThreadNum = aInterMeta_.slaveThreadNum + 1;
    u32 bThreadNum = bMeta_.slaveThreadNum + 1;
    u32 requiredThreadNum = 1 + aIntraThreadNum + aInterThreadNum + bThreadNum;
    CHK_PRT_RET(threads_.size() < requiredThreadNum,
                HCCL_ERROR("[A2AV_AB][PrepareTemplateResources] threads[%zu] < required[%u].",
                           threads_.size(), requiredThreadNum),
                HcclResult::HCCL_E_INTERNAL);
    mainThread_ = threads_[0];
    auto it = threads_.begin() + 1;
    aIntraThreads_.assign(it, it + aIntraThreadNum);
    it += aIntraThreadNum;
    aInterThreads_.assign(it, it + aInterThreadNum);
    it += aInterThreadNum;
    bThreads_.assign(it, it + bThreadNum);
    templateMainThreads_ = {aIntraThreads_[0], aInterThreads_[0], bThreads_[0]};
    syncNotifyOnTemplates_ = {aIntraMeta_.notifyNumOnMainThread, aInterMeta_.notifyNumOnMainThread,
                              bMeta_.notifyNumOnMainThread};
    syncNotifyOnMain_ = {0, 1, 2};
    HCCL_WARNING("[A2AV_AB][PrepareTemplateResources] rank=%u totalThreads=%zu required=%u "
                 "aIntraThreads=%u aInterThreads=%u bThreads=%u main=%llu aMain=%llu iMain=%llu bMain=%llu",
                 resCtx.topoInfo.userRank, threads_.size(), requiredThreadNum,
                 aIntraThreadNum, aInterThreadNum, bThreadNum,
                 static_cast<unsigned long long>(mainThread_),
                 static_cast<unsigned long long>(aIntraThreads_[0]),
                 static_cast<unsigned long long>(aInterThreads_[0]),
                 static_cast<unsigned long long>(bThreads_[0]));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::BuildBaseParams(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, TemplateDataParams &params) const
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
    params.sendCounts.resize(rankSize_, 0);
    params.recvCounts.resize(rankSize_, 0);
    params.sdispls.resize(rankSize_, 0);
    params.rdispls.resize(rankSize_, 0);
    for (u64 i = 0; i < rankSize_; ++i) {
        params.sendCounts[i] = sendCounts[i];
        params.recvCounts[i] = recvCounts[i];
        params.sdispls[i] = sdispls[i];
        params.rdispls[i] = rdispls[i];
        if (sendCounts[i] != 0 || recvCounts[i] != 0 || i == myRank_) {
            HCCL_INFO("[A2AV_AB][BaseParam] rank=%u peer=%llu send=%llu recv=%llu sdispl=%llu rdispl=%llu",
                      myRank_, i, sendCounts[i], recvCounts[i], sdispls[i], rdispls[i]);
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::SplitABParams(
    const TemplateDataParams &baseParams, double ratio, TemplateDataParams &aParams, TemplateDataParams &bParams) const
{
    aParams = baseParams;
    bParams = baseParams;
    u64 aTotalCount = 0;
    u64 bTotalCount = 0;
    for (u64 i = 0; i < rankSize_; ++i) {
        u64 aSend = static_cast<u64>(static_cast<double>(baseParams.sendCounts[i]) * ratio);
        u64 aRecv = static_cast<u64>(static_cast<double>(baseParams.recvCounts[i]) * ratio);
        aParams.sendCounts[i] = std::min(aSend, baseParams.sendCounts[i]);
        aParams.recvCounts[i] = std::min(aRecv, baseParams.recvCounts[i]);
        if (i != myRank_ && (aParams.sendCounts[i] == 0 || aParams.recvCounts[i] == 0)) {
            HCCL_WARNING("[A2AV_AB][Split] rank=%u peer=%llu fallback to B because A is asymmetric. "
                         "send=%llu recv=%llu aSend=%llu aRecv=%llu",
                         myRank_, i, baseParams.sendCounts[i], baseParams.recvCounts[i],
                         aParams.sendCounts[i], aParams.recvCounts[i]);
            aParams.sendCounts[i] = 0;
            aParams.recvCounts[i] = 0;
        }
        bParams.sendCounts[i] = baseParams.sendCounts[i] - aParams.sendCounts[i];
        bParams.recvCounts[i] = baseParams.recvCounts[i] - aParams.recvCounts[i];
        bParams.sdispls[i] = baseParams.sdispls[i] + aParams.sendCounts[i];
        bParams.rdispls[i] = baseParams.rdispls[i] + aParams.recvCounts[i];
        aTotalCount += aParams.sendCounts[i];
        bTotalCount += bParams.sendCounts[i];
        if (baseParams.sendCounts[i] != 0 || baseParams.recvCounts[i] != 0 || i == myRank_) {
            HCCL_INFO("[A2AV_AB][SplitPeer] rank=%u peer=%llu aSend=%llu aRecv=%llu bSend=%llu bRecv=%llu "
                      "aSdispl=%llu aRdispl=%llu bSdispl=%llu bRdispl=%llu",
                      myRank_, i, aParams.sendCounts[i], aParams.recvCounts[i],
                      bParams.sendCounts[i], bParams.recvCounts[i],
                      aParams.sdispls[i], aParams.rdispls[i], bParams.sdispls[i], bParams.rdispls[i]);
        }
    }
    aParams.count = aTotalCount;
    bParams.count = bTotalCount;
    aParams.sliceSize = aTotalCount * dataTypeSize_;
    bParams.sliceSize = bTotalCount * dataTypeSize_;
    HCCL_WARNING("[A2AV_AB][Split] rank=%u ratio=%f aCount=%llu bCount=%llu selfA=%llu selfB=%llu",
                 myRank_, ratio, aTotalCount, bTotalCount, aParams.sendCounts[myRank_],
                 bParams.sendCounts[myRank_]);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::GetMaxSendRecvDataCount(
    const TemplateDataParams &params, u64 &maxCount) const
{
    maxCount = 0;
    for (u64 i = 0; i < rankSize_; ++i) {
        maxCount = std::max(maxCount, params.sendCounts[i]);
        maxCount = std::max(maxCount, params.recvCounts[i]);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::CalcMaxDataCountPerLoop(
    const OpParam &param, u64 scratchMultiple, u64 &maxDataCountPerLoop) const
{
    (void)param;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxDataSizePerLoop = transportBoundDataSize;
    if (scratchMultiple != 0) {
        u64 scratchBoundDataSize = maxTmpMemSize_ / scratchMultiple / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    }
    maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    CHK_PRT_RET(maxDataCountPerLoop == 0,
                HCCL_ERROR("[A2AV_AB][CalcMaxDataCountPerLoop] maxDataCountPerLoop is 0. "
                           "maxTmpMemSize=%llu scratchMultiple=%llu dataTypeSize=%u rankSize=%llu",
                           maxTmpMemSize_, scratchMultiple, dataTypeSize_, rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    HCCL_WARNING("[A2AV_AB][CalcMaxDataCountPerLoop] maxDataCountPerLoop=%llu", maxDataCountPerLoop);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::SetLoopParams(
    const TemplateDataParams &srcParams, u64 processedCount, u64 currCount, TemplateDataParams &dstParams) const
{
    dstParams = srcParams;
    u64 loopTotalCount = 0;
    dstParams.buffInfo.hcclBuffBaseOff = 0;
    dstParams.inputSliceStride = currCount * dataTypeSize_;
    dstParams.outputSliceStride = currCount * dataTypeSize_;
    for (u64 i = 0; i < rankSize_; ++i) {
        if (srcParams.sendCounts[i] > processedCount) {
            dstParams.sendCounts[i] = std::min(currCount, srcParams.sendCounts[i] - processedCount);
            dstParams.sdispls[i] = srcParams.sdispls[i] + processedCount;
        } else {
            dstParams.sendCounts[i] = 0;
            dstParams.sdispls[i] = srcParams.sdispls[i] + srcParams.sendCounts[i];
        }

        if (srcParams.recvCounts[i] > processedCount) {
            dstParams.recvCounts[i] = std::min(currCount, srcParams.recvCounts[i] - processedCount);
            dstParams.rdispls[i] = srcParams.rdispls[i] + processedCount;
        } else {
            dstParams.recvCounts[i] = 0;
            dstParams.rdispls[i] = srcParams.rdispls[i] + srcParams.recvCounts[i];
        }
        loopTotalCount += dstParams.sendCounts[i];
    }
    dstParams.count = loopTotalCount;
    dstParams.sliceSize = loopTotalCount * dataTypeSize_;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::RunATemplates(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const TemplateDataParams &aParams)
{
    if (aParams.count == 0) {
        HCCL_WARNING("[A2AV_AB][RunA] skip empty A segment.");
        return HCCL_SUCCESS;
    }
    InsTempAlltoAllMesh2DV3NoMemcpy aIntraTemp(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    InsTempAlltoAllMeshClosV3NoMemcpy aInterTemp(param, resCtx.topoInfo.userRank, interHierarchyInfo_);
    aIntraTemp.SetMeshDimensions(rankSize_, myRank_, rankSizeLevel0_, rankSizeLevel1_);
    aInterTemp.SetMeshDimensions(rankSize_, myRank_, rankSizeLevel0_, rankSizeLevel1_);
    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS || param.engine == CommEngine::COMM_ENGINE_AIV) {
        aInterTemp.SetchannelsPerRank(interLinkMap_);
    }
    TemplateResource aIntraRes;
    aIntraRes.channels = intraLinkMap_;
    aIntraRes.threads = aIntraThreads_;
    aIntraRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    TemplateResource aInterRes;
    aInterRes.channels = interLinkMap_;
    aInterRes.threads = aInterThreads_;
    aInterRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    HCCL_WARNING("[A2AV_AB][RunA][PRE] rank=%u aCount=%llu aSize=%llu intraChannels=%zu interChannels=%zu",
                 myRank_, aParams.count, aParams.sliceSize, intraLinkMap_.size(), interLinkMap_.size());
    CHK_RET(PreSyncInterThreads(mainThread_, {aIntraThreads_[0], aInterThreads_[0]},
                                {aIntraMeta_.notifyNumOnMainThread, aInterMeta_.notifyNumOnMainThread}));
    CHK_RET(aIntraTemp.KernelRun(param, aParams, aIntraRes));
    CHK_RET(aInterTemp.KernelRun(param, aParams, aInterRes));
    CHK_RET(PostSyncInterThreads(mainThread_, {aIntraThreads_[0], aInterThreads_[0]}, {0, 1}));
    HCCL_WARNING("[A2AV_AB][RunA][POST] rank=%u", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::RunBTemplate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const TemplateDataParams &bParams)
{
    if (bParams.count == 0) {
        HCCL_WARNING("[A2AV_AB][RunB] skip empty B segment.");
        return HCCL_SUCCESS;
    }
    InsTempAlltoAllVMesh1D bTemp(param, resCtx.topoInfo.userRank, bRunHierarchyInfo_);
    TemplateResource bRes;
    bRes.channels = fullLinkMap_;
    bRes.threads = bThreads_;
    bRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    u64 maxCount = 0;
    CHK_RET(GetMaxSendRecvDataCount(bParams, maxCount));
    u64 maxDataCountPerLoop = 1;
    CHK_RET(CalcMaxDataCountPerLoop(param, bTemp.CalcScratchMultiple(bParams.buffInfo.inBuffType,
                                                                      bParams.buffInfo.outBuffType),
                                    maxDataCountPerLoop));
    u64 loopTimes = (maxCount + maxDataCountPerLoop - 1) / maxDataCountPerLoop;
    HCCL_WARNING("[A2AV_AB][RunB] maxCount=%llu maxLoopCount=%llu loopTimes=%llu",
                 maxCount, maxDataCountPerLoop, loopTimes);
    CHK_RET(PreSyncInterThreads(mainThread_, {bThreads_[0]}, {bMeta_.notifyNumOnMainThread}));
    u64 processedCount = 0;
    for (u64 loop = 0; loop < loopTimes; ++loop) {
        u64 currCount = (loop == loopTimes - 1) ? (maxCount - processedCount) : maxDataCountPerLoop;
        TemplateDataParams loopParams;
        CHK_RET(SetLoopParams(bParams, processedCount, currCount, loopParams));
        HCCL_WARNING("[A2AV_AB][RunB][LOOP_PRE] rank=%u loop=%llu/%llu processed=%llu curr=%llu "
                     "loopCount=%llu loopSize=%llu fullChannels=%zu",
                     myRank_, loop, loopTimes, processedCount, currCount, loopParams.count,
                     loopParams.sliceSize, fullLinkMap_.size());
        CHK_RET(bTemp.KernelRun(param, loopParams, bRes));
        HCCL_WARNING("[A2AV_AB][RunB][LOOP_POST] rank=%u loop=%llu/%llu", myRank_, loop, loopTimes);
        processedCount += currCount;
    }
    CHK_RET(PostSyncInterThreads(mainThread_, {bThreads_[0]}, {2}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelABExecutor<AlgTopoMatch>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_WARNING("[A2AV_AB][Orchestrate] start alg=%s", param.algName);
    CHK_PRT_RET(param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV || !IsA2AVABAlg(param),
                HCCL_ERROR("[A2AV_AB][Orchestrate] unsupported op/alg. opType=%d alg=%s",
                           param.opType, param.algName),
                HcclResult::HCCL_E_NOT_SUPPORT);
    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataType_ >= HCCL_DATA_TYPE_RESERVED || dataTypeSize_ == 0,
                HCCL_ERROR("[A2AV_AB][Orchestrate] invalid dataType=%d", static_cast<int>(dataType_)),
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
    CHK_RET(PrepareTemplateResources(resCtx));

    TemplateDataParams baseParams;
    CHK_RET(BuildBaseParams(param, resCtx, baseParams));
    TemplateDataParams aParams;
    TemplateDataParams bParams;
    CHK_RET(SplitABParams(baseParams, GetABRatio(), aParams, bParams));
    CHK_RET(RunATemplates(param, resCtx, aParams));
    CHK_RET(RunBTemplate(param, resCtx, bParams));
    HCCL_WARNING("[A2AV_AB][Orchestrate] end.");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV3ABNoMemcpy,
                          InsV2AlltoAllVParallelABExecutor,
                          TopoMatchUBX_V2);

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV3ABNoMemcpyPodUbxV2,
                          InsV2AlltoAllVParallelABExecutor,
                          TopoMatchUBX_V2);

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV3ABNoMemcpyPodDirect,
                          InsV2AlltoAllVParallelABExecutor,
                          TopoMatchAlltoAllPodDirect);

}  // namespace ops_hccl
