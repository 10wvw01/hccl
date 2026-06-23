/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#include "ins_v2_alltoallv_parallel_v2_stage1_no_memcpy_executor.h"
#include <algorithm>
#include <cstring>
#include "ins_temp_all_to_all_v_v2_stage1_no_memcpy.h"
#include "template_utils.h"
#include "topo_match_alltoall_pod_direct.h"
#include "topo_match_ubx_v2.h"

namespace ops_hccl {
namespace {
constexpr u32 TEMPLATE_NUM = 2;
constexpr u32 MIN_TEMPLATE_THREAD_NUM = 1;

bool IsA2AVV2Stage1NoMemcpyAlg(const OpParam &param)
{
    return std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpy") == 0 ||
           std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpyPodUbxV2") == 0 ||
           std::strcmp(param.algName, "InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpyPodDirect") == 0;
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
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
u64 InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    u64 count = 1;
    for (const auto &ranks : vTopo) {
        count *= ranks.size();
    }
    return count;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::BuildHierarchyInfo(
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    CHK_PTR_NULL(topoInfo);
    intraHierarchyInfo_.clear();
    interHierarchyInfo_.clear();
    CHK_PRT_RET(algHierarchyInfo.infos.empty() || algHierarchyInfo.infos[0].empty(),
                HCCL_ERROR("[A2AV_V2_STAGE1_NM][BuildHierarchyInfo] invalid hierarchy."),
                HcclResult::HCCL_E_INTERNAL);
    if (IsSingleLayerUbx4x2(topoInfo)) {
        BuildSingleLayerUbx4x2Hierarchy(topoInfo->userRank, intraHierarchyInfo_, interHierarchyInfo_);
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        CHK_PRT_RET(algHierarchyInfo.infos[0].size() < 2 || algHierarchyInfo.infos[0][0].empty(),
                    HCCL_ERROR("[A2AV_V2_STAGE1_NM][BuildHierarchyInfo] invalid MESH_1D_CLOS hierarchy."),
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
                    HCCL_ERROR("[A2AV_V2_STAGE1_NM][BuildHierarchyInfo] invalid direct hierarchy."),
                    HcclResult::HCCL_E_INTERNAL);
        interHierarchyInfo_ = algHierarchyInfo.infos[1];
    }
    meshSize_ = GetRankSize(intraHierarchyInfo_);
    rankSize_ = topoInfo->userRankSize;
    CHK_PRT_RET(meshSize_ == 0 || rankSize_ % meshSize_ != 0,
                HCCL_ERROR("[A2AV_V2_STAGE1_NM][BuildHierarchyInfo] invalid mesh/rank. rankSize=%llu mesh=%llu",
                           rankSize_, meshSize_),
                HcclResult::HCCL_E_INTERNAL);
    groupNum_ = rankSize_ / meshSize_;
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][BuildHierarchyInfo] rank=%u rankSize=%llu mesh=%llu group=%llu",
                 topoInfo->userRank, rankSize_, meshSize_, groupNum_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    resourceRequest = AlgResourceRequest{};
    CHK_PRT_RET(param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV || !IsA2AVV2Stage1NoMemcpyAlg(param),
                HCCL_ERROR("[A2AV_V2_STAGE1_NM][CalcRes] unsupported op/alg. opType=%d alg=%s",
                           param.opType, param.algName),
                HcclResult::HCCL_E_NOT_SUPPORT);
    myRank_ = topoInfo->userRank;
    CHK_RET(BuildHierarchyInfo(topoInfo, algHierarchyInfo));

    std::vector<HcclChannelDesc> meshChannels;
    std::vector<HcclChannelDesc> closChannels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, intraHierarchyInfo_, meshChannels));
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, interHierarchyInfo_, closChannels));
    CHK_PRT_RET(meshChannels.empty() || closChannels.empty(),
                HCCL_ERROR("[A2AV_V2_STAGE1_NM][CalcRes] empty channel request. mesh=%zu clos=%zu",
                           meshChannels.size(), closChannels.size()),
                HcclResult::HCCL_E_INTERNAL);

    u32 maxPeerNum = rankSize_ > 0 ? static_cast<u32>(rankSize_ - 1) : 0;
    stage0Meta_.slaveThreadNum = std::max(MIN_TEMPLATE_THREAD_NUM, maxPeerNum) - 1;
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
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][CalcRes] rank=%u stage0Slave=%u stage1Slave=%u totalSlave=%u "
                 "meshChannels=%zu closChannels=%zu",
                 myRank_, stage0Meta_.slaveThreadNum, stage1Meta_.slaveThreadNum,
                 resourceRequest.slaveThreadNum, meshChannels.size(), closChannels.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::RestoreChannelMaps(
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
                HCCL_ERROR("[A2AV_V2_STAGE1_NM][RestoreChannelMaps] expected 2 channel levels, got %zu.",
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
                    HCCL_ERROR("[A2AV_V2_STAGE1_NM][RestoreChannelMaps] missing remote a2av info. rank=%u peer=%u",
                               myRank_, item.first),
                    HcclResult::HCCL_E_INTERNAL);
        remoteMaxSendCountsWithoutSelf_[item.first] = channel.remoteAlltoAllVMaxSendCountWithoutSelf;
        remoteCountInfoValid_[item.first] = true;
        HCCL_WARNING("[A2AV_V2_STAGE1_NM][RestoreChannelMaps] rank=%u peer=%u links=%zu remoteMax=%llu",
                     myRank_, item.first, item.second.size(), channel.remoteAlltoAllVMaxSendCountWithoutSelf);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::BuildRuntimeMetas()
{
    u32 threadNum = std::max(MIN_TEMPLATE_THREAD_NUM, static_cast<u32>(allLinkMap_.size()));
    stage0Meta_.slaveThreadNum = threadNum > 0 ? threadNum - 1 : 0;
    stage0Meta_.notifyNumOnMainThread = stage0Meta_.slaveThreadNum;
    stage0Meta_.notifyNumPerThread.assign(stage0Meta_.slaveThreadNum, 1);
    stage1Meta_ = stage0Meta_;
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][BuildRuntimeMetas] rank=%u threadNum=%u peers=%zu",
                 myRank_, threadNum, allLinkMap_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::PrepareTemplateResources(
    const AlgResourceCtxSerializable &resCtx)
{
    threads_ = resCtx.threads;
    u32 stage0ThreadNum = stage0Meta_.slaveThreadNum + 1;
    u32 stage1ThreadNum = stage1Meta_.slaveThreadNum + 1;
    u32 required = 1 + stage0ThreadNum + stage1ThreadNum;
    CHK_PRT_RET(threads_.size() < required,
                HCCL_ERROR("[A2AV_V2_STAGE1_NM][PrepareTemplateResources] threads[%zu] < required[%u].",
                           threads_.size(), required),
                HcclResult::HCCL_E_INTERNAL);
    mainThread_ = threads_[0];
    auto it = threads_.begin() + 1;
    stage0Threads_.assign(it, it + stage0ThreadNum);
    it += stage0ThreadNum;
    stage1Threads_.assign(it, it + stage1ThreadNum);
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][PrepareTemplateResources] rank=%u totalThreads=%zu required=%u "
                 "stage0=%u stage1=%u",
                 myRank_, threads_.size(), required, stage0ThreadNum, stage1ThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::BuildBaseParams(
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
                    HCCL_ERROR("[A2AV_V2_STAGE1_NM][BaseParam] missing remote count info. rank=%u peer=%llu",
                               myRank_, i),
                    HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::GetGlobalMaxSendCount(
    u64 &globalMaxSend) const
{
    globalMaxSend = 0;
    for (u64 count : remoteMaxSendCountsWithoutSelf_) {
        globalMaxSend = std::max(globalMaxSend, count);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::BuildExactSlotOffsets(
    TemplateDataParams &params)
{
    const u64 slotNum = rankSize_ * rankSize_ * 2;
    params.alltoAllVV2SlotOffsets.assign(slotNum, 0);
    exactSlotBytes_ = 0;
    auto slotIndex = [this](u64 srcRank, u64 dstRank, u64 partIdx) {
        return (srcRank * rankSize_ + dstRank) * 2 + partIdx;
    };
    auto splitPair = [](u64 count, u64 &part0, u64 &part1) {
        part0 = count / 2;
        part1 = count - part0;
    };
    for (u64 srcRank = 0; srcRank < rankSize_; ++srcRank) {
        for (u64 dstRank = 0; dstRank < rankSize_; ++dstRank) {
            if (srcRank == dstRank) {
                continue;
            }
            u64 count = 0;
            if (srcRank == myRank_) {
                count = params.sendCounts[dstRank];
            } else {
                auto it = allLinkMap_.find(static_cast<u32>(srcRank));
                CHK_PRT_RET(it == allLinkMap_.end() || it->second.empty() ||
                                it->second[0].remoteAlltoAllVRecvCounts.size() <= dstRank,
                            HCCL_ERROR("[A2AV_V2_STAGE1_NM][ExactSlot] missing recv count. "
                                       "rank=%u src=%llu dst=%llu", myRank_, srcRank, dstRank),
                            HcclResult::HCCL_E_INTERNAL);
                count = it->second[0].remoteAlltoAllVRecvCounts[dstRank];
            }
            u64 part0 = 0;
            u64 part1 = 0;
            splitPair(count, part0, part1);
            u64 parts[2] = {part0, part1};
            for (u64 partIdx = 0; partIdx < 2; ++partIdx) {
                u64 bytes = parts[partIdx] * dataTypeSize_;
                if (bytes == 0) {
                    continue;
                }
                u64 idx = slotIndex(srcRank, dstRank, partIdx);
                params.alltoAllVV2SlotOffsets[idx] = exactSlotBytes_;
                exactSlotBytes_ += AlignUp(bytes, HCCL_MIN_SLICE_ALIGN);
            }
        }
    }
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][ExactSlot] rank=%u compactBytes=%llu slotNum=%llu rankSize=%llu",
                 myRank_, exactSlotBytes_, slotNum, rankSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::CheckScratch(
    const AlgResourceCtxSerializable &resCtx) const
{
    CHK_PRT_RET(exactSlotBytes_ > resCtx.cclMem.size,
                HCCL_ERROR("[A2AV_V2_STAGE1_NM][Scratch] insufficient ccl buffer. rank=%u required=%llu "
                           "cclSize=%llu slotStride=%llu rankSize=%llu",
                           myRank_, exactSlotBytes_, resCtx.cclMem.size, slotStride_, rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::RunStage0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const TemplateDataParams &params)
{
    InsTempAlltoAllVV2Stage1NoMemcpy temp(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    temp.SetV2Stage1NoMemcpyInfo(A2AVV2Stage1NoMemcpyPhase::STAGE0_TO_RELAY,
                                 static_cast<u32>(rankSize_), static_cast<u32>(meshSize_), slotStride_);
    TemplateResource resource;
    resource.channels = allLinkMap_;
    resource.threads = stage0Threads_;
    resource.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][Stage0][PRE] rank=%u count=%llu slotStride=%llu",
                 myRank_, params.count, slotStride_);
    CHK_RET(PreSyncInterThreads(mainThread_, {stage0Threads_[0]}, {stage0Meta_.notifyNumOnMainThread}));
    CHK_RET(temp.KernelRun(param, params, resource));
    CHK_RET(PostSyncInterThreads(mainThread_, {stage0Threads_[0]}, {0}));
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][Stage0][POST] rank=%u", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::RunStage1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const TemplateDataParams &params)
{
    InsTempAlltoAllVV2Stage1NoMemcpy temp(param, resCtx.topoInfo.userRank, interHierarchyInfo_);
    temp.SetV2Stage1NoMemcpyInfo(A2AVV2Stage1NoMemcpyPhase::STAGE1_TO_OUTPUT,
                                 static_cast<u32>(rankSize_), static_cast<u32>(meshSize_), slotStride_);
    TemplateResource resource;
    resource.channels = allLinkMap_;
    resource.threads = stage1Threads_;
    resource.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][Stage1][PRE] rank=%u count=%llu slotStride=%llu",
                 myRank_, params.count, slotStride_);
    CHK_RET(PreSyncInterThreads(mainThread_, {stage1Threads_[0]}, {stage1Meta_.notifyNumOnMainThread}));
    CHK_RET(temp.KernelRun(param, params, resource));
    CHK_RET(PostSyncInterThreads(mainThread_, {stage1Threads_[0]}, {1}));
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][Stage1][POST] rank=%u", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor<AlgTopoMatch>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][Orchestrate] start alg=%s", param.algName);
    CHK_PRT_RET(param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV || !IsA2AVV2Stage1NoMemcpyAlg(param),
                HCCL_ERROR("[A2AV_V2_STAGE1_NM][Orchestrate] unsupported op/alg. opType=%d alg=%s",
                           param.opType, param.algName),
                HcclResult::HCCL_E_NOT_SUPPORT);
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataType_ >= HCCL_DATA_TYPE_RESERVED || dataTypeSize_ == 0,
                HCCL_ERROR("[A2AV_V2_STAGE1_NM][Orchestrate] invalid dataType=%d", static_cast<int>(dataType_)),
                HcclResult::HCCL_E_INTERNAL);
    CHK_RET(BuildHierarchyInfo(&resCtx.topoInfo, resCtx.algHierarchyInfo));
    CHK_RET(RestoreChannelMaps(resCtx));
    CHK_RET(BuildRuntimeMetas());
    CHK_RET(PrepareTemplateResources(resCtx));
    TemplateDataParams params;
    CHK_RET(BuildBaseParams(param, resCtx, params));
    dataCount_ = params.count;
    u64 globalMaxSend = 0;
    CHK_RET(GetGlobalMaxSendCount(globalMaxSend));
    for (u64 count : params.sendCounts) {
        globalMaxSend = std::max(globalMaxSend, count);
    }
    slotStride_ = AlignUp((globalMaxSend + 1) / 2 * dataTypeSize_, HCCL_MIN_SLICE_ALIGN);
    if (slotStride_ == 0) {
        slotStride_ = HCCL_MIN_SLICE_ALIGN;
    }
    CHK_RET(BuildExactSlotOffsets(params));
    CHK_RET(CheckScratch(resCtx));
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][Orchestrate] rank=%u total=%llu globalMax=%llu "
                 "fallbackSlotStride=%llu exactSlotBytes=%llu",
                 myRank_, dataCount_, globalMaxSend, slotStride_, exactSlotBytes_);
    CHK_RET(RunStage0(param, resCtx, params));
    CHK_RET(RunStage1(param, resCtx, params));
    HCCL_WARNING("[A2AV_V2_STAGE1_NM][Orchestrate] end.");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpy,
                          InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor,
                          TopoMatchUBX_V2);

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpyPodUbxV2,
                          InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor,
                          TopoMatchUBX_V2);

REGISTER_EXECUTOR_BY_TOPO(HcclCMDType::HCCL_CMD_ALLTOALLV,
                          InsAlltoAllVParallelMesh2DClosV2Stage1NoMemcpyPodDirect,
                          InsV2AlltoAllVParallelV2Stage1NoMemcpyExecutor,
                          TopoMatchAlltoAllPodDirect);

} // namespace ops_hccl
