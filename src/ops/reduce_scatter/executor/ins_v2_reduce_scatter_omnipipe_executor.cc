/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_omnipipe_executor.h"
#include "topo_match_3_level.h"
#include "ins_temp_reduce_scatter_omnipipe_mesh_1D.h"
#include "ins_temp_reduce_scatter_omnipipe_mesh_1d_dpu.h"
#include "ins_temp_reduce_scatter_omnipipe_nhr.h"
#include "topo_match_pcie_mix.h"
#include "omnipipe_template_utils.h"
#include "alg_data_trans_wrapper.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "hccl_sym_win.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
namespace ops_hccl {
constexpr uint32_t HIERARCHY_SIZE_3 = 3;
constexpr uint64_t RANK_SIZE_LEVEL_2 = 2;
constexpr uint64_t RANK_SIZE_LEVEL_4 = 4;

namespace {
std::vector<std::vector<u32>> BuildFullRankSubComm(u32 rankSize)
{
    std::vector<std::vector<u32>> subCommInfo(1);
    subCommInfo[0].reserve(rankSize);
    for (u32 rank = 0; rank < rankSize; ++rank) {
        subCommInfo[0].push_back(rank);
    }
    return subCommInfo;
}
} // namespace

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                   InsAlgTemplate2>::InsV2ReduceScatterOmniPipeExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[%s]myRank[%u] userRankSize[%u] devType[%u] redOp[%u] dataType[%u] dataTypeSize[%u]", __func__, myRank_,
              rankSize_, devType_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

// 实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                   InsAlgTemplate2>::CalcAlgHierarchyInfo(HcclComm comm,
                                                                          TopoInfoWithNetLayerDetails* topoInfo,
                                                                          AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                   InsAlgTemplate2>::CalcSymmetricDirectRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest) const
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.channels.clear();

    if (rankSize_ <= 1) {
        HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][CalcSymmetricDirectRes] rankSize[%u], no peer channel needed.",
                  static_cast<u32>(rankSize_));
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> channels;
    std::vector<std::vector<u32>> fullSubCommInfo = BuildFullRankSubComm(static_cast<u32>(rankSize_));
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, fullSubCommInfo, channels));
    resourceRequest.channels.push_back(channels);
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][CalcSymmetricDirectRes] rankSize[%u], channels[%u]",
              static_cast<u32>(rankSize_), static_cast<u32>(channels.size()));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::BuildSubCommAndTempMap(
    const OpParam& param,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    std::vector<std::vector<u32>>& subCommRanks0,
    std::vector<std::vector<u32>>& subCommRanks1,
    std::vector<std::vector<u32>>& subCommRanks2,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    subCommRanks0.clear();
    subCommRanks1.clear();
    subCommRanks2.clear();
    tempMap.clear();

    HCCL_INFO("[BuildSubCommAndTempMap]infos,%s", ThreeDVecToStrOmni(algHierarchyInfo_.infos).c_str());
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<u32> closRanks;
        if (!algHierarchyInfo_.infos[0].empty() && !algHierarchyInfo_.infos[0][0].empty()) {
            subCommRanks0 = {algHierarchyInfo_.infos[0][0]};
            u32 meshSize = algHierarchyInfo_.infos[0][0].size();
            if (!algHierarchyInfo_.infos[0][1].empty()) {
                for (auto rank : algHierarchyInfo_.infos[0][1]) {
                    if (rank % meshSize == topoInfo->userRank % meshSize) {
                        closRanks.push_back(rank);
                    }
                }
            }
        }
        subCommRanks1 = {closRanks};
        omniNeedSetStepNum_ = (subCommRanks1[0].size() == RANK_SIZE_LEVEL_4) ? OmniNeedSetStepNum::OMNIPIPE_UBX_16P
                                                        : OmniNeedSetStepNum::OMNIPIPE_DEFAULT;
        if (!algHierarchyInfo_.infos[1].empty()){
            subCommRanks2 = algHierarchyInfo_.infos[1];
        } else {
            subCommRanks2.emplace_back(std::vector<u32>{myRank_});
        }
    } else if (topoType_ == TopoType::THREE_LEVEL) {
        if (!algHierarchyInfo.infos[0].empty() && !algHierarchyInfo.infos[0][0].empty()) {
            subCommRanks0.push_back(algHierarchyInfo.infos[0][0]);
        } else {
            subCommRanks0.emplace_back(std::vector<u32>{myRank_});
        }
        if (!algHierarchyInfo.infos[1].empty() && !algHierarchyInfo.infos[1][0].empty()) {
            subCommRanks1.push_back(algHierarchyInfo.infos[1][0]);
        } else {
            subCommRanks1.emplace_back(std::vector<u32>{myRank_});
        }
        if (!algHierarchyInfo.infos[2].empty() && !algHierarchyInfo.infos[2][0].empty()) {
            subCommRanks2.push_back(algHierarchyInfo.infos[2][0]);
        } else {
            subCommRanks2.emplace_back(std::vector<u32>{myRank_});
        }
    } else {
        if (!algHierarchyInfo_.infos[0].empty()) {
            subCommRanks0 = algHierarchyInfo_.infos[0];
        }
        if (!algHierarchyInfo_.infos[1].empty()) {
            subCommRanks1 = algHierarchyInfo_.infos[1];
        }
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }

    rankSizeLevel0_ = subCommRanks0[0].size();
    rankSizeLevel1_ = subCommRanks1[0].size();
    rankSizeLevel2_ = subCommRanks2[0].size();

    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ % (rankSizeLevel0_ * rankSizeLevel1_) / rankSizeLevel0_;
    rankIdxLevel2_ = myRank_ / (rankSizeLevel0_ * rankSizeLevel1_);

    if (rankSizeLevel0_ > 1) {
        tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel1_ > 1) {
        tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
    }
    if (rankSizeLevel2_ > 1) {
        tempMap[OMNIPIPE_LEVEL2] = std::make_shared<InsAlgTemplate2>(param, myRank_, subCommRanks2);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("CalcRes");
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);

    if (algHierarchyInfo_.infos.size() == HIERARCHY_SIZE_3 &&
        !algHierarchyInfo_.infos[2].empty() && !algHierarchyInfo_.infos[2][0].empty()) {
        topoType_ = TopoType::THREE_LEVEL;
    } else {
        topoType_ = TopoType::UBX_2LEVEL;
    }

    if (param.supportSymmetricMemory) {
        return CalcSymmetricDirectRes(comm, param, topoInfo, resourceRequest);
    }

    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo,
            subCommRanks0, subCommRanks1, subCommRanks2, tempMap, topoInfo));

    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    // 对称内存要求所有层的 channel 合并到一次 HcclChannelAcquire（pending 一次性交换/回填），
    // 否则多层多次建链只有首层 peer 会被回填，后续层 remoteMems 为空。此处合并进唯一 channels[0]。
    resourceRequest.channels.clear();

    for (auto& temp : tempMap) {
        AlgResourceRequest resReqlevel;
        CHK_RET(temp.second->CalcRes(comm, param, topoInfo, resReqlevel));
        resourceRequest.slaveThreadNum += 1 + resReqlevel.slaveThreadNum;
        resourceRequest.notifyNumPerThread.emplace_back(resReqlevel.notifyNumOnMainThread + 1);
        resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                                  resReqlevel.notifyNumPerThread.begin(),
                                                  resReqlevel.notifyNumPerThread.end());
        resourceRequest.notifyNumOnMainThread++;
        if (!resReqlevel.channels.empty()) {
            if (resourceRequest.channels.empty()) {
                resourceRequest.channels.resize(1);
            }
            resourceRequest.channels[0].insert(resourceRequest.channels[0].end(),
                resReqlevel.channels[0].begin(), resReqlevel.channels[0].end());
        }
    }

    return HCCL_SUCCESS;
}

// 该函数必须按照level0、level1、level2的顺序调用
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    PrepareResForTemplateLevel(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    u32 levelThreadNum = tempBase->GetThreadNum();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreads_[OMNIPIPE_LEVEL0].assign(threads_.begin() + 1, threads_.begin() + 1 + levelThreadNum);
        tempMainThreadsLevel01_.push_back(levelThreads_[0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreads_[OMNIPIPE_LEVEL1].assign(threads_.begin() + 1 + levelThreads_[0].size(),
                                              threads_.begin() + 1 + levelThreads_[0].size() + levelThreadNum);
        tempMainThreadsLevel01_.push_back(levelThreads_[1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreads_[OMNIPIPE_LEVEL2].assign(threads_.begin() + 1 + levelThreads_[0].size() + levelThreads_[1].size(),
                                              threads_.end());
        tempMainThreadsLevel2_.push_back(levelThreads_[OMNIPIPE_LEVEL2].at(0));
    }

    // 获取当前template各自的主thread上有多少notify
    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    if (level < OMNIPIPE_LEVEL2) {
        notifyIdxCtrlToTempLevel01_.push_back(levelTempRequest.notifyNumOnMainThread);
        notifyIdxTempToCtrlLevel01_.push_back(tempMainThreadsLevel01_.size() + tempMainThreadsLevel2_.size() - 1);
    } else {
        notifyIdxCtrlToTempLevel2_.push_back(levelTempRequest.notifyNumOnMainThread);
        notifyIdxTempToCtrlLevel2_.push_back(tempMainThreadsLevel01_.size() + tempMainThreadsLevel2_.size() - 1);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    threads_ = resCtx.threads;
    controlThread_ = threads_.at(0);

    if (param.supportSymmetricMemory) {
        HcclResult ret = OrchestrateSymmetricDirect(param, resCtx);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[InsV2ReduceScatterOmniPipeExecutor][Orchestrate]errNo[0x%016llx] symmetric "
                               "reduce scatter direct kernel run failed",
                               HCCL_ERROR_CODE(ret)),
                    ret);
        return HCCL_SUCCESS;
    }
    
    if (algHierarchyInfo_.infos.size() == HIERARCHY_SIZE_3 &&
        !algHierarchyInfo_.infos[2].empty() && !algHierarchyInfo_.infos[2][0].empty()) {
        topoType_ = TopoType::THREE_LEVEL;
    } else {
        topoType_ = TopoType::UBX_2LEVEL;
    }

    // 计算subCommRanks和template
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo_,
            subCommRanks0, subCommRanks1, subCommRanks2, tempMap, &resCtx.topoInfo));
    // 保存各层 rank 列表，供 RestoreChannelMap 按 remoteRank 归层使用（channels 已扁平合并为一次建链）
    subCommRanks0_ = subCommRanks0;
    subCommRanks1_ = subCommRanks1;
    subCommRanks2_ = subCommRanks2;
    threads_ = resCtx.threads;
    controlThread_ = threads_.at(0);
    levelThreads_.resize(OMNIPIPE_LEVEL_NUM);

    // 先初始化remoteRankToChannelInfo_，然后为nhr赋值多channel，最后再计算资源，这样计算线程资源的时候就能获取到多channel需要的线程数
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    for (u32 level = 0; level < remoteRankToChannelInfo_.size(); ++level) {
        u32 channelCount = 0;
        for (const auto& rankChannels : remoteRankToChannelInfo_[level]) {
            channelCount += rankChannels.second.size();
        }
        HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][Orchestrate] level[%u] remoteRanks[%u] channels[%u]",
            level, remoteRankToChannelInfo_[level].size(), channelCount);
    }
    // todo 这边写死了
    if (rankSizeLevel1_ > 1) {
        tempMap[OMNIPIPE_LEVEL1]->SetchannelsPerRank(remoteRankToChannelInfo_[1]);
    }

    for (auto& temp : tempMap) {
        CHK_RET(PrepareResForTemplateLevel(temp.first, temp.second));
    }

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, tempMap);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[InsV2ReduceScatterOmniPipeExecutor][Orchestrate]errNo[0x%016llx] Reduce scatter executor "
                           "kernel run failed",
                           HCCL_ERROR_CODE(ret)),
                ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::RestoreChannelMap(
    const AlgResourceCtxSerializable& resCtx, std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const
{
    rankIdToChannelInfo.resize(OMNIPIPE_LEVEL_NUM);
    if (resCtx.channels.empty()) {
        return HCCL_SUCCESS;
    }
    // channels 已扁平合并到 channels[0]（一次建链）。按 remoteRank 归到对应 Omnipipe 层
    // （同一对端 rank 只会出现在一层子通信域里）。
    auto contains = [](const std::vector<u32>& group, u32 r) -> bool {
        for (u32 v : group) { if (v == r) { return true; } }
        return false;
    };
    auto tryLevel = [&](u32 i, uint64_t rankSize, const std::vector<std::vector<u32>>& subComms,
                        u32 remoteRank, const ChannelInfo& ch) -> bool {
        if (rankSize <= 1) { return false; }
        for (const auto& group : subComms) {
            if (contains(group, static_cast<u32>(myRank_)) && contains(group, remoteRank)) {
                rankIdToChannelInfo[i][remoteRank].push_back(ch);
                return true;
            }
        }
        return false;
    };
    for (const auto& channel : resCtx.channels[0]) {
        u32 rr = channel.remoteRank;
        if (tryLevel(OMNIPIPE_LEVEL0, rankSizeLevel0_, subCommRanks0_, rr, channel)) { continue; }
        if (tryLevel(OMNIPIPE_LEVEL1, rankSizeLevel1_, subCommRanks1_, rr, channel)) { continue; }
        if (tryLevel(OMNIPIPE_LEVEL2, rankSizeLevel2_, subCommRanks2_, rr, channel)) { continue; }
        HCCL_WARNING("[RestoreChannelMap] remoteRank[%u] not found in any active level, dropped.", rr);
    }
    return HCCL_SUCCESS;
}

// Symmetric memory RS directly reduces peer input slices into local user output.
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                   InsAlgTemplate2>::RestoreSymmetricDirectChannelMap(
    const AlgResourceCtxSerializable& resCtx, std::map<u32, std::vector<ChannelInfo>>& rankIdToChannelInfo) const
{
    rankIdToChannelInfo.clear();
    if (resCtx.channels.empty()) {
        return HCCL_SUCCESS;
    }
    for (const auto& channel : resCtx.channels[0]) {
        if (channel.remoteRank == myRank_) {
            continue;
        }
        rankIdToChannelInfo[channel.remoteRank].push_back(channel);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                   InsAlgTemplate2>::OrchestrateSymmetricDirect(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateSymmetricDirect] Start");
    if (dataCount_ == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(param.inputPtr == nullptr || param.outputPtr == nullptr,
                HCCL_ERROR("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateSymmetricDirect] input/output ptr is null, "
                           "input[%p], output[%p]",
                           param.inputPtr, param.outputPtr),
                HcclResult::HCCL_E_PARA);
    CHK_PRT_RET(rankSize_ > 1 && param.inputSymWindow == nullptr,
                HCCL_ERROR("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateSymmetricDirect] inputSymWindow is null"),
                HcclResult::HCCL_E_INTERNAL);

    std::map<u32, std::vector<ChannelInfo>> remoteRankToChannelInfo;
    CHK_RET(RestoreSymmetricDirectChannelMap(resCtx, remoteRankToChannelInfo));
    if (rankSize_ > 1) {
        CHK_PRT_RET(remoteRankToChannelInfo.empty(),
                    HCCL_ERROR("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateSymmetricDirect] no peer channel"),
                    HcclResult::HCCL_E_INTERNAL);
    }

    const u64 sliceBytes = dataCount_ * dataTypeSize_;
    const u64 myInputOffset = static_cast<u64>(myRank_) * sliceBytes;
    DataSlice localSrc(param.inputPtr, myInputOffset, sliceBytes, dataCount_);
    DataSlice localDst(param.outputPtr, 0, sliceBytes, dataCount_);
    CHK_RET(LocalCopy(controlThread_, localSrc, localDst));

    for (u32 remoteRank = 0; remoteRank < static_cast<u32>(rankSize_); ++remoteRank) {
        if (remoteRank == myRank_) {
            continue;
        }
        auto channelIter = remoteRankToChannelInfo.find(remoteRank);
        CHK_PRT_RET(channelIter == remoteRankToChannelInfo.end() || channelIter->second.empty(),
                    HCCL_ERROR("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateSymmetricDirect] missing channel for "
                               "remoteRank[%u]",
                               remoteRank),
                    HcclResult::HCCL_E_INTERNAL);

        void* remoteInput = nullptr;
        HcclResult ret = HcclSymWinGetPeerPointer(param.inputSymWindow, param.inputOffset, remoteRank, &remoteInput);
        CHK_PRT_RET(ret != HCCL_SUCCESS || remoteInput == nullptr,
                    HCCL_ERROR("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateSymmetricDirect] "
                               "HcclSymWinGetPeerPointer failed, remoteRank[%u], ret[%d], input[%p]",
                               remoteRank, ret, remoteInput),
                    ret != HCCL_SUCCESS ? ret : HcclResult::HCCL_E_INTERNAL);

        std::vector<DataSlice> emptySlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        rxSrcSlices.emplace_back(remoteInput, myInputOffset, sliceBytes, dataCount_);
        rxDstSlices.emplace_back(param.outputPtr, 0, sliceBytes, dataCount_);
        SlicesList txSlices(emptySlices, emptySlices);
        SlicesList rxSlices(rxSrcSlices, rxDstSlices);
        TxRxSlicesList sendRecvSlices(txSlices, rxSlices);
        const ChannelInfo& linkRemote = channelIter->second[0];
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvReduceInfo sendRecvInfo(sendRecvChannels, sendRecvSlices, dataType_, reduceOp_);
        HcclResult reduceRet = SendRecvReadReduce(sendRecvInfo, controlThread_);
        CHK_PRT_RET(reduceRet != HCCL_SUCCESS,
                    HCCL_ERROR("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateSymmetricDirect] "
                               "SendRecvReadReduce failed, remoteRank[%u], ret[%d]",
                               remoteRank, reduceRet),
                    reduceRet);
    }
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateSymmetricDirect] End");
    return HCCL_SUCCESS;
#else
    (void)param;
    (void)resCtx;
    HCCL_ERROR("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateSymmetricDirect] symmetric memory is not supported "
               "before CANN 9.0.0");
    return HcclResult::HCCL_E_NOT_SUPPORT;
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                   InsAlgTemplate2>::GenTemplateAlgParamsByDimData(TemplateDataParams& tempAlgParams,
                                                                                   const StepSliceInfo& stepSliceInfo) const
{
    // rs特殊处理，过程中的所有step都在ccl中进行数据搬运，在template中只使用ccl的起始地址就可以了，in和out不用赋值
    CHK_RET(FillOmniPipeTemplateAlgParams(tempAlgParams, stepSliceInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap)
{
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] Start");
    // 1.计算带宽
    double bw_rs_l0 = BW_OMNI_DEFAULT;
    double bw_rs_l1 = BW_OMNI_DEFAULT;
    double bw_rs_l2 = BW_OMNI_DEFAULT;

    if (resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel1_==RANK_SIZE_LEVEL_2) {
            bw_rs_l1 = BW_OMNI_PCIE_EIGHT_RS_CLOS;
        } else if (rankSizeLevel1_==RANK_SIZE_LEVEL_4) {
            bw_rs_l1 = BW_OMNI_PCIE_SIXTEEN_RS_CLOS;
        }
    } else if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS) {
        bw_rs_l1 = BW_OMNI_UBX_RS_CLOS;
    }

    //计算等价带宽
    double eqBw0 = bw_rs_l0;//L0 mesh
    double eqBw1 = bw_rs_l1;//L1 NHR
    double eqBw2 = bw_rs_l2;//L2 NHR

    //level0为mesh,等价mesh为其本身
    //level1为nhr
    //level2, ranksize = 1
    eqBw1 = rankSizeLevel1_ > 1 ? eqBw1 / (rankSizeLevel1_ - 1) : eqBw1;
    eqBw2 = rankSizeLevel2_ > 1 ? eqBw2 / (rankSizeLevel2_ - 1) : eqBw2;

    std::vector<double> endpointAttrBwNew{eqBw0, eqBw1, eqBw2};

    // 2、计算scratch 返回的数组0是maxCountPerloop, 1是loopTimes
    OmniPipeScratchParam scratchParam;
    scratchParam.endpointAttrBw = endpointAttrBwNew;
    scratchParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    std::vector<u64> levelAlgType;
    (tempMap.count(OMNIPIPE_LEVEL0) > 0) ?
        levelAlgType.push_back(tempMap[OMNIPIPE_LEVEL0]->CalcScratchMultiple(BufferType::DEFAULT, BufferType::DEFAULT)) :
        levelAlgType.push_back(0);
    (tempMap.count(OMNIPIPE_LEVEL1) > 0) ?
        levelAlgType.push_back(tempMap[OMNIPIPE_LEVEL1]->CalcScratchMultiple(BufferType::DEFAULT, BufferType::DEFAULT)) :
        levelAlgType.push_back(0);
    (tempMap.count(OMNIPIPE_LEVEL2) > 0) ?
        levelAlgType.push_back(tempMap[OMNIPIPE_LEVEL2]->CalcScratchMultiple(BufferType::DEFAULT, BufferType::DEFAULT)) :
        levelAlgType.push_back(0);
    scratchParam.levelAlgType = levelAlgType;
    // 手动转成数组，这边只给reducescatter用
    std::vector<u64> dataSizeVec;
    for (int i = 0; i < rankSize_; i++) {
        dataSizeVec.push_back(dataSize_);
    }
    scratchParam.dataSize = dataSizeVec;
    scratchParam.dataTypeSize = dataTypeSize_;
    scratchParam.maxTmpMemSize = resCtx.cclMem.size;
    scratchParam.opMode = param.opMode;
    scratchParam.engine = param.engine;
    std::vector<u64> loopInfo = CalcOmniPipeScratchInfo(scratchParam);
    u64 maxCountPerLoop = loopInfo[0];
    u64 loopTimes = loopInfo[1];

    // 3、计算n-1次loop的slice信息
    OmniPipeSliceParam sliceParam;
    std::vector<u64> dataSizePerLoop;
    std::vector<u64> dataWholeSize;
    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;

    for (int i = 0; i < rankSize_; i++) {
        dataSizePerLoop.push_back(perLoopSize);
        dataWholeSize.push_back(perLoopSize);
    }
    sliceParam.dataSizePerLoop = dataSizePerLoop;
    sliceParam.dataWholeSize = dataWholeSize;  // rs这个值和peerloop一致，已对齐
    sliceParam.endpointAttrBw = endpointAttrBwNew;
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, rankIdxLevel2_};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = param.engine;
    sliceParam.needSetStepNum = omniNeedSetStepNum_;
    OmniPipeSliceInfo alignSliceInfo = CalcRSOmniPipeSliceInfo(sliceParam);

    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        std::vector<u64> dataSizePerLoop;
        std::vector<u64> dataWholeSize;
        u64 perLoopSize = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        for (int i = 0; i < rankSize_; i++) {
            dataSizePerLoop.push_back(perLoopSize);
            dataWholeSize.push_back(perLoopSize);
        }
        sliceParam.dataSizePerLoop = dataSizePerLoop;
        sliceParam.dataWholeSize = dataWholeSize;
        tailSliceInfo = CalcRSOmniPipeSliceInfo(sliceParam);
    }

    u64 processedDataCount = 0;
    OmniPipeSliceInfo omnipipeSliceInfo;
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop]loopTimes = [%u]", loopTimes);
    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;
    for (auto& temp : tempMap) {
        tempResMap[temp.first].channels = remoteRankToChannelInfo_[temp.first];
        tempResMap[temp.first].threads = levelThreads_[temp.first];
        tempResMap[temp.first].npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
        tempResMap[temp.first].dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
        tempAlgParamMap[temp.first].buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamMap[temp.first].buffInfo.inputPtr = param.inputPtr;
        tempAlgParamMap[temp.first].buffInfo.outputPtr = param.outputPtr;
        tempAlgParamMap[temp.first].supportSymmetricMemory = false;
    }
    // RS can read from peer user-input only before any partial reduction has been produced.
    // Later levels must consume the ccl-buffer partial result from the previous level.
    if (param.supportSymmetricMemory) {
        if (rankSizeLevel2_ > 1 && tempAlgParamMap.count(OMNIPIPE_LEVEL2) > 0) {
            tempAlgParamMap[OMNIPIPE_LEVEL2].supportSymmetricMemory = true;
        } else if (rankSizeLevel0_ > 1 && tempAlgParamMap.count(OMNIPIPE_LEVEL0) > 0) {
            tempAlgParamMap[OMNIPIPE_LEVEL0].supportSymmetricMemory = true;
        } else if (rankSizeLevel1_ > 1 && tempAlgParamMap.count(OMNIPIPE_LEVEL1) > 0) {
            tempAlgParamMap[OMNIPIPE_LEVEL1].supportSymmetricMemory = true;
        }
    }

    TemplateDataParams tempParamLocalcopy;
    tempParamLocalcopy.buffInfo.hcclBuff = resCtx.cclMem;
    tempParamLocalcopy.buffInfo.inputPtr = param.inputPtr;
    tempParamLocalcopy.buffInfo.outputPtr = param.outputPtr;
    // 5、进行一次loop的数据处理
    for (u64 loop = 0; loop < loopTimes; loop++) {
        // localcopy前同步，这里默认xy不会同时为1，因此使用01的线程组处理localcopy的前后同步
        CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxCtrlToTempLevel01_));
        // 5.1 RS在每次loop进行之前先将所有数据从usrin拷贝到ccl，2.3已修改下列参数
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        auto loopSize = currDataCount * dataTypeSize_;
        for (auto& temp : tempAlgParamMap) {
            temp.second.processedDataCount = processedDataCount;
            temp.second.inputSliceStride = dataCount_ * dataTypeSize_;
            temp.second.outputSliceStride = loopSize;
        }
        const bool level2Symmetric = tempAlgParamMap.count(OMNIPIPE_LEVEL2) > 0 &&
            tempAlgParamMap[OMNIPIPE_LEVEL2].supportSymmetricMemory;
        const bool level0Symmetric = tempAlgParamMap.count(OMNIPIPE_LEVEL0) > 0 &&
            tempAlgParamMap[OMNIPIPE_LEVEL0].supportSymmetricMemory;
        const bool level1Symmetric = tempAlgParamMap.count(OMNIPIPE_LEVEL1) > 0 &&
            tempAlgParamMap[OMNIPIPE_LEVEL1].supportSymmetricMemory;
        HCCL_INFO("[ReduceScatterOmniPipeSymmetricAddrCheck][ExecutorLoop] loop[%llu] currDataCount[%llu] "
                  "processedDataCount[%llu] dataCount[%llu] dataTypeSize[%u] inputSliceStride[%llu] "
                  "compactLoopStride[%llu] supportSymmetricMemory[%d] level2Symmetric[%d] level0Symmetric[%d] "
                  "level1Symmetric[%d]",
                  loop, currDataCount, processedDataCount, dataCount_, dataTypeSize_,
                  dataCount_ * dataTypeSize_, loopSize, param.supportSymmetricMemory,
                  level2Symmetric, level0Symmetric, level1Symmetric);
        tempParamLocalcopy.buffInfo.inBuffType = BufferType::INPUT;
        tempParamLocalcopy.count = currDataCount;
        tempParamLocalcopy.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempParamLocalcopy.inputSliceStride = dataCount_ * dataTypeSize_;
        tempParamLocalcopy.buffInfo.outBuffBaseOff = 0;
        tempParamLocalcopy.outputSliceStride = loopSize;
        tempParamLocalcopy.repeatNum = rankSize_;
        tempParamLocalcopy.sliceSize = loopSize;
        // 这边不论三层为什么拓扑，都使用第一个template去做localcopy
        if (rankSizeLevel0_ > 1) {
            auto temp0 = std::dynamic_pointer_cast<InsAlgTemplate0>(tempMap.begin()->second);
            CHK_RET(temp0->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
        } else {
            auto temp1 = std::dynamic_pointer_cast<InsAlgTemplate1>(tempMap.begin()->second);
            CHK_RET(temp1->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
        }
        // localcopy后同步
        CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxTempToCtrlLevel01_));

        // 5.2 确定当前是前n-1次loop的slice结果，还是存在尾块时最后一次loop的slice结果
        if (loop == loopTimes - 1 && !tailSliceInfo.isEmpty()) {
            omnipipeSliceInfo = tailSliceInfo;
        } else {
            omnipipeSliceInfo = alignSliceInfo;
        }
        u32 level2StepCount = omnipipeSliceInfo.dataSliceLevel2.size();
        u32 level0StepCount = omnipipeSliceInfo.dataSliceLevel0.size() / omnipipeSliceInfo.dataSliceLevel2.size();
        HCCL_INFO(
            "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop]level2 step count = [%u], level0 step count = [%u]",
            level2StepCount, level0StepCount);

        // 5.3 for外层2d
        u32 axisReduceId = 0;  // 轴间reduce从计算slice的结果中获取
        std::vector<TemplateDataParams> axisReduceTempParams;
        for (int i = 0; i < level2StepCount; i++) {
            HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop]Step [%u] in level2", i);
            if (rankSizeLevel2_ > 1) {
                HCCL_DEBUG("rankSizeLevel2_ > 1");
                GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_LEVEL2], omnipipeSliceInfo.dataSliceLevel2[i]);
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel2_, notifyIdxCtrlToTempLevel2_));
                CHK_RET(tempMap[OMNIPIPE_LEVEL2]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL2], tempResMap[OMNIPIPE_LEVEL2]));
            }
            // 5.4 for内层2d
            for (int j = 0; j < level0StepCount; j++) {
                // level0、1前同步
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxCtrlToTempLevel01_));
                // 初始化并执行机内template任务
                if (rankSizeLevel0_ > 1) {
                    HCCL_DEBUG("rankSizeLevel0_ > 1");
                    CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[0],
                                                          omnipipeSliceInfo.dataSliceLevel0[i * level0StepCount + j]));
                    CHK_RET(
                        tempMap[0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL0], tempResMap[OMNIPIPE_LEVEL0]));
                }
                if (rankSizeLevel1_ > 1) {
                    HCCL_DEBUG("rankSizeLevel1_ > 1");
                    CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[1],
                                                          omnipipeSliceInfo.dataSliceLevel1[i * level0StepCount + j]));
                    CHK_RET(
                        tempMap[1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL1], tempResMap[OMNIPIPE_LEVEL1]));
                }
                // level0、1尾同步
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxTempToCtrlLevel01_));
            }
            if (rankSizeLevel2_ > 1) {
                // z轴尾同步
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel2_, notifyIdxTempToCtrlLevel2_));
                HCCL_DEBUG("PostSyncInterThreads z success.");
            }
        }
        // localcopy前同步
        CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxCtrlToTempLevel01_));
        // 5.5 将当前这个loop在ccl中的数据一次性拷贝到userout中
        tempParamLocalcopy.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempParamLocalcopy.buffInfo.inBuffBaseOff = loopSize * myRank_;
        tempParamLocalcopy.inputSliceStride = 0;
        tempParamLocalcopy.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempParamLocalcopy.outputSliceStride = 0;
        tempParamLocalcopy.sliceSize = loopSize;  // 尾拷贝数据量变成1/rankSize
        // repeat=1，temp内部已经没有和rankid相关的处理
        tempParamLocalcopy.repeatNum = 1;
        if (rankSizeLevel0_ > 1) {
            auto temp0 = std::dynamic_pointer_cast<InsAlgTemplate0>(tempMap.begin()->second);
            CHK_RET(temp0->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
        } else {
            auto temp1 = std::dynamic_pointer_cast<InsAlgTemplate1>(tempMap.begin()->second);
            CHK_RET(temp1->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
        }
        // localcopy后同步
        CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxTempToCtrlLevel01_));
        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, InsV2ReduceScatterOmniPipeMultilevel,
                       InsV2ReduceScatterOmniPipeExecutor, TopoMatchMultilevel, InsTempReduceScatterOmniPipeMesh1D,
                       InsTempReduceScatterOmniPipeNHR, InsTempReduceScatterOmniPipeMesh1dDpu);
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, InsV2ReduceScatterOmniPipePcie,
                       InsV2ReduceScatterOmniPipeExecutor, TopoMatchPcieMix, InsTempReduceScatterOmniPipeMesh1D,
                       InsTempReduceScatterOmniPipeNHR, InsTempReduceScatterOmniPipeMesh1dDpu);
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, InsV2ReduceScatterOmniPipe,
                       InsV2ReduceScatterOmniPipeExecutor, TopoMatchUBX, InsTempReduceScatterOmniPipeMesh1D,
                       InsTempReduceScatterOmniPipeMesh1D, InsTempReduceScatterOmniPipeMesh1dDpu);

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, InsV2ReduceScatterOmniPipeUboe,
                       InsV2ReduceScatterOmniPipeExecutor, TopoMatch3Level, InsTempReduceScatterOmniPipeMesh1D,
                       InsTempReduceScatterOmniPipeNHR, InsTempReduceScatterOmniPipeMesh1D);

}  // namespace ops_hccl
