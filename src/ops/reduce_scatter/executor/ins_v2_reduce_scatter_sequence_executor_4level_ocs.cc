/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_sequence_executor_4level_ocs.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_mesh_1D_ocs.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 4;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
InsV2ReduceScatterSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::InsV2ReduceScatterSequenceExecutor4LevelOCS()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2ReduceScatterSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS][InitCommInfo] myRank [%u], rankSize [%u], redOp [%u], "
        "dataType [%u] dataTypeSize [%u]", myRank_, rankSize_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2ReduceScatterSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2ReduceScatterSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("[InsV2ReduceScatterSequenceExecutor4LevelOCS][CalcRes] myRank[%u] start", myRank_);
    InitCommInfo(param, topoInfo, algHierarchyInfo);
    if (algHierarchyInfo.infos.size() != SEQUENCE_EXECUTOR_LEVEL_NUM) {
        HCCL_ERROR("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] algHierarchyInfo size should be %u",
            myRank_, SEQUENCE_EXECUTOR_LEVEL_NUM);
        return HCCL_E_INTERNAL;
    }
    skipLevel1_ = (algHierarchyInfo.infos[1][0].size() == 1);
    skipLevel2_ = (algHierarchyInfo.infos[2][0].size() == 1);
    skipLevel3_ = (algHierarchyInfo.infos[3][0].size() == 1);
    std::shared_ptr<InsAlgTemplate0> tempAlgLevel0 = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> tempAlgLevel1 = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate2> tempAlgLevel2 = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[2]);
    std::shared_ptr<InsAlgTemplate3> tempAlgLevel3 = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo.infos[3]);

    AlgResourceRequest resReq0;
    AlgResourceRequest resReq1;
    AlgResourceRequest resReq2;
    AlgResourceRequest resReq3;
    CHK_RET(tempAlgLevel0->CalcRes(comm, param, topoInfo, resReq0));
    if (skipLevel1_) {
        HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS][CalcRes] myRank[%u] level1 rankSize is 1, skip level1 CalcRes",
            myRank_);
    } else {
        CHK_RET(tempAlgLevel1->CalcRes(comm, param, topoInfo, resReq1));
    }
    if (skipLevel2_) {
        HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS][CalcRes] myRank[%u] level2 rankSize is 1, skip level2 CalcRes",
            myRank_);
    } else {
        CHK_RET(tempAlgLevel2->CalcRes(comm, param, topoInfo, resReq2));
    }
    if (skipLevel3_) {
        HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS][CalcRes] myRank[%u] level3 rankSize is 1, skip level3 CalcRes",
            myRank_);
    } else {
        CHK_RET(tempAlgLevel3->CalcRes(comm, param, topoInfo, resReq3));
    }

    u32 slaveThreadNum = resReq0.slaveThreadNum;
    if (!skipLevel1_) {
        slaveThreadNum = std::max(slaveThreadNum, resReq1.slaveThreadNum);
    }
    if (!skipLevel2_) {
        slaveThreadNum = std::max(slaveThreadNum, resReq2.slaveThreadNum);
    }
    if (!skipLevel3_) {
        slaveThreadNum = std::max(slaveThreadNum, resReq3.slaveThreadNum);
    }
    resourceRequest.slaveThreadNum = slaveThreadNum;
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReq0.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReq0.notifyNumPerThread[i]);
        }
        if (!skipLevel1_ && i < resReq1.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReq1.notifyNumPerThread[i]);
        }
        if (!skipLevel2_ && i < resReq2.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReq2.notifyNumPerThread[i]);
        }
        if (!skipLevel3_ && i < resReq3.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReq3.notifyNumPerThread[i]);
        }
    }
    resourceRequest.notifyNumOnMainThread = resReq0.notifyNumOnMainThread;
    if (!skipLevel1_) {
        resourceRequest.notifyNumOnMainThread = std::max(resourceRequest.notifyNumOnMainThread, resReq1.notifyNumOnMainThread);
    }
    if (!skipLevel2_) {
        resourceRequest.notifyNumOnMainThread = std::max(resourceRequest.notifyNumOnMainThread, resReq2.notifyNumOnMainThread);
    }
    if (!skipLevel3_) {
        resourceRequest.notifyNumOnMainThread = std::max(resourceRequest.notifyNumOnMainThread, resReq3.notifyNumOnMainThread);
    }
    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] notifyNumOnMainThread is %u",
        myRank_, resourceRequest.notifyNumOnMainThread);
    resourceRequest.channels.resize(SEQUENCE_EXECUTOR_LEVEL_NUM);
    if (resReq0.channels.empty()) {
        HCCL_ERROR("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] channels empty, level0[%u]",
            myRank_, resReq0.channels.size());
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[0] = resReq0.channels[0];
    if (!skipLevel1_) {
        if (resReq1.channels.empty()) {
            HCCL_ERROR("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] channels empty, level1[%u]",
                myRank_, resReq1.channels.size());
            return HCCL_E_INTERNAL;
        }
        resourceRequest.channels[1] = resReq1.channels[0];
    }
    if (!skipLevel2_) {
        if (resReq2.channels.empty()) {
            HCCL_ERROR("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] channels empty, level2[%u]",
                myRank_, resReq2.channels.size());
            return HCCL_E_INTERNAL;
        }
        resourceRequest.channels[2] = resReq2.channels[0];
    }
    if (!skipLevel3_) {
        if (resReq3.channels.empty()) {
            HCCL_ERROR("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] channels empty, level3[%u]",
                myRank_, resReq3.channels.size());
            return HCCL_E_INTERNAL;
        }
        resourceRequest.channels[3] = resReq3.channels[0];
    }
    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] slaveThreadNum is [%u], notifyNumOnMainThread is [%u], "
        "level0 chanel size [%u], level1 channel size [%u], level2 channel size [%u], level3 channel size [%u]",
        myRank_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread,
        resourceRequest.channels[0].size(), resourceRequest.channels[1].size(),
        resourceRequest.channels[2].size(), resourceRequest.channels[3].size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2ReduceScatterSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = (myRank_ / algHierarchyInfo_.infos[0][0].size()) % algHierarchyInfo_.infos[1][0].size();
    rankIdxLevel2_ = (myRank_ / (algHierarchyInfo_.infos[0][0].size() * algHierarchyInfo_.infos[1][0].size())) %
        algHierarchyInfo_.infos[2][0].size();
    rankIdxLevel3_ = myRank_ /
        (algHierarchyInfo_.infos[0][0].size() * algHierarchyInfo_.infos[1][0].size() * algHierarchyInfo_.infos[2][0].size());

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
    rankSizeLevel3_ = algHierarchyInfo_.infos[3][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);
    skipLevel2_ = (rankSizeLevel2_ == 1);
    skipLevel3_ = (rankSizeLevel3_ == 1);
    if (skipLevel1_) {
        HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS] [Orchestrate] myRank[%u] level1 rankSize is 1, skip level1",
            myRank_);
    }
    if (skipLevel2_) {
        HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS] [Orchestrate] myRank[%u] level2 rankSize is 1, skip level2",
            myRank_);
    }
    if (skipLevel3_) {
        HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS] [Orchestrate] myRank[%u] level3 rankSize is 1, skip level3",
            myRank_);
    }
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS] [Orchestrate] myRank_[%u] rankIdxLevel0_[%u] "
        "rankIdxLevel1_[%u] rankIdxLevel2_[%u] rankIdxLevel3_[%u] rankSizeLevel0_[%u] rankSizeLevel1_[%u] "
        "rankSizeLevel2_[%u] rankSizeLevel3_[%u]",
        myRank_, rankIdxLevel0_, rankIdxLevel1_, rankIdxLevel2_, rankIdxLevel3_,
        rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_, rankSizeLevel3_);
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2ReduceScatterSequenceExecutor4LevelOCS][Orchestrate] myRank[%u] errNo[0x%016llx] "
            "Reduce scatter excutor kernel run failed", myRank_, HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

// level0 intra (INPUT -> HCCL): 每个 rank 在 HCCL 中持有 (s1*s2*s3) 个同序号块
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2ReduceScatterSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::GenIntraTemplateParams(
    TemplateDataParams &tempAlgParamsIntra, const u64 processedDataCount, const u64 currDataCount, const u64 loop) const
{
    tempAlgParamsIntra.count = currDataCount;
    tempAlgParamsIntra.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsIntra.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsIntra.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsIntra.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsIntra.tailSize = tempAlgParamsIntra.sliceSize;

    tempAlgParamsIntra.inputSliceStride = dataSize_;
    tempAlgParamsIntra.outputSliceStride = currDataCount * dataTypeSize_;
    tempAlgParamsIntra.repeatNum = rankSizeLevel1_ * rankSizeLevel2_ * rankSizeLevel3_;
    tempAlgParamsIntra.inputRepeatStride = rankSizeLevel0_ * dataSize_;
    tempAlgParamsIntra.outputRepeatStride = rankSizeLevel0_ * currDataCount * dataTypeSize_;

    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] loop[%llu] Intra inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsIntra.inputSliceStride, tempAlgParamsIntra.outputSliceStride,
        tempAlgParamsIntra.sliceSize, tempAlgParamsIntra.buffInfo.inBuffBaseOff,
        tempAlgParamsIntra.buffInfo.outBuffBaseOff, tempAlgParamsIntra.repeatNum,
        tempAlgParamsIntra.inputRepeatStride, tempAlgParamsIntra.outputRepeatStride);
}

// level1 inter (HCCL -> HCCL): reduce 范围 [i0 + i1*s0], repeatNum = s2*s3
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2ReduceScatterSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::GenInterTemplateParams1(
    TemplateDataParams &tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount, const u64 loop) const
{
    tempAlgParamsInter.count = currDataCount;
    tempAlgParamsInter.buffInfo.inBuffBaseOff = rankIdxLevel0_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.outBuffBaseOff = (rankIdxLevel0_ + (rankIdxLevel1_ * rankSizeLevel0_)) * currDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff = rankIdxLevel0_ * currDataCount * dataTypeSize_;

    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = rankSizeLevel0_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = rankSizeLevel2_ * rankSizeLevel3_;
    tempAlgParamsInter.inputRepeatStride = rankSizeLevel0_ * rankSizeLevel1_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.outputRepeatStride = rankSizeLevel0_ * rankSizeLevel1_ * currDataCount * dataTypeSize_;

    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] loop[%llu] Inter1 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum,
        tempAlgParamsInter.inputRepeatStride, tempAlgParamsInter.outputRepeatStride);
}

// level2 inter (HCCL -> HCCL): reduce 范围 [i0 + i1*s0 + i2*s0*s1], repeatNum = s3
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2ReduceScatterSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::GenInterTemplateParams2(
    TemplateDataParams &tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount, const u64 loop) const
{
    u64 level1GroupOffset = (rankIdxLevel0_ + (rankIdxLevel1_ * rankSizeLevel0_)) * currDataCount * dataTypeSize_;
    tempAlgParamsInter.count = currDataCount;
    tempAlgParamsInter.buffInfo.inBuffBaseOff = level1GroupOffset;
    tempAlgParamsInter.buffInfo.outBuffBaseOff =
        (rankIdxLevel0_ + (rankIdxLevel1_ * rankSizeLevel0_) + (rankIdxLevel2_ * rankSizeLevel0_ * rankSizeLevel1_)) *
        currDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff = level1GroupOffset;

    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = rankSizeLevel0_ * rankSizeLevel1_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = rankSizeLevel3_;
    tempAlgParamsInter.inputRepeatStride = rankSizeLevel0_ * rankSizeLevel1_ * rankSizeLevel2_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.outputRepeatStride = rankSizeLevel0_ * rankSizeLevel1_ * rankSizeLevel2_ * currDataCount * dataTypeSize_;

    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] loop[%llu] Inter2 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum,
        tempAlgParamsInter.inputRepeatStride, tempAlgParamsInter.outputRepeatStride);
}

// level3 inter (HCCL -> OUTPUT): reduce 范围 [i0 + i1*s0 + i2*s0*s1], repeatNum = 1, 写到 OUTPUT
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2ReduceScatterSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::GenInterTemplateParams3(
    TemplateDataParams &tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount, const u64 loop) const
{
    u64 level2GroupOffset =
        (rankIdxLevel0_ + (rankIdxLevel1_ * rankSizeLevel0_) + (rankIdxLevel2_ * rankSizeLevel0_ * rankSizeLevel1_)) *
        currDataCount * dataTypeSize_;
    tempAlgParamsInter.count = currDataCount;
    tempAlgParamsInter.buffInfo.inBuffBaseOff = level2GroupOffset;
    tempAlgParamsInter.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff = level2GroupOffset;

    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = rankSizeLevel0_ * rankSizeLevel1_ * rankSizeLevel2_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = 1;
    tempAlgParamsInter.inputRepeatStride = 0;
    tempAlgParamsInter.outputRepeatStride = 0;

    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] loop[%llu] Inter3 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum,
        tempAlgParamsInter.inputRepeatStride, tempAlgParamsInter.outputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
template <typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::GenTempResource
    (const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
    const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempResource) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR("[InsV2ReduceScatterSequenceExecutor4LevelOCS][GenTempResource] myRank[%u] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%u]", myRank_, channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2ReduceScatterSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    TemplateDataParams tempAlgParamsLevel0;
    tempAlgParamsLevel0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsLevel0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel0.buffInfo.hcclBuff = resCtx.cclMem;

    std::shared_ptr<InsAlgTemplate0> algTemplateLevel0 = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    if (rankSizeLevel0_ > 1) {
        CHK_RET(algTemplateLevel0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));
    }

    TemplateDataParams tempAlgParamsLevel1;
    tempAlgParamsLevel1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.hcclBuff = resCtx.cclMem;

    std::shared_ptr<InsAlgTemplate1> algTemplateLevel1 = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
    if (!skipLevel1_) {
        CHK_RET(algTemplateLevel1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    TemplateDataParams tempAlgParamsLevel2;
    tempAlgParamsLevel2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel2.buffInfo.hcclBuff = resCtx.cclMem;

    std::shared_ptr<InsAlgTemplate2> algTemplateLevel2 = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);
    if (!skipLevel2_) {
        CHK_RET(algTemplateLevel2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));
    }

    TemplateDataParams tempAlgParamsLevel3;
    tempAlgParamsLevel3.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel3.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsLevel3.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel3.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel3.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsLevel3.buffInfo.hcclBuff = resCtx.cclMem;

    std::shared_ptr<InsAlgTemplate3> algTemplateLevel3 = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[3]);
    if (!skipLevel3_) {
        CHK_RET(algTemplateLevel3->SetchannelsPerRank(remoteRankToChannelInfo_[3]));
    }

    u32 templateScratchMultiplier0 = algTemplateLevel0->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier1 = skipLevel1_ ? 1 :
        algTemplateLevel1->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier2 = skipLevel2_ ? 1 :
        algTemplateLevel2->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier3 = skipLevel3_ ? 1 :
        algTemplateLevel3->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);
    u32 templateScratchMultiplier = templateScratchMultiplier0 * templateScratchMultiplier1 *
        templateScratchMultiplier2 * templateScratchMultiplier3;

    TemplateResource templateResource0;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateLevel0, templateResource0));
    TemplateResource templateResource1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(resCtx, 1, algTemplateLevel1, templateResource1));
    }
    TemplateResource templateResource2;
    if (!skipLevel2_) {
        CHK_RET(GenTempResource(resCtx, 2, algTemplateLevel2, templateResource2));
    }
    TemplateResource templateResource3;
    if (!skipLevel3_) {
        CHK_RET(GenTempResource(resCtx, 3, algTemplateLevel3, templateResource3));
    }

    u64 maxCountPerLoop = tempAlgParamsLevel3.buffInfo.hcclBuff.size / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR("[InsV2ReduceScatterSequenceExecutor4LevelOCS] myRank[%u] maxCountPerLoop is 0, "
            "scratchMultiplier[%u] too large for cclBuffSize[%llu]",
            myRank_, templateScratchMultiplier, tempAlgParamsLevel3.buffInfo.hcclBuff.size);
        return HCCL_E_INTERNAL;
    }
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        GenIntraTemplateParams(tempAlgParamsLevel0, processedDataCount, currDataCount, loop);
        CHK_RET(algTemplateLevel0->KernelRun(param, tempAlgParamsLevel0, templateResource0));

        if (!skipLevel1_) {
            GenInterTemplateParams1(tempAlgParamsLevel1, processedDataCount, currDataCount, loop);
            CHK_RET(algTemplateLevel1->KernelRun(param, tempAlgParamsLevel1, templateResource1));
        }

        if (!skipLevel2_) {
            GenInterTemplateParams2(tempAlgParamsLevel2, processedDataCount, currDataCount, loop);
            CHK_RET(algTemplateLevel2->KernelRun(param, tempAlgParamsLevel2, templateResource2));
        }

        if (!skipLevel3_) {
            GenInterTemplateParams3(tempAlgParamsLevel3, processedDataCount, currDataCount, loop);
            CHK_RET(algTemplateLevel3->KernelRun(param, tempAlgParamsLevel3, templateResource3));
        }
        processedDataCount += currDataCount;
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
    InsReduceScatterSequenceMesh1DNHRNHRMesh1D,
    InsV2ReduceScatterSequenceExecutor4LevelOCS,
    TopoMatchMultilevel,
    InsTempReduceScatterMesh1DZAxisDetour,
    InsTempReduceScatterNHR,
    InsTempReduceScatterNHR,
    InsTempReduceScatterMesh1DOcs);

}
