/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_sequence_executor_aicpu_3level.h"
#include "ins_temp_scatter_nhr.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "aicpu_temp_scatter_mesh_1D_Z_axis_detour.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::InsV2BroadcastSequenceExecutorAicpu3Level() {}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::InitCommInfo(HcclComm comm, const OpParam &param,
        const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][InitCommInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]",
        myRank_,
        rankSize_,
        dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
template <typename InsAlgTemplate>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempResource
    (const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
    const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempResource) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR("[InsV2BroadcastSequenceExecutorAicpu3Level][GenTempResource] myRank[%u] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%u]", myRank_, channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

// 实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    rankSizeLevel0_ = algHierarchyInfo.infos[0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1].size();
    rankSizeLevel2_ = algHierarchyInfo.infos[2].size();
    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][CalcRes] rankSizeLevel0 [%u], rankSizeLevel1 [%u], rankSizeLevel2 [%u]", rankSizeLevel0_,
        rankSizeLevel1_, rankSizeLevel2_);

    std::shared_ptr<InsAlgTemplate0> ScatterL0TempAlg =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> ScatterL1TempAlg =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate2> ScatterL2TempAlg =
        std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[2]);
    std::shared_ptr<InsAlgTemplate3> agL2TempAlg =
        std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo.infos[2]);
    std::shared_ptr<InsAlgTemplate4> agL1TempAlg =
        std::make_shared<InsAlgTemplate4>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate5> agL0TempAlg =
        std::make_shared<InsAlgTemplate5>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqScatterL0;
    AlgResourceRequest resReqScatterL1;
    AlgResourceRequest resReqScatterL2;
    AlgResourceRequest resReqAGL2;
    AlgResourceRequest resReqAGL1;
    AlgResourceRequest resReqAGL0;

    CHK_RET(ScatterL0TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL0));
    CHK_RET(ScatterL1TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL1));
    CHK_RET(ScatterL2TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL2));
    CHK_RET(agL2TempAlg->CalcRes(comm, param, topoInfo, resReqAGL2));
    CHK_RET(agL1TempAlg->CalcRes(comm, param, topoInfo, resReqAGL1));
    CHK_RET(agL0TempAlg->CalcRes(comm, param, topoInfo, resReqAGL0));

    // step1在完成后，完成后同步后展开step2，因此slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum = std::max({resReqScatterL0.slaveThreadNum,
        resReqScatterL1.slaveThreadNum,
        resReqScatterL2.slaveThreadNum,
        resReqAGL2.slaveThreadNum,
        resReqAGL1.slaveThreadNum,
        resReqAGL0.slaveThreadNum});

    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReqScatterL0.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqScatterL0.notifyNumPerThread[i]);
        }
        if (i < resReqScatterL1.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqScatterL1.notifyNumPerThread[i]);
        }
        if (i < resReqScatterL2.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqScatterL2.notifyNumPerThread[i]);
        }
        if (i < resReqAGL2.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqAGL2.notifyNumPerThread[i]);
        }
        if (i < resReqAGL1.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqAGL1.notifyNumPerThread[i]);
        }
        if (i < resReqAGL0.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqAGL0.notifyNumPerThread[i]);
        }
    }

    resourceRequest.notifyNumOnMainThread = std::max({resReqScatterL0.notifyNumOnMainThread,
        resReqScatterL1.notifyNumOnMainThread,
        resReqScatterL2.notifyNumOnMainThread,
        resReqAGL2.notifyNumOnMainThread,
        resReqAGL1.notifyNumOnMainThread,
        resReqAGL0.notifyNumOnMainThread});

    u64 channelsSize = 3;
    resourceRequest.channels.resize(channelsSize);
    resourceRequest.channels[0] = resReqScatterL0.channels[0];
    resourceRequest.channels[1] = resReqScatterL1.channels[0];
    resourceRequest.channels[2] = resReqScatterL2.channels[0];

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][Orchestrate] Orchestrate Start");
    // 参数填充
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    CHK_RET(InitExecutorInfo(param, resCtx));
    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2BroadcastSequenceExecutorAicpu3Level][Orchestrate]errNo[0x%016llx] Broadcast excutor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::InitExecutorInfo(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = (myRank_ / algHierarchyInfo_.infos[0][0].size()) % algHierarchyInfo_.infos[1][0].size();

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
        
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][InitExecutorInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]",
        +myRank_,
        rankSize_,
        dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsScatterL0(u64 currDataCount, u64 processedDataCount, TemplateDataParams &params) const
{
    u64 sliceCnt = currDataCount / rankSizeLevel0_;
    u64 remCnt = currDataCount % rankSizeLevel0_;
    params.sliceSize = sliceCnt * dataTypeSize_;
    params.tailSize = (sliceCnt + remCnt) * dataTypeSize_;

    params.count = currDataCount;
    params.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    params.buffInfo.outBuffBaseOff = 0;
    params.buffInfo.hcclBuffBaseOff = 0;

    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = 0;

    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsScatterL1(u64 level1TotalCnt, u64 l0SliceByte, TemplateDataParams &params) const
{
    u64 sliceCnt = level1TotalCnt / rankSizeLevel1_;
    u64 remCnt = level1TotalCnt % rankSizeLevel1_;
    params.sliceSize = sliceCnt * dataTypeSize_;
    params.tailSize = (sliceCnt + remCnt) * dataTypeSize_;

    params.count = level1TotalCnt;
    params.buffInfo.inBuffBaseOff = 0;
    params.buffInfo.outBuffBaseOff = 0;
    params.buffInfo.hcclBuffBaseOff = rankIdxLevel0_ * l0SliceByte;

    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;

    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsScatterL2(u64 level2TotalCnt, u64 l0SliceByte, u64 l1SliceByte, TemplateDataParams &params) const
{
    u64 sliceCnt = level2TotalCnt / rankSizeLevel2_;
    u64 remCnt = level2TotalCnt % rankSizeLevel2_;
    params.sliceSize = sliceCnt * dataTypeSize_;
    params.tailSize = (sliceCnt + remCnt) * dataTypeSize_;

    params.count = level2TotalCnt;
    params.buffInfo.inBuffBaseOff = 0;
    params.buffInfo.outBuffBaseOff = 0;
    params.buffInfo.hcclBuffBaseOff = rankIdxLevel0_ * l0SliceByte + rankIdxLevel1_ * l1SliceByte;

    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;

    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
}


template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsAGL2(u64 ag2Cnt, u64 l0SliceByte, u64 l1SliceByte, TemplateDataParams &params) const
{
    u64 sliceCnt = ag2Cnt / rankSizeLevel2_;
    u64 remCnt = ag2Cnt % rankSizeLevel2_;
    params.sliceSize = sliceCnt * dataTypeSize_;
    params.tailSize = (sliceCnt + remCnt) * dataTypeSize_;

    params.count = ag2Cnt;
    params.buffInfo.inBuffBaseOff = 0;
    params.buffInfo.outBuffBaseOff = 0;
    params.buffInfo.hcclBuffBaseOff = rankIdxLevel0_ * l0SliceByte + rankIdxLevel1_ * l1SliceByte;

    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;
    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsAGL1(u64 ag1Cnt, u64 l0SliceByte, TemplateDataParams &params) const
{
    u64 sliceCnt = ag1Cnt / rankSizeLevel1_;
    u64 remCnt = ag1Cnt % rankSizeLevel1_;
    params.sliceSize = sliceCnt * dataTypeSize_;
    params.tailSize = (sliceCnt + remCnt) * dataTypeSize_;

    params.count = ag1Cnt;
    params.buffInfo.inBuffBaseOff = 0;
    params.buffInfo.outBuffBaseOff = 0;
    params.buffInfo.hcclBuffBaseOff = rankIdxLevel0_ * l0SliceByte;

    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;
    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsAGL0(u64 currDataCount, u64 processedDataCount, TemplateDataParams &params) const
{
    u64 sliceCnt = currDataCount / rankSizeLevel0_;
    u64 remCnt = currDataCount % rankSizeLevel0_;
    params.sliceSize = sliceCnt * dataTypeSize_;
    params.tailSize = (sliceCnt + remCnt) * dataTypeSize_;

    params.count = currDataCount;
    params.buffInfo.inBuffBaseOff = 0;
    params.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    params.buffInfo.hcclBuffBaseOff = 0;

    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;
    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][OrchestrateLoop] Start");
    // scatter L0
    TemplateDataParams tempAlgParamsScatterL0;
    tempAlgParamsScatterL0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsScatterL0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL0.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate0> algTemplateScatterL0 =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    CHK_RET(algTemplateScatterL0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));

    // scatter L1
    TemplateDataParams tempAlgParamsScatterL1;
    tempAlgParamsScatterL1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL1.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate1> algTemplateScatterL1 =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);

    // scatter L2
    TemplateDataParams tempAlgParamsScatterL2;
    tempAlgParamsScatterL2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL2.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate2> algTemplateScatterL2 =
        std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);

    // AG L2
    TemplateDataParams tempAlgParamsAllGatherL2;
    tempAlgParamsAllGatherL2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL2.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate3> algTemplateAllGatherL2 =
        std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[2]);

    // AG L1
    TemplateDataParams tempAlgParamsAllGatherL1;
    tempAlgParamsAllGatherL1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL1.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate4> algTemplateAllGatherL1 =
        std::make_shared<InsAlgTemplate4>(param, myRank_, algHierarchyInfo_.infos[1]);

    // AG L0
    TemplateDataParams tempAlgParamsAllGatherL0;
    tempAlgParamsAllGatherL0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsAllGatherL0.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate5> algTemplateAllGatherL0 =
        std::make_shared<InsAlgTemplate5>(param, myRank_, algHierarchyInfo_.infos[0]);
    CHK_RET(algTemplateAllGatherL0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));

    // 构造L0 template资源
    TemplateResource templateResourceL0;
    templateResourceL0.channels = remoteRankToChannelInfo_[0];
    templateResourceL0.threads = resCtx.threads;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateScatterL0, templateResourceL0));
    // 构造L1 template资源
    TemplateResource templateResourceL1;
    templateResourceL1.channels = remoteRankToChannelInfo_[1];
    templateResourceL1.threads = resCtx.threads;
    CHK_RET(GenTempResource(resCtx, 1, algTemplateScatterL1, templateResourceL1));

    // 构造L2 template资源
    TemplateResource templateResourceL2;
    templateResourceL2.channels = remoteRankToChannelInfo_[2];
    templateResourceL2.threads = resCtx.threads;
    CHK_RET(GenTempResource(resCtx, 2, algTemplateScatterL2, templateResourceL2));

    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    u64 dataCount_ = param.DataDes.count;
    // 当前公式是未优化版本
    u64 maxCountPerLoop = tempAlgParamsScatterL0.buffInfo.hcclBuff.size / HCCL_MIN_SLICE_ALIGN *
                          HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    // 已处理的元素数
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        // ---------------------- Scatter L0 标量分片 ----------------------
        GenTempAlgParamsScatterL0(currDataCount, processedDataCount, tempAlgParamsScatterL0);
        algTemplateScatterL0->SetRoot(param.root);
        CHK_RET(algTemplateScatterL0->KernelRun(param, tempAlgParamsScatterL0, templateResourceL0));
        u64 l0SliceByte = tempAlgParamsScatterL0.sliceSize;
        u64 l0SliceCnt = currDataCount / rankSizeLevel0_;
        // ---------------------- Scatter L1 标量分片 ----------------------
        u64 l1TotalCnt = l0SliceCnt;
        GenTempAlgParamsScatterL1(l1TotalCnt, l0SliceByte, tempAlgParamsScatterL1);
        u32 localRootL1 = (param.root / rankSizeLevel0_) * rankSizeLevel0_ + (myRank_ % rankSizeLevel0_);
        algTemplateScatterL1->SetRoot(localRootL1);
        if (l1TotalCnt != 0) {
            CHK_RET(algTemplateScatterL1->KernelRun(param, tempAlgParamsScatterL1, templateResourceL1));
        }
        u64 l1SliceByte = tempAlgParamsScatterL1.sliceSize;
        u64 l1SliceCnt = l1TotalCnt / rankSizeLevel1_;
        // ---------------------- Scatter L2 标量分片 ----------------------
        u64 l2TotalCnt = l1SliceCnt;
        GenTempAlgParamsScatterL2(l2TotalCnt, l0SliceByte, l1SliceByte, tempAlgParamsScatterL2);
        u64 rankPerPod = rankSizeLevel0_ * rankSizeLevel1_;
        u32 localRootL2 = (param.root / rankPerPod) * rankPerPod + (myRank_ % rankPerPod);
        algTemplateScatterL2->SetRoot(localRootL2);
        if (l2TotalCnt != 0) {
            CHK_RET(algTemplateScatterL2->KernelRun(param, tempAlgParamsScatterL2, templateResourceL2));
        }
        // ---------------------- AllGather L2 ----------------------
        GenTempAlgParamsAGL2(l2TotalCnt, l0SliceByte, l1SliceByte, tempAlgParamsAllGatherL2);
        CHK_RET(algTemplateAllGatherL2->KernelRun(param, tempAlgParamsAllGatherL2, templateResourceL2));
        // ---------------------- AllGather L1 ----------------------
        GenTempAlgParamsAGL1(l1TotalCnt, l0SliceByte, tempAlgParamsAllGatherL1);
        CHK_RET(algTemplateAllGatherL1->KernelRun(param, tempAlgParamsAllGatherL1, templateResourceL1));
        // ---------------------- AllGather L0 ----------------------
        GenTempAlgParamsAGL0(currDataCount, processedDataCount, tempAlgParamsAllGatherL0);
        CHK_RET(algTemplateAllGatherL0->KernelRun(param, tempAlgParamsAllGatherL0, templateResourceL0));
        processedDataCount += currDataCount;
    }

    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_BROADCAST,
    InsBroadcastSequenceMesh1DNHRNHR,
    InsV2BroadcastSequenceExecutorAicpu3Level,
    TopoMatchMultilevel,
    AicpuTempScatterMesh1DZAxisDetour,      // Scatter L0 (框内)
    InsTempScatterNHR,          // Scatter L1 (框间)
    InsTempScatterNHR,          // Scatter L2 (跨超节点)
    InsTempAllGatherNHR,        // AllGather L2 (跨超节点)
    InsTempAllGatherNHR,        // AllGather L1 (框间)
    InsTempAllGatherMesh1D1DZAxisDetour);  // AllGather L0 (框内, Z 轴绕路)
}  // namespace ops_hccl