/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_sequence_executor_3level.h"
#include "ins_temp_scatter_mesh_1D.h"
#include "ins_temp_scatter_nhr.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_mesh_1D.h"

namespace ops_hccl {

constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 3;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::InsV2BroadcastSequenceExecutor3Level()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::InitCommInfo(const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level][InitCommInfo] myRank [%u], rankSize [%u], "
        "dataType [%u] dataTypeSize [%u]", myRank_, rankSize_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    InitCommInfo(param, topoInfo, algHierarchyInfo);
    if (algHierarchyInfo.infos.size() != SEQUENCE_EXECUTOR_LEVEL_NUM) {
        HCCL_ERROR("[InsV2BroadcastSequenceExecutor3Level] algHierarchyInfo size should be %u",
            SEQUENCE_EXECUTOR_LEVEL_NUM);
        return HCCL_E_INTERNAL;
    }
    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1][0].size();
    rankSizeLevel2_ = algHierarchyInfo.infos[2][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);

    std::shared_ptr<InsAlgTemplate0> scatterL0TempAlg =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> scatterL1TempAlg;
    if (!skipLevel1_) {
        scatterL1TempAlg = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    }
    std::shared_ptr<InsAlgTemplate2> scatterL2TempAlg =
        std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[2]);
    std::shared_ptr<InsAlgTemplate3> agL2TempAlg =
        std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo.infos[2]);
    std::shared_ptr<InsAlgTemplate4> agL1TempAlg;
    if (!skipLevel1_) {
        agL1TempAlg = std::make_shared<InsAlgTemplate4>(param, myRank_, algHierarchyInfo.infos[1]);
    }
    std::shared_ptr<InsAlgTemplate5> agL0TempAlg =
        std::make_shared<InsAlgTemplate5>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqScatterL0;
    AlgResourceRequest resReqScatterL1;
    AlgResourceRequest resReqScatterL2;
    AlgResourceRequest resReqAGL2;
    AlgResourceRequest resReqAGL1;
    AlgResourceRequest resReqAGL0;
    CHK_RET(scatterL0TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL0));
    if (!skipLevel1_) {
        CHK_RET(scatterL1TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL1));
        CHK_RET(agL1TempAlg->CalcRes(comm, param, topoInfo, resReqAGL1));
    }
    CHK_RET(scatterL2TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL2));
    CHK_RET(agL2TempAlg->CalcRes(comm, param, topoInfo, resReqAGL2));
    CHK_RET(agL0TempAlg->CalcRes(comm, param, topoInfo, resReqAGL0));

    std::vector<AlgResourceRequest> activeReqs = {resReqScatterL0, resReqScatterL2, resReqAGL2, resReqAGL0};
    if (!skipLevel1_) {
        activeReqs.push_back(resReqScatterL1);
        activeReqs.push_back(resReqAGL1);
    }
    resourceRequest.slaveThreadNum = 0;
    for (auto &req : activeReqs) {
        resourceRequest.slaveThreadNum = std::max(resourceRequest.slaveThreadNum, req.slaveThreadNum);
    }
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        for (auto &req : activeReqs) {
            if (i < req.notifyNumPerThread.size()) {
                resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                    req.notifyNumPerThread[i]);
            }
        }
    }
    u32 mainNotifyNum = 0;
    for (auto &req : activeReqs) {
        mainNotifyNum = std::max(mainNotifyNum, req.notifyNumOnMainThread);
    }
    resourceRequest.notifyNumOnMainThread = mainNotifyNum;

    resourceRequest.channels.resize(SEQUENCE_EXECUTOR_LEVEL_NUM);
    if (resReqScatterL0.channels.empty()) {
        HCCL_ERROR("[InsV2BroadcastSequenceExecutor3Level] level0 channels empty");
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[0] = resReqScatterL0.channels[0];
    if (!skipLevel1_) {
        if (resReqScatterL1.channels.empty()) {
            HCCL_ERROR("[InsV2BroadcastSequenceExecutor3Level] level1 channels empty");
            return HCCL_E_INTERNAL;
        }
        resourceRequest.channels[1] = resReqScatterL1.channels[0];
    }
    if (resReqScatterL2.channels.empty()) {
        HCCL_ERROR("[InsV2BroadcastSequenceExecutor3Level] level2 channels empty");
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[2] = resReqScatterL2.channels[0];
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::Orchestrate(const OpParam &param,
    const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level][Orchestrate] Orchestrate Start");
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataCount_ = param.DataDes.count;
    dataSize_ = dataCount_ * dataTypeSize_;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);
    if (skipLevel1_) {
        HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level][Orchestrate] level1 rankSize is 1, skip level1");
    }
    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = (myRank_ / algHierarchyInfo_.infos[0][0].size()) % algHierarchyInfo_.infos[1][0].size();
    rankIdxLevel2_ = myRank_ / (algHierarchyInfo_.infos[0][0].size() * algHierarchyInfo_.infos[1][0].size());

    root_ = param.root;
    rootIdx0_ = root_ % rankSizeLevel0_;
    rootIdx1_ = (root_ / rankSizeLevel0_) % rankSizeLevel1_;
    rootIdx2_ = root_ / (rankSizeLevel0_ * rankSizeLevel1_);

    HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level][Orchestrate] myRank [%u], root [%u], "
        "rankIdx [%u/%u/%u], rootIdx [%u/%u/%u]",
        myRank_, root_, rankIdxLevel0_, rankIdxLevel1_, rankIdxLevel2_,
        rootIdx0_, rootIdx1_, rootIdx2_);

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2BroadcastSequenceExecutor3Level][Orchestrate]errNo[0x%016llx] "
            "Broadcast executor kernel run failed", HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenBaseTempAlgParams(const OpParam &param,
    const AlgResourceCtxSerializable &resCtx, TemplateDataParams &tempAlgParamsScatterL0,
    TemplateDataParams &tempAlgParamsScatterL1, TemplateDataParams &tempAlgParamsScatterL2,
    TemplateDataParams &tempAlgParamsAGL2, TemplateDataParams &tempAlgParamsAGL1,
    TemplateDataParams &tempAlgParamsAGL0) const
{
    tempAlgParamsScatterL0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsScatterL0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterL0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterL0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsScatterL0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL0.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsScatterL1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterL1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterL1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterL1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL1.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsScatterL2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterL2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterL2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterL2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL2.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsAGL2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAGL2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAGL2.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsAGL1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAGL1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAGL1.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsAGL0.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsAGL0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAGL0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsAGL0.buffInfo.hcclBuff = resCtx.cclMem;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsScatterL0(const u64 loop,
    const u64 currDataCount, const u64 processedDataCount,
    TemplateDataParams &tempAlgParamsScatterL0) const
{
    tempAlgParamsScatterL0.count = currDataCount;
    tempAlgParamsScatterL0.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsScatterL0.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsScatterL0.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsScatterL0.sliceSize = currDataCount / rankSizeLevel0_ * dataTypeSize_;
    tempAlgParamsScatterL0.tailSize = (currDataCount / rankSizeLevel0_ + currDataCount % rankSizeLevel0_) * dataTypeSize_;

    tempAlgParamsScatterL0.inputSliceStride = tempAlgParamsScatterL0.sliceSize;
    tempAlgParamsScatterL0.outputSliceStride = 0;

    HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level] loop [%u] ScatterL0.sliceSize [%u], "
        "ScatterL0.tailSize [%u], ScatterL0.inBuffBaseOff [%u], ScatterL0.hcclBuffBaseOff [%u]",
        loop, tempAlgParamsScatterL0.sliceSize, tempAlgParamsScatterL0.tailSize,
        tempAlgParamsScatterL0.buffInfo.inBuffBaseOff, tempAlgParamsScatterL0.buffInfo.hcclBuffBaseOff);

    tempAlgParamsScatterL0.repeatNum = 1;
    tempAlgParamsScatterL0.inputRepeatStride = 0;
    tempAlgParamsScatterL0.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsScatterL1(const u64 loop,
    const u64 currDataCount, const u64 sliceSizeL0, const u64 tailSizeL0,
    TemplateDataParams &tempAlgParamsScatterL1) const
{
    tempAlgParamsScatterL1.count = currDataCount;
    if (rankIdxLevel0_ == rankSizeLevel0_ - 1) {
        u64 tailCountL0 = tailSizeL0 / dataTypeSize_;
        tempAlgParamsScatterL1.sliceSize = tailCountL0 / rankSizeLevel1_ * dataTypeSize_;
        tempAlgParamsScatterL1.tailSize = tempAlgParamsScatterL1.sliceSize + tailCountL0 % rankSizeLevel1_ * dataTypeSize_;
    } else {
        u64 sliceCountL0 = sliceSizeL0 / dataTypeSize_;
        tempAlgParamsScatterL1.sliceSize = sliceCountL0 / rankSizeLevel1_ * dataTypeSize_;
        tempAlgParamsScatterL1.tailSize = tempAlgParamsScatterL1.sliceSize + sliceCountL0 % rankSizeLevel1_ * dataTypeSize_;
    }
    tempAlgParamsScatterL1.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsScatterL1.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsScatterL1.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsScatterL1.inputSliceStride = tempAlgParamsScatterL1.sliceSize;
    tempAlgParamsScatterL1.outputSliceStride = tempAlgParamsScatterL1.sliceSize;

    HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level] loop [%u] ScatterL1.sliceSize [%u], "
        "ScatterL1.tailSize [%u], ScatterL1.hcclBuffBaseOff [%u]",
        loop, tempAlgParamsScatterL1.sliceSize, tempAlgParamsScatterL1.tailSize,
        tempAlgParamsScatterL1.buffInfo.hcclBuffBaseOff);

    tempAlgParamsScatterL1.repeatNum = 1;
    tempAlgParamsScatterL1.inputRepeatStride = 0;
    tempAlgParamsScatterL1.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsScatterL2(const u64 loop,
    const u64 currDataCount, const u64 sliceSizeL1, const u64 tailSizeL1,
    TemplateDataParams &tempAlgParamsScatterL2) const
{
    tempAlgParamsScatterL2.count = currDataCount;
    if (rankIdxLevel1_ == rankSizeLevel1_ - 1) {
        u64 tailCountL1 = tailSizeL1 / dataTypeSize_;
        tempAlgParamsScatterL2.sliceSize = tailCountL1 / rankSizeLevel2_ * dataTypeSize_;
        tempAlgParamsScatterL2.tailSize = tempAlgParamsScatterL2.sliceSize +
            tailCountL1 % rankSizeLevel2_ * dataTypeSize_;
    } else {
        u64 sliceCountL1 = sliceSizeL1 / dataTypeSize_;
        tempAlgParamsScatterL2.sliceSize = sliceCountL1 / rankSizeLevel2_ * dataTypeSize_;
        tempAlgParamsScatterL2.tailSize = tempAlgParamsScatterL2.sliceSize +
            sliceCountL1 % rankSizeLevel2_ * dataTypeSize_;
    }
    u64 offsetL1 = rankIdxLevel1_ * sliceSizeL1;
    tempAlgParamsScatterL2.buffInfo.inBuffBaseOff = offsetL1;
    tempAlgParamsScatterL2.buffInfo.outBuffBaseOff = offsetL1;
    tempAlgParamsScatterL2.buffInfo.hcclBuffBaseOff = offsetL1;

    tempAlgParamsScatterL2.inputSliceStride = tempAlgParamsScatterL2.sliceSize;
    tempAlgParamsScatterL2.outputSliceStride = tempAlgParamsScatterL2.sliceSize;

    HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level] loop [%u] ScatterL2.sliceSize [%u], "
        "ScatterL2.tailSize [%u], ScatterL2.hcclBuffBaseOff [%u]",
        loop, tempAlgParamsScatterL2.sliceSize, tempAlgParamsScatterL2.tailSize,
        tempAlgParamsScatterL2.buffInfo.hcclBuffBaseOff);

    tempAlgParamsScatterL2.repeatNum = 1;
    tempAlgParamsScatterL2.inputRepeatStride = 0;
    tempAlgParamsScatterL2.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsAGL2(const u64 loop,
    const u64 currDataCount, const u64 sliceSizeL2, const u64 tailSizeL2,
    const u64 sliceSizeL1, TemplateDataParams &tempAlgParamsAGL2) const
{
    tempAlgParamsAGL2.count = currDataCount;
    u64 offsetL1 = rankIdxLevel1_ * sliceSizeL1;
    tempAlgParamsAGL2.buffInfo.inBuffBaseOff = offsetL1;
    tempAlgParamsAGL2.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsAGL2.buffInfo.hcclBuffBaseOff = offsetL1;

    tempAlgParamsAGL2.sliceSize = sliceSizeL2;
    tempAlgParamsAGL2.tailSize = tailSizeL2;

    tempAlgParamsAGL2.inputSliceStride = tempAlgParamsAGL2.sliceSize;
    tempAlgParamsAGL2.outputSliceStride = sliceSizeL1;

    HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level] loop [%u] AGL2.sliceSize [%u], "
        "AGL2.tailSize [%u], AGL2.hcclBuffBaseOff [%u]",
        loop, tempAlgParamsAGL2.sliceSize, tempAlgParamsAGL2.tailSize,
        tempAlgParamsAGL2.buffInfo.hcclBuffBaseOff);

    tempAlgParamsAGL2.repeatNum = 1;
    tempAlgParamsAGL2.inputRepeatStride = 0;
    tempAlgParamsAGL2.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsAGL1(const u64 loop,
    const u64 currDataCount, const u64 sliceSize, const u64 tailSize,
    TemplateDataParams &tempAlgParamsAGL1) const
{
    tempAlgParamsAGL1.count = currDataCount;
    tempAlgParamsAGL1.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsAGL1.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsAGL1.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsAGL1.sliceSize = sliceSize;
    tempAlgParamsAGL1.tailSize = tailSize;

    tempAlgParamsAGL1.inputSliceStride = tempAlgParamsAGL1.sliceSize;
    tempAlgParamsAGL1.outputSliceStride = tempAlgParamsAGL1.sliceSize;

    HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level] loop [%u] AGL1.sliceSize [%u], "
        "AGL1.tailSize [%u], AGL1.hcclBuffBaseOff [%u]",
        loop, tempAlgParamsAGL1.sliceSize, tempAlgParamsAGL1.tailSize,
        tempAlgParamsAGL1.buffInfo.hcclBuffBaseOff);

    tempAlgParamsAGL1.repeatNum = 1;
    tempAlgParamsAGL1.inputRepeatStride = 0;
    tempAlgParamsAGL1.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempAlgParamsAGL0(const u64 loop,
    const u64 currDataCount, const u64 processedDataCount, const u64 sliceSize, const u64 tailSize,
    TemplateDataParams &tempAlgParamsAGL0) const
{
    tempAlgParamsAGL0.count = currDataCount;
    tempAlgParamsAGL0.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsAGL0.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsAGL0.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsAGL0.sliceSize = sliceSize;
    tempAlgParamsAGL0.tailSize = tailSize;

    tempAlgParamsAGL0.inputSliceStride = 0;
    tempAlgParamsAGL0.outputSliceStride = tempAlgParamsAGL0.sliceSize;

    HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level] loop [%u] AGL0.sliceSize [%u], "
        "AGL0.tailSize [%u], AGL0.inBuffBaseOff [%u], AGL0.outBuffBaseOff [%u]",
        loop, tempAlgParamsAGL0.sliceSize, tempAlgParamsAGL0.tailSize,
        tempAlgParamsAGL0.buffInfo.inBuffBaseOff, tempAlgParamsAGL0.buffInfo.outBuffBaseOff);

    tempAlgParamsAGL0.repeatNum = 1;
    tempAlgParamsAGL0.inputRepeatStride = 0;
    tempAlgParamsAGL0.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
template <typename InsAlgTemplate>
HcclResult InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::GenTempResource(
    const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
    const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempResource) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR("[InsV2BroadcastSequenceExecutor3Level][GenTempResource] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%u]", channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::OrchestrateLoop(const OpParam &param,
    const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level][OrchestrateLoop] Start");

    TemplateDataParams tempAlgParamsScatterL0;
    TemplateDataParams tempAlgParamsScatterL1;
    TemplateDataParams tempAlgParamsScatterL2;
    TemplateDataParams tempAlgParamsAGL2;
    TemplateDataParams tempAlgParamsAGL1;
    TemplateDataParams tempAlgParamsAGL0;
    GenBaseTempAlgParams(param, resCtx, tempAlgParamsScatterL0, tempAlgParamsScatterL1,
        tempAlgParamsScatterL2, tempAlgParamsAGL2, tempAlgParamsAGL1, tempAlgParamsAGL0);

    std::shared_ptr<InsAlgTemplate0> algTemplateScatterL0 =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    CHK_RET(algTemplateScatterL0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));

    std::shared_ptr<InsAlgTemplate1> algTemplateScatterL1;
    if (!skipLevel1_) {
        algTemplateScatterL1 = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
        CHK_RET(algTemplateScatterL1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    std::shared_ptr<InsAlgTemplate2> algTemplateScatterL2 =
        std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);
    CHK_RET(algTemplateScatterL2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));

    std::shared_ptr<InsAlgTemplate3> algTemplateAGL2 =
        std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[2]);
    CHK_RET(algTemplateAGL2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));

    std::shared_ptr<InsAlgTemplate4> algTemplateAGL1;
    if (!skipLevel1_) {
        algTemplateAGL1 = std::make_shared<InsAlgTemplate4>(param, myRank_, algHierarchyInfo_.infos[1]);
        CHK_RET(algTemplateAGL1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    std::shared_ptr<InsAlgTemplate5> algTemplateAGL0 =
        std::make_shared<InsAlgTemplate5>(param, myRank_, algHierarchyInfo_.infos[0]);
    CHK_RET(algTemplateAGL0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));

    TemplateResource templateResourceScatterL0;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateScatterL0, templateResourceScatterL0));
    TemplateResource templateResourceScatterL1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(resCtx, 1, algTemplateScatterL1, templateResourceScatterL1));
    }
    TemplateResource templateResourceScatterL2;
    CHK_RET(GenTempResource(resCtx, 2, algTemplateScatterL2, templateResourceScatterL2));
    TemplateResource templateResourceAGL2;
    CHK_RET(GenTempResource(resCtx, 2, algTemplateAGL2, templateResourceAGL2));
    TemplateResource templateResourceAGL1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(resCtx, 1, algTemplateAGL1, templateResourceAGL1));
    }
    TemplateResource templateResourceAGL0;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateAGL0, templateResourceAGL0));

    // SetRoot for Scatter templates (before loop, called once)
    algTemplateScatterL0->SetRoot(root_);
    if (!skipLevel1_) {
        u32 scatterL1Root = root_ - rootIdx0_ + rankIdxLevel0_;
        algTemplateScatterL1->SetRoot(scatterL1Root);
    }
    u32 scatterL2Root = rootIdx2_ * (rankSizeLevel0_ * rankSizeLevel1_)
        + rankIdxLevel1_ * rankSizeLevel0_ + rankIdxLevel0_;
    algTemplateScatterL2->SetRoot(scatterL2Root);

    // Buffer sizing: single-segment, totalMult = ScatterL0's CalcScratchMultiple (=1)
    u64 totalMult = algTemplateScatterL0->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    if (totalMult == 0) {
        totalMult = 1;
    }
    u32 totalRankAlign = rankSizeLevel0_ * rankSizeLevel1_ * rankSizeLevel2_;
    u64 maxCountPerLoop = resCtx.cclMem.size / totalMult / HCCL_MIN_SLICE_ALIGN *
                          HCCL_MIN_SLICE_ALIGN / dataTypeSize_ / totalRankAlign * totalRankAlign;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR("[InsV2BroadcastSequenceExecutor3Level] maxCountPerLoop is 0, cclMemSize[%llu] too small",
            resCtx.cclMem.size);
        return HCCL_E_INTERNAL;
    }

    bool inRootFrame = (rankIdxLevel2_ == rootIdx2_ && rankIdxLevel1_ == rootIdx1_);
    bool inRootSupernode = (rankIdxLevel2_ == rootIdx2_);

    u64 processedDataCount = 0;
    u64 loop = 0;
    while (processedDataCount < dataCount_) {
        u64 remaining = dataCount_ - processedDataCount;
        u64 currDataCount = (remaining <= maxCountPerLoop) ? remaining : maxCountPerLoop;

        // ----------- ScatterL0: level0 Scatter (only root's frame) -----------
        GenTempAlgParamsScatterL0(loop, currDataCount, processedDataCount, tempAlgParamsScatterL0);
        if (inRootFrame) {
            CHK_RET(algTemplateScatterL0->KernelRun(param, tempAlgParamsScatterL0, templateResourceScatterL0));
        }

        // ----------- ScatterL1: level1 Scatter (only root's supernode) -----------
        u64 sliceSizeL1 = tempAlgParamsScatterL0.sliceSize;
        u64 tailSizeL1 = tempAlgParamsScatterL0.tailSize;
        if (!skipLevel1_) {
            GenTempAlgParamsScatterL1(loop, currDataCount, tempAlgParamsScatterL0.sliceSize,
                tempAlgParamsScatterL0.tailSize, tempAlgParamsScatterL1);
            if (inRootSupernode) {
                CHK_RET(algTemplateScatterL1->KernelRun(param, tempAlgParamsScatterL1, templateResourceScatterL1));
            }
            sliceSizeL1 = tempAlgParamsScatterL1.sliceSize;
            tailSizeL1 = tempAlgParamsScatterL1.tailSize;
        } else {
            sliceSizeL1 = tempAlgParamsScatterL0.sliceSize;
            tailSizeL1 = (rankIdxLevel0_ == rankSizeLevel0_ - 1) ?
                tempAlgParamsScatterL0.tailSize : tempAlgParamsScatterL0.sliceSize;
        }

        // ----------- ScatterL2: level2 Scatter (all ranks) -----------
        GenTempAlgParamsScatterL2(loop, currDataCount, sliceSizeL1, tailSizeL1, tempAlgParamsScatterL2);
        CHK_RET(algTemplateScatterL2->KernelRun(param, tempAlgParamsScatterL2, templateResourceScatterL2));

        // ----------- AGL2: level2 AllGather (all ranks) -----------
        GenTempAlgParamsAGL2(loop, currDataCount, tempAlgParamsScatterL2.sliceSize,
            tempAlgParamsScatterL2.tailSize, sliceSizeL1, tempAlgParamsAGL2);
        CHK_RET(algTemplateAGL2->KernelRun(param, tempAlgParamsAGL2, templateResourceAGL2));

        // ----------- AGL1: level1 AllGather (all ranks) -----------
        if (!skipLevel1_) {
            GenTempAlgParamsAGL1(loop, currDataCount, sliceSizeL1, tailSizeL1, tempAlgParamsAGL1);
            CHK_RET(algTemplateAGL1->KernelRun(param, tempAlgParamsAGL1, templateResourceAGL1));
        }

        // ----------- AGL0: level0 AllGather (all ranks) -----------
        GenTempAlgParamsAGL0(loop, currDataCount, processedDataCount, tempAlgParamsScatterL0.sliceSize,
            tempAlgParamsScatterL0.tailSize, tempAlgParamsAGL0);
        CHK_RET(algTemplateAGL0->KernelRun(param, tempAlgParamsAGL0, templateResourceAGL0));

        processedDataCount += currDataCount;
        loop++;
    }
    HCCL_INFO("[InsV2BroadcastSequenceExecutor3Level][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_BROADCAST,
    InsBroadcastSequenceMesh1DNHRNHR,
    InsV2BroadcastSequenceExecutor3Level,
    TopoMatchMultilevel,
    InsTempScatterMesh1D,
    InsTempScatterNHR,
    InsTempScatterNHR,
    InsTempAllGatherNHR,
    InsTempAllGatherNHR,
    InsTempAllGatherMesh1D);

}
