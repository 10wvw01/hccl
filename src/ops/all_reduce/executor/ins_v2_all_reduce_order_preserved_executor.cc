/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_order_preserved_executor.h"
#include "ins_temp_reduce_scatter_order_preserved_level1.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "alg_env_config.h"
#include "order_preserved_common.h"
#include <cmath>
#include <algorithm>

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::InsV2AllReduceOrderPreservedExecutor()
{
    deterministicStrict_ = true;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcAlgHierarchyInfo] myRank[%u], rankSize[%u] (flat level1 only)",
        myRank_, rankSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    InitExecutorInfo(param);
    CalcSizePerBlock(param);
    CalcGroupSlices(param);

    std::shared_ptr<InsAlgTemplateRS> rsTempAlg =
        std::make_shared<InsAlgTemplateRS>(param, myRank_, algHierarchyInfo.infos[0]);

    std::shared_ptr<InsAlgTemplateAG> agTempAlg =
        std::make_shared<InsAlgTemplateAG>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqRS;
    AlgResourceRequest resReqAG;

    CHK_RET(rsTempAlg->CalcRes(comm, param, topoInfo, resReqRS));
    CHK_RET(agTempAlg->CalcRes(comm, param, topoInfo, resReqAG));

    resourceRequest.slaveThreadNum = std::max(resReqRS.slaveThreadNum, resReqAG.slaveThreadNum);

    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReqRS.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqRS.notifyNumPerThread[i]);
        }
        if (i < resReqAG.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqAG.notifyNumPerThread[i]);
        }
    }

    resourceRequest.notifyNumOnMainThread = std::max(resReqRS.notifyNumOnMainThread,
        resReqAG.notifyNumOnMainThread);

    resourceRequest.channels.clear();
    if (resReqRS.channels.size() > 0) {
        resourceRequest.channels.push_back(resReqRS.channels[0]);
    }
    if (resReqAG.channels.size() > 0) {
        resourceRequest.channels.push_back(resReqAG.channels[0]);
    }

    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] slaveThreadNum[%u], notifyNumOnMainThread[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] Start");

    OrderPreservedBaseParams baseParams = InitOrderPreservedBaseParams(param, resCtx);
    SetOrderPreservedBaseParams(baseParams);
    
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    InitExecutorInfo(param);
    CalcSizePerBlock(param);
    CalcGroupSlices(param);

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] kernel run failed, err[0x%016llx]",
            HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
template <typename InsAlgTemplate>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::GenTempResource(
    const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
    const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempResource)
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR("[GenTempResource] channelLevelIdx[%u] should be lower than remoteRankToChannelInfo_.size()[%u]",
            channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    tempResource.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
void InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::InitTemplateDataParams(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, TemplateDataParams &tempAlgParams)
{
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}


template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] Start, deterministicStrict[%d] (flat level1)",
        deterministicStrict_);

    std::shared_ptr<InsAlgTemplateRS> rsTempAlg =
        std::make_shared<InsAlgTemplateRS>(param, myRank_, resCtx.algHierarchyInfo.infos[0]);
    rsTempAlg->SetchannelsPerRank(remoteRankToChannelInfo_[0]);

    std::shared_ptr<InsAlgTemplateAG> agTempAlg =
        std::make_shared<InsAlgTemplateAG>(param, myRank_, resCtx.algHierarchyInfo.infos[0]);
    agTempAlg->SetchannelsPerRank(remoteRankToChannelInfo_[0]);

    TemplateResource rsTemplateAlgRes;
    CHK_RET(GenTempResource(resCtx, 0, rsTempAlg, rsTemplateAlgRes));

    TemplateResource agTemplateAlgRes;
    CHK_RET(GenTempResource(resCtx, 0, agTempAlg, agTemplateAlgRes));

    TemplateDataParams tempAlgParams;
    InitTemplateDataParams(param, resCtx, tempAlgParams);

    outCclBuffSize_ = tempAlgParams.buffInfo.hcclBuff.size / 2;
    inCclBuffSize_ = tempAlgParams.buffInfo.hcclBuff.size - outCclBuffSize_;
    outCclBuffOffset_ = 0;
    inCclBuffOffset_ = outCclBuffSize_;
    HCCL_INFO("[OrchestrateLoop] outCclBuffSize_[%llu], inCclBuffSize_[%llu], "
        "outCclBuffOffset_[%llu], inCclBuffOffset_[%llu]",
        outCclBuffSize_, inCclBuffSize_, outCclBuffOffset_, inCclBuffOffset_);

    u64 maxCountPerLoop = outCclBuffSize_ / HCCL_MIN_SLICE_ALIGN *
                        HCCL_MIN_SLICE_ALIGN / dataTypeSize_ / rankSize_ * rankSize_;
    HCCL_INFO(
        "[OrchestrateLoop] maxCountPerLoop[%llu], outCclBuffSize_[%llu], dataTypeSize_[%llu], rankSize[%u]",
        maxCountPerLoop, outCclBuffSize_, dataTypeSize_, rankSize_);
    CHK_PRT_RET(maxCountPerLoop == 0,
        HCCL_ERROR("[OrchestrateLoop] maxCountPerLoop is 0"), HCCL_E_INTERNAL);

    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        u64 alignedCount = (currDataCount / rankSize_) * rankSize_;
        u64 tailCount = currDataCount - alignedCount;

        if (alignedCount > 0) {
            CHK_RET(RunReduceScatter(param, resCtx, alignedCount, 
                processedDataCount, rsTempAlg, rsTemplateAlgRes));
            CHK_RET(RunAllGather(param, resCtx, alignedCount,
                processedDataCount, agTempAlg, agTemplateAlgRes));
            processedDataCount += alignedCount;
        }

        if (tailCount > 0) {
            for (u64 tailIdx = 0; tailIdx < tailCount; tailIdx++) {
                u64 singleCount = 1;
                CHK_RET(RunReduceScatterSingle(param, resCtx, singleCount,
                    processedDataCount, rsTempAlg, rsTemplateAlgRes, tailIdx));
                CHK_RET(RunAllGatherSingle(param, resCtx, singleCount,
                    processedDataCount, agTempAlg, agTemplateAlgRes, tailIdx));
                processedDataCount += singleCount;
            }
        }
    }

    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] Success");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::InitExecutorInfo(const OpParam &param)
{
    deterministicStrict_ = true;
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][InitExecutorInfo] deterministicStrict[%d]",
        deterministicStrict_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcSizePerBlock(const OpParam &param)
{
    u64 sizePerBlock = (dataCount_ + rankSize_ - 1) / rankSize_ * dataTypeSize_;
    memInfo_.sizePerBlock = RoundUpWithDivisor(sizePerBlock, HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED);
    memInfo_.scratchMemFlag = false;
    memInfo_.totalSize = 0;
    HCCL_INFO("[CalcSizePerBlock] sizePerBlock[%llu], dataCount[%llu], rankSize[%u]",
        memInfo_.sizePerBlock, dataCount_, rankSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcGroupSlices(const OpParam &param)
{
    memInfo_.groupSize.clear();
    u64 sizeRemain = dataSize_;
    for (u32 rankId = 0; rankId < rankSize_; rankId++) {
        u64 size = (sizeRemain > memInfo_.sizePerBlock) ? memInfo_.sizePerBlock : sizeRemain;
        memInfo_.groupSize.push_back(size);
        sizeRemain -= size;
    }
    memInfo_.totalSize = std::max(memInfo_.sizePerBlock * rankSize_, dataSize_);
    HCCL_INFO("[CalcGroupSlices] groupSize.size[%u], totalSize[%llu]",
        memInfo_.groupSize.size(), memInfo_.totalSize);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RunReduceScatter(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    u64 currDataCount, u64 processedDataCount,
    std::shared_ptr<InsAlgTemplateRS> rsTempAlg, TemplateResource &rsTemplateAlgRes)
{
    TemplateDataParams rsTempAlgParams;
    rsTempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    rsTempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    rsTempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    rsTempAlgParams.buffInfo.inputPtr = param.inputPtr;
    rsTempAlgParams.buffInfo.outputPtr = resCtx.cclMem.addr;
    rsTempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    rsTempAlgParams.buffInfo.inputSize = param.inputSize;
    rsTempAlgParams.buffInfo.outputSize = outCclBuffSize_;
    rsTempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    rsTempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    rsTempAlgParams.buffInfo.outBuffBaseOff = outCclBuffOffset_;
    rsTempAlgParams.buffInfo.hcclBuffBaseOff = inCclBuffOffset_;

    // ReduceScatter: INPUT -> HCCL_BUFFER(outCclBuff)
    // memBlockInfo中的offset是相对于整个hcclBuff.addr的绝对偏移
    MemBlockInfo memBlockInfo;
    memBlockInfo.size.clear();
    memBlockInfo.userInputOffsets.clear();
    memBlockInfo.inputOffsets.clear();
    memBlockInfo.outputOffsets.clear();
    
    u64 unitSize = dataTypeSize_;
    u64 sliceCount = currDataCount / rankSize_;
    u64 rsSliceSize = sliceCount * unitSize;

    for (u32 dataId = 0; dataId < rankSize_; dataId++) {
        u64 userMemInOffset = processedDataCount * unitSize + dataId * rsSliceSize;
        
        memBlockInfo.size.push_back(rsSliceSize);
        memBlockInfo.userInputOffsets.push_back(userMemInOffset);
        memBlockInfo.inputOffsets.push_back(inCclBuffOffset_ + dataId * rsSliceSize);
        memBlockInfo.outputOffsets.push_back(inCclBuffOffset_ + dataId * rsSliceSize);
    }
    
    std::vector<u64> rsSliceSizes(rankSize_, rsSliceSize);
    rsTempAlgParams.allRankSliceSize = rsSliceSizes;
    rsTempAlgParams.sliceSize = rsSliceSize;
    rsTempAlgParams.tailSize = rsSliceSize;
    rsTempAlgParams.inputSliceStride = rsSliceSize;
    rsTempAlgParams.outputSliceStride = 0;
    rsTempAlgParams.repeatNum = 1;
    rsTempAlgParams.inputRepeatStride = 0;
    rsTempAlgParams.outputRepeatStride = 0;
    rsTempAlgParams.count = sliceCount;
    
    rsTempAlg->SetMemBlockInfo(memBlockInfo);
    CHK_RET(rsTempAlg->KernelRun(param, rsTempAlgParams, rsTemplateAlgRes));
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RunReduceScatterSingle(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    u64 currDataCount, u64 processedDataCount,
    std::shared_ptr<InsAlgTemplateRS> rsTempAlg, TemplateResource &rsTemplateAlgRes,
    u64 tailIdx)
{
    TemplateDataParams rsTempAlgParams;
    rsTempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    rsTempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    rsTempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    rsTempAlgParams.buffInfo.inputPtr = param.inputPtr;
    rsTempAlgParams.buffInfo.outputPtr = resCtx.cclMem.addr;
    rsTempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    rsTempAlgParams.buffInfo.inputSize = param.inputSize;
    rsTempAlgParams.buffInfo.outputSize = outCclBuffSize_;
    rsTempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    rsTempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    rsTempAlgParams.buffInfo.outBuffBaseOff = outCclBuffOffset_;
    rsTempAlgParams.buffInfo.hcclBuffBaseOff = inCclBuffOffset_;

    MemBlockInfo memBlockInfo;
    memBlockInfo.size.clear();
    memBlockInfo.userInputOffsets.clear();
    memBlockInfo.inputOffsets.clear();
    memBlockInfo.outputOffsets.clear();
    
    u64 unitSize = dataTypeSize_;
    u64 sliceSize = currDataCount * unitSize;

    for (u32 dataId = 0; dataId < rankSize_; dataId++) {
        u64 userMemInOffset = processedDataCount * unitSize;
        
        memBlockInfo.size.push_back(sliceSize);
        memBlockInfo.userInputOffsets.push_back(userMemInOffset);
        memBlockInfo.inputOffsets.push_back(inCclBuffOffset_ + dataId * sliceSize);
        memBlockInfo.outputOffsets.push_back(inCclBuffOffset_ + dataId * sliceSize);
    }
    
    std::vector<u64> rsSliceSizes(rankSize_, sliceSize);
    rsTempAlgParams.allRankSliceSize = rsSliceSizes;
    rsTempAlgParams.sliceSize = sliceSize;
    rsTempAlgParams.tailSize = sliceSize;
    rsTempAlgParams.inputSliceStride = 0;
    rsTempAlgParams.outputSliceStride = 0;
    rsTempAlgParams.repeatNum = 1;
    rsTempAlgParams.inputRepeatStride = 0;
    rsTempAlgParams.outputRepeatStride = 0;
    rsTempAlgParams.count = currDataCount;
    
    rsTempAlg->SetMemBlockInfo(memBlockInfo);
    CHK_RET(rsTempAlg->KernelRun(param, rsTempAlgParams, rsTemplateAlgRes));
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RunAllGather(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    u64 currDataCount, u64 processedDataCount,
    std::shared_ptr<InsAlgTemplateAG> agTempAlg, TemplateResource &agTemplateAlgRes)
{
    TemplateDataParams agTempAlgParams;
    agTempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    agTempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    agTempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    agTempAlgParams.buffInfo.inputPtr = resCtx.cclMem.addr;
    agTempAlgParams.buffInfo.outputPtr = param.outputPtr;
    agTempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    agTempAlgParams.buffInfo.inputSize = outCclBuffSize_;
    agTempAlgParams.buffInfo.outputSize = param.outputSize;
    agTempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    agTempAlgParams.buffInfo.inBuffBaseOff = outCclBuffOffset_;
    agTempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    agTempAlgParams.buffInfo.hcclBuffBaseOff = inCclBuffOffset_;

    u64 agSliceSize = currDataCount / rankSize_ * dataTypeSize_;
    u64 agTailSize = (currDataCount / rankSize_ + currDataCount % rankSize_) * dataTypeSize_;
    
    std::vector<u64> agSliceSizes(rankSize_, agSliceSize);
    agTempAlgParams.allRankSliceSize = agSliceSizes;
    agTempAlgParams.sliceSize = agSliceSize;
    agTempAlgParams.tailSize = agSliceSize;
    agTempAlgParams.inputSliceStride = 0;
    agTempAlgParams.outputSliceStride = agSliceSize;
    agTempAlgParams.repeatNum = 1;
    agTempAlgParams.inputRepeatStride = 0;
    agTempAlgParams.outputRepeatStride = 0;
    agTempAlgParams.count = sliceCount;
    
    CHK_RET(agTempAlg->KernelRun(param, agTempAlgParams, agTemplateAlgRes));
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RunAllGatherSingle(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    u64 currDataCount, u64 processedDataCount,
    std::shared_ptr<InsAlgTemplateAG> agTempAlg, TemplateResource &agTemplateAlgRes,
    u64 tailIdx)
{
    TemplateDataParams agTempAlgParams;
    agTempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    agTempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    agTempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    agTempAlgParams.buffInfo.inputPtr = resCtx.cclMem.addr;
    agTempAlgParams.buffInfo.outputPtr = param.outputPtr;
    agTempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    agTempAlgParams.buffInfo.inputSize = outCclBuffSize_;
    agTempAlgParams.buffInfo.outputSize = param.outputSize;
    agTempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    agTempAlgParams.buffInfo.inBuffBaseOff = outCclBuffOffset_;
    agTempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    agTempAlgParams.buffInfo.hcclBuffBaseOff = inCclBuffOffset_;

    u64 agSliceSize = currDataCount * dataTypeSize_;
    
    std::vector<u64> agSliceSizes(rankSize_, agSliceSize);
    agTempAlgParams.allRankSliceSize = agSliceSizes;
    agTempAlgParams.sliceSize = agSliceSize;
    agTempAlgParams.tailSize = agSliceSize;
    agTempAlgParams.inputSliceStride = 0;
    agTempAlgParams.outputSliceStride = 0;
    agTempAlgParams.repeatNum = 1;
    agTempAlgParams.inputRepeatStride = 0;
    agTempAlgParams.outputRepeatStride = 0;
    agTempAlgParams.count = currDataCount;
    
    CHK_RET(agTempAlg->KernelRun(param, agTempAlgParams, agTemplateAlgRes));
    
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, AllReduceOrderPreserved,
    InsV2AllReduceOrderPreservedExecutor, TopoMatch1D,
    InsTempReduceScatterOrderPreservedLevel1, InsTempAllGatherMesh1D);

}