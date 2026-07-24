/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "ins_v2_reduce_omnipipe_executor.h"
#include "alg_data_trans_wrapper.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_reduce_scatter_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_omnipipe_nhr1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_omnipipe_mesh1d.h"
#include "ccu_temp_gather_omnipipe_mesh_1d_mem2mem.h"
#include "ccu_temp_gather_omnipipe_nhr1d_mem2mem.h"
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
namespace ops_hccl {

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::CcuV2ReduceOmniPipeExecutor()
{
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::CalcAlgHierarchyInfo(HcclComm comm, 
            TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    u32 userrank = topoInfo->userRank;
    HCCL_DEBUG("[%s] myRank[%u]", __func__, userrank);
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));

    for (auto i = 0; i < algHierarchyInfo.infos.size(); ++i) {
        for (auto j = 0; j < algHierarchyInfo.infos[i].size(); ++j) {
            for (auto k = 0; k < algHierarchyInfo.infos[i][j].size(); ++k) {
                HCCL_DEBUG("[%s] myRank[%u] (%d, %d, %d) %u", __func__, topoInfo->userRank, i, j, k,
                    algHierarchyInfo.infos[i][j][k]);
            }
        }
    }

    return HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::InitCommInfo(
            const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    
    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
		HCCL_ERROR("[%s] rankSizeLevel0 is 0", __func__);
		return HcclResult::HCCL_E_PARA;
	}
    rankSizeLevel1_ = algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    if (rankSizeLevel1_ == 0) {
		HCCL_ERROR("[%s] rankSizeLevel1 is 0", __func__);
		return HcclResult::HCCL_E_PARA;
	}
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;

    rootx = param.root % rankSizeLevel0_;
    rooty = param.root / rankSizeLevel0_;

    bool isRoot = (myRank_ == param.root);
    isSameYAxisAsRoot = (rankIdxLevel0_ == rootx && !isRoot); // 同x，走NHR
    isSameXAxisAsRoot = (rankIdxLevel1_ == rooty && !isRoot); // 同y，走mesh
    

    HCCL_DEBUG("[%s]myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] "
        "rankIdxLevel1[%u] devType[%u] dataCount[%u] dataType[%u] dataTypeSize[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_, devType_,
        dataCount_, dataType_, dataTypeSize_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::CalcResLevel(
            HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
            AlgResourceRequest& resReqlevel, AlgResourceRequest& resourceReq, const int& curLevel)
{
    resourceReq.slaveThreadNum += resReqlevel.slaveThreadNum; // 从流数 一般是0
    resourceReq.notifyNumOnMainThread += resReqlevel.notifyNumOnMainThread; // 一般是0
    resourceReq.notifyNumPerThread.insert(resourceReq.notifyNumPerThread.end(), // 一般是0
                                            resReqlevel.notifyNumPerThread.begin(),
                                            resReqlevel.notifyNumPerThread.end());
    
    //资源组的值一样就一起申请，资源组的值不一样就串行申请，前一个销毁后后一个申请
    HCCL_DEBUG("[%s] currTemplate has [%d] kernels.", __func__, resReqlevel.ccuKernelNum[0]);
    if (curLevel == OMNIPIPE_RS_LEVEL0 || curLevel == OMNIPIPE_RS_LEVEL1) {
        std::for_each(resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end(), [](CcuKernelInfo &info) {
            info.resGroup = 0;
        });
    } else {
        std::for_each(resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end(), [](CcuKernelInfo &info) {
            info.resGroup = 1;
        });
    }
    resourceReq.ccuKernelInfos.insert(resourceReq.ccuKernelInfos.end(), resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end()); //不需要改，无脑放
    resourceReq.ccuKernelNum.insert(resourceReq.ccuKernelNum.end(), resReqlevel.ccuKernelNum.begin(), resReqlevel.ccuKernelNum.end());

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::InitSubCommRanks(
            std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    subCommRanks0.clear();
    subCommRanks1.clear();

    subCommRanks0.push_back(algHierarchyInfo.infos[0][0]);
    subCommRanks1.resize(1);
    for (auto i = myRank_ % rankSizeLevel0_; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0].push_back(algHierarchyInfo.infos[0][1][i]);
    }

    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    HCCL_DEBUG("start CalcRes");
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));

    // 初始化通信域subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    CHK_RET(InitSubCommRanks(subCommRanks0, subCommRanks1, algHierarchyInfo));

    CcuRsAlgTemplateX rsAlgTempLevelX(param, myRank_, subCommRanks0);
	CcuRsAlgTemplateY rsAlgTempLevelY(param, myRank_, subCommRanks1);
    CcuGAlgTemplateX gAlgTempLevelX(param, myRank_, subCommRanks0);
	CcuGAlgTemplateY gAlgTempLevelY(param, myRank_, subCommRanks1);

    // 计算调用每一个template的资源
    resourceRequest.slaveThreadNum = 0; //ccu内部没有从流和notify
    resourceRequest.notifyNumOnMainThread = 0;

    AlgResourceRequest resRsReqLevelX;
    CHK_RET(rsAlgTempLevelX.CalcRes(comm, param, topoInfo, resRsReqLevelX));
    AlgResourceRequest resRsReqLevelY;
    CHK_RET(rsAlgTempLevelY.CalcRes(comm, param, topoInfo, resRsReqLevelY));
    AlgResourceRequest resGReqLevelX;
    CHK_RET(gAlgTempLevelX.CalcRes(comm, param, topoInfo, resGReqLevelX));
    AlgResourceRequest resGReqLevelY;
    gAlgTempLevelY.SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
    CHK_RET(gAlgTempLevelY.CalcRes(comm, param, topoInfo, resGReqLevelY));
    

    CHK_RET(CalcResLevel(comm, param, topoInfo, resRsReqLevelX, resourceRequest, 0));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resRsReqLevelY, resourceRequest, 1));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resGReqLevelX, resourceRequest, 2));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resGReqLevelY, resourceRequest, 3));

    resourceRequest.slaveThreadNum += 1; // 需要一个主流和一个从流来并行2d   
    resourceRequest.notifyNumOnMainThread += 1; 
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[%s] slaveThreadNum:%d, notifyNumOnMainThread:%d", __func__, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    HCCL_DEBUG("end CalcRes");
    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_DEBUG("[%s] start", __func__);
    threads_ = resCtx.threads;
    HCCL_DEBUG("[%s]threads size: %u", __func__, threads_.size());
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    maxTmpMemSize_ = resCtx.cclMem.size;
    rankSizeLevel0_ = resCtx.algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    rankSizeLevel1_ = resCtx.algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel1 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;

    rootx = param.root % rankSizeLevel0_;
    rooty = param.root / rankSizeLevel0_;
    bool isRoot = (myRank_ == param.root);
    isSameYAxisAsRoot = (rankIdxLevel0_ == rootx && !isRoot);
    isSameXAxisAsRoot = (rankIdxLevel1_ == rooty && !isRoot);
    
    HCCL_DEBUG("[%s] myRank[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        __func__, myRank_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CcuV2ReduceOmniPipeExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    return HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::GenTemplateAlgParamsByDimData(
            TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount)
{
    tempAlgParams.count = 0;

    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.localCopyFlag = 0;
    return HcclResult::HCCL_SUCCESS;
}
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::GenTempAlgParamsIn2HCCLBuff(
    TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount, const AlgResourceCtxSerializable &resCtx, const OpParam &param)
{
    tempAlgParams.count = 0;
    tempAlgParams.dataType = dataType_;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = param.inputPtr;
    stepSliceInfo.buffInfo.inputSize = param.inputSize;
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.outputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.inBuffType = BufferType::INPUT;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    u64 inputOffset = stepSliceInfo.buffInfo.inBuffBaseOff +  processedDataCount * dataTypeSize_;
    u64 outputOffset = stepSliceInfo.buffInfo.outBuffBaseOff;
    // Gather 可能读取 stepSliceInfo
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff = inputOffset;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff = outputOffset;
    // FastLaunch 或其他模板可能读取顶层 buffInfo
    tempAlgParams.buffInfo.inBuffBaseOff = inputOffset;
    tempAlgParams.buffInfo.outBuffBaseOff = outputOffset;

    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;

    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::GenTempAlgParamsHCCLBuff2HCCLBuff(
    TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount, const AlgResourceCtxSerializable &resCtx, const OpParam &param)
{
    tempAlgParams.count = 0;
    tempAlgParams.dataType = dataType_;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.inputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.outputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;

    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::CalcEndpointBandwidth(std::vector<double> &endpointAttrBwAvgRS,
    std::vector<double> &endpointAttrBwAvgG)
{
    // RS带宽: Level0走mesh, Level1走clos（按rankSizeLevel1_-1均摊）
    double eqBwLevel0RS = BW_OMNI_UBX_CCU_SCHED_RS_MESH;
    double eqBwLevel1RS = BW_OMNI_UBX_CCU_SCHED_R_RS_CLOS;
    eqBwLevel1RS = rankSizeLevel1_ > 1 ? eqBwLevel1RS / (rankSizeLevel1_ - 1) : eqBwLevel1RS;
    endpointAttrBwAvgRS = {eqBwLevel0RS, eqBwLevel1RS, 1.0};

    // G带宽: Level0走mesh, Level1走clos（按rankSizeLevel1_-1均摊）
    double eqBwLevel0G = BW_OMNI_UBX_CCU_SCHED_G_MESH;
    double eqBwLevel1G = BW_OMNI_UBX_CCU_SCHED_G_CLOS;
    eqBwLevel1G = rankSizeLevel1_ > 1 ? eqBwLevel1G / (rankSizeLevel1_ - 1) : eqBwLevel1G;
    endpointAttrBwAvgG = {eqBwLevel0G, eqBwLevel1G, 1.0};

    HCCL_DEBUG("[%s] eqBwLevel0RS:%f, eqBwLevel1RS:%f, eqBwLevel0G:%f, eqBwLevel1G:%f", __func__, eqBwLevel0RS, eqBwLevel1RS, eqBwLevel0G, eqBwLevel1G);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::OrchestrateLoop(
            const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_DEBUG("[%s] Start", __func__);
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    subCommRanks0.push_back(resCtx.algHierarchyInfo.infos[0][0]);
    subCommRanks1.resize(1);
    for (auto i = myRank_ % rankSizeLevel0_; i < resCtx.algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0].push_back(resCtx.algHierarchyInfo.infos[0][1][i]);
    }
    bool isRoot = (myRank_ == param.root);
    CcuRsAlgTemplateX rsAlgTempX(param, myRank_, subCommRanks0);
    CcuRsAlgTemplateY rsAlgTempY(param, myRank_, subCommRanks1);
    CcuGAlgTemplateX gAlgTempX(param, myRank_, subCommRanks0);
    CcuGAlgTemplateY gAlgTempY(param, myRank_, subCommRanks1);
    rootx = param.root % rankSizeLevel0_;
    rooty = param.root / rankSizeLevel0_;
    TemplateDataParams tempAlgParamsCommon;
    tempAlgParamsCommon.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsCommon.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsCommon.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsCommon.buffInfo.hcclBuffSize = resCtx.cclMem.size;
    TemplateResource templateResourceRsX, templateResourceRsY, templateResourceGX, templateResourceGY;
    CHK_RET(InitLoopResources(resCtx, templateResourceRsX, templateResourceRsY, templateResourceGX,
                              templateResourceGY));
    std::vector<u64> allRankSplitData;
    std::vector<std::vector<u64>> multiLoopAllRankSplitData;
    u64 maxCountPerLoop = 0;
    u32 loopTimes = 0;
    std::vector<double> endpointAttrBwAvgRS, endpointAttrBwAvgG;
    CHK_RET(CalcLoopDataParams(param, resCtx, allRankSplitData, multiLoopAllRankSplitData, maxCountPerLoop,
                               loopTimes, endpointAttrBwAvgRS, endpointAttrBwAvgG));
    OmniPipeSliceParam sliceParam;
    BuildSliceParam(param, allRankSplitData, multiLoopAllRankSplitData, sliceParam);
    CHK_RET(RunMainLoop(param, resCtx, rsAlgTempX, rsAlgTempY, gAlgTempX, gAlgTempY, tempAlgParamsCommon,
        templateResourceRsX, templateResourceRsY, templateResourceGX, templateResourceGY, allRankSplitData,
        multiLoopAllRankSplitData, maxCountPerLoop, loopTimes, endpointAttrBwAvgRS, endpointAttrBwAvgG,
        sliceParam, isRoot));
    HCCL_DEBUG("[%s][OrchestrateLoop] Endxx.", __func__);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::RunMainLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, CcuRsAlgTemplateX& rsAlgTempX,
    CcuRsAlgTemplateY& rsAlgTempY, CcuGAlgTemplateX& gAlgTempX, CcuGAlgTemplateY& gAlgTempY,
    const TemplateDataParams& tempAlgParamsCommon, TemplateResource& templateResourceRsX,
    TemplateResource& templateResourceRsY, TemplateResource& templateResourceGX,
    TemplateResource& templateResourceGY, const std::vector<u64>& allRankSplitData,
    const std::vector<std::vector<u64>>& multiLoopAllRankSplitData, u64 maxCountPerLoop, u32 loopTimes,
    const std::vector<double>& endpointAttrBwAvgRS, const std::vector<double>& endpointAttrBwAvgG,
    const OmniPipeSliceParam& sliceParam, bool isRoot)
{
    TemplateDataParams tempRsAlgParamsX = tempAlgParamsCommon;
    TemplateDataParams tempRsAlgParamsY = tempAlgParamsCommon;
    TemplateDataParams tempGAlgParamsX = tempAlgParamsCommon;
    TemplateDataParams tempGAlgParamsY = tempAlgParamsCommon;
    OmniPipeSliceInfo omniPipeSliceInfoRS, omniPipeSliceInfoG;
    std::vector<u64> processedDataCountTmp(rankSize_, 0);
    u64 processedDataCount = 0;
    OmniPipeSliceParam sp = sliceParam;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        gAlgTempX.SetRoot(rankIdxLevel1_ * rankSizeLevel0_ + rootx);
        gAlgTempY.SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
        CHK_PRT_RET(multiLoopAllRankSplitData.size() <= loop,
            HCCL_ERROR("[CcuV2ReduceOmniPipeExecutor][Orchestrate] multiLoopAllRankSplitData.size() <= loop"),
            HCCL_E_PARA);
        if (loop == 0 || !isSameLoop(multiLoopAllRankSplitData[loop - 1], multiLoopAllRankSplitData[loop])) {
            sp.dataSizePerLoop = CalcCountToDataSize(multiLoopAllRankSplitData[loop], dataTypeSize_);
            sp.dataWholeSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
            sp.endpointAttrBw = endpointAttrBwAvgRS;
            omniPipeSliceInfoRS = CalcRSOmniPipeSliceInfo(sp);
            sp.endpointAttrBw = endpointAttrBwAvgG;
            omniPipeSliceInfoG = CalcGatherOmniPipeSliceInfo(sp);
        }
        CHK_RET(RunRsInnerLoop(param, omniPipeSliceInfoRS, processedDataCount, tempRsAlgParamsX,
                               tempRsAlgParamsY, rsAlgTempX, rsAlgTempY, templateResourceRsX,
                               templateResourceRsY));
        CHK_RET(RunGatherInnerLoop(param, resCtx, omniPipeSliceInfoG, processedDataCount, loop,
            tempGAlgParamsX, tempGAlgParamsY, gAlgTempX, gAlgTempY, templateResourceGX, templateResourceGY,
            isRoot, isSameXAxisAsRoot, isSameYAxisAsRoot));
        if (myRank_ == param.root) {
            CHK_RET(RunGatherLocalCopy(param, resCtx, allRankSplitData, multiLoopAllRankSplitData, loop,
                processedDataCount, tempAlgParamsCommon, gAlgTempX, templateResourceGX,
                processedDataCountTmp));
        }
        processedDataCount += maxCountPerLoop;
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::InitLoopResources(
    const AlgResourceCtxSerializable& resCtx, TemplateResource& templateResourceRsX,
    TemplateResource& templateResourceRsY, TemplateResource& templateResourceGX,
    TemplateResource& templateResourceGY)
{
    auto fillResource = [&resCtx](TemplateResource& r, u32 threadIdx, u64 begin, u64 end) {
        r.threads.push_back(resCtx.threads[threadIdx]);
        r.ccuKernels.insert(r.ccuKernels.end(), resCtx.ccuKernels.begin() + begin,
                            resCtx.ccuKernels.begin() + end);
    };
    fillResource(templateResourceRsX, 0, 0, resCtx.ccuKernelNum[0]);
    fillResource(templateResourceRsY, 1, resCtx.ccuKernelNum[0], resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
    u64 off = resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1];
    fillResource(templateResourceGX, 0, off, off + resCtx.ccuKernelNum[2]);
    fillResource(templateResourceGY, 1, off + resCtx.ccuKernelNum[2],
                 off + resCtx.ccuKernelNum[2] + resCtx.ccuKernelNum[3]);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::CalcLoopDataParams(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, std::vector<u64>& allRankSplitData,
    std::vector<std::vector<u64>>& multiLoopAllRankSplitData, u64& maxCountPerLoop, u32& loopTimes,
    std::vector<double>& endpointAttrBwAvgRS, std::vector<double>& endpointAttrBwAvgG)
{
    CHK_RET(CalcEndpointBandwidth(endpointAttrBwAvgRS, endpointAttrBwAvgG));
    allRankSplitData = OmniPipeSplitData(rankSize_, dataCount_, dataTypeSize_);
    maxTmpMemSize_ = resCtx.cclMem.size;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 scratchBoundDataSize = maxTmpMemSize_ / rankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    HCCL_DEBUG("[%s] myRank[%u] transportBoundDataSize[%u] scratchBoundDataSize[%u]", __func__, myRank_,
               transportBoundDataSize, scratchBoundDataSize);
    maxCountPerLoop = std::min(transportBoundDataSize, scratchBoundDataSize) / dataTypeSize_;
    CHK_PRT_RET(maxCountPerLoop == 0, HCCL_ERROR("[%s] maxCountPerLoop is 0", __func__), HCCL_E_INTERNAL);
    loopTimes = allRankSplitData[0] / maxCountPerLoop + ((allRankSplitData[0] % maxCountPerLoop == 0) ? 0 : 1);
    HCCL_DEBUG("[%s] myRank[%u] maxCountPerLoop[%u] loopTimes[%u]", __func__, myRank_, maxCountPerLoop, loopTimes);
    multiLoopAllRankSplitData = OmniPipeSplitRankDataLoop(allRankSplitData, maxCountPerLoop, loopTimes,
                                                          dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
void CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::BuildSliceParam(
    const OpParam& param, const std::vector<u64>& allRankSplitData,
    const std::vector<std::vector<u64>>& multiLoopAllRankSplitData, OmniPipeSliceParam& sliceParam)
{
    sliceParam.dataSizePerLoop = CalcCountToDataSize(multiLoopAllRankSplitData[0], dataTypeSize_);
    sliceParam.dataWholeSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    sliceParam.levelAlgType = {1, 0, 1};
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = CommEngine::COMM_ENGINE_CCU;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::RunRsInnerLoop(
    const OpParam& param, const OmniPipeSliceInfo& sliceInfoRS, u64 processedDataCount,
    TemplateDataParams& tempRsAlgParamsX, TemplateDataParams& tempRsAlgParamsY,
    CcuRsAlgTemplateX& rsAlgTempX, CcuRsAlgTemplateY& rsAlgTempY,
    TemplateResource& templateResourceRsX, TemplateResource& templateResourceRsY)
{
    ThreadHandle mainThread = threads_[0];
    std::vector<ThreadHandle> syncThreads{threads_[1]};
    std::vector<u32> notifyIdxesMainToSub{0};
    std::vector<u32> notifyIdxesSubToMain{0};
    auto level0StepCountRS = sliceInfoRS.dataSliceLevel0.size();
    for (auto i = 0; i < level0StepCountRS; ++i) {
        CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
        GenTemplateAlgParamsByDimData(tempRsAlgParamsX, sliceInfoRS.dataSliceLevel0[i], processedDataCount);
        CHK_RET(rsAlgTempX.KernelRun(param, tempRsAlgParamsX, templateResourceRsX));
        GenTemplateAlgParamsByDimData(tempRsAlgParamsY, sliceInfoRS.dataSliceLevel1[i], processedDataCount);
        CHK_RET(rsAlgTempY.KernelRun(param, tempRsAlgParamsY, templateResourceRsY));
        CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::RunGatherInnerLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, const OmniPipeSliceInfo& sliceInfoG,
    u64 processedDataCount, u64 loop, TemplateDataParams& tempGAlgParamsX, TemplateDataParams& tempGAlgParamsY,
    CcuGAlgTemplateX& gAlgTempX, CcuGAlgTemplateY& gAlgTempY, TemplateResource& templateResourceGX,
    TemplateResource& templateResourceGY, bool isRoot, bool isSameXAxisAsRoot, bool isSameYAxisAsRoot)
{
    ThreadHandle mainThread = threads_[0];
    std::vector<ThreadHandle> syncThreads{threads_[1]};
    std::vector<u32> notifyIdxesMainToSub{0};
    std::vector<u32> notifyIdxesSubToMain{0};
    u32 level0StepCountAG = sliceInfoG.dataSliceLevel0.size();
    for (u32 i = 0; i < level0StepCountAG; i++) {
        CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
        CHK_RET(GenTempAlgParamsIn2HCCLBuff(tempGAlgParamsX, sliceInfoG.dataSliceLevel0[i], processedDataCount,
                                            resCtx, param));
        CHK_RET(GenTempAlgParamsIn2HCCLBuff(tempGAlgParamsY, sliceInfoG.dataSliceLevel1[i], processedDataCount,
                                            resCtx, param));
        gAlgTempX.SetRoot(rankIdxLevel1_ * rankSizeLevel0_ + rootx);
        gAlgTempY.SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
        CHK_RET(SetupGatherStepParams(param, resCtx, sliceInfoG, processedDataCount, i, level0StepCountAG,
            tempGAlgParamsX, tempGAlgParamsY, gAlgTempX, gAlgTempY, isRoot, isSameXAxisAsRoot,
            isSameYAxisAsRoot));
        gAlgTempX.isStepOne_ = (i == 0);
        gAlgTempX.isloopOne_ = (loop == 0);
        gAlgTempX.isLastStep_ = (i == level0StepCountAG - 1);
        CHK_RET(gAlgTempX.KernelRun(param, tempGAlgParamsX, templateResourceGX));
        gAlgTempY.isStepOne_ = (i == 0);
        gAlgTempY.isloopOne_ = (loop == 0);
        gAlgTempY.isLastStep_ = (i == level0StepCountAG - 1);
        CHK_RET(gAlgTempY.KernelRun(param, tempGAlgParamsY, templateResourceGY));
        CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::SetupGatherStepParams(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, const OmniPipeSliceInfo& sliceInfoG,
    u64 processedDataCount, u32 i, u32 level0StepCountAG, TemplateDataParams& tempGAlgParamsX,
    TemplateDataParams& tempGAlgParamsY, CcuGAlgTemplateX& gAlgTempX, CcuGAlgTemplateY& gAlgTempY,
    bool isRoot, bool isSameXAxisAsRoot, bool isSameYAxisAsRoot)
{
    gAlgTempY.ifDoTask_ = (isSameYAxisAsRoot || isRoot);
    if (i == 0) {
        gAlgTempY.ifDoTask_ = true;
        return HCCL_SUCCESS;
    }
    if (i == level0StepCountAG - 1) {
        if (isSameYAxisAsRoot && !isRoot) {
            CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(tempGAlgParamsY, sliceInfoG.dataSliceLevel1[i],
                                                       processedDataCount, resCtx, param));
            gAlgTempX.UnsetRoot(myRank_);
        } else if (isSameXAxisAsRoot && !isRoot) {
            CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(tempGAlgParamsX, sliceInfoG.dataSliceLevel0[i],
                                                       processedDataCount, resCtx, param));
            gAlgTempY.UnsetRoot(myRank_);
        } else if (!isRoot) {
            gAlgTempX.UnsetRoot(myRank_);
            gAlgTempY.UnsetRoot(myRank_);
        }
        return HCCL_SUCCESS;
    }
    if (isRoot) {
        return HCCL_SUCCESS;
    }
    if (isSameYAxisAsRoot) {
        CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(tempGAlgParamsY, sliceInfoG.dataSliceLevel1[i],
                                                   processedDataCount, resCtx, param));
    } else if (isSameXAxisAsRoot) {
        CHK_RET(GenTempAlgParamsIn2HCCLBuff(tempGAlgParamsX, sliceInfoG.dataSliceLevel0[i],
                                             processedDataCount, resCtx, param));
        gAlgTempY.UnsetRoot(myRank_);
    } else {
        CHK_RET(GenTempAlgParamsIn2HCCLBuff(tempGAlgParamsX, sliceInfoG.dataSliceLevel0[i],
                                             processedDataCount, resCtx, param));
        gAlgTempY.UnsetRoot(myRank_);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::RunGatherLocalCopy(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, const std::vector<u64>& allRankSplitData,
    const std::vector<std::vector<u64>>& multiLoopAllRankSplitData, u64 loop, u64 processedDataCount,
    const TemplateDataParams& tempAlgParamsCommon, CcuGAlgTemplateX& gAlgTempX,
    TemplateResource& templateResourceGX, std::vector<u64>& processedDataCountTmp)
{
    ThreadHandle mainThread = threads_[0];
    std::vector<ThreadHandle> syncThreads{threads_[1]};
    std::vector<u32> notifyIdxesMainToSub{0};
    std::vector<u32> notifyIdxesSubToMain{0};
    u64 rankOffset = 0;
    u64 rankLoopOffset = 0;
    CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
    for (u32 i = 0; i < rankSize_; i++) {
        if (loop != 0) {
            processedDataCountTmp[i] += multiLoopAllRankSplitData[loop - 1][i];
        }
    }
    for (u32 i = 0; i < rankSize_; i++) {
        u64 currDataCountTmp = multiLoopAllRankSplitData[loop][i];
        if (currDataCountTmp == 0) {
            rankOffset += allRankSplitData[i] * dataTypeSize_;
            continue;
        }
        TemplateDataParams tempAlgParamLocalCopy = tempAlgParamsCommon;
        BuildLocalCopyParam(param, resCtx, tempAlgParamLocalCopy, i, currDataCountTmp, rankOffset,
                            rankLoopOffset, processedDataCountTmp);
        CHK_RET(gAlgTempX.KernelRun(param, tempAlgParamLocalCopy, templateResourceGX));
        rankOffset += allRankSplitData[i] * dataTypeSize_;
        rankLoopOffset += multiLoopAllRankSplitData[loop][i] * dataTypeSize_;
    }
    CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
void CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::BuildLocalCopyParam(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& p, u32 i,
    u64 currDataCountTmp, u64 rankOffset, u64 rankLoopOffset, const std::vector<u64>& processedDataCountTmp)
{
    p.localCopyFlag = 1;
    p.dataType = dataType_;
    p.buffInfo.inputSize = param.inputSize;
    p.buffInfo.outputSize = param.outputSize;
    p.buffInfo.hcclBuff = resCtx.cclMem;
    p.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    p.inputSliceStride = 0;
    p.outputSliceStride = 0;
    p.count = currDataCountTmp;
    p.sliceSize = currDataCountTmp * dataTypeSize_;
    p.buffInfo.outputPtr = param.outputPtr;
    p.buffInfo.outBuffType = BufferType::OUTPUT;
    u64 off = rankOffset + processedDataCountTmp[i] * dataTypeSize_;
    p.buffInfo.outBuffBaseOff = off;
    p.stepSliceInfo.buffInfo.outBuffBaseOff = off;
    if (i == param.root) {
        p.buffInfo.inputPtr = param.inputPtr;
        p.buffInfo.inBuffType = BufferType::INPUT;
        p.buffInfo.inBuffBaseOff = off;
        p.stepSliceInfo.buffInfo.inBuffBaseOff = off;
    } else {
        p.buffInfo.inputPtr = resCtx.cclMem.addr;
        p.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        p.buffInfo.inBuffBaseOff = rankLoopOffset;
        p.stepSliceInfo.buffInfo.inBuffBaseOff = rankLoopOffset;
    }
}


#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE, 
                                CcuV2ReduceOmniPipe2D,
                                CcuV2ReduceOmniPipeExecutor, 
                                TopoMatchUBX, 
                                CcuTempReduceScatterOmniPipeMesh1DMem2Mem, 
                                CcuTempReduceScatterOmniPipeNHR1DMem2Mem, 
                                CcuTempGatherOmniPipeMesh1DMem2Mem,
                                CcuTempGatherOmniPipeNHR1DMem2Mem);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE, 
                                CcuV2ReduceOmniPipe2DMs,
                                CcuV2ReduceOmniPipeExecutor, 
                                TopoMatchUBX, 
                                CcuTempReduceScatterOmniPipeMesh1D,
                                CcuTempReduceScatterOmniPipeNHR1DMem2Mem, 
                                CcuTempGatherOmniPipeMesh1DMem2Mem,
                                CcuTempGatherOmniPipeNHR1DMem2Mem);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif 
}
