/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_omnipipe_2d_executor.h"
#include "alg_data_trans_wrapper.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_scatter_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_scatter_omnipipe_nhr1d_mem2mem.h"
#include "ccu_temp_all_gather_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_all_gather_omnipipe_nhr1d_mem2mem.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif

namespace ops_hccl {
 
template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InsV2BroadcastOmniPipe2dExecutor()
{
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CalcAlgHierarchyInfo(HcclComm comm, 
            TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
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
 
template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitCommInfo(
            const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
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

    u64 rootx = param.root % rankSizeLevel0_;
	u64 rooty = param.root / rankSizeLevel0_;
	bool isRoot = (myRank_ == param.root);
	isSameYAxisAsRoot = (rankIdxLevel0_ == rootx) && !isRoot;
	isSameXAxisAsRoot = (rankIdxLevel1_ == rooty) && !isRoot;
 
    HCCL_DEBUG("[%s]myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] "
        "rankIdxLevel1[%u] devType[%u] dataCount[%u] dataType[%u] dataTypeSize[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_, devType_,
        dataCount_, dataType_, dataTypeSize_);
    return HcclResult::HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CalcResLevel(
            HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
            AlgResourceRequest& resReqlevel, AlgResourceRequest& resourceReq, const int& curLevel)
{
    resourceReq.slaveThreadNum += resReqlevel.slaveThreadNum;
    resourceReq.notifyNumOnMainThread += resReqlevel.notifyNumOnMainThread;
    resourceReq.notifyNumPerThread.insert(resourceReq.notifyNumPerThread.end(),
                                            resReqlevel.notifyNumPerThread.begin(),
                                            resReqlevel.notifyNumPerThread.end());
    if (curLevel == OMNIPIPE_SC_LEVEL0 || curLevel == OMNIPIPE_SC_LEVEL1) {
        std::for_each(resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end(), [](CcuKernelInfo &info) {
            info.resGroup = 0;
        });
    } else {
        std::for_each(resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end(), [](CcuKernelInfo &info) {
            info.resGroup = 1;
        });
    }
    resourceReq.ccuKernelInfos.insert(resourceReq.ccuKernelInfos.end(), resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end());
    resourceReq.ccuKernelNum.insert(resourceReq.ccuKernelNum.end(), resReqlevel.ccuKernelNum.begin(), resReqlevel.ccuKernelNum.end());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitSubCommRanks(
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
 
template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));
 
    // 初始化通信域subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    CHK_RET(InitSubCommRanks(subCommRanks0, subCommRanks1, algHierarchyInfo));
 
    CcuScatterAlgTemplateX scatterAlgTempLevelX(param, myRank_, subCommRanks0);
	CcuScatterAlgTemplateY scatterAlgTempLevelY(param, myRank_, subCommRanks1);
    CcuAgAlgTemplateX agAlgTempLevelX(param, myRank_, subCommRanks0);
	CcuAgAlgTemplateY agAlgTempLevelY(param, myRank_, subCommRanks1);
 
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    AlgResourceRequest resScatterReqLevelX;
    CHK_RET(scatterAlgTempLevelX.CalcRes(comm, param, topoInfo, resScatterReqLevelX));

    AlgResourceRequest resScatterReqLevelY;
    scatterAlgTempLevelY.SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
    CHK_RET(scatterAlgTempLevelY.CalcRes(comm, param, topoInfo, resScatterReqLevelY));

    AlgResourceRequest resAgReqLevelX;
    CHK_RET(agAlgTempLevelX.CalcRes(comm, param, topoInfo, resAgReqLevelX));
    AlgResourceRequest resAgReqLevelY;
    CHK_RET(agAlgTempLevelY.CalcRes(comm, param, topoInfo, resAgReqLevelY));

    CHK_RET(CalcResLevel(comm, param, topoInfo, resScatterReqLevelX, resourceRequest, 0));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resScatterReqLevelY, resourceRequest, 1));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resAgReqLevelX, resourceRequest, 2));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resAgReqLevelY, resourceRequest, 3));

    resourceRequest.slaveThreadNum += 1; // 需要一个主流和一个从流来并行2d
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[%s] slaveThreadNum:%d, notifyNumOnMainThread:%d", __func__, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
 
    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    threads_ = resCtx.threads;
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
    u64 rootx = param.root % rankSizeLevel0_;
    u64 rooty = param.root / rankSizeLevel0_;
    bool isRoot = (myRank_ == param.root);
    isSameYAxisAsRoot = (rankIdxLevel0_ == rootx) && !isRoot;
    isSameXAxisAsRoot = (rankIdxLevel1_ == rooty) && !isRoot;
    HCCL_DEBUG("[%s]myRank[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]", __func__,
        myRank_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2BroadcastOmniPipe2dExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
 
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitTemplate(
            const OpParam& param, std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap,
            const std::vector<std::vector<u32>>& subCommRanks0, const std::vector<std::vector<u32>>& subCommRanks1)
{
    // if (rankSizeLevel0_ > 1) {
    //     tempMap[OMNIPIPE_SC_LEVEL0] = std::make_shared<CcuScatterAlgTemplateX>(param, myRank_, subCommRanks0);
    //     tempMap[OMNIPIPE_AG_LEVEL0] = std::make_shared<CcuAgAlgTemplateX>(param, myRank_, subCommRanks0);
    // }
    // if (rankSizeLevel1_ > 1) {
    //     tempMap[OMNIPIPE_SC_LEVEL1] = std::make_shared<CcuScatterAlgTemplateY>(param, myRank_, subCommRanks1);
    //     tempMap[OMNIPIPE_AG_LEVEL1] = std::make_shared<CcuAgAlgTemplateY>(param, myRank_, subCommRanks1);
    // }
    // HCCL_DEBUG("[InsV2BroadcastOmniPipe2dExecutor][%s] tempMap.size:%u", __func__, tempMap.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::GenTempAlgParamsIn2HCCLBuff(
	TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount,
	const AlgResourceCtxSerializable &resCtx, const OpParam &param)
{
    tempAlgParams.count = processedDataCount;
    tempAlgParams.dataType = dataType_;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = param.inputPtr;
    stepSliceInfo.buffInfo.inputSize = param.inputSize;
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.outputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff
        = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff
        = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();

	return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::GenTempAlgParamsHCCLBuff2HCCLBuff(
	TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount,
	const AlgResourceCtxSerializable &resCtx, const OpParam &param)
{
    tempAlgParams.count = processedDataCount;
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
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff
        = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff
        = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();

	return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::GenTemplateAlgParamsByDimData(
            TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount)
{
    tempAlgParams.count = 0;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.buffInfo.inBuffBaseOff
        = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff
        = stepSliceInfo.buffInfo.outBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.localCopyFlag = 0;
    return HcclResult::HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitTemplateResources(
            const AlgResourceCtxSerializable& resCtx,
            TemplateResource& templateResourceScatterX, TemplateResource& templateResourceScatterY,
            TemplateResource& templateResourceAgX, TemplateResource& templateResourceAgY)
{
    TemplateResource templateResourceCommon;
    // TODO 设置下offsetStart、设置下offsetEnd

    // 按段顺序切分ccuKernels: 用kernelOffset累加器避免长偏移表达式
    // X维度(Level0)走thread[0], Y维度(Level1)走thread[1]
    u32 kernelOffset = 0;
    auto sliceKernels = [&](TemplateResource& res, u32 threadIdx, u32 segIdx) {
        res = templateResourceCommon;
        res.threads.push_back(resCtx.threads[threadIdx]);
        res.ccuKernels.insert(res.ccuKernels.end(),
            resCtx.ccuKernels.begin() + kernelOffset,
            resCtx.ccuKernels.begin() + kernelOffset + resCtx.ccuKernelNum[segIdx]);
        kernelOffset += resCtx.ccuKernelNum[segIdx];
    };
    sliceKernels(templateResourceScatterX, 0, OMNIPIPE_SC_LEVEL0);
    sliceKernels(templateResourceScatterY, 1, OMNIPIPE_SC_LEVEL1);
    sliceKernels(templateResourceAgX,      0, OMNIPIPE_AG_LEVEL0);
    sliceKernels(templateResourceAgY,      1, OMNIPIPE_AG_LEVEL1);

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CalcEndpointBandwidth(
            std::vector<double>& endpointAttrBwAvgSC, std::vector<double>& endpointAttrBwAvgAG)
{
    // SC带宽: Level0走mesh, Level1走clos（按rankSizeLevel1_-1均摊）
    double eqBwLevel0SC = BW_OMNI_UBX_CCU_SCHED_SC_MESH;
    double eqBwLevel1SC = BW_OMNI_UBX_CCU_SCHED_SC_CLOS;
    eqBwLevel1SC = rankSizeLevel1_ > 1 ? eqBwLevel1SC / (rankSizeLevel1_ - 1) : eqBwLevel1SC;
    endpointAttrBwAvgSC = {eqBwLevel0SC, eqBwLevel1SC, 1.0};

    // AG带宽: Level0走mesh, Level1走clos（按rankSizeLevel1_-1均摊）
    double eqBwLevel0AG = BW_OMNI_UBX_CCU_SCHED_AG_MESH;
    double eqBwLevel1AG = BW_OMNI_UBX_CCU_SCHED_AG_CLOS;
    eqBwLevel1AG = rankSizeLevel1_ > 1 ? eqBwLevel1AG / (rankSizeLevel1_ - 1) : eqBwLevel1AG;
    endpointAttrBwAvgAG = {eqBwLevel0AG, eqBwLevel1AG, 1.0};

    HCCL_INFO("[%s] eqBwLevel0SC:%f, eqBwLevel1SC:%f, eqBwLevel0AG:%f, eqBwLevel1AG:%f",
                __func__, eqBwLevel0SC, eqBwLevel1SC, eqBwLevel0AG, eqBwLevel1AG);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CalcLoopSplitData(
            u64 maxTmpMemSize, LoopSplitData& loopSplitData)
{
    // 1. 每个rank切分的总count
    loopSplitData.allRankSplitData = OmniPipeSplitData(rankSize_, dataCount_, dataTypeSize_);
    for (int i = 0; i < loopSplitData.allRankSplitData.size(); i++) {
        HCCL_DEBUG("[%s] rankId[%d], allRankSplitData[%d]:%d", __func__, myRank_, i, loopSplitData.allRankSplitData[i]);
    }

    // 2. 计算loop次数: 受UB单次传输上限和scratch显存上限双约束
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 scratchBoundDataSize = maxTmpMemSize / rankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    HCCL_DEBUG("[%s] myRank[%u] transportBoundDataSize[%u] scratchBoundDataSize[%u]",
                __func__, myRank_, transportBoundDataSize, scratchBoundDataSize);
    loopSplitData.maxCountPerLoop = std::min(transportBoundDataSize, scratchBoundDataSize) / dataTypeSize_;
    CHK_PRT_RET(loopSplitData.maxCountPerLoop == 0,
                HCCL_ERROR("[%s] maxCountPerLoop is 0", __func__), HCCL_E_INTERNAL);
    HCCL_DEBUG("[%s] myRank[%u] maxCountPerLoop[%u]", __func__, myRank_, loopSplitData.maxCountPerLoop);

    loopSplitData.loopTimes = loopSplitData.allRankSplitData[0] / loopSplitData.maxCountPerLoop +
        ((loopSplitData.allRankSplitData[0] % loopSplitData.maxCountPerLoop == 0) ? 0 : 1);
    HCCL_DEBUG("[%s] myRank[%u] loopTimes[%u]", __func__, myRank_, loopSplitData.loopTimes);

    // 3. 每个rank每个loop切分的count
    loopSplitData.multiLoopAllRankSplitData = OmniPipeSplitRankDataLoop(
        loopSplitData.allRankSplitData, loopSplitData.maxCountPerLoop, loopSplitData.loopTimes, dataTypeSize_);
    for (int i = 0; i < loopSplitData.multiLoopAllRankSplitData.size(); i++) {
        for (int j = 0; j < loopSplitData.multiLoopAllRankSplitData[i].size(); j++) {
            HCCL_DEBUG("[%s] rankId[%d], multiLoopAllRankSplitData[%d][%d]:%d",
                        __func__, myRank_, i, j, loopSplitData.multiLoopAllRankSplitData[i][j]);
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitSliceParam(
            const OpParam& param, const std::vector<u64>& allRankSplitData,
            const std::vector<std::vector<u64>>& multiLoopAllRankSplitData, OmniPipeSliceParam& sliceParam)
{
    // 每loop数据量: 取切分计划首loop首rank的大小，但不超过总数据量
    u64 perLoopSize = multiLoopAllRankSplitData[0][0] * dataTypeSize_;
    perLoopSize = dataSize_ > perLoopSize ? perLoopSize : dataSize_;
    HCCL_DEBUG("[%s] perLoopSize[%u] dataSize_[%u]", __func__, perLoopSize, dataSize_);

    sliceParam.dataSizePerLoop = std::vector<u64>(rankSize_, perLoopSize);
    sliceParam.dataWholeSize = std::vector<u64>(rankSize_, allRankSplitData[myRank_] * dataTypeSize_);
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    // levelAlgType: Level0=Scatter(1), Level1=AG(0), Level2=Scatter(1)
    sliceParam.levelAlgType = std::vector<u64>{1, 0, 1};
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = CommEngine::COMM_ENGINE_CCU;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::PrepareSliceInfoForLoop(
            u64 loop, u64 root, const std::vector<u64>& allRankSplitData,
            const std::vector<std::vector<u64>>& multiLoopAllRankSplitData,
            const std::vector<double>& endpointAttrBwAvgSC, const std::vector<double>& endpointAttrBwAvgAG,
            OmniPipeSliceParam& sliceParam, OmniPipeSliceInfo& omniPipeSliceInfoSC,
            OmniPipeSliceInfo& omniPipeSliceInfoAG)
{
    // 首轮loop或与上轮切分不同时重新计算sliceInfo
    if (loop != 0 && isSameLoop(multiLoopAllRankSplitData[loop - 1], multiLoopAllRankSplitData[loop])) {
        return HCCL_SUCCESS;
    }
    sliceParam.dataSizePerLoop = CalcCountToDataSize(multiLoopAllRankSplitData[loop], dataTypeSize_);
    sliceParam.dataWholeSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
    sliceParam.endpointAttrBw = endpointAttrBwAvgSC;
    omniPipeSliceInfoSC = CalcScatterOmniPipeSliceInfo(sliceParam, root);
    sliceParam.endpointAttrBw = endpointAttrBwAvgAG;
    omniPipeSliceInfoAG = CalcAGOmniPipeSliceInfo(sliceParam);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::RunScatterPhase(
            const OpParam& param, const AlgResourceCtxSerializable& resCtx,
            ScatterPhaseCtx& scatterCtx, const OmniPipeSliceInfo& omniPipeSliceInfoSC,
            u64 processedDataCount, const ThreadSyncCtx& syncCtx)
{
    auto level0StepCountSC = omniPipeSliceInfoSC.dataSliceLevel0.size();
    HCCL_DEBUG("[%s] myRank[%u] level0StepCountSC[%u]", __func__, myRank_, level0StepCountSC);

    for (auto i = 0; i < level0StepCountSC; ++i) {
        CHK_RET(PreSyncInterThreads(syncCtx.mainThread, syncCtx.syncThreads, syncCtx.notifyIdxesMainToSub));
        CHK_RET(GenTempAlgParamsIn2HCCLBuff(
            scatterCtx.tempScatterAlgParamsX, omniPipeSliceInfoSC.dataSliceLevel0[i], processedDataCount, resCtx, param));
        CHK_RET(GenTempAlgParamsIn2HCCLBuff(
            scatterCtx.tempScatterAlgParamsY, omniPipeSliceInfoSC.dataSliceLevel1[i], processedDataCount, resCtx, param));

        // NHR算法时，root的同y轴都需要执行y轴任务
        if (isSameYAxisAsRoot || myRank_ == param.root) {
            scatterCtx.scatterAlgTempY.ifDoTask_ = true;
        }

        if (i == level0StepCountSC - 1) {
            // 末步: 同x轴非root沿y轴转发(NHR/templateY); 同y轴非root沿x轴转发(mesh/templateX)
            HCCL_DEBUG("[%s] myRank[%u] StepNum[%u]", __func__, myRank_, i);
            scatterCtx.scatterAlgTempY.ifDoTask_ = true;
            if (isSameXAxisAsRoot) {
                CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                    scatterCtx.tempScatterAlgParamsY, omniPipeSliceInfoSC.dataSliceLevel1[i], processedDataCount, resCtx, param));
            }
            if (isSameYAxisAsRoot && rankSizeLevel0_ > 1) {
                HCCL_DEBUG("[%s] set myRank[%u] as root", __func__, myRank_);
                scatterCtx.scatterAlgTempX.SetRoot(myRank_);
                CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                    scatterCtx.tempScatterAlgParamsX, omniPipeSliceInfoSC.dataSliceLevel0[i], processedDataCount, resCtx, param));
            }
        } else if (i != 0) {
            // 中间步: 同y轴非root往x轴方向发送部分转发数据(mesh/templateX)
            HCCL_DEBUG("[%s] myRank[%u] StepNum[%u]", __func__, myRank_, i);
            if (isSameYAxisAsRoot && rankSizeLevel0_ > 1) {
                HCCL_DEBUG("[%s] set myRank[%u] as root", __func__, myRank_);
                scatterCtx.scatterAlgTempX.SetRoot(myRank_);
                CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                    scatterCtx.tempScatterAlgParamsX, omniPipeSliceInfoSC.dataSliceLevel0[i], processedDataCount, resCtx, param));
            }
        }

        // 执行X/Y维度通信
        scatterCtx.scatterAlgTempX.isStepOne_ = (i == 0);
        scatterCtx.scatterAlgTempX.isLastStep_ = (i == level0StepCountSC - 1);
        CHK_RET(scatterCtx.scatterAlgTempX.KernelRun(param, scatterCtx.tempScatterAlgParamsX, scatterCtx.templateResourceScatterX));
        scatterCtx.scatterAlgTempY.isStepOne_ = (i == 0);
        scatterCtx.scatterAlgTempY.isLastStep_ = (i == level0StepCountSC - 1);
        scatterCtx.scatterAlgTempY.xRankSize_ = rankSizeLevel0_;
        CHK_RET(scatterCtx.scatterAlgTempY.KernelRun(param, scatterCtx.tempScatterAlgParamsY, scatterCtx.templateResourceScatterY));

        CHK_RET(PostSyncInterThreads(syncCtx.mainThread, syncCtx.syncThreads, syncCtx.notifyIdxesSubToMain));
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::RunAGLocalCopyPhase(
            const OpParam& param, const AlgResourceCtxSerializable& resCtx,
            ScatterPhaseCtx& scatterCtx, const LoopSplitData& loopSplitData,
            const LoopProgress& loopProgress, const ThreadSyncCtx& syncCtx)
{
    HCCL_DEBUG("[%s] AG local copy start, myRank[%d], currDataCount %llu, processedDataCount %llu",
                __func__, myRank_, loopProgress.currDataCount, loopProgress.processedDataCount);

    if (myRank_ != param.root) {
        CHK_RET(PreSyncInterThreads(syncCtx.mainThread, syncCtx.syncThreads, syncCtx.notifyIdxesMainToSub));
        TemplateDataParams tempAlgParamLocalCopy = scatterCtx.tempAlgParamsCommon;
        tempAlgParamLocalCopy.localCopyFlag = 1;
        tempAlgParamLocalCopy.buffInfo.inputPtr = resCtx.cclMem.addr;
        tempAlgParamLocalCopy.buffInfo.inputSize = resCtx.cclMem.size;
        tempAlgParamLocalCopy.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamLocalCopy.buffInfo.outputSize = param.outputSize;
        // perRankOffset: 每rank在总数据中的累计偏移
        std::vector<u64> perRankOffset(rankSize_, 0);
        for (u32 i = 1; i < rankSize_; i++) {
            perRankOffset[i] = perRankOffset[i - 1] + loopSplitData.allRankSplitData[i - 1];
        }
        // perInnnerLoopOffset: 当前loop内每rank的累计偏移
        std::vector<u64> perInnnerLoopOffset(rankSize_, 0);
        for (u32 i = 1; i < rankSize_; i++) {
            perInnnerLoopOffset[i] = perInnnerLoopOffset[i - 1] + loopSplitData.multiLoopAllRankSplitData[loopProgress.loop][i - 1];
        }
        tempAlgParamLocalCopy.buffInfo.outBuffBaseOff = perRankOffset[myRank_] * dataTypeSize_ + loopProgress.processedDataCount * dataTypeSize_;
        tempAlgParamLocalCopy.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamLocalCopy.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamLocalCopy.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamLocalCopy.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamLocalCopy.count = loopProgress.currDataCount;
        tempAlgParamLocalCopy.sliceSize = loopProgress.currDataCount * dataTypeSize_;
        tempAlgParamLocalCopy.buffInfo.inBuffBaseOff = perInnnerLoopOffset[myRank_] * dataTypeSize_;
        tempAlgParamLocalCopy.outputSliceStride = 0;
        HCCL_DEBUG("[%s] myRank[%u] localCopy inBuffBaseOff[%lu] outBuffBaseOff[%lu] sliceSize[%lu] inputSliceStride [%lu]", __func__,
            myRank_, tempAlgParamLocalCopy.buffInfo.inBuffBaseOff, tempAlgParamLocalCopy.buffInfo.outBuffBaseOff,
            tempAlgParamLocalCopy.sliceSize, tempAlgParamLocalCopy.inputSliceStride);
        if (rankSizeLevel0_ > 1) {
            CHK_RET(scatterCtx.scatterAlgTempX.KernelRun(param, tempAlgParamLocalCopy, scatterCtx.templateResourceScatterX));
        } else if (rankSizeLevel1_ > 1) {
            CHK_RET(scatterCtx.scatterAlgTempY.KernelRun(param, tempAlgParamLocalCopy, scatterCtx.templateResourceScatterY));
        }
        CHK_RET(PostSyncInterThreads(syncCtx.mainThread, syncCtx.syncThreads, syncCtx.notifyIdxesSubToMain));
    }

    // 重置scatterAlgTemp状态，为下轮loop准备
    if (rankSizeLevel0_ > 1) {
        scatterCtx.scatterAlgTempX.UnsetRoot(myRank_);
    }
    scatterCtx.scatterAlgTempY.ifDoTask_ = false;
    HCCL_DEBUG("[%s] AG local copy end", __func__);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::RunAGPhase(
            const OpParam& param,
            CcuAgAlgTemplateX& agAlgTempX, CcuAgAlgTemplateY& agAlgTempY,
            TemplateDataParams& tempAgAlgParamsX, TemplateDataParams& tempAgAlgParamsY,
            const TemplateResource& templateResourceAgX, const TemplateResource& templateResourceAgY,
            const OmniPipeSliceInfo& omniPipeSliceInfoAG, u64 processedDataCount, const ThreadSyncCtx& syncCtx)
{
    u32 level0StepCountAG = omniPipeSliceInfoAG.dataSliceLevel0.size();
    HCCL_DEBUG("[%s] level0StepCountAG %u", __func__, level0StepCountAG);
    for (u32 i = 0; i < level0StepCountAG; i++) {
        // 初始化机内template param
        GenTemplateAlgParamsByDimData(tempAgAlgParamsX, omniPipeSliceInfoAG.dataSliceLevel0[i], processedDataCount);
        GenTemplateAlgParamsByDimData(tempAgAlgParamsY, omniPipeSliceInfoAG.dataSliceLevel1[i], processedDataCount);

        // 步前同步 -> X/Y维度通信 -> 步后同步
        CHK_RET(PreSyncInterThreads(syncCtx.mainThread, syncCtx.syncThreads, syncCtx.notifyIdxesMainToSub));
        CHK_RET(agAlgTempX.KernelRun(param, tempAgAlgParamsX, templateResourceAgX));
        CHK_RET(agAlgTempY.KernelRun(param, tempAgAlgParamsY, templateResourceAgY));
        CHK_RET(PostSyncInterThreads(syncCtx.mainThread, syncCtx.syncThreads, syncCtx.notifyIdxesSubToMain));
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2BroadcastOmniPipe2dExecutor<AlgTopoMatch, CcuScatterAlgTemplateX, CcuScatterAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::OrchestrateLoop(
            const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_DEBUG("[%s] Start", __func__);

    // 初始化通信域subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    auto& algHierarchyInfo_local = const_cast<ops_hccl::AlgHierarchyInfoForAllLevel&>(resCtx.algHierarchyInfo);
    CHK_RET(InitSubCommRanks(subCommRanks0, subCommRanks1, algHierarchyInfo_local));

    // 初始化template
    CcuScatterAlgTemplateX scatterAlgTempX(param, myRank_, subCommRanks0);
    CcuScatterAlgTemplateY scatterAlgTempY(param, myRank_, subCommRanks1);
    scatterAlgTempY.SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
    CcuAgAlgTemplateX agAlgTempX(param, myRank_, subCommRanks0);
    CcuAgAlgTemplateY agAlgTempY(param, myRank_, subCommRanks1);

    // 公共参数初始化
    TemplateDataParams tempAlgParamsCommon;
    tempAlgParamsCommon.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsCommon.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsCommon.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsCommon.buffInfo.hcclBuffSize = resCtx.cclMem.size;

    // 资源模板初始化
    TemplateResource templateResourceScatterX, templateResourceScatterY, templateResourceAgX, templateResourceAgY;
    CHK_RET(InitTemplateResources(resCtx, templateResourceScatterX, templateResourceScatterY,
                                  templateResourceAgX, templateResourceAgY));

    // 1、计算带宽
    std::vector<double> endpointAttrBwAvgSC;
    std::vector<double> endpointAttrBwAvgAG;
    CHK_RET(CalcEndpointBandwidth(endpointAttrBwAvgSC, endpointAttrBwAvgAG));

    // 2、计算数据切分与loop次数
    maxTmpMemSize_ = resCtx.cclMem.size;
    LoopSplitData loopSplitData;
    CHK_RET(CalcLoopSplitData(maxTmpMemSize_, loopSplitData));
    const auto& allRankSplitData = loopSplitData.allRankSplitData;
    const auto& multiLoopAllRankSplitData = loopSplitData.multiLoopAllRankSplitData;
    u64 maxCountPerLoop = loopSplitData.maxCountPerLoop;
    u32 loopTimes = loopSplitData.loopTimes;

    // 3 计算n-1次loop的slice信息
    OmniPipeSliceParam sliceParam;
    CHK_RET(InitSliceParam(param, allRankSplitData, multiLoopAllRankSplitData, sliceParam));

    // 4 进行一次loop的数据处理
    u64 processedDataCount = 0;
    TemplateDataParams tempScatterAlgParamsX = tempAlgParamsCommon;
    TemplateDataParams tempScatterAlgParamsY = tempAlgParamsCommon;
    TemplateDataParams tempAgAlgParamsX = tempAlgParamsCommon;
    TemplateDataParams tempAgAlgParamsY = tempAlgParamsCommon;

    // 主从线程同步上下文（Scatter/AGLocalCopy/AG三阶段共用）
    ThreadSyncCtx syncCtx{threads_[0], {threads_[1]}, {0}, {0}};

    // Scatter阶段上下文（聚合scatterAlgTemp + tempAlgParams + templateResource）
    ScatterPhaseCtx scatterCtx{scatterAlgTempX, scatterAlgTempY,
                               tempScatterAlgParamsX, tempScatterAlgParamsY,
                               tempAlgParamsCommon,
                               templateResourceScatterX, templateResourceScatterY};

    OmniPipeSliceInfo omniPipeSliceInfoSC;
    OmniPipeSliceInfo omniPipeSliceInfoAG;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        CHK_PRT_RET(
            multiLoopAllRankSplitData.size() <= loop,
            HCCL_ERROR("[InsV2BroadcastOmniPipe2dExecutor][Orchestrate] multiLoopAllRankSplitData.size() <= loop"),
            HCCL_E_PARA);
        // 4.1首轮计算, 或者与上轮不同loop重新计算OmniPipeSliceInfoRS、OmniPipeSliceInfoAG
        CHK_RET(PrepareSliceInfoForLoop(loop, param.root, allRankSplitData, multiLoopAllRankSplitData,
                                        endpointAttrBwAvgSC, endpointAttrBwAvgAG,
                                        sliceParam, omniPipeSliceInfoSC, omniPipeSliceInfoAG));
        u64 currDataCount = multiLoopAllRankSplitData[loop][myRank_];
        HCCL_DEBUG("[%s] dataCount_ %llu, processedDataCount %llu, maxCountPerLoop %llu, currDataCount %llu",
                        __func__, dataCount_, processedDataCount, maxCountPerLoop, currDataCount);

        // 4.2 Scatter阶段
        CHK_RET(RunScatterPhase(param, resCtx, scatterCtx,
                                omniPipeSliceInfoSC, processedDataCount, syncCtx));

        // 4.3 AG本地拷贝
        LoopProgress loopProgress{loop, currDataCount, processedDataCount};
        CHK_RET(RunAGLocalCopyPhase(param, resCtx, scatterCtx, loopSplitData, loopProgress, syncCtx));

        // 4.4 AG阶段
        CHK_RET(RunAGPhase(param, agAlgTempX, agAlgTempY, tempAgAlgParamsX, tempAgAlgParamsY,
                           templateResourceAgX, templateResourceAgY,
                           omniPipeSliceInfoAG, processedDataCount, syncCtx));

        processedDataCount += currDataCount;
    }
 
    HCCL_DEBUG("[%s][OrchestrateLoop] End.", __func__);
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_BROADCAST, CcuBroadcastOmniPipe2D,
                                InsV2BroadcastOmniPipe2dExecutor, TopoMatchUBX,
                                CcuTempScatterOmniPipeMesh1DMem2Mem, CcuTempScatterOmniPipeNHR1DMem2Mem,
                                CcuTempAllGatherOmniPipeMesh1DMem2Mem, CcuTempAllGatherOmniPipeNHR1DMem2Mem);
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif 
} // namespace ops_hccl