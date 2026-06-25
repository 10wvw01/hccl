/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "template_utils.h"

#include "aicpu/omni_temp_aicpu.h"
#include "omni_sole_executor.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
OmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OmniSoleExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult OmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult OmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    HCCL_INFO("[InitCommInfo] begin ");

    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];

    // dataCount_ 使用 param.inputSize（CalcInputOutputSize 计算的 per-rank 总元素数），
    // 而非 varData[0]（即 sendCounts[0]，仅为发往 rank0 的元素数）
    dataCount_ = param.inputSize;
    dataSize_ = dataCount_ * dataTypeSize_;
    HCCL_INFO("[OmniSoleExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], dataType_ [%u], "
        "dataCount_ [%llu]", myRank_, rankSize_, devType_, dataType_, dataCount_);

    HCCL_INFO("[OmniSoleExecutor][InitCommInfo] dataTypeSize_ [%u], dataSize_ [%llu]", dataTypeSize_, dataSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult OmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam& param,
                       const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                       AlgResourceRequest& resourceRequest)
{
    HCCL_INFO("CalcRes BEGIN");
    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo));

    std::vector<std::vector<u32>> tempAlgHierachyInfo;
    // if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
    //     tempAlgHierachyInfo.push_back(algHierarchyInfo.infos[0][1]);    // clos拓扑，包含所有rank
    // } else {
    //     tempAlgHierachyInfo = algHierarchyInfo.infos[0];
    // }

    tempAlgHierachyInfo = algHierarchyInfo.infos[0];

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, tempAlgHierachyInfo);
    // 调用计算资源的函数
    algTemplate->CalcRes(comm, param, topoInfo, resourceRequest, xmlInfo_);

    HCCL_INFO("CalcRes END");

    return HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult OmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[OmniSoleExecutor][Orchestrate] Orchestrate Start, rankid [%u]", myRank_);

    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, &resCtx.topoInfo));

    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU && param.engine != CommEngine::COMM_ENGINE_AICPU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[OmniSoleExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("[OmniSoleExecutor][Orchestrate] Orchestrate End, rankid [%u]", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult OmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[OmniSoleExecutor][OrchestrateLoop] Start, rankid [%u]", myRank_);

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);

    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) {
        // 准备资源
        TemplateResource templateAlgRes;
        if (remoteRankToChannelInfo_.size() > 0) {
            templateAlgRes.channels = remoteRankToChannelInfo_[0];
        }
        templateAlgRes.threads = resCtx.threads;
        templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

        TemplateDataParams tempAlgParams;
        tempAlgParams.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams.buffInfo.outputPtr = param.outputPtr;
        tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParams.buffInfo.hcclBuffSize = resCtx.cclMem.size / rankSize_;
        // param.inputSize/outputSize 是 CalcInputOutputSize 算出的 per-rank 总元素数，
        // 需转换为 per-slice 字节步长用于地址计算：(inputSize / rankSize_) * dataTypeSize_
        tempAlgParams.buffInfo.inputSize = param.inputSize / rankSize_ * dataTypeSize_;
        tempAlgParams.buffInfo.outputSize = param.outputSize / rankSize_ * dataTypeSize_;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
        tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParams.repeatNum = 1;  // 不需要重复
        tempAlgParams.inputRepeatStride = 0;
        tempAlgParams.outputRepeatStride = 0;

        // maxCountPerSlice: 每个 CCL buffer slot 能容纳的最大元素数（per-slice 上限）
        // maxCountPerLoop: 每轮循环能处理的总元素数（跨所有 slice，sliceNum = rankSize_）
        u64 sliceNum = rankSize_;
        u64 maxCountPerSlice = static_cast<u64>(tempAlgParams.buffInfo.hcclBuffSize) / dataTypeSize_;
        u64 maxCountPerLoop = maxCountPerSlice * sliceNum;
        u32 loopTimes = (dataCount_ + maxCountPerLoop - 1) / maxCountPerLoop;
        int64_t remainDataCount = dataCount_;
        for (uint32_t loop = 0; 0 < remainDataCount; ++loop, remainDataCount -= maxCountPerLoop) {
            u64 currDataCount = (remainDataCount < static_cast<int64_t>(maxCountPerLoop))
                ? remainDataCount : maxCountPerLoop;
            // sliceSize 为 per-slice 传输字节数，确保不超出 slot 步长(inputSize/outputSize)
            tempAlgParams.sliceSize = currDataCount * dataTypeSize_ / sliceNum;
            // inBuffBaseOff/outBuffBaseOff 按 per-slice 数据量推进
            tempAlgParams.buffInfo.inBuffBaseOff = loop * maxCountPerSlice * dataTypeSize_;
            tempAlgParams.buffInfo.outBuffBaseOff = loop * maxCountPerSlice * dataTypeSize_;
            HCCL_INFO("YHB-DEBUG: KernelRun, loop(%u), loopTimes(%u) sliceSize(%u) sliceNum(%u) origin-size(%u * %u) ccl-size(%u) in-size(%u) out-size(%u)",
                loop, loopTimes, tempAlgParams.sliceSize, sliceNum, dataCount_, dataTypeSize_,
                tempAlgParams.buffInfo.hcclBuffSize, tempAlgParams.buffInfo.inputSize,
                tempAlgParams.buffInfo.outputSize);
            CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes, resCtx.topoInfo.xmlInfo));
        }
        HCCL_INFO("[OmniSoleExecutor][OrchestrateLoop] End, rankid [%u]", myRank_);
        return HCCL_SUCCESS;
    }
    // 准备资源
    TemplateResource templateAlgRes;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    if (param.engine == COMM_ENGINE_CCU) {
        templateAlgRes.ccuKernels = resCtx.ccuKernels;
    }
    templateAlgRes.threads = resCtx.threads;

    //计算loop  ccu 不用cclbuff，根据UB_MAX_DATA_SIZE来计算
    u64 maxCountPerLoop = static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_;
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);
    HCCL_INFO("[OmniSoleExecutor][OrchestrateLoop]loopTimes = [%u]", loopTimes);

    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        TemplateDataParams tempAlgParams;
        tempAlgParams.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams.buffInfo.outputPtr = param.outputPtr;
        tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParams.sliceSize = currDataCount * dataTypeSize_ / xmlInfo_.vecNormalInstruction[0].sendRecvInfo.sliceNum * rankSize_;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
        tempAlgParams.repeatNum = 1;  // 不需要重复
        tempAlgParams.inputRepeatStride = 0;
        tempAlgParams.outputRepeatStride = 0;
        tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes, xmlInfo_));
        processedDataCount += currDataCount;
    }

    HCCL_INFO("[OmniSoleExecutor][OrchestrateLoop] End, rankid [%u]", myRank_);
    return HCCL_SUCCESS;
}


REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLV,
                OmniRunAicpu,
                OmniSoleExecutor,
                TopoMatch1D,
                OmniTempAicpu);

}
