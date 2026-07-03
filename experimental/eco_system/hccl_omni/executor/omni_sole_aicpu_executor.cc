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

#include "omni_temp_aicpu.h"
#include "omni_sole_aicpu_executor.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsOmniSoleAicpuExecutor<AlgTopoMatch, InsAlgTemplate>::InsOmniSoleAicpuExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleAicpuExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleAicpuExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo)
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
    HCCL_INFO("[InsOmniSoleAicpuExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], dataType_ [%u], "
              "dataCount_ [%llu]",
        myRank_, rankSize_, devType_, dataType_, dataCount_);

    HCCL_INFO(
        "[InsOmniSoleAicpuExecutor][InitCommInfo] dataTypeSize_ [%u], dataSize_ [%llu]", dataTypeSize_, dataSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleAicpuExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
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
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, tempAlgHierachyInfo);
    // 调用计算资源的函数
    algTemplate->CalcRes(comm, param, topoInfo, resourceRequest, xmlInfo_);

    HCCL_INFO("CalcRes END");

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleAicpuExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsOmniSoleAicpuExecutor][Orchestrate] Orchestrate Start, rankid [%u]", myRank_);

    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, &resCtx.topoInfo));

    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU
        && param.engine != CommEngine::COMM_ENGINE_AICPU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsOmniSoleAicpuExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed", HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO("[InsOmniSoleAicpuExecutor][Orchestrate] Orchestrate End, rankid [%u]", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleAicpuExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsOmniSoleAicpuExecutor][OrchestrateLoop] Start, rankid [%u]", myRank_);

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);

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
    tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.repeatNum = 1; // 不需要重复
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;
    tempAlgParams.dataType = dataType_;

    // 从 param.all2AllVDataDes 提取 sendCounts/recvCounts/sdispls/rdispls，
    // 用于支持 unequal AlltoAllV 的 displs-based 地址计算
    const u64 *scPtr = static_cast<const u64 *>(param.all2AllVDataDes.sendCounts);
    const u64 *rcPtr = static_cast<const u64 *>(param.all2AllVDataDes.recvCounts);
    const u64 *sdPtr = static_cast<const u64 *>(param.all2AllVDataDes.sdispls);
    const u64 *rdPtr = static_cast<const u64 *>(param.all2AllVDataDes.rdispls);
    u64 maxSendCount = 0;
    u64 maxRecvCount = 0;
    for (u32 i = 0; i < rankSize_; i++) {
        tempAlgParams.sendCounts.push_back(scPtr[i]);
        tempAlgParams.recvCounts.push_back(rcPtr[i]);
        tempAlgParams.sdispls.push_back(sdPtr[i]);
        tempAlgParams.rdispls.push_back(rdPtr[i]);
        if (scPtr[i] > maxSendCount)
            maxSendCount = scPtr[i];
        if (rcPtr[i] > maxRecvCount)
            maxRecvCount = rcPtr[i];
    }
    // inputSize/outputSize: 有 displs 时设为最大 per-slice 字节大小（用于 HCCL_BUFFER slot 对齐等场景）
    // 无 displs 时保持原 fixed-stride 逻辑
    tempAlgParams.buffInfo.inputSize = maxSendCount * dataTypeSize_;
    tempAlgParams.buffInfo.outputSize = maxRecvCount * dataTypeSize_;
    // maxCountPerSlice: 每个 CCL buffer slot 能容纳的最大元素数（per-slice 上限）
    // maxCountPerLoop: 每轮循环能处理的总元素数（跨所有 slice，sliceNum = rankSize_）
    u64 sliceNum = rankSize_;
    u64 maxCountPerSlice = static_cast<u64>(tempAlgParams.buffInfo.hcclBuffSize) / dataTypeSize_;
    u64 maxCountPerLoop = maxCountPerSlice * sliceNum;
    // loop 次数基于最大 per-slice 元素数（取 send/recv 较大者），确保所有 slice 都能处理完
    u64 maxSliceCount = (maxSendCount > maxRecvCount) ? maxSendCount : maxRecvCount;
    u32 loopTimes = (maxSliceCount + maxCountPerSlice - 1) / maxCountPerSlice;
    if (loopTimes == 0)
        loopTimes = 1;
    HCCL_INFO("YHB-DEBUG: AICPU_TS unequal loop, maxSendCount(%llu) maxRecvCount(%llu) maxSliceCount(%llu) "
              "maxCountPerSlice(%llu) loopTimes(%u)",
        maxSendCount, maxRecvCount, maxSliceCount, maxCountPerSlice, loopTimes);

    for (uint32_t loop = 0; loop < loopTimes; ++loop) {
        // sliceSize 为本轮 per-slice 最大传输字节数（用于 HCCL_BUFFER 和 fallback 场景）
        u64 currSliceCount = (maxSliceCount > loop * maxCountPerSlice)
                                 ? ((maxSliceCount - loop * maxCountPerSlice < maxCountPerSlice)
                                           ? maxSliceCount - loop * maxCountPerSlice
                                           : maxCountPerSlice)
                                 : 0;
        tempAlgParams.sliceSize = currSliceCount * dataTypeSize_;
        // inBuffBaseOff/outBuffBaseOff: 用于 displs-based 地址的 loop 内偏移推进
        // 每轮推进 maxCountPerSlice 个元素，叠加到 sdispls/rdispls 基础偏移上
        tempAlgParams.buffInfo.inBuffBaseOff = loop * maxCountPerSlice * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = loop * maxCountPerSlice * dataTypeSize_;
        HCCL_INFO("YHB-DEBUG: KernelRun, loop(%u/%u) sliceSize(%u) sliceNum(%u) inOff(%u) outOff(%u) in-size(%u) "
                  "out-size(%u)",
            loop, loopTimes, tempAlgParams.sliceSize, sliceNum, tempAlgParams.buffInfo.inBuffBaseOff,
            tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.buffInfo.inputSize, tempAlgParams.buffInfo.outputSize);
        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes, resCtx.topoInfo.xmlInfo));
    }
    HCCL_INFO("[OmniSoleExecutor][OrchestrateLoop] End, rankid [%u]", myRank_);
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLV, OmniRunAicpu, InsOmniSoleAicpuExecutor, TopoMatch1D, OmniTempAicpu);

} // namespace ops_hccl
