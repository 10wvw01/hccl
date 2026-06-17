/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 包含本类的头文件声明
#include "ins_v2_all_reduce_order_preserved_executor.h"
#include "ins_temp_reduce_scatter_order_preserved_level1.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "alg_env_config.h"
#include "order_preserved_common.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <sstream>

namespace ops_hccl {

static constexpr u64 MAX_PRINT_DATA_ELEMENTS = 256; // 小数据量时可以打印全部数据

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
void InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::PrintBufferData(
    const char *tag, const void *dataPtr, u64 offset, u64 elementCount,
    HcclDataType dataType, u64 maxPrintElements)
{
    if (dataPtr == nullptr) {
        HCCL_INFO("[%s] dataPtr is NULL, skip print", tag);
        return;
    }
    u64 printCount = std::min(elementCount, maxPrintElements);
    if (printCount == 0) {
        HCCL_INFO("[%s] elementCount is 0, skip print", tag);
        return;
    }

    const u8 *baseAddr = static_cast<const u8 *>(dataPtr) + offset;
    u32 typeSize = DATATYPE_SIZE_TABLE[dataType];

    std::stringstream ss;
    ss << "[" << tag << "] offset[" << offset << "] elementCount[" << elementCount
       << "] dataType[" << static_cast<u32>(dataType) << "] typeSize[" << typeSize << "]";

    switch (dataType) {
        case HcclDataType::HCCL_DATA_TYPE_FP32: {
            const float *ptr = reinterpret_cast<const float *>(baseAddr);
            ss << " values[float32]: [";
            for (u64 i = 0; i < printCount; i++) {
                if (i > 0) ss << ", ";
                ss << ptr[i];
            }
            ss << "]";
            break;
        }
        case HcclDataType::HCCL_DATA_TYPE_FP16: {
            // FP16无法直接打印为浮点数，以hex方式打印
            const u16 *ptr = reinterpret_cast<const u16 *>(baseAddr);
            ss << " values[fp16_hex]: [";
            for (u64 i = 0; i < printCount; i++) {
                if (i > 0) ss << ", ";
                ss << "0x" << std::hex << ptr[i] << std::dec;
            }
            ss << "]";
            break;
        }
        case HcclDataType::HCCL_DATA_TYPE_FP64: {
            const double *ptr = reinterpret_cast<const double *>(baseAddr);
            ss << " values[float64]: [";
            for (u64 i = 0; i < printCount; i++) {
                if (i > 0) ss << ", ";
                ss << ptr[i];
            }
            ss << "]";
            break;
        }
        case HcclDataType::HCCL_DATA_TYPE_INT32: {
            const int32_t *ptr = reinterpret_cast<const int32_t *>(baseAddr);
            ss << " values[int32]: [";
            for (u64 i = 0; i < printCount; i++) {
                if (i > 0) ss << ", ";
                ss << ptr[i];
            }
            ss << "]";
            break;
        }
        case HcclDataType::HCCL_DATA_TYPE_INT8: {
            const int8_t *ptr = reinterpret_cast<const int8_t *>(baseAddr);
            ss << " values[int8]: [";
            for (u64 i = 0; i < printCount; i++) {
                if (i > 0) ss << ", ";
                ss << static_cast<int>(ptr[i]);
            }
            ss << "]";
            break;
        }
        default: {
            // 未知类型，以hex方式按字节打印前几个字节
            ss << " values[raw_hex]: [";
            u64 byteCount = std::min(printCount * typeSize, static_cast<u64>(64));
            for (u64 i = 0; i < byteCount; i++) {
                if (i > 0) ss << " ";
                ss << std::hex << static_cast<unsigned>(baseAddr[i]) << std::dec;
            }
            ss << "]";
            break;
        }
    }

    HCCL_INFO("%s", ss.str().c_str());
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::InsV2AllReduceOrderPreservedExecutor()
{
    deterministicStrict_ = true;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcAlgHierarchyInfo] === ENTRY ===");
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcAlgHierarchyInfo] topoInfo: userRank[%u], userRankSize[%u], "
        "deviceType[%u], serverIdx[%u], moduleIdx[%u]",
        topoInfo->userRank, topoInfo->userRankSize, topoInfo->deviceType,
        topoInfo->serverIdx, topoInfo->moduleIdx);
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcAlgHierarchyInfo] topoMatch succeeded, "
        "algHierarchyInfo.infos.size[%zu], level0 infos count[%zu]",
        algHierarchyInfo.infos.size(), algHierarchyInfo.infos.empty() ? 0 : algHierarchyInfo.infos[0].size());
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcAlgHierarchyInfo] myRank[%u], rankSize[%u] (flat level1 only)",
        myRank_, rankSize_);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcAlgHierarchyInfo] === EXIT ===");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] === ENTRY ===");
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] input params: myRank[%u], rankSize[%u], "
        "devType[%u], reduceOp[%u], dataType[%u], dataCount[%llu], dataTypeSize[%llu]",
        myRank_, rankSize_, devType_, static_cast<u32>(reduceOp_), static_cast<u32>(dataType_),
        dataCount_, dataTypeSize_);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] OpParam details: inputPtr[%p], outputPtr[%p], "
        "inputSize[%llu], outputSize[%llu], opMode[%u], engine[%u]",
        param.inputPtr, param.outputPtr, param.inputSize, param.outputSize,
        static_cast<u32>(param.opMode), static_cast<u32>(param.engine));

    // 初始化执行器信息（检查是否启用严格模式）
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] >>> calling InitExecutorInfo");
    InitExecutorInfo(param);
    // 计算每个数据块的大小
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] >>> calling CalcSizePerBlock");
    CalcSizePerBlock(param);
    // 计算每个rank的数据切片大小
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] >>> calling CalcGroupSlices");
    CalcGroupSlices(param);

    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] memInfo: sizePerBlock[%llu], "
        "scratchMemFlag[%d], totalSize[%llu], groupSize.size[%zu]",
        memInfo_.sizePerBlock, memInfo_.scratchMemFlag, memInfo_.totalSize, memInfo_.groupSize.size());
    for (u32 i = 0; i < memInfo_.groupSize.size(); i++) {
        HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] groupSize[%u] = [%llu]",
            i, memInfo_.groupSize[i]);
    }

    // 创建ReduceScatter算法模板实例
    std::shared_ptr<InsAlgTemplateRS> rsTempAlg =
        std::make_shared<InsAlgTemplateRS>(param, myRank_, algHierarchyInfo.infos[0]);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] ReduceScatter template created, "
        "algHierarchyInfo.infos[0].size[%zu]", algHierarchyInfo.infos[0].size());

    // 创建AllGather算法模板实例
    std::shared_ptr<InsAlgTemplateAG> agTempAlg =
        std::make_shared<InsAlgTemplateAG>(param, myRank_, algHierarchyInfo.infos[0]);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] AllGather template created");

    AlgResourceRequest resReqRS;
    AlgResourceRequest resReqAG;

    // 调用ReduceScatter模板的CalcRes函数计算所需资源
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] >>> calling rsTempAlg->CalcRes");
    CHK_RET(rsTempAlg->CalcRes(comm, param, topoInfo, resReqRS));
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] rsTempAlg->CalcRes done: "
        "slaveThreadNum[%u], notifyNumOnMainThread[%u], channels.size[%zu]",
        resReqRS.slaveThreadNum, resReqRS.notifyNumOnMainThread, resReqRS.channels.size());
    // 调用AllGather模板的CalcRes函数计算所需资源
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] >>> calling agTempAlg->CalcRes");
    CHK_RET(agTempAlg->CalcRes(comm, param, topoInfo, resReqAG));
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] agTempAlg->CalcRes done: "
        "slaveThreadNum[%u], notifyNumOnMainThread[%u], channels.size[%zu]",
        resReqAG.slaveThreadNum, resReqAG.notifyNumOnMainThread, resReqAG.channels.size());

    // 设置从线程数为两个模板的最大值
    resourceRequest.slaveThreadNum = std::max(resReqRS.slaveThreadNum, resReqAG.slaveThreadNum);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] merged slaveThreadNum[%u] "
        "(RS[%u], AG[%u])", resourceRequest.slaveThreadNum,
        resReqRS.slaveThreadNum, resReqAG.slaveThreadNum);

    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    // 遍历每个从线程，设置通知数为两个模板的最大值
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReqRS.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqRS.notifyNumPerThread[i]);
        }
        if (i < resReqAG.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqAG.notifyNumPerThread[i]);
        }
        HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] notifyNumPerThread[%u] = [%u] "
            "(RS[%u], AG[%u])", i, resourceRequest.notifyNumPerThread[i],
            (i < resReqRS.notifyNumPerThread.size()) ? resReqRS.notifyNumPerThread[i] : 0,
            (i < resReqAG.notifyNumPerThread.size()) ? resReqAG.notifyNumPerThread[i] : 0);
    }

    // 设置主线程通知数为两个模板的最大值
    resourceRequest.notifyNumOnMainThread = std::max(resReqRS.notifyNumOnMainThread,
        resReqAG.notifyNumOnMainThread);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] merged notifyNumOnMainThread[%u] "
        "(RS[%u], AG[%u])", resourceRequest.notifyNumOnMainThread,
        resReqRS.notifyNumOnMainThread, resReqAG.notifyNumOnMainThread);

    resourceRequest.channels.clear();
    if (resReqRS.channels.size() > 0) {
        resourceRequest.channels.push_back(resReqRS.channels[0]);
    }
    if (resReqAG.channels.size() > 0) {
        resourceRequest.channels.push_back(resReqAG.channels[0]);
    }
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] merged channels.size[%zu] "
        "(RS[%zu], AG[%zu])", resourceRequest.channels.size(),
        resReqRS.channels.size(), resReqAG.channels.size());

    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] slaveThreadNum[%u], notifyNumOnMainThread[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] === EXIT ===");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] === ENTRY ===");
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] input params: inputPtr[%p], outputPtr[%p], "
        "inputSize[%llu], outputSize[%llu], dataCount[%llu], dataType[%u], reduceOp[%u]",
        param.inputPtr, param.outputPtr, param.inputSize, param.outputSize,
        param.DataDes.count, static_cast<u32>(param.DataDes.dataType), static_cast<u32>(param.reduceType));
    // 打印输入数据的整体前N个元素（用param直接获取count和dataType，因为成员变量可能还没初始化）
    PrintBufferData("Orchestrate_INPUT_FULL", param.inputPtr, 0, param.DataDes.count, param.DataDes.dataType, MAX_PRINT_DATA_ELEMENTS);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] resCtx: cclMem.addr[%p], cclMem.size[%llu], "
        "threads.size[%zu], aivCommInfoPtr[%p]",
        resCtx.cclMem.addr, resCtx.cclMem.size, resCtx.threads.size(), resCtx.aivCommInfoPtr);

    OrderPreservedBaseParams baseParams = InitOrderPreservedBaseParams(param, resCtx);
    SetOrderPreservedBaseParams(baseParams);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] baseParams: myRank[%u], rankSize[%u], "
        "dataCount[%llu], dataTypeSize[%llu], dataSize[%llu], dataType[%u], reduceOp[%u], maxTmpMemSize[%llu]",
        baseParams.myRank, baseParams.rankSize, baseParams.dataCount, baseParams.dataTypeSize,
        baseParams.dataSize, static_cast<u32>(baseParams.dataType), static_cast<u32>(baseParams.reduceOp),
        baseParams.maxTmpMemSize);
    
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] remoteRankToChannelInfo_.size[%zu], "
        "engine[%u]", remoteRankToChannelInfo_.size(), static_cast<u32>(param.engine));

    InitExecutorInfo(param);
    // 根据rank，把总数据切分
    CalcSizePerBlock(param);
    CalcGroupSlices(param);

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] kernel run failed, err[0x%016llx]",
            HCCL_ERROR_CODE(ret)), ret);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] === EXIT SUCCESS ===");
    // 打印最终输出数据的整体前N个元素（用param直接获取count和dataType）
    PrintBufferData("Orchestrate_OUTPUT_FULL", param.outputPtr, 0, param.DataDes.count, param.DataDes.dataType, MAX_PRINT_DATA_ELEMENTS);
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
    
    // 设置通道信息，只要level0
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
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] === ENTRY ===, deterministicStrict[%d] (flat level1)",
        deterministicStrict_);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] dataCount[%llu], dataSize[%llu], "
        "dataTypeSize[%llu], rankSize[%u], myRank[%u], reduceOp[%u], dataType[%u]",
        dataCount_, dataSize_, dataTypeSize_, rankSize_, myRank_,
        static_cast<u32>(reduceOp_), static_cast<u32>(dataType_));

    // 创建ReduceScatter算法模板实例
    std::shared_ptr<InsAlgTemplateRS> rsTempAlg =
        std::make_shared<InsAlgTemplateRS>(param, myRank_, resCtx.algHierarchyInfo.infos[0]);
    // 设置ReduceScatter模板的通道映射
    rsTempAlg->SetchannelsPerRank(remoteRankToChannelInfo_[0]);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] RS template created and channels set, "
        "remoteRankToChannelInfo_[0].size[%zu]", remoteRankToChannelInfo_[0].size());

    // 创建AllGather算法模板实例
    std::shared_ptr<InsAlgTemplateAG> agTempAlg =
        std::make_shared<InsAlgTemplateAG>(param, myRank_, resCtx.algHierarchyInfo.infos[0]);
    // 设置AllGather模板的通道映射
    agTempAlg->SetchannelsPerRank(remoteRankToChannelInfo_[0]);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] AG template created and channels set");

    TemplateResource rsTemplateAlgRes;
    CHK_RET(GenTempResource(resCtx, 0, rsTempAlg, rsTemplateAlgRes));
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] RS template resource generated: "
        "channels.size[%zu], threads.size[%zu], aivCommInfoPtr[%p]",
        rsTemplateAlgRes.channels.size(), rsTemplateAlgRes.threads.size(), rsTemplateAlgRes.aivCommInfoPtr);

    TemplateResource agTemplateAlgRes;
    CHK_RET(GenTempResource(resCtx, 0, agTempAlg, agTemplateAlgRes));
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] AG template resource generated: "
        "channels.size[%zu], threads.size[%zu], aivCommInfoPtr[%p]",
        agTemplateAlgRes.channels.size(), agTemplateAlgRes.threads.size(), agTemplateAlgRes.aivCommInfoPtr);

    TemplateDataParams tempAlgParams;
    InitTemplateDataParams(param, resCtx, tempAlgParams);
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] InitTemplateDataParams done: "
        "inputPtr[%p], outputPtr[%p], inputSize[%llu], outputSize[%llu], "
        "hcclBuff.addr[%p], hcclBuff.size[%llu]",
        tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.buffInfo.inputSize, tempAlgParams.buffInfo.outputSize,
        tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuff.size);

    // CCL buffer切分为2块：前1块作为ReduceScatter输出，后1块作为AllGather输入
    outCclBuffSize_ = tempAlgParams.buffInfo.hcclBuff.size / 2;
    inCclBuffSize_ = tempAlgParams.buffInfo.hcclBuff.size - outCclBuffSize_;
    outCclBuffOffset_ = 0;
    inCclBuffOffset_ = outCclBuffSize_;
    HCCL_INFO("[OrchestrateLoop] CCL buffer split: totalSize[%llu], outCclBuffSize[%llu], inCclBuffSize[%llu], "
        "outCclBuffOffset[%llu], inCclBuffOffset[%llu]",
        tempAlgParams.buffInfo.hcclBuff.size, outCclBuffSize_, inCclBuffSize_, outCclBuffOffset_, inCclBuffOffset_);

    // 计算单次循环最大数据元素个数
    // outCclBuff存储ReduceScatter输出（每个rank的归约结果大小 = currDataCount/rankSize）
    // 所以最大总数据量 = outCclBuffSize_ * rankSize_ / dataTypeSize_
    // 向下对齐到rankSize的倍数，方便数据切分
    u64 rawMaxCount = outCclBuffSize_ / HCCL_MIN_SLICE_ALIGN *
                        HCCL_MIN_SLICE_ALIGN / dataTypeSize_ / rankSize_ * rankSize_;
    HCCL_INFO("[OrchestrateLoop] maxCountPerLoop calculation: outCclBuffSize[%llu] / HCCL_MIN_SLICE_ALIGN[%llu] "
        "* HCCL_MIN_SLICE_ALIGN[%llu] / dataTypeSize[%llu] / rankSize[%u] * rankSize[%u] = rawMaxCount[%llu]",
        outCclBuffSize_, HCCL_MIN_SLICE_ALIGN, HCCL_MIN_SLICE_ALIGN, dataTypeSize_, rankSize_, rankSize_, rawMaxCount);
    u64 maxCountPerLoop = rawMaxCount;
    HCCL_INFO("[OrchestrateLoop] maxCountPerLoop[%llu], outCclBuffSize[%llu], dataTypeSize[%llu], rankSize[%u]",
        maxCountPerLoop, outCclBuffSize_, dataTypeSize_, rankSize_);
    CHK_PRT_RET(maxCountPerLoop == 0,
        HCCL_ERROR("[OrchestrateLoop] maxCountPerLoop is 0, outCclBuffSize[%llu], dataTypeSize[%llu], "
            "rankSize[%u], HCCL_MIN_SLICE_ALIGN[%llu]",
            outCclBuffSize_, dataTypeSize_, rankSize_, HCCL_MIN_SLICE_ALIGN), HCCL_E_INTERNAL);

    // 计算循环次数：总数据量 / 单次最大数据量，向上取整
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    HCCL_INFO("[OrchestrateLoop] loopTimes[%llu] = dataCount[%llu] / maxCountPerLoop[%llu] "
        "+ remainder[%llu]", loopTimes, dataCount_, maxCountPerLoop,
        static_cast<u64>(dataCount_ % maxCountPerLoop != 0));
    // 初始化已处理的数据元素个数
    u64 processedDataCount = 0;

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        HCCL_INFO("[OrchestrateLoop] >>> LOOP[%llu/%llu]: currDataCount[%llu], processedDataCount[%llu], "
            "dataCount[%llu], maxCountPerLoop[%llu]",
            loop, loopTimes - 1, currDataCount, processedDataCount, dataCount_, maxCountPerLoop);
        HCCL_INFO("[OrchestrateLoop] LOOP[%llu]: currDataCount / rankSize = [%llu], "
            "remainder = [%llu]",
            loop, currDataCount / rankSize_, currDataCount % rankSize_);

        HCCL_INFO("[OrchestrateLoop] LOOP[%llu]: >>> calling RunReduceScatter", loop);
        // 打印ReduceScatter输入数据：整体 + 每个rank的slice
        PrintBufferData("Loop_RS_INPUT_FULL", param.inputPtr, processedDataCount * dataTypeSize_,
            currDataCount, dataType_, MAX_PRINT_DATA_ELEMENTS);
        u64 rsSliceCount = currDataCount / rankSize_;
        u64 rsSliceSize = rsSliceCount * dataTypeSize_;
        for (u32 r = 0; r < rankSize_; r++) {
            PrintBufferData("Loop_RS_INPUT_slice", param.inputPtr,
                processedDataCount * dataTypeSize_ + r * rsSliceSize,
                rsSliceCount, dataType_, MAX_PRINT_DATA_ELEMENTS);
        }
        CHK_RET(RunReduceScatter(param, resCtx, currDataCount, 
            processedDataCount, rsTempAlg, rsTemplateAlgRes));
        HCCL_INFO("[OrchestrateLoop] LOOP[%llu]: RunReduceScatter done", loop);
        // 打印ReduceScatter输出数据：本rank的归约结果 + CCL buffer中各rank区域
        PrintBufferData("Loop_RS_OUTPUT_myRank", resCtx.cclMem.addr, outCclBuffOffset_,
            rsSliceCount, dataType_, MAX_PRINT_DATA_ELEMENTS);
        PrintBufferData("Loop_RS_OUTPUT_CCLBUFF_full", resCtx.cclMem.addr, inCclBuffOffset_,
            currDataCount, dataType_, MAX_PRINT_DATA_ELEMENTS);

        HCCL_INFO("[OrchestrateLoop] LOOP[%llu]: >>> calling RunAllGather", loop);
        CHK_RET(RunAllGather(param, resCtx, currDataCount,
            processedDataCount, agTempAlg, agTemplateAlgRes));
        HCCL_INFO("[OrchestrateLoop] LOOP[%llu]: RunAllGather done", loop);
        // 打印AllGather输出数据：整体 + 每个rank的slice
        PrintBufferData("Loop_AG_OUTPUT_FULL", param.outputPtr, processedDataCount * dataTypeSize_,
            currDataCount, dataType_, MAX_PRINT_DATA_ELEMENTS);
        for (u32 r = 0; r < rankSize_; r++) {
            PrintBufferData("Loop_AG_OUTPUT_slice", param.outputPtr,
                processedDataCount * dataTypeSize_ + r * rsSliceSize,
                rsSliceCount, dataType_, MAX_PRINT_DATA_ELEMENTS);
        }

        processedDataCount += currDataCount;
        HCCL_INFO("[OrchestrateLoop] LOOP[%llu]: processedDataCount updated to [%llu]", loop, processedDataCount);
    }

    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] === EXIT SUCCESS ===");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::InitExecutorInfo(const OpParam &param)
{
    // 规约保序判断已在 selector 中完成，executor 直接启用保序模式
    deterministicStrict_ = true;
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][InitExecutorInfo] deterministicStrict[%d]",
        deterministicStrict_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcSizePerBlock(const OpParam &param)
{
    HCCL_INFO("[CalcSizePerBlock] === ENTRY ===");
    // 计算单卡数据量：总数据量 / rank数，向上取整
    u64 rawSizePerBlock = (dataCount_ + rankSize_ - 1) / rankSize_ * dataTypeSize_;
    HCCL_INFO("[CalcSizePerBlock] rawSizePerBlock[%llu] = (dataCount[%llu] + rankSize[%u] - 1) / rankSize[%u] "
        "* dataTypeSize[%llu]", rawSizePerBlock, dataCount_, rankSize_, rankSize_, dataTypeSize_);
    memInfo_.sizePerBlock = RoundUpWithDivisor(rawSizePerBlock, HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED);
    HCCL_INFO("[CalcSizePerBlock] sizePerBlock after RoundUp[%llu] (divisor[%llu])",
        memInfo_.sizePerBlock, HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED);
    memInfo_.scratchMemFlag = false;
    memInfo_.totalSize = 0;
    HCCL_INFO("[CalcSizePerBlock] sizePerBlock[%llu], dataCount[%llu], rankSize[%u]",
        memInfo_.sizePerBlock, dataCount_, rankSize_);
    HCCL_INFO("[CalcSizePerBlock] === EXIT ===");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcGroupSlices(const OpParam &param)
{
    HCCL_INFO("[CalcGroupSlices] === ENTRY === dataSize[%llu], sizePerBlock[%llu], rankSize[%u]",
        dataSize_, memInfo_.sizePerBlock, rankSize_);
    memInfo_.groupSize.clear();
    // 初始化剩余数据大小为总数据大小
    u64 sizeRemain = dataSize_;
    for (u32 rankId = 0; rankId < rankSize_; rankId++) {
        u64 size = (sizeRemain > memInfo_.sizePerBlock) ? memInfo_.sizePerBlock : sizeRemain;
        memInfo_.groupSize.push_back(size);
        HCCL_INFO("[CalcGroupSlices] rankId[%u]: size[%llu], sizeRemain(before)[%llu], sizePerBlock[%llu]",
            rankId, size, sizeRemain, memInfo_.sizePerBlock);
        sizeRemain -= size;
    }
    memInfo_.totalSize = std::max(memInfo_.sizePerBlock * rankSize_, dataSize_);
    HCCL_INFO("[CalcGroupSlices] groupSize.size[%u], totalSize[%llu], sizePerBlock*rankSize[%llu], dataSize[%llu]",
        memInfo_.groupSize.size(), memInfo_.totalSize, memInfo_.sizePerBlock * rankSize_, dataSize_);
    HCCL_INFO("[CalcGroupSlices] === EXIT ===");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RunReduceScatter(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    u64 currDataCount, u64 processedDataCount,
    std::shared_ptr<InsAlgTemplateRS> rsTempAlg, TemplateResource &rsTemplateAlgRes)
{
    HCCL_INFO("[RunReduceScatter] === ENTRY === currDataCount[%llu], processedDataCount[%llu]",
        currDataCount, processedDataCount);
    // 准备ReduceScatter模板数据参数结构体
    // ReduceScatter: INPUT -> HCCL_BUFFER (outCclBuff部分)
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

    HCCL_INFO("[RunReduceScatter] buffInfo: inputPtr[%p], outputPtr[%p], hcclBuff.addr[%p], "
        "hcclBuff.size[%llu], inputSize[%llu], outputSize[%llu]",
        rsTempAlgParams.buffInfo.inputPtr, rsTempAlgParams.buffInfo.outputPtr,
        rsTempAlgParams.buffInfo.hcclBuff.addr, rsTempAlgParams.buffInfo.hcclBuff.size,
        rsTempAlgParams.buffInfo.inputSize, rsTempAlgParams.buffInfo.outputSize);

    rsTempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    // 输出缓冲区基址偏移：归约结果输出到outCclBuff的起始位置
    rsTempAlgParams.buffInfo.outBuffBaseOff = outCclBuffOffset_;
    // 临时缓冲区基址偏移：使用inCclBuff部分作为临时缓冲区
    rsTempAlgParams.buffInfo.hcclBuffBaseOff = inCclBuffOffset_;

    HCCL_INFO("[RunReduceScatter] buffInfo offsets: inBuffBaseOff[%llu] = processedDataCount[%llu] * dataTypeSize[%llu], "
        "outBuffBaseOff[%llu], hcclBuffBaseOff[%llu]",
        rsTempAlgParams.buffInfo.inBuffBaseOff, processedDataCount, dataTypeSize_,
        rsTempAlgParams.buffInfo.outBuffBaseOff, rsTempAlgParams.buffInfo.hcclBuffBaseOff);



    // ReduceScatter: INPUT -> HCCL_BUFFER(outCclBuff)
    // memBlockInfo中的offset是相对于整个hcclBuff.addr的绝对偏移
    MemBlockInfo memBlockInfo;
    memBlockInfo.size.clear();
    memBlockInfo.userInputOffsets.clear();
    memBlockInfo.inputOffsets.clear();
    memBlockInfo.outputOffsets.clear();
    
    u64 unitSize = dataTypeSize_;
    // 计算sliceSize和tailSize（尾块处理）
    u64 rsSliceSize = currDataCount / rankSize_ * unitSize;
    u64 rsTailSize = (currDataCount / rankSize_ + currDataCount % rankSize_) * unitSize;
    HCCL_INFO("[RunReduceScatter] slice calculation: currDataCount[%llu], rankSize[%u], "
        "currDataCount/rankSize[%llu], remainder[%llu], unitSize[%llu], "
        "rsSliceSize[%llu], rsTailSize[%llu]",
        currDataCount, rankSize_, currDataCount / rankSize_, currDataCount % rankSize_,
        unitSize, rsSliceSize, rsTailSize);
    
    memBlockInfo.outputOffsets.resize(rankSize_, 0);
    
    for (u32 outputIndex = 0; outputIndex < rankSize_; outputIndex++) {
        memBlockInfo.outputOffsets[outputIndex] = inCclBuffOffset_ + outputIndex * rsTailSize;
    }
    
    HCCL_INFO("[RunReduceScatter] memBlockInfo.outputOffsets:");
    for (u32 i = 0; i < rankSize_; i++) {
        HCCL_INFO("[RunReduceScatter] outputOffsets[%u] = inCclBuffOffset[%llu] + %u * rsTailSize[%llu] = [%llu]",
            i, inCclBuffOffset_, i, rsTailSize, memBlockInfo.outputOffsets[i]);
    }

    // 设置输入偏移和大小：每个rank对应一个数据切片，最后一个rank处理剩余的尾块
    for (u32 dataId = 0; dataId < rankSize_; dataId++) {
        // 实际数据大小：最后一个rank包含余数
        u64 actualSize = (dataId == rankSize_ - 1) ? rsTailSize : rsSliceSize;
        // 用户输入偏移：已处理数据偏移 + 当前数据块在输入数据中的偏移
        u64 userMemInOffset = processedDataCount * unitSize + dataId * rsSliceSize;
        
        memBlockInfo.size.push_back(actualSize);
        memBlockInfo.userInputOffsets.push_back(userMemInOffset);
        memBlockInfo.inputOffsets.push_back(inCclBuffOffset_ + dataId * rsTailSize);
        HCCL_INFO("[RunReduceScatter] dataId[%u]: actualSize[%llu] (%s), userMemInOffset[%llu], "
            "inputOffset[%llu]",
            dataId, actualSize, (dataId == rankSize_ - 1) ? "TAIL" : "SLICE",
            userMemInOffset, inCclBuffOffset_ + dataId * rsTailSize);
    }
    
    // 设置所有rank的数据切片大小向量（考虑尾块）
    std::vector<u64> rsSliceSizes;
    for (u32 rankId = 0; rankId < rankSize_; rankId++) {
        u64 size = (rankId == rankSize_ - 1) ? rsTailSize : rsSliceSize;
        rsSliceSizes.push_back(size);
        HCCL_INFO("[RunReduceScatter] rsSliceSizes[%u] = [%llu] (%s)",
            rankId, size, (rankId == rankSize_ - 1) ? "TAIL" : "SLICE");
    }
    rsTempAlgParams.allRankSliceSize = rsSliceSizes;
    rsTempAlgParams.sliceSize = rsSliceSize;
    rsTempAlgParams.tailSize = rsTailSize;
    rsTempAlgParams.inputSliceStride = rsSliceSize;
    rsTempAlgParams.outputSliceStride = 0;
    rsTempAlgParams.repeatNum = 1;
    rsTempAlgParams.inputRepeatStride = 0;
    rsTempAlgParams.outputRepeatStride = 0;
    rsTempAlgParams.count = currDataCount / rankSize_;

    HCCL_INFO("[RunReduceScatter] rsTempAlgParams summary: sliceSize[%llu], tailSize[%llu], "
        "inputSliceStride[%llu], outputSliceStride[%llu], repeatNum[%u], "
        "inputRepeatStride[%llu], outputRepeatStride[%llu], count[%llu]",
        rsTempAlgParams.sliceSize, rsTempAlgParams.tailSize,
        rsTempAlgParams.inputSliceStride, rsTempAlgParams.outputSliceStride,
        rsTempAlgParams.repeatNum, rsTempAlgParams.inputRepeatStride,
        rsTempAlgParams.outputRepeatStride, rsTempAlgParams.count);
    
    rsTempAlg->SetMemBlockInfo(memBlockInfo);
    HCCL_INFO("[RunReduceScatter] >>> calling rsTempAlg->KernelRun");
    CHK_RET(rsTempAlg->KernelRun(param, rsTempAlgParams, rsTemplateAlgRes));
    HCCL_INFO("[RunReduceScatter] rsTempAlg->KernelRun done");
    
    HCCL_INFO("[RunReduceScatter] === EXIT ===");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RunAllGather(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    u64 currDataCount, u64 processedDataCount,
    std::shared_ptr<InsAlgTemplateAG> agTempAlg, TemplateResource &agTemplateAlgRes)
{
    HCCL_INFO("[RunAllGather] === ENTRY === currDataCount[%llu], processedDataCount[%llu]",
        currDataCount, processedDataCount);
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

    HCCL_INFO("[RunAllGather] buffInfo: inputPtr[%p], outputPtr[%p], hcclBuff.addr[%p], "
        "hcclBuff.size[%llu], inputSize[%llu], outputSize[%llu]",
        agTempAlgParams.buffInfo.inputPtr, agTempAlgParams.buffInfo.outputPtr,
        agTempAlgParams.buffInfo.hcclBuff.addr, agTempAlgParams.buffInfo.hcclBuff.size,
        agTempAlgParams.buffInfo.inputSize, agTempAlgParams.buffInfo.outputSize);

    // 输入缓冲区基址偏移：从outCclBuff位置读取ReduceScatter的归约结果
    agTempAlgParams.buffInfo.inBuffBaseOff = outCclBuffOffset_;
    // 输出缓冲区基址偏移：输出到用户输出缓冲区的已处理数据位置
    agTempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    // 临时缓冲区基址偏移：使用inCclBuff部分，避免与outCclBuff冲突
    agTempAlgParams.buffInfo.hcclBuffBaseOff = inCclBuffOffset_;

    HCCL_INFO("[RunAllGather] buffInfo offsets: inBuffBaseOff[%llu], "
        "outBuffBaseOff[%llu] = processedDataCount[%llu] * dataTypeSize[%llu], hcclBuffBaseOff[%llu]",
        agTempAlgParams.buffInfo.inBuffBaseOff,
        agTempAlgParams.buffInfo.outBuffBaseOff, processedDataCount, dataTypeSize_,
        agTempAlgParams.buffInfo.hcclBuffBaseOff);


    u64 agSliceSize = currDataCount / rankSize_ * dataTypeSize_;
    u64 agTailSize = (currDataCount / rankSize_ + currDataCount % rankSize_) * dataTypeSize_;
    HCCL_INFO("[RunAllGather] slice calculation: currDataCount[%llu], rankSize[%u], "
        "currDataCount/rankSize[%llu], remainder[%llu], dataTypeSize[%llu], "
        "agSliceSize[%llu], agTailSize[%llu]",
        currDataCount, rankSize_, currDataCount / rankSize_, currDataCount % rankSize_,
        dataTypeSize_, agSliceSize, agTailSize);
    
    std::vector<u64> agSliceSizes;
    for (u32 rankId = 0; rankId < rankSize_; rankId++) {
        u64 size = (rankId == rankSize_ - 1) ? agTailSize : agSliceSize;
        agSliceSizes.push_back(size);
        HCCL_INFO("[RunAllGather] agSliceSizes[%u] = [%llu] (%s)",
            rankId, size, (rankId == rankSize_ - 1) ? "TAIL" : "SLICE");
    }
    agTempAlgParams.allRankSliceSize = agSliceSizes;
    // 设置allRankDispls和allRankProcessedDataCount（AllGather模板依赖这两个字段）
    agTempAlgParams.allRankDispls.clear();
    agTempAlgParams.allRankProcessedDataCount.clear();
    u64 agSliceCount = currDataCount / rankSize_;
    u64 agTailCount = currDataCount / rankSize_ + currDataCount % rankSize_;
    u64 displsOffset = 0;
    for (u32 rankId = 0; rankId < rankSize_; rankId++) {
        u64 size = (rankId == rankSize_ - 1) ? agTailSize : agSliceSize;
        u64 count = (rankId == rankSize_ - 1) ? agTailCount : agSliceCount;
        agTempAlgParams.allRankDispls.push_back(displsOffset);
        agTempAlgParams.allRankProcessedDataCount.push_back(count);
        HCCL_INFO("[RunAllGather] allRankDispls[%u] = [%llu], allRankProcessedDataCount[%u] = [%llu] (%s)",
            rankId, displsOffset, rankId, count, (rankId == rankSize_ - 1) ? "TAIL" : "SLICE");
        displsOffset += size;
    }
    agTempAlgParams.sliceSize = agSliceSize;
    agTempAlgParams.tailSize = agTailSize;
    agTempAlgParams.inputSliceStride = 0;
    agTempAlgParams.outputSliceStride = agSliceSize;
    agTempAlgParams.repeatNum = 1;
    agTempAlgParams.inputRepeatStride = 0;
    agTempAlgParams.outputRepeatStride = 0;
    agTempAlgParams.count = currDataCount / rankSize_;

    HCCL_INFO("[RunAllGather] agTempAlgParams summary: sliceSize[%llu], tailSize[%llu], "
        "inputSliceStride[%llu], outputSliceStride[%llu], repeatNum[%u], "
        "inputRepeatStride[%llu], outputRepeatStride[%llu], count[%llu]",
        agTempAlgParams.sliceSize, agTempAlgParams.tailSize,
        agTempAlgParams.inputSliceStride, agTempAlgParams.outputSliceStride,
        agTempAlgParams.repeatNum, agTempAlgParams.inputRepeatStride,
        agTempAlgParams.outputRepeatStride, agTempAlgParams.count);

    HCCL_INFO("[RunAllGather] >>> calling agTempAlg->KernelRun");
    CHK_RET(agTempAlg->KernelRun(param, agTempAlgParams, agTemplateAlgRes));
    HCCL_INFO("[RunAllGather] agTempAlg->KernelRun done");
    
    HCCL_INFO("[RunAllGather] === EXIT ===");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, AllReduceOrderPreserved,
    InsV2AllReduceOrderPreservedExecutor, TopoMatch1D,
    InsTempReduceScatterOrderPreservedLevel1, InsTempAllGatherMesh1D);

}