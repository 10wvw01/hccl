/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "template_utils.h"
#include "aicpu/ins_temp_all_to_all_v_omni.h"
#include "ins_omni_sole_executor.h"


namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsOmniSoleExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    HCCL_INFO("[InitCommInfo] begin ");
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];

    devType_ = topoInfo->deviceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;

    // dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    const u64* data = reinterpret_cast<const u64*>(param.varData);
    dataCount_ = data[0];
    dataSize_ = dataCount_ * dataTypeSize_;
    HCCL_INFO("[InsOmniSoleExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], dataType_ [%u], "
        "dataCount_ [%llu]", myRank_, rankSize_, devType_, dataType_, dataCount_);

    HCCL_INFO("[InsOmniSoleExecutor][InitCommInfo] dataTypeSize_ [%u], dataSize_ [%llu]", dataTypeSize_, dataSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ParseXmlInfo(const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    // 解析bin文件 读取内容存入xmlInfo结构体中
    xmlInfo_.resInfo.slaveThreadNum = 3;
    xmlInfo_.resInfo.notifyNumOnMainThread = 3;
    xmlInfo_.resInfo.notifyNumPerThread = 3;
    xmlInfo_.resInfo.netLayerNum = 1;

    if (myRank_ == 0) {
        std::map<u32, OmniChannelInfo> tmp;
        OmniChannelInfo tmpInfo;
        tmpInfo.channelId = 0;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 1;
        tmp[1] = tmpInfo;

        tmpInfo.channelId = 1;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 2;
        tmp[2] = tmpInfo;

        tmpInfo.channelId = 2;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 3;
        tmp[3] = tmpInfo;
        xmlInfo_.resInfo.mapchannelInfo.push_back(tmp);
    } else if (myRank_ == 1) {
        std::map<u32, OmniChannelInfo> tmp;
        OmniChannelInfo tmpInfo;
        tmpInfo.channelId = 0;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 0;
        tmp[0] = tmpInfo;

        tmpInfo.channelId = 1;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 2;
        tmp[2] = tmpInfo;

        tmpInfo.channelId = 2;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 3;
        tmp[3] = tmpInfo;
        xmlInfo_.resInfo.mapchannelInfo.push_back(tmp);
    } else if (myRank_ == 2) {
        std::map<u32, OmniChannelInfo> tmp;
        OmniChannelInfo tmpInfo;
        tmpInfo.channelId = 0;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 0;
        tmp[0] = tmpInfo;

        tmpInfo.channelId = 1;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 1;
        tmp[1] = tmpInfo;

        tmpInfo.channelId = 2;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 3;
        tmp[3] = tmpInfo;
        xmlInfo_.resInfo.mapchannelInfo.push_back(tmp);
    } else if (myRank_ == 3) {
        std::map<u32, OmniChannelInfo> tmp;
        OmniChannelInfo tmpInfo;
        tmpInfo.channelId = 0;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 0;
        tmp[0] = tmpInfo;

        tmpInfo.channelId = 1;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 1;
        tmp[1] = tmpInfo;

        tmpInfo.channelId = 2;
        tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
        tmpInfo.remoteRank = 2;
        tmp[2] = tmpInfo;
        xmlInfo_.resInfo.mapchannelInfo.push_back(tmp);
    }

    if (myRank_ == 0) {        
        OmniSendRecvInfo tmpInfo;
        tmpInfo.optype = OP_LOCAL_COPY;
        tmpInfo.srcSliceInfo.resize(1);
        tmpInfo.srcSliceInfo[0].sliceType = 0;
        tmpInfo.srcSliceInfo[0].sliceIdx = 0;

        tmpInfo.dstSliceInfo.resize(1);
        tmpInfo.dstSliceInfo[0].sliceType = 1;
        tmpInfo.dstSliceInfo[0].sliceIdx = 0;
        tmpInfo.remoteRank = 1;

        tmpInfo.sliceNum = 4;

        xmlInfo_.vecSendRecvInfo.push_back(tmpInfo);
    } else if (myRank_ == 1) {        
        OmniSendRecvInfo tmpInfo;
        tmpInfo.optype = OP_LOCAL_COPY;
        tmpInfo.srcSliceInfo.resize(1);
        tmpInfo.srcSliceInfo[0].sliceType = 0;
        tmpInfo.srcSliceInfo[0].sliceIdx = 1;

        tmpInfo.dstSliceInfo.resize(1);
        tmpInfo.dstSliceInfo[0].sliceType = 1;
        tmpInfo.dstSliceInfo[0].sliceIdx = 1;
        tmpInfo.remoteRank = 1;
        tmpInfo.sliceNum = 4;

        xmlInfo_.vecSendRecvInfo.push_back(tmpInfo);
    } else if (myRank_ == 2) {        
        OmniSendRecvInfo tmpInfo;
        tmpInfo.optype = OP_LOCAL_COPY;
        tmpInfo.srcSliceInfo.resize(1);
        tmpInfo.srcSliceInfo[0].sliceType = 0;
        tmpInfo.srcSliceInfo[0].sliceIdx = 2;

        tmpInfo.dstSliceInfo.resize(1);
        tmpInfo.dstSliceInfo[0].sliceType = 1;
        tmpInfo.dstSliceInfo[0].sliceIdx = 2;
        tmpInfo.remoteRank = 1;
        tmpInfo.sliceNum = 4;

        xmlInfo_.vecSendRecvInfo.push_back(tmpInfo);
    } else if (myRank_ == 3) {        
        OmniSendRecvInfo tmpInfo;
        tmpInfo.optype = OP_LOCAL_COPY;
        tmpInfo.srcSliceInfo.resize(1);
        tmpInfo.srcSliceInfo[0].sliceType = 0;
        tmpInfo.srcSliceInfo[0].sliceIdx = 3;

        tmpInfo.dstSliceInfo.resize(1);
        tmpInfo.dstSliceInfo[0].sliceType = 1;
        tmpInfo.dstSliceInfo[0].sliceIdx = 3;
        tmpInfo.remoteRank = 1;
        tmpInfo.sliceNum = 4;

        xmlInfo_.vecSendRecvInfo.push_back(tmpInfo);
    }
    


    return HCCL_SUCCESS;

}


template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam& param,
                       const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                       AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo));
    CHK_RET(ParseXmlInfo(param, topoInfo)); // 解析xml

    std::vector<std::vector<u32>> tempAlgHierachyInfo;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        tempAlgHierachyInfo.push_back(algHierarchyInfo.infos[0][1]);    // clos拓扑，包含所有rank
    } else {
        tempAlgHierachyInfo = algHierarchyInfo.infos[0];
    }

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate = 
        std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, tempAlgHierachyInfo);
    // 调用计算资源的函数
    algTemplate->CalcRes(comm, param, topoInfo, resourceRequest, xmlInfo_);

    return HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsOmniSoleExecutor][Orchestrate] Orchestrate Start, rankid [%u]", myRank_);

    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, &resCtx.topoInfo));

    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU && param.engine != CommEngine::COMM_ENGINE_AICPU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsOmniSoleExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("[InsOmniSoleExecutor][Orchestrate] Orchestrate End, rankid [%u]", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsOmniSoleExecutor][OrchestrateLoop] Start, rankid [%u]", myRank_);
    
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);

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
    HCCL_INFO("[InsOmniSoleExecutor][OrchestrateLoop]loopTimes = [%u]", loopTimes);

    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        TemplateDataParams tempAlgParams;
        tempAlgParams.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams.buffInfo.outputPtr = param.outputPtr;
        tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
        // tempAlgParams.sliceSize = currDataCount * dataTypeSize_ / xmlInfo_.vecSendRecvInfo[0].sliceNum;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
        tempAlgParams.repeatNum = 1;  // 不需要重复
        tempAlgParams.inputRepeatStride = 0;
        tempAlgParams.outputRepeatStride = 0;
        tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;

        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));

        processedDataCount += currDataCount;
    }

    HCCL_INFO("[InsOmniSoleExecutor][OrchestrateLoop] End, rankid [%u]", myRank_);
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLVC,
                CcuOMNI,
                InsOmniSoleExecutor,
                TopoMatch1D,
                CcuTempOmni);

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLV,
                AicpuOMNI,
                InsOmniSoleExecutor,
                TopoMatch1D,
                InsTempAlltoAllVOmni);

}