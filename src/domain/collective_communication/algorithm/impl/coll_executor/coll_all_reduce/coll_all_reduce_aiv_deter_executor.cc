/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_aiv_deter_executor.h"

namespace hccl {

CollAllReduceAivDeterExecutor::CollAllReduceAivDeterExecutor(const HcclDispatcher dispatcher,
                                                           std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
}

HcclResult CollAllReduceAivDeterExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO("[CollAllReduceAivDeterExecutor][CalcStreamNum] tag[%s] streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceAivDeterExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceAivDeterExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO("[CollAllReduceAivDeterExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceAivDeterExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceAivDeterExecutor::CalBlockDim(u32& blockDim, u32 rankSize, u64 dataSize, HcclCMDType cmdType)
{
    blockDim = rankSize; // 默认情况使用rankSize个AIV

    blockDim = BLOCK_DIM_FACTOR_TWO * rankSize; // 小数据量使用2倍rankSize的AIV core
    if (dataSize >= AIV_ALL_REDUCE_DETER_SIZE) {
        blockDim = BLOCK_DIM_FACTOR_THREE * rankSize;
    }

    HCCL_INFO("[CollAllReduceAivDeterExecutor][CalBlockDim] datasize is [%u], blockDim is set to [%u]", dataSize, blockDim);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceAivDeterExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    // 确定性场景不支持图模式
    execMem.inputMem = algRes.cclInputMem; 
    execMem.outputMem = algRes.aivOutputMem; // 使用AIVOUT
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceAivDeterExecutor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], AllReduce executor orchestrate success, take time [%lld]us",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceAivDeterExecutor::GetAivExecParam(const OpParam& param, AlgResourceResponse& algRes, AivSuperKernelArgs &args)
{
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceAivDeterExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllReduceAivDeterExecutor][KernelRun]allreduce aiv enter.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    void *buffersIn[MAX_RANK_SIZE];
    void *buffersOut[MAX_RANK_SIZE];

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_CONFIG_DEBUG(HCCL_ALG,
        "[CollAllReduceAivDeterExecutor][KernelRun] userRank [%u] localRank [%u]", topoAttr_.userRank, localRank);

    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersIn[i])));
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersOut[i])));
        } else {
            buffersIn[i] = execMem.inputMem.ptr();
            buffersOut[i] = execMem.outputMem.ptr();
        }
    }

    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    AivOpArgs opArgs {
        HcclCMDType::HCCL_CMD_ALLREDUCE, execMem.inputPtr, execMem.outputPtr, execMem.count,
        param.DataDes.dataType, param.reduceType, 0, isOpbase
    };
    AivTopoArgs topoArgs { localRank, localRankSize, MAX_RANK_SIZE, 0, 1, topoAttr_.deviceType };
    topoArgs.identify = algoAttr_.identifier;
    
    u64 dataSize = SIZE_TABLE[param.DataDes.dataType] * execMem.count;
    u32 blockDim;
    CHK_RET(CalBlockDim(blockDim, localRankSize, dataSize));
    blockDim_ = blockDim;
    AivResourceArgs resourceArgs {
        param.tag, param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(), blockDim_, param.aivTag
    };
    AivAlgArgs algArgs {};
    algArgs.deterministic = 1;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;
    HCCL_INFO("[CollAllReduceAivDeterExecutor][KernelRun]allreduce bufferin[%d] bufferout[%d]",execMem.inputMem.size(), execMem.outputMem.size());

    if (aivClearEnable_) {
        ClearAivSyncBuf(buffersOut, param.stream.ptr(), topoArgs);
    }

    HcclResult ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo);
    
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceAivDeterExecutor][KernelRun]allreduce aiv failed, return[%d]", ret), ret);

    HCCL_INFO("[CollAllReduceAivDeterExecutor][KernelRun]allreduce aiv run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceAivDeterExecutor", AllReduceAivDeter, CollAllReduceAivDeterExecutor);

} // namespace hccl