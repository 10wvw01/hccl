/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_mesh_aiv_smallcount_executor.h"

namespace hccl {

CollAllReduceMeshAivSmallCountExecutor::CollAllReduceMeshAivSmallCountExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher): CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
}

HcclResult CollAllReduceMeshAivSmallCountExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO("[CollAllReduceMeshAivSmallCountExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshAivSmallCountExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshAivSmallCountExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    inputType = TransportMemType::AIV_INPUT;
    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO("[CollAllReduceMeshAivSmallCountExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d].", tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshAivSmallCountExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshAivSmallCountExecutor::CalBlockDim(u32& blockDim, u32 rankSize, u64 dataSize, HcclCMDType cmdType)
{
    blockDim = rankSize; // 默认情况使用rankSize个AIV
    u32 bestBlockDim = blockDim;

    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    if (isOpbase) {
        CHK_PRT_RET(blockDim_ < blockDim,
            HCCL_ERROR("[CollAllReduceMeshAivSmallCountExecutor][CalBlockDim]aivCore[%u] is less than need[%u].",
            blockDim_, blockDim), HCCL_E_PARA);
    }

    HCCL_INFO("[CollAllReduceMeshAivSmallCountExecutor][CalBlockDim] blockDim is set to [%u], limit[%u], best[%u]",
        blockDim, blockDim_, bestBlockDim);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshAivSmallCountExecutor::GetAivExecParam(const OpParam& param, AlgResourceResponse& algRes, AivSuperKernelArgs &args)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
 
    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
 
    // 单算子大数据量
    execMem.inputMem = algRes.aivInputMem;
    execMem.outputMem = algRes.aivOutputMem;

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
 
    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG("[CollAllReduceMeshAivSmallCountExecutor][GetAivExecParam] userRank [%d] localRank [%d]",
        topoAttr_.userRank, localRank);
 
    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(args.buffersIn[i])));
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(args.buffersOut[i])));
        } else {
            args.buffersIn[i] = execMem.inputMem.ptr();
            args.buffersOut[i] = execMem.outputMem.ptr();
        }
    }

    HCCL_INFO("SPK, buffersIn [%p] [%p] [%p] [%p]"
        "buffersOut [%p] [%p] [%p] [%p]", args.buffersIn[0], args.buffersIn[1], args.buffersIn[2], args.buffersIn[3],
        args.buffersOut[0], args.buffersOut[1], args.buffersOut[2], args.buffersOut[3]);
    args.rank = localRank;
    args.rankSize = localRankSize;
    args.len = execMem.count;
    args.dataType = param.DataDes.dataType;
    args.unitSize = SIZE_TABLE[param.DataDes.dataType];
    args.reduceOp = param.reduceType;

    HCCL_INFO("SPK [CollAllReduceMeshAivSmallCountExecutor][GetAivExecParam], rank[%llu], rankSize[%llu], len[%llu],datatype[%llu], op[%llu]", args.rank, args.rankSize, args.len, args.dataType, args.reduceOp);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceMeshAivSmallCountExecutor][Orchestrate]errNo[0x%016llx] tag[%s] excutor kernel "
            "run failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
 
    HCCL_INFO("tag[%s], AllReduce executor getalgexecparam success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshAivSmallCountExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    // 小数据量都用aiv
    execMem.inputMem = algRes.aivInputMem;
    execMem.outputMem = algRes.aivOutputMem;
    execMem.scratchMem = algRes.scratchMem; // 不需要
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceMeshAivSmallCountExecutor][Orchestrate]errNo[0x%016llx] tag[%s] excutor kernel "
            "run failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], AllReduce executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshAivSmallCountExecutor::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshAivSmallCountExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllReduceMeshAivSmallCountExecutor][KernelRun]allreduce aiv enter.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    void *buffersIn[MAX_RANK_SIZE];
    void *buffersOut[MAX_RANK_SIZE];

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG("[CollAllReduceMeshAivSmallCountExecutor][KernelRun] userRank [%u] localRank [%u]", topoAttr_.userRank, localRank);

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
    AivTopoArgs topoArgs { localRank, localRankSize, MAX_RANK_SIZE, 0, 1, topoAttr_.deviceType, algoAttr_.identifier};
    u32 blockDim;
    CHK_RET(CalBlockDim(blockDim, localRankSize));
    blockDim_ = blockDim;
    AivResourceArgs resourceArgs {
        param.tag, param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(), blockDim_, param.aivTag
    };
    AivAlgArgs algArgs {};
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;
    if (aivClearEnable_) {
        ClearAivSyncBuf(buffersOut, param.stream.ptr(), topoArgs);
    }

    HcclResult ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceMeshAivSmallCountExecutor][KernelRun]allreduce aiv failed, return[%d]", ret), ret);

    HCCL_INFO("[CollAllReduceMeshAivSmallCountExecutor][KernelRun]allreduce aiv run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceMeshAivSmallCountExecutor", AllReduceMeshAivSmallCount,
    CollAllReduceMeshAivSmallCountExecutor);

} // namespace hccl