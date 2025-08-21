/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "coll_all_gather_mesh_aiv_for_910_93_executor.h"

namespace hccl {
CollAllGatherMeshAivFor91093Executor::CollAllGatherMeshAivFor91093Executor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher): 
    CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
    desc_.isAivCrossNode = true;
}
 
HcclResult CollAllGatherMeshAivFor91093Executor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO("[CollAllGatherMeshAivFor91093Executor][CalcStreamNum] tag[%s] streamNum[%u].",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}
 
HcclResult CollAllGatherMeshAivFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}
 
HcclResult CollAllGatherMeshAivFor91093Executor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    inputType = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ?
        TransportMemType::CCL_INPUT : TransportMemType::PARAM_INPUT);
    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO("[CollAllGatherMeshAivFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d].", tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshAivFor91093Executor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
    commCombinePara.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
 
    LevelNSubCommTransport &commTransportLevel0 = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto &transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = false;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshAivFor91093Executor::CalBlockDim(u32& blockDim, u32 rankSize, u64 dataSize, HcclCMDType cmdType)
{
    blockDim = rankSize; // 默认情况使用rankSize个AIV

    bool isOpBase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93 && topoAttr_.serverNum > 1) { // A3超节点内多机场景
        // 多核并行优化
        if (rankSize > HALF_MAX_BLOCK_DIM || dataSize < AIV_A3_CROSSNODE_TINY_SIZE) {
            blockDim = rankSize;
        } else if (rankSize > ONE_THIRD_MAX_BLOCK_DIM || dataSize < AIV_A3_CROSSNODE_SMALL_SIZE) {
            blockDim = rankSize * BLOCK_DIM_FACTOR_TWO;
        } else if (rankSize > ONE_FOURTH_MAX_BLOCK_DIM) {
            blockDim = rankSize * BLOCK_DIM_FACTOR_THREE;
        } else if (rankSize > ONE_SIXTH_MAX_BLOCK_DIM || dataSize < AIV_A3_CROSSNODE_MID_SIZE) {
            blockDim = rankSize * BLOCK_DIM_FACTOR_FOUR;
        } else if (rankSize > ONE_EIGHTH_MAX_BLOCK_DIM) {
            blockDim = rankSize * BLOCK_DIM_FACTOR_SIX;
        } else {
            blockDim = rankSize * BLOCK_DIM_FACTOR_EIGHT;
        }

        // baseline
        if (rankSize > HALF_MAX_BLOCK_DIM) { // 当ranksize大于了一半的aiv核数时，走baseline，block_num需要为偶数
            blockDim =  (blockDim < MAX_BLOCK_DIM ? blockDim + blockDim % BLOCK_DIM_FACTOR_TWO : MAX_BLOCK_DIM);
        }
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93 && !isOpBase) {
        blockDim = rankSize * BLOCK_DIM_FOUR_PER_RANK_A3 > MAX_BLOCK_DIM ?
            rankSize * BLOCK_DIM_THREE_PER_RANK_A3 : rankSize * BLOCK_DIM_FOUR_PER_RANK_A3;
    } else if (isOpBase) {
        blockDim += 1; // 单机场景，单算子AllGather大数据使用(rankSize + 1)个aiv
    }

    u32 bestBlockDim = blockDim;
    if (isOpBase) {
        CHK_PRT_RET(rankSize > HALF_MAX_BLOCK_DIM && blockDim_ < BLOCK_DIM_FACTOR_TWO,
            HCCL_ERROR("[CollAllGatherMeshAivFor91093Executor][CalBlockDim]aivCore[%u] is invalid, at lest need 2.",
            blockDim_), HCCL_E_PARA);
        CHK_PRT_RET(rankSize <= HALF_MAX_BLOCK_DIM && blockDim_ < rankSize,
            HCCL_ERROR("[CollAllGatherMeshAivFor91093Executor][CalBlockDim]aivCore[%u] is invalid, at lest need [%u].",
            blockDim_, rankSize), HCCL_E_PARA);
        if (blockDim_ < blockDim) {
            if (rankSize > HALF_MAX_BLOCK_DIM) {
                blockDim = blockDim_ / BLOCK_DIM_FACTOR_TWO * BLOCK_DIM_FACTOR_TWO;
            } else {
                blockDim = blockDim_ / rankSize * rankSize;
            }
        }
    }

    HCCL_INFO("[CollAllGatherMeshAivFor91093Executor][CalBlockDim] blockDim is set to [%u], limit[%u], best[%u]",
        blockDim, blockDim_, bestBlockDim);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshAivFor91093Executor::PrepareCommInfoToDevice(AlgResourceResponse& algResource)
{
    HCCL_INFO("[CollAllGatherMeshAivFor91093Executor][PrepareCommInfoToDevice]allgather aiv copy comm info to device.");
    CHK_RET(CopyAivCommInfoToDevice(COMM_COMBINE_ORDER, COMM_INDEX_0, algResource));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshAivFor91093Executor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
 
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
 
    execMem.inputMem = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ?
        algRes.paramInputMem : algRes.cclInputMem);
    execMem.outputMem = algRes.aivOutputMem;
    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherMeshAivFor91093Executor][Orchestrate]errNo[0x%016llx] tag[%s] excutor kernel "
            "run failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
 
    HCCL_INFO("tag[%s], AllGather executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
 
HcclResult CollAllGatherMeshAivFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,"[CollAllGatherMeshAivFor91093Executor][KernelRun]allgather aiv enter.");
 
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
 
    void *buffersIn[MAX_RANK_SIZE];
    void *buffersOut[MAX_RANK_SIZE];
 
    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG("[CollAllGatherMeshAivFor91093Executor][KernelRun] userRank [%d] localRank [%d]",
        topoAttr_.userRank, localRank);

    buffersIn[0] = execMem.inputMem.ptr();
    buffersOut[0] = execMem.outputMem.ptr();
    constexpr u32 BUFFER_IDX_ONE = 1;
    buffersOut[BUFFER_IDX_ONE] = algResResp_->aivCommInfoMem.ptr(); // 通信域信息

    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    AivOpArgs opArgs {
        HcclCMDType::HCCL_CMD_ALLGATHER, execMem.inputPtr, execMem.outputPtr, execMem.count,
        param.DataDes.dataType, param.reduceType, 0, isOpbase
    };

    AivTopoArgs topoArgs { localRank, localRankSize, MAX_RANK_SIZE, 0, topoAttr_.serverNum, topoAttr_.deviceType, algoAttr_.identifier };
    u32 blockDim;
    CHK_RET(CalBlockDim(blockDim, localRankSize, opArgs.count * sizeof(opArgs.dataType)));
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
        HCCL_ERROR("[CollAllGatherMeshAivFor91093Executor][KernelRun]allgather aiv failed, return[%d]", ret),
        ret);
 
    HCCL_INFO("[CollAllGatherMeshAivFor91093Executor][KernelRun]allgather aiv run success.");
    return HCCL_SUCCESS;
}
 
REGISTER_EXEC("AllGatherMeshAivFor91093Executor", AllGatherMeshAivFor91093, CollAllGatherMeshAivFor91093Executor);
 
} // namespace hccl