/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_executor_base.h"
#include "hccl_aiv.h"

namespace hccl {

CollExecutorBase::CollExecutorBase(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : dispatcher_(dispatcher), topoMatcher_(topoMatcher)
{
}

HcclResult CollExecutorBase::SetAlgType(const AlgType algType)
{
    const std::vector<AlgTypeLevel0> &l0Algo = desc_.level0SupportedAlgos;
    const std::vector<AlgTypeLevel1> &l1Algo = desc_.level1SupportedAlgos;
    const std::vector<AlgTypeLevel2> &l2Algo = desc_.level2SupportedAlgos;

    algType_ = algType;
    if (!l0Algo.empty()
        && std::find(l0Algo.begin(), l0Algo.end(), algType_.algoLevel0) == l0Algo.end()) {
        HCCL_WARNING("[%s] not support level0 algo[%d], reset to algo[%d]", __func__,
            algType_.algoLevel0, l0Algo[0]);
        algType_.algoLevel0 = l0Algo[0];
    }
    if (!l1Algo.empty()
        && std::find(l1Algo.begin(), l1Algo.end(), algType_.algoLevel1) == l1Algo.end()) {
        HCCL_WARNING("[%s] not support level1 algo[%d], reset to algo[%d]", __func__,
            algType_.algoLevel1, l1Algo[0]);
        algType_.algoLevel1 = l1Algo[0];
    }
    if (!l2Algo.empty()
        && std::find(l2Algo.begin(), l2Algo.end(), algType_.algoLevel2) == l2Algo.end()) {
        HCCL_WARNING("[%s] not support level2 algo[%d], reset to algo[%d]", __func__,
            algType_.algoLevel2, l2Algo[0]);
        algType_.algoLevel2 = l2Algo[0];
    }
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetCCLInBuffer(u64 cclbufferSize)
{
    inCCLbufferSize_ = cclbufferSize;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetIsSupportSDMAReduce(bool isSupportSDMAReduce)
{
    isSupportSDMAReduce_ = isSupportSDMAReduce;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::RunTemplate(const std::unique_ptr<AlgTemplateBase> &tempAlg, const SubCommInfo &commInfo)
{
    HcclResult ret = tempAlg->RunAsync(commInfo.localRank, commInfo.localRankSize, commInfo.links);
    CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("[CollExecutorBase][RunTemplate]" \
        "group has been destroyed. Break!"), ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollExecutorBase][RunTemplate]run tempAlg rank[%u] rank size[%u] failed",
        commInfo.localRank, commInfo.localRankSize), ret);
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::GetAivExecParam(const OpParam& param, AlgResourceResponse& algRes, AivSuperKernelArgs &args)
{
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::PrepareCommInfoToDevice(AlgResourceResponse& algResource)
{
    (void) algResource;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::CalcIncreLinkRequest(const OpParam& param, AlgResourceRequest &resourceRequest)
{
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::CreatePairWiseList(HcclSendRecvItem *sendRecvInfo, u32 itemNum)
{
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::GetPairWiseList(std::vector<std::vector<HcclSendRecvItem*>> &sendRecvPairList)
{
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetAlgOpContext(AlgOpContext algOpContext)
{
    algOpContext_ = algOpContext;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetAivClearEnable(bool aivClearEnable)
{
    aivClearEnable_ = aivClearEnable;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::CalBlockDim(u32& blockDim, u32 rankSize, u64 dataSize, HcclCMDType cmdType)
{
    blockDim = rankSize;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::GetBlockDim(u32& blockDim)
{
    blockDim = blockDim_;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetBlockDim(const u32& blockDim)
{
    blockDim_ = blockDim;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetOpCounter(const OpCounterInfo& opCounter)
{
    opCounter_ = opCounter;
    return HCCL_SUCCESS;
}
HcclResult CollExecutorBase::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    return HCCL_SUCCESS;
}

}