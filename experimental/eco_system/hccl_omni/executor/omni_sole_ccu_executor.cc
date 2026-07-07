/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <fstream>
#include "template_utils.h"

#include "omni_sole_ccu_executor.h"
#include "ccu_temp_omni.h"
#include "topo_match_ubx.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsOmniSoleCcuExecutor<AlgTopoMatch, InsAlgTemplate>::InsOmniSoleCcuExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleCcuExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleCcuExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo)
{
    HCCL_INFO("[InitCommInfo] begin ");

    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];

    // dataCount_ 来自 OpParamBlob.dataCount（经 omni_run 透传至 param.dataCount）：
    // 非 0 表示等分，按 dataCount_/sliceNum 切分；0 表示不等分，按 sendCounts/recvCounts 切分。
    dataCount_ = param.dataCount;
    dataSize_ = dataCount_ * dataTypeSize_;
    HCCL_INFO("[InsOmniSoleCcuExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], dataType_ [%u], "
              "dataCount_ [%llu]",
        myRank_, rankSize_, devType_, dataType_, dataCount_);

    HCCL_INFO("[InsOmniSoleCcuExecutor][InitCommInfo] dataTypeSize_ [%u], dataSize_ [%llu]", dataTypeSize_, dataSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleCcuExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("CalcRes BEGIN");
    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo));

    std::vector<std::vector<u32>> tempAlgHierachyInfo;
    tempAlgHierachyInfo = algHierarchyInfo.infos[0];

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, tempAlgHierachyInfo);
    // 调用计算资源的函数
    algTemplate->CalcRes(comm, param, topoInfo, resourceRequest);

    HCCL_INFO("CalcRes END");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleCcuExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    if (resCtx.topoInfo.xmlInfo.vecNormalInstruction.size() == 0) {
        INFO("[InsOmniSoleCcuExecutor][Orchestrate] xmlInfo Instruction is empty");
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[InsOmniSoleCcuExecutor][Orchestrate] Orchestrate Start, rankid [%u]", myRank_);

    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, &resCtx.topoInfo));

    // 给channels_和threads_赋值
    threads_ = resCtx.threads;

    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        syncNum_ = resCtx.topoInfo.xmlInfo.syncNum;
        syncNum_ = syncNum_ / 2; // 因为一对同步有post和pre
        if (syncNum_ == 0) {
            syncNum_ = 1;
        }
        HCCL_DEBUG("rankid [%u] thread size is %u, syncnum [%u]", myRank_, threads_.size(), syncNum_);

        if (syncNum_ > 1) {
            // 用于两个算法同步
            controlThread_ = threads_.at(0);
            tmpThread_.push_back(threads_.at(1));
            notifyIdxControlToTemplates_.push_back(0);
            notifyIdxTemplatesToControl_.push_back(0);
        }
    }

    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU
        && param.engine != CommEngine::COMM_ENGINE_AICPU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsOmniSoleCcuExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed", HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO("[InsOmniSoleCcuExecutor][Orchestrate] Orchestrate End, rankid [%u]", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleCcuExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsOmniSoleCcuExecutor][OrchestrateLoop] Start, rankid [%u]", myRank_);

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);

    // 准备资源
    TemplateResource templateAlgRes;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    if (param.engine == COMM_ENGINE_CCU) {
        templateAlgRes.ccuKernels = resCtx.ccuKernels;
    }
    templateAlgRes.threads = resCtx.threads;

    // 从 param.all2AllVDataDes 提取 sendCounts/recvCounts/sdispls/rdispls
    const u64 *scPtr = static_cast<const u64 *>(param.all2AllVDataDes.sendCounts);
    const u64 *rcPtr = static_cast<const u64 *>(param.all2AllVDataDes.recvCounts);
    const u64 *sdPtr = static_cast<const u64 *>(param.all2AllVDataDes.sdispls);
    const u64 *rdPtr = static_cast<const u64 *>(param.all2AllVDataDes.rdispls);

    u64 sliceNum = resCtx.topoInfo.xmlInfo.vecNormalInstruction[0].sendRecvInfo.sliceNum;
    if (sliceNum == 0) {
        HCCL_INFO("[InsOmniSoleCcuExecutor][OrchestrateLoop] sliceNum is 0, skip");
        return HCCL_SUCCESS;
    }

    std::vector<u64> sendSliceData;
    std::vector<u64> recvSliceData;
    std::vector<u64> sendSdispls;
    std::vector<u64> localSdispls;
    CHK_RET(BuildOmniSliceMeta(param, rankSize_, sliceNum, dataCount_, sendSliceData, recvSliceData, sendSdispls,
        localSdispls));

    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;

    tempAlgParams.sendCounts.clear();
    tempAlgParams.recvCounts.clear();
    tempAlgParams.sdispls.clear();
    tempAlgParams.rdispls.clear();
    for (u32 i = 0; i < rankSize_; i++) {
        tempAlgParams.sendCounts.push_back(scPtr[i]);
        tempAlgParams.recvCounts.push_back(rcPtr[i]);
        tempAlgParams.sdispls.push_back(sdPtr[i]);
        tempAlgParams.rdispls.push_back(rdPtr[i]);
    }

    for (u32 i = 0; i < syncNum_; i++) {
        if (syncNum_ > 1) {
            CHK_RET(PreSyncInterThreads(controlThread_, tmpThread_, notifyIdxControlToTemplates_));
        }
        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes, resCtx.topoInfo.xmlInfo, sendSliceData,
            recvSliceData, sendSdispls, localSdispls, i));
        if (syncNum_ > 1) {
            CHK_RET(PostSyncInterThreads(controlThread_, tmpThread_, notifyIdxTemplatesToControl_));
        }
    }

#ifndef AICPU_COMPILE
    if (syncNum_ == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, templateAlgRes, resCtx.notifyNumOnMainThread));
    }
#endif

    HCCL_INFO("[InsOmniSoleCcuExecutor][OrchestrateLoop] End, rankid [%u]", myRank_);
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleCcuExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunchSaveCtx(
    const OpParam &param, const TemplateResource &templateAlgRes, u32 notifyNumOnMainThread)
{
    HCCL_INFO("[InsOmniSoleCcuExecutor] loopTimes==1, save fast launch ctx.");
    u32 threadNum = 1;
    u32 ccuKernelNum = templateAlgRes.submitInfos.size();
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsOmniSoleCcuExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[InsOmniSoleCcuExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void *ctxPtr = nullptr;
    HCCL_INFO("[InsOmniSoleCcuExecutor][HcclEngineCtxCreate] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx *>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[InsOmniSoleCcuExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

    // 2 thread
    ccuFastLaunchCtx->threadNum = threadNum;
    ccuFastLaunchCtx->notifyNumOnMainThread = notifyNumOnMainThread;
    ThreadHandle *threads = ccuFastLaunchCtx->GetThreadHandlePtr();
    threads[0] = templateAlgRes.threads[0];

    // 3 ccu kernel handle, taskArg入参
    ccuFastLaunchCtx->ccuKernelNum[0] = ccuKernelNum;
    CcuKernelSubmitInfo *kernelSubmitInfos = ccuFastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    for (u32 i = 0; i < ccuKernelNum; i++) {
        kernelSubmitInfos[i] = templateAlgRes.submitInfos[i];
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleCcuExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunch(
    const OpParam &param, const CcuFastLaunchCtx *fastLaunchCtx)
{
    HCCL_INFO("[InsOmniSoleCcuExecutor][FastLaunch] Start.");
    TemplateFastLaunchCtx tempFastLaunchCtx;
    // 1 取thread
    ThreadHandle *threads = fastLaunchCtx->GetThreadHandlePtr();
    tempFastLaunchCtx.threads.assign(threads, threads + fastLaunchCtx->threadNum);
    HCCL_INFO("[InsOmniSoleCcuExecutor][FastLaunch] threadNum[%llu]", fastLaunchCtx->threadNum);

    // 2 取arg
    CcuKernelSubmitInfo *ccuKernelSubmitInfos = fastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    tempFastLaunchCtx.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + fastLaunchCtx->ccuKernelNum[0]);
    HCCL_INFO("[InsOmniSoleCcuExecutor][FastLaunch] ccuKernelNum[%llu]", fastLaunchCtx->ccuKernelNum[0]);
    tempFastLaunchCtx.buffInfo.inputPtr = param.inputPtr;
    tempFastLaunchCtx.buffInfo.outputPtr = param.outputPtr;
    tempFastLaunchCtx.buffInfo.hcclBuff = param.hcclBuff;

    // 3 调template
    std::unique_ptr<InsAlgTemplate> algTemplate = std::make_unique<InsAlgTemplate>();
    CHK_RET(algTemplate->FastLaunch(param, tempFastLaunchCtx));
    HCCL_INFO("[InsOmniSoleCcuExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

#ifndef AICPU_COMPILE
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, OmniRunCcu, InsOmniSoleCcuExecutor, TopoMatchUBX, CcuTempOmni);
#endif
} // namespace ops_hccl
