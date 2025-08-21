/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_aicpu_interface.h"

#include <sstream>
#include "common/aicpu_hccl_common.h"
#include "common/aicpu_hccl_def.h"
#include "common/sqe_context.h"
#include "mc2_trace_utils.h"
#include "profiling_manager_device.h"
#include "framework/aicpu_hccl_process.h"
#include "utils/mc2_aicpu_utils.h"
#include "framework/aicpu_communicator.h"
#include "utils/mc2_tiling_utils.h"
#include "framework/aicpu_prof.h"
#include "utils/aicpu_hdc_utils.h"

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuKfcResInit(void *args)
{
    if (args == nullptr) {
        HCCL_ERROR("args is null.");
        return HCCL_E_PARA;
    }

    KFCResInitTask *ctxArgs = reinterpret_cast<KFCResInitTask *>(args);
    AicpuHcclProcess aicpuHcclProcess;
    return aicpuHcclProcess.AicpuRpcResInit(reinterpret_cast<HccCommResParamTask *>(ctxArgs->context));
}

u64 GetTensorAddr(uint16_t index, uint8_t *tensorPtr) {
    uint64_t* dataAddr = reinterpret_cast<uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    uint64_t* resPtr = dataAddr + (tensorPtrOffset >> 3);
    return u64(*(resPtr + index));
}

__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvLaunch(void *args)
{
    KfcState state;
    static uint32_t aicpuOpIdx = 0;
    u64 launchEntryTime = GetCurCpuTimestamp(true);

    if (args == nullptr) {
        HCCL_ERROR("args is null.");
        return HCCL_E_PARA;
    }

    KFCTask *task = reinterpret_cast<KFCTask *>(args);
    HCCL_INFO("KFCTask inputA %p, outputC %p, commOut %p, context %p, workSpace %p, tilingData %p",
        task->inputA, task->outputC, task->commOut, task->context, task->workSpace, task->tilingData);
    HcclKFCTilingData *tilingData = reinterpret_cast<HcclKFCTilingData *>(task->tilingData);
    HccCommResParamTask *contextParam = reinterpret_cast<HccCommResParamTask *>(task->context);
    if (tilingData == nullptr || contextParam == nullptr) {
        HCCL_ERROR("tilingData or context args is null.");
        return HCCL_E_PARA;
    }

    std::string key = contextParam->hcomId;
    AicpuComContext *ctx = AicpuHcclProcess::AicpuGetInnerComContext(key);
    if (ctx == nullptr) {
        HCCL_ERROR("the comm domain have not exist.");
        return HCCL_E_PARA;
    }
    if ((ctx->dfxExtendInfo.cqeStatus != dfx::CqeStatus::kDefault) ||
        (ctx->dfxExtendInfo.pollStatus == dfx::PollStatus::kStopAsException)) {
        HCCL_ERROR("Exist errors before, cqeStatus:%d, pollStatus:%d, group[%s]", ctx->dfxExtendInfo.cqeStatus,
                   ctx->dfxExtendInfo.pollStatus, contextParam->hcomId);
        return HCCL_E_INTERNAL;
    }
    ctx->debugMode = tilingData->debugMode;
    if (ctx->debugMode == MC2_DEBUG_ONLY_CUBE) {
        HCCL_INFO("DebugMode is set to be 1 (i.e. computation only).");
        return HCCL_SUCCESS;
    }
    if (!tilingData->useBufferType) {
        ctx->gatherOut = task->commOut;
    } else {
        ctx->gatherOut = task->outputC;
    }
    // 这里就是判断Suspending通道的内容
    HCCL_DEBUG("[NsRecovery]check the suspending status");
    HcclComSuspendingFlag kfcFlag = HcclComSuspendingFlag ::isNull;
    CHK_RET(AicpuHdcUtils::GetSuspendingStatus(ctx->kfcControlTransferH2D, kfcFlag));
    if (kfcFlag == HcclComSuspendingFlag::isSuspending) {
        HCCL_WARNING("[NsRecovery] the op should not be launched in the suspending status");
        return 0;
    }
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), false);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), false);
    SqeContextUtils::SyncVariable();
    if (MC2AicpuUtils::NeedRecordTimeTaken(*ctx)) {
        ctx->acprof[g_proxLoopCnt].tid = syscall(__NR_gettid);
        ctx->acprof[g_proxLoopCnt].clusterId = ctx->clusterId;
        ctx->acprof[g_proxLoopCnt].rankId = ctx->rankId;
        ctx->acprof[g_proxLoopCnt].launchEntryTime = launchEntryTime;
    }
    HCCL_INFO("RunAicpuRpcSrvLaunch, preparePosition %u", tilingData->preparePosition);
    if (tilingData->preparePosition > 1) {
        HCCL_ERROR("invalid preparePosition %u", tilingData->preparePosition);
        return HCCL_E_PARA;
    }
    ctx->preparePosition = static_cast<TASK_PREPARE_POSITION>(tilingData->preparePosition);
    if (ctx->preparePosition == TASK_PREPARE_HOST) {
        ctx->notifyOff = tilingData->notifyOff;
        ctx->notifyBeginCnt = tilingData->notifyBeginCnt;
        ctx->notifyEndCnt = tilingData->notifyEndCnt;
        ctx->totalCnt = tilingData->totalCnt;
    } else {
        ctx->notifyOff = 0;
        ctx->notifyBeginCnt = 0;
        ctx->notifyEndCnt = 0;
        ctx->totalCnt = 0;
        u64 newAddr = ctx->workSpaceAddr;
        if (newAddr & 0x1ff) {
            newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
            HCCL_INFO("Align hcclmsgarea from %p to %p", ctx->workSpaceAddr, newAddr);
        }
        ctx->workSpaceAddr = newAddr;
    }
    tilingData->commAlg = (ctx->devType == DevType::DEV_TYPE_910B) ? COMM_ALG_FULL_MESH : tilingData->commAlg;
    ctx->commAlg = tilingData->commAlg;
    ctx->skipLocalDataCopy = tilingData->hasCommOut ? false : true;
    ctx->curTurnCnt = 0;
    ctx->acprof[g_proxLoopCnt].workCnt = 0;
    ctx->msgPosForKernel = 0;
    ctx->curTurnCntForKernel = 0;
    ctx->sendCntRecord[0] = MC2AicpuUtils::GetSendCnt(ctx);
    ctx->recvCntRecord[0] = MC2AicpuUtils::GetRecvCnt(ctx);
    ctx->totalTurnCntForKernel = 0;
    MC2AicpuUtils::PrintTilingData(*tilingData);
    MC2AicpuUtils::PrintMC2AicpuContext(*ctx);

    CHK_RET(MC2TraceUtils::Submit(task, tilingData));
    CHK_RET(MC2TraceUtils::Submit(ctx)); // 上报ctx消息

    aicpuOpIdx++;
    if (ctx->debugMode == MC2_DEBUG_PRINT_MSG || ctx->debugMode == MC2_DEBUG_PRINT_BUFF) {
        HCCL_RUN_INFO("Server start, MC2 opIdx:%u", aicpuOpIdx);
    }
    HCCL_DEBUG("[RunAicpuRpcSrvLaunch] g_proxLoopCnt:%u, workCnt:%u.", g_proxLoopCnt,
               ctx->acprof[g_proxLoopCnt].workCnt);
    auto ret = AicpuHcclProcess::AicpuRunRpcServer(ctx, task);
    ctx->sendCntRecord[3] = MC2AicpuUtils::GetSendCnt(ctx); // 3 记录执行结束时的sendCnt
    ctx->recvCntRecord[3] = MC2AicpuUtils::GetRecvCnt(ctx); // 3 记录执行结束时的recvCnt
    if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_SUSPENDING)) {
        MC2AicpuUtils::PrintTilingDataError(*tilingData);
        MC2AicpuUtils::PrintMC2AicpuContextError(*ctx);
        if (ctx->preparePosition == TASK_PREPARE_HOST) { // host展开时aicore会通过workspace回传维测信息, 解析后打印
            MC2AicpuUtils::PrintAicDebugInfo(ctx);
            HCCL_ERROR("Run rpc error, opIdx:%u, sndCnt:%d %d %d %d, rcvCnt:%d %d %d %d", aicpuOpIdx,
            ctx->sendCntRecord[0], ctx->sendCntRecord[1], ctx->sendCntRecord[2], ctx->sendCntRecord[3],
            ctx->recvCntRecord[0], ctx->recvCntRecord[1], ctx->recvCntRecord[2], ctx->recvCntRecord[3]);
        }
        HCCL_ERROR("AicpuRunRpcServer failed, MC2 opIdx:%u", aicpuOpIdx);
        return ret;
    } else if (ret == HCCL_E_SUSPENDING) {
        HCCL_INFO("mc2 opp is suspended");
        return AICPUSUSPENDING_ERROR;
    }
    if (MC2AicpuUtils::NeedRecordTimeTaken(*ctx)) {
        ctx->acprof[g_proxLoopCnt].endTime = GetCurCpuTimestamp(true);
    }
    CHK_RET(dfx::ProfilingManager::ReportTaskExecTimeLine(&ctx->acprof[g_proxLoopCnt]));
    AicpuHcclProcess::OutputProfLog(*ctx);
    AicpuHcclProcess::IncProfCnt(*ctx);
    SqeContextUtils::SaveVariable();
    CHK_RET(SqeContextUtils::ClearLocalBuff());
    HCCL_INFO("end RunAicpuRpcSrvLaunch");
    return 0;
}

__attribute__((visibility("default"))) uint32_t RunAicpuInnerRpcSrvGroupLaunch(void *args[],
                                                                               KFCGroupTilingDataAuto *tilingData,
                                                                               HcclCommParamDesc* desc)
{
    HCCL_INFO("Start RunAicpuInnerRpcSrvGroupLaunch");
    constexpr int DESC_POS = 0;

    if (tilingData == nullptr || tilingData->groupNum == 0) {
        HCCL_ERROR("tilingData is nullptr or groupNum is 0.");
        return HCCL_E_PARA;
    }
    for (uint32_t i = 0; i < tilingData->groupNum; ++i) {
        KFCTask singleTask;
        singleTask.inputA = u64(args[tilingData->msg[i].sendArgIndex + desc->hasFfts + desc->groupNum + 1]);
        singleTask.outputC = GetTensorAddr(i,
                             reinterpret_cast<uint8_t*>(args[tilingData->msg[i].recvArgIndex + desc->hasFfts + desc->groupNum + 1]));
        singleTask.commOut = 0;
        singleTask.context = u64(args[DESC_POS + desc->hasFfts + desc->groupNum]);
        singleTask.workSpace = u64(args[desc->tilingOff - 1]);
        singleTask.tilingData = u64(&tilingData->msg[i]);
        uint32_t ret = RunAicpuRpcSrvLaunch(&singleTask);
        if (ret != 0) {
            HCCL_ERROR("RunAicpuRpcSrvGroupLaunch runs failed.");
            return HCCL_E_PARA;
        }
    }
    return 0;
}

__attribute__((visibility("default"))) uint32_t RunKernelAicpuServerV2(void *args[], HcclCommParamDesc *desc,
    void *tilingData)
{
    u64 launchEntryTime = GetCurCpuTimestamp(true);
    // MC2目前只支持OP_BASE
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    static uint64_t aicpuOpIdx = 0;

    Mc2ServerCfg *cfg = MC2TilingGetServerCfg(tilingData);
    HcclCommProf::SetDebugMode(cfg->debugMode);
    if (HcclCommProf::IsDebugModeEquals(MC2_DEBUG_ONLY_CUBE)) {
        HCCL_INFO("DebugMode is set to be 1 (i.e. computation only).");
        return HCCL_SUCCESS;
    }

    KFCTaskV2 task;
    task.tilingData = reinterpret_cast<u64>(tilingData);
    if (tilingData == nullptr) {
        HCCL_ERROR("tilingData is null.");
        return HCCL_E_PARA;
    }
    task.ctxNum = desc->groupNum;
    if (task.ctxNum > MC2_RES_CTX_MAX) {
        HCCL_ERROR("group num must be smaller than %u.", MC2_RES_CTX_MAX);
        return HCCL_E_PARA;
    }
    for (int i = 0; i < desc->groupNum; i++) {
        task.context[i] = reinterpret_cast<u64>(args[desc->hasFfts + i + 1]);
        if (task.context[i] == 0) {
            HCCL_ERROR("idx %d ctx is null, please check the input ctx.", i);
            return HCCL_E_PARA;
        }
        MC2AicpuUtils::PrintMC2HcclOpResParam(reinterpret_cast<HcclOpResParam *>(task.context[i]));
    }
    task.workSpace = reinterpret_cast<u64>(args[desc->tilingOff - 1]);
    aicpuOpIdx++;
    if (HcclCommProf::IsDebugModeEquals(MC2_DEBUG_PRINT_MSG) || HcclCommProf::IsDebugModeEquals(MC2_DEBUG_PRINT_BUFF)) {
        HCCL_RUN_INFO("Server start, MC2 opIdx:%lu", aicpuOpIdx);
    }
    HCCL_INFO("Start launch RunAicpuInnerKfcSrvLaunch");
    HcclCommProf::GetCurrentAicpuProf()->workCnt = 0;
    auto ret = AicpuHcclProcess::AicpuRunRpcServerForMC2(&task);
    if (AicpuHcclProcess::CheckNsStopLaunchStatus(desc->groupNum) == HCCL_E_SUSPENDING) {
        HCCL_INFO("mc2 opp is suspended");
        return AICPUSUSPENDING_ERROR;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Server runs failed.");
        return ret;
    }
    if (HcclCommProf::NeedRecordTimeTaken()) {
        HcclCommProf::SetCurrentProf(launchEntryTime);
        HcclOpResParam *commParam = reinterpret_cast<HcclOpResParam *>(task.context[0]);
        HcclCommProf::GetCurrentAicpuProf()->rankId = commParam->topoInfo.userRank;
        HcclCommProf::GetCurrentAicpuProf()->endTime = GetCurCpuTimestamp(true);
    }

    CHK_RET(dfx::ProfilingManager::ReportTaskExecTimeLine(HcclCommProf::GetCurrentAicpuProf()));
    HcclCommProf::OutputProfLog();
    HcclCommProf::AddProfLoopCnt(1);
    HCCL_INFO("end kfc server.");
    return 0;
}

__attribute__((visibility("default"))) uint32_t RunAicpuApiRpcSrvLaunchV1(void *args[], HcclCommParamDesc *desc)
{
    static u32 aicpuOpIdx = 0;
    u64 launchEntryTime = GetCurCpuTimestamp(true);

    HccCommResParamTask *contextParam = nullptr;
    for (uint64_t i = 1; i <= desc->groupNum; ++i) {
        contextParam = reinterpret_cast<HccCommResParamTask *>(args[desc->hasFfts + i]);
        if (contextParam != nullptr) {
            HCCL_INFO("Idx %llu ctx addr %p.", i, contextParam);
            break;
        }
    }
    if (contextParam == nullptr) {
        HCCL_ERROR("Context args is null.");
        return HCCL_E_PARA;
    }

    std::string key = contextParam->hcomId;
    AicpuComContext *ctx = AicpuHcclProcess::AicpuGetInnerComContext(key);
    if (ctx == nullptr) {
        HCCL_ERROR("The comm domain %s have not exist.", key.c_str());
        return HCCL_E_PARA;
    }
	AicpuServerRole role = AicpuHcclProcess::GetVerifiedServerRole(*ctx);
	if (role == AicpuServerRole::INVALID) {
        HCCL_INFO("aicpu server role is invalid, return");
        return HCCL_SUCCESS;
    } else if (role == AicpuServerRole::SLAVE) {
        return AicpuHcclProcess::RunSlaveRpcServerForApi(ctx);
    }
    if (ctx->dfxExtendInfo.cqeStatus != dfx::CqeStatus::kDefault ||
        ctx->dfxExtendInfo.pollStatus == dfx::PollStatus::kStopAsException) {
        HCCL_ERROR("Exist errors before, cqeStatus:%d, pollStatus:%d, group[%s]", ctx->dfxExtendInfo.cqeStatus,
                   ctx->dfxExtendInfo.pollStatus, key.c_str());
        return HCCL_E_INTERNAL;
    }

    Mc2InitTilingInner *tilingData = reinterpret_cast<Mc2InitTilingInner *>(args[desc->tilingOff]);
    ctx->debugMode = tilingData->debugMode;
    if (ctx->debugMode == MC2_DEBUG_ONLY_CUBE) {
        HCCL_INFO("DebugMode is set to be 1 (i.e. computation only).");
        return HCCL_SUCCESS;
    }

    HcclComSuspendingFlag kfcFlag;
    CHK_RET(AicpuHdcUtils::GetSuspendingStatus(ctx->kfcControlTransferH2D, kfcFlag));
    if (kfcFlag == HcclComSuspendingFlag::isSuspending) {
        HCCL_WARNING("[NsRecovery] the op should not be launched in the suspending status");
        return HCCL_SUCCESS;
    }
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), false);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), false);
    SqeContextUtils::SyncVariable();
    if (MC2AicpuUtils::NeedRecordTimeTaken(*ctx)) {
        ctx->acprof[g_proxLoopCnt].tid = syscall(__NR_gettid);
        ctx->acprof[g_proxLoopCnt].clusterId = ctx->clusterId;
        ctx->acprof[g_proxLoopCnt].rankId = ctx->rankId;
        ctx->acprof[g_proxLoopCnt].launchEntryTime = launchEntryTime;
    }

    ctx->preparePosition = TASK_PREPARE_KERNEL;
    ctx->notifyOff = 0;
    ctx->notifyBeginCnt = 0;
    ctx->notifyEndCnt = 0;
    ctx->totalCnt = 0;
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
        HCCL_INFO("Align hcclmsgarea from %p to %p", ctx->workSpaceAddr, newAddr);
    }
    ctx->workSpaceAddr = newAddr;
    ctx->commAlg = COMM_ALG_FULL_MESH;
    ctx->curTurnCnt = 0;
    ctx->acprof[g_proxLoopCnt].workCnt = 0;
    ctx->msgPosForKernel = 0;
    ctx->curTurnCntForKernel = 0;
    ctx->totalTurnCntForKernel = 0;
    ctx->gatherOut = 0UL;
    MC2AicpuUtils::PrintTilingData(*tilingData);
    MC2AicpuUtils::PrintMC2AicpuContext(*ctx);

    CHK_RET(MC2TraceUtils::Submit(ctx)); // 上报ctx消息

    aicpuOpIdx++;
    if (ctx->debugMode == MC2_DEBUG_PRINT_MSG || ctx->debugMode == MC2_DEBUG_PRINT_BUFF) {
        HCCL_RUN_INFO("Server start, MC2 opIdx:%u", aicpuOpIdx);
    }
    HCCL_DEBUG("[RunAicpuRpcSrvLaunch] g_proxLoopCnt:%u, workCnt:%u.", g_proxLoopCnt,
        ctx->acprof[g_proxLoopCnt].workCnt);
    HcclResult ret = AicpuHcclProcess::AicpuRunRpcServerForApi(ctx);
    if (ret == HCCL_E_SUSPENDING) {
        HCCL_INFO("mc2 opp is suspended");
        return AICPUSUSPENDING_ERROR;
    } else if (ret != HCCL_SUCCESS) {
        MC2AicpuUtils::PrintTilingDataError(*tilingData);
        MC2AicpuUtils::PrintMC2AicpuContextError(*ctx);
        HCCL_ERROR("Server failed, MC2 opIdx:%u", aicpuOpIdx);
        return ret;
    }
    if (MC2AicpuUtils::NeedRecordTimeTaken(*ctx)) {
        ctx->acprof[g_proxLoopCnt].endTime = GetCurCpuTimestamp(true);
    }
    CHK_RET(dfx::ProfilingManager::ReportTaskExecTimeLine(&ctx->acprof[g_proxLoopCnt]));
    AicpuHcclProcess::OutputProfLog(*ctx);
    AicpuHcclProcess::IncProfCnt(*ctx);
    SqeContextUtils::SaveVariable();
    CHK_RET(SqeContextUtils::ClearLocalBuff());
    HCCL_INFO("Kfc server ends successfully.");
    return HCCL_SUCCESS;
}

__attribute__((visibility("default"))) uint32_t RunAicpuApiRpcSrvLaunchV2(void *args[], HcclCommParamDesc *desc)
{
    // MC2目前只支持OP_BASE
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    u64 launchEntryTime = GetCurCpuTimestamp(true);
    static uint64_t aicpuOpIdx = 0;

    const Mc2InitTilingInner *tilingData = static_cast<const Mc2InitTilingInner *>(args[desc->tilingOff]);
    HcclCommProf::SetDebugMode(tilingData->debugMode);
    if (HcclCommProf::IsDebugModeEquals(MC2_DEBUG_ONLY_CUBE)) {
        HCCL_INFO("DebugMode is set to be 1 (i.e. computation only).");
        return HCCL_SUCCESS;
    }

    KFCTaskV2 task;
    task.ctxNum = desc->groupNum;
    if (task.ctxNum > MC2_RES_CTX_MAX) {
        HCCL_ERROR("group num must be smaller than %u.", MC2_RES_CTX_MAX);
        return HCCL_E_PARA;
    }
    for (uint64_t i = 0; i < desc->groupNum; i++) {
        task.context[i] = reinterpret_cast<u64>(args[desc->hasFfts + i + 1]);
        if (task.context[i] == 0) {
            HCCL_ERROR("idx %d ctx is null, please check the input ctx.", i);
            return HCCL_E_PARA;
        }
    }
    task.workSpace = 0;
    HcclCommProf::GetCurrentAicpuProf()->workCnt = 0;
    aicpuOpIdx++;
    if (HcclCommProf::IsDebugModeEquals(MC2_DEBUG_PRINT_MSG) || HcclCommProf::IsDebugModeEquals(MC2_DEBUG_PRINT_BUFF)) {
        HCCL_RUN_INFO("Server start, MC2 opIdx:%lu", aicpuOpIdx);
    }
    HCCL_INFO("Start launch RunAicpuApiRpcSrvLaunchV2, aicpuOpIdx %lu", aicpuOpIdx);
    auto ret = AicpuHcclProcess::AicpuRunRpcServerForMC2V2(&task, tilingData);
    if (AicpuHcclProcess::CheckNsStopLaunchStatus(desc->groupNum) == HCCL_E_SUSPENDING) {
        HCCL_INFO("mc2 opp is suspended");
        return AICPUSUSPENDING_ERROR;
    }

    if (ret != HCCL_SUCCESS) {
        MC2AicpuUtils::PrintTilingDataError(*tilingData);
        HCCL_ERROR("[RunAicpuApiRpcSrvLaunchV2] aicpuOpIdx %lu", aicpuOpIdx);
        return ret;
    }
    if (HcclCommProf::NeedRecordTimeTaken()) {
        HcclCommProf::SetCurrentProf(launchEntryTime);
        HcclOpResParam *commParam = reinterpret_cast<HcclOpResParam *>(task.context[0]);
        HcclCommProf::GetCurrentAicpuProf()->rankId = commParam->topoInfo.userRank;
        HcclCommProf::GetCurrentAicpuProf()->endTime = GetCurCpuTimestamp(true);
    }
    const u32 blockNum = MC2AicpuUtils::GetBlockNum();
    const u32 blockIdx = MC2AicpuUtils::GetBlockIdx();
    const u32 totalQueueNum = tilingData->commBlockNum * tilingData->queueNum;
    const u32 turnOffset = blockIdx * (totalQueueNum / blockNum) + std::min(blockIdx, totalQueueNum % blockNum);
    CHK_RET(dfx::ProfilingManager::ReportTaskExecTimeLine(HcclCommProf::GetCurrentAicpuProf(), turnOffset));
    HcclCommProf::OutputProfLog();
    HcclCommProf::AddProfLoopCnt(1);
    HCCL_INFO("end RunAicpuApiRpcSrvLaunchV2");
    return 0;
}

__attribute__((visibility("default"))) uint32_t RunKernelAicpuServerForTilingApi(void *args[], HcclCommParamDesc* desc)
{
    DevType devType = AicpuHcclProcess::AicpuGetInnerDevType();
    if (devType == DevType::DEV_TYPE_910_93) {
        return RunAicpuApiRpcSrvLaunchV2(args, desc);
    } else {
        return RunAicpuApiRpcSrvLaunchV1(args, desc);
    }
}

__attribute__((visibility("default"))) uint32_t RunKernelAicpuServerV1(void *args[], HcclCommParamDesc *desc)
{
    HcclKFCTilingData *tilingData = static_cast<HcclKFCTilingData *>(args[desc->tilingOff]);
    HCCL_INFO("RunAicpuKfcSrvLaunch, tiling.sendArgIndex %lu", tilingData->sendArgIndex);
    HCCL_INFO("RunAicpuKfcSrvLaunch, tiling.recvArgIndex %lu", tilingData->recvArgIndex);
    HCCL_INFO("RunAicpuKfcSrvLaunch, tiling.commOutArgIndex %lu", tilingData->commOutArgIndex);
    HCCL_INFO("RunAicpuKfcSrvLaunch, tiling.hasCommOut %lu", tilingData->hasCommOut);
    KFCTask task;
    task.tilingData = u64(tilingData);
    task.inputA =  u64(args[tilingData->sendArgIndex + desc->hasFfts + desc->groupNum + 1]);
    task.outputC = u64(args[tilingData->recvArgIndex + desc->hasFfts + desc->groupNum + 1]);
    task.context = u64(args[desc->hasFfts + desc->groupNum]);
    task.workSpace = u64(args[desc->tilingOff - 1]);
    if (tilingData->commOutArgIndex != u64(0xff)) {
        task.commOut = u64(args[tilingData->commOutArgIndex + desc->hasFfts + desc->groupNum + 1]);
    } else {
        task.commOut = 0;
    }
    MC2AicpuUtils::PrintKFCTask(task);
    HCCL_INFO("Task Assembled. Start to launch RunAicpuRpcSrvLaunch");
    const uint32_t ret = RunAicpuRpcSrvLaunch(&task);
    HCCL_INFO("RunAicpuKfcSrvLaunch ends with result %lu.", ret);
    return ret;
}

constexpr u32 GROUP_DYN_FLAG = 23U;
constexpr u32 GROUP_TILING_MAGIC_NUM = 99U;
__attribute__((visibility("default"))) uint32_t RunAicpuKfcSrvLaunch(void *args[])
{
    KfcState state;
    bool profL1Open = dfx::ProfilingManager::IsProfL1On();
    bool profL0Open = dfx::ProfilingManager::IsProfL0On();
    HCCL_INFO("profL1Open:%d, profL0Open:%d", profL1Open, profL0Open);
    if (args == nullptr) {
        HCCL_ERROR("args is null.");
        return HCCL_E_PARA;
    }
    constexpr int DESC_POS = 0;
    uint64_t desc_value = u64(args[DESC_POS]);
    uint64_t *desc_addr = &desc_value;
    HcclCommParamDesc *desc = reinterpret_cast<HcclCommParamDesc*>(desc_addr);
    MC2AicpuUtils::PrintHcclCommParamDesc(*desc);
    void *tiling = reinterpret_cast<void *>(args[desc->tilingOff]);
    if (tiling == nullptr) {
        HCCL_ERROR("tiling is null.");
        return HCCL_E_PARA;
    }
    const uint32_t ver = MC2TilingGetVer(tiling);
    HCCL_INFO("Start RunAicpuKfcSrvLaunch with tiling version %u.", ver);
    uint32_t ret;
    switch (ver) {
        case TILING_DATA_VER_OLD_FOR_HOST: {
            KFCGroupTilingDataAuto *tilingData = static_cast<KFCGroupTilingDataAuto *>(tiling);
            if (desc->isDyn == GROUP_DYN_FLAG &&
                tilingData->groupTilingMagicNum == GROUP_TILING_MAGIC_NUM) {
                ret = RunAicpuInnerRpcSrvGroupLaunch(args, tilingData, desc);
            } else {
                ret = RunKernelAicpuServerV1(args, desc);
            }
            break;
        }
        case TILING_DATA_VER_OLD_FOR_KERNEL:
            ret = RunKernelAicpuServerV1(args, desc);
            break;
        case TILING_DATA_VER_OLD_FOR_KERNEL_V2:
            ret = RunKernelAicpuServerV2(args, desc, tiling);
            break;
        case TILING_DATA_VER_OLD_FOR_TILING_API:
            ret = RunKernelAicpuServerForTilingApi(args, desc);
            break;
        default:
            HCCL_ERROR("Invalid tiling version %u.", ver);
            ret = HCCL_E_PARA;
    }
    return ret;
}

__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvGroupLaunch(void *args)
{
    KfcState state;
    if (args == nullptr) {
        HCCL_ERROR("args is null.");
        return HCCL_E_PARA;
    }

    KFCTask *task = reinterpret_cast<KFCTask *>(args);
    HCCL_INFO("KFCTask inputA %p, outputC %p, commOut %p, context %p, workSpace %p, tilingData %p",
        task->inputA, task->outputC, task->commOut, task->context, task->workSpace, task->tilingData);
    KFCGroupTilingData *tilingData = reinterpret_cast<KFCGroupTilingData *>(task->tilingData);

    if (tilingData == nullptr || tilingData->groupNum == 0) {
        HCCL_ERROR("tilingData is nullptr or groupNum is 0.");
        return HCCL_E_PARA;
    }

    for (uint32_t i = 0; i < tilingData->groupNum; ++i) {
        KFCTask singleTask;
        singleTask.inputA = task->inputA;
        singleTask.outputC = *reinterpret_cast<u64*>(task->outputC + sizeof(void*) * i);
        singleTask.commOut = task->commOut;
        singleTask.context = task->context;
        singleTask.workSpace = task->workSpace;
        singleTask.tilingData = reinterpret_cast<u64>(&tilingData->msg[i]);
        uint32_t ret = RunAicpuRpcSrvLaunch(&singleTask);
        if (ret != 0) {
            HCCL_ERROR("RunAicpuRpcSrvGroupLaunch runs failed.");
            return HCCL_E_PARA;
        }
    }

    return 0;
}

__attribute__((visibility("default"))) uint32_t RunAicpuKfcResInitV2(void *args)
{
    if (args == nullptr) {
        HCCL_ERROR("args is null.");
        return HCCL_E_PARA;
    }

    KFCResInitTask *ctxArgs = reinterpret_cast<KFCResInitTask *>(args);
    HCCL_INFO("RunAicpuKfcResInitV2 isCustom %u", ctxArgs->isCustom);
    return AicpuHcclProcess::AicpuRpcResInitV2(reinterpret_cast<HcclOpResParam *>(ctxArgs->context),
        ctxArgs->isCustom);
}

__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvLaunchV2(void *args)
{
    if (args == nullptr) {
        HCCL_ERROR("RunAicpuRpcSrvLaunchV2 args is null.");
        return HCCL_E_PARA;
    }

    KFCTaskComm *task = reinterpret_cast<KFCTaskComm *>(args);
    HCCL_INFO("RunAicpuRpcSrvLaunchV2 KFCTask task %p, context %p, tilingData %p", task, task->context, task->tilingData);

    OpTilingData *tilingData = reinterpret_cast<OpTilingData *>(task->tilingData);
    HcclOpResParam *commParam = reinterpret_cast<HcclOpResParam *>(task->context);
    if (tilingData == nullptr || commParam == nullptr) {
        HCCL_ERROR("RunAicpuRpcSrvLaunchV2 tilingData or context args is null.");
        return HCCL_E_PARA;
    }

    std::string group = commParam->hcomId;
    hccl::HcclCommAicpu *hcclCommAicpu = AicpuHcclProcess::AicpuGetCommbyGroup(group);
    if (hcclCommAicpu == nullptr) {
        HCCL_ERROR("RunAicpuRpcSrvLaunchV2 get Hcclcomm error group[%s], tag[%s]", commParam->hcomId, tilingData->tag);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[RunAicpuRpcSrvLaunchV2] isZeroCopy [%d], workflowMode[%d]", tilingData->isZeroCopy, tilingData->workflowMode);
    hcclCommAicpu->SetZeroCopyEnable(tilingData->isZeroCopy);
    dfx::DfxExtendInfo* dfxInfo = hcclCommAicpu->GetDfxExtendInfo();
    if ((dfxInfo->cqeStatus != dfx::CqeStatus::kDefault) ||
        (dfxInfo->pollStatus == dfx::PollStatus::kStopAsException)) {
        HCCL_ERROR("RunAicpuRpcSrvLaunchV2 exist errors before, cqeStatus:%d, pollStatus:%d, group[%s]",
                   dfxInfo->cqeStatus, dfxInfo->pollStatus, commParam->hcomId);
        return HCCL_E_INTERNAL;
    }
    SetWorkflowMode(static_cast<HcclWorkflowMode>(tilingData->workflowMode));
    HCCL_DEBUG("[NsRecovery]check the suspending status in hcclCommAicpu");
    HcclComSuspendingFlag kfcFlag = HcclComSuspendingFlag ::isResume;
    CHK_RET(hcclCommAicpu->GetSuspendingFlag(kfcFlag));
    if (kfcFlag == HcclComSuspendingFlag::isSuspending) {
        HCCL_WARNING("[NsRecovery] the op should not be launched in hcclCommAicpu on the suspending status");
        AicpuHcclProcess::AicpuReleaseCommbyGroup(group);
        return 0;
    }
    hcclCommAicpu->SetNsStopLaunchStatus(false);
    HcclResult res = AicpuHcclProcess::AicpuRunRpcServerV2(hcclCommAicpu, tilingData, commParam);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(group);
    if (res != HCCL_SUCCESS) {
        if (res == HCCL_E_OPRETRY_FAIL) {
            HCCL_RUN_INFO("Retry failed, support step retry");
            return TS_ERROR_RETRY_CONSTRAINT;
        } else if (res != HCCL_E_SUSPENDING) {
            HCCL_ERROR("run AicpuRunRpcServerV2 failed. ret[%d]", res);
            return res;
        } else {
            HCCL_INFO("aicpu is suspended");
            return AICPUSUSPENDING_ERROR;
        }
    }
    HCCL_INFO("end RunAicpuRpcSrvLaunchV2");
    return 0;
}
}  // extern "C"
