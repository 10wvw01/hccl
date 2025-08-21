/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <slog.h>

#include "algorithm/aicpu_hccl_dispatcher.h"
#include "common/sqe_context.h"
#include "common/aicpu_hccl_common.h"
#include "mc2_trace_utils.h"
#include "profiling_manager_device.h"
#include "framework/aicpu_communicator.h"
#include "log.h"
#include "hccl_types.h"
#include "utils/mc2_aicpu_utils.h"
#include "transport_pub.h"

namespace {
#define HCCL_LOG_BY_LEVEL(error, format, ...) do {  \
        if (error) {                                \
            HCCL_ERROR(format, ##__VA_ARGS__);      \
        } else {                                    \
            HCCL_RUN_INFO(format, ##__VA_ARGS__);   \
        }                                           \
    } while (0)

void PrintAllHcclMsgArea(HcclMsgArea *hcclMsgArea, u32 rankSize, bool errorFlag)
{
    if (hcclMsgArea == nullptr) {
        return;
    }
    HCCL_LOG_BY_LEVEL(errorFlag, "********* msgArea %p start print **********", hcclMsgArea);
    for (uint32_t i = 0; i < AC_MSG_CNT; ++i) {
        if (hcclMsgArea->sendMsgList[i].commType == HCCL_CMD_INVALID ||
            hcclMsgArea->sendMsgList[i].commType >= HCCL_CMD_MAX) {
            continue;
        }
        HCCL_LOG_BY_LEVEL(errorFlag, "SendMsg[%d]: %s", i, hcclMsgArea->sendMsgList[i].MsgToStr().c_str());
    }
    // recvMsgList暂不支持，不处理了
    for (uint32_t i = 0; i < AC_MSG_CNT; ++i) {
        if (hcclMsgArea->sendMsgList[i].commType != HCCL_CMD_ALLTOALLV) {
            continue;
        }
        HCCL_LOG_BY_LEVEL(errorFlag, "HcclMsgExt[%d]: %s",
            i, hcclMsgArea->paramExtMsgList[i].MsgToStr(rankSize).c_str());
    }

    std::stringstream ssCommitCnt;
    for (uint32_t i = 0; i < AC_MSG_CNT; ++i) {
        ssCommitCnt << hcclMsgArea->commitTurnCnt[i].cnt << ",";
    }
    HCCL_LOG_BY_LEVEL(errorFlag, "commitTurnCnt: %s", ssCommitCnt.str().c_str());
    std::stringstream ssFinishCnt;
    for (uint32_t i = 0; i < AC_MSG_CNT; ++i) {
        ssFinishCnt << hcclMsgArea->finishedTurnCnt[i].cnt << ",";
    }
    HCCL_LOG_BY_LEVEL(errorFlag, "finishedTurnCnt: %s", ssFinishCnt.str().c_str());
    HCCL_LOG_BY_LEVEL(errorFlag, "********* msgArea %p end print **********", hcclMsgArea);
}

void PrintAllHcclMsgArea(HcclMsgAreaForMultiQue *hcclMsgArea, u32 rankSize, bool errorFlag)
{
    if (hcclMsgArea == nullptr) {
        return;
    }
    HCCL_LOG_BY_LEVEL(errorFlag, "********* msgArea %p start print **********", hcclMsgArea);
    for (u32 i = 0U; i < MAX_QUE_NUM; ++i) {
        for (u32 j = 0; j < AC_MSG_CNT; ++j) {
            if (hcclMsgArea->sendMsgList[i][j].commType == HCCL_CMD_INVALID ||
                hcclMsgArea->sendMsgList[i][j].commType >= HCCL_CMD_MAX) {
                continue;
            }
            HCCL_LOG_BY_LEVEL(errorFlag, "SendMsg[%u/%u]: %s", i, j, hcclMsgArea->sendMsgList[i][j].MsgToStr().c_str());
        }
    }
    HCCL_LOG_BY_LEVEL(errorFlag, "********* msgArea %p end print **********", hcclMsgArea);
}
}

struct hcclCommAicpuRecovery {
    std::mutex commAicpuRecoverMutex;
    std::unordered_map<std::string, hccl::HcclCommAicpu *> recoveryCommMap;  // 【NsTest】这里Ns快恢时候的通信域
};
hcclCommAicpuRecovery g_commAicpuRecovery;

int32_t MC2AicpuUtils::GetCpuId()
{
    static thread_local int32_t curCpu = -1;
    if (curCpu < 0) {
        curCpu = sched_getcpu();
    }
    return curCpu;
}

int32_t MC2AicpuUtils::GetCurClusterId()
{
    return !!(GetCpuId() & (AICPU_CNT / CLUSTER_CNT));
}

void MC2AicpuUtils::PrintHcclCombinOpParam(const HccCommResParamTask &commParam)
{
	if (!HcclCheckLogLevel(HCCL_LOG_INFO)) {
        return;
    }
    HCCL_INFO("HccCommResParamTask.workSpace %p", commParam.mc2WorkSpace.workSpace);
    HCCL_INFO("HccCommResParamTask.workSpaceSize %lu", commParam.mc2WorkSpace.workSpaceSize);
    HCCL_INFO("HccCommResParamTask.rankId %u", commParam.rankId);
    HCCL_INFO("HccCommResParamTask.rankNum %u", commParam.rankNum);
    HCCL_INFO("HccCommResParamTask.winSize %lu", commParam.winSize);
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        HCCL_INFO("HccCommResParamTask.windowsIn[%u] %p, windowsOut[%u] %p",
            i, commParam.windowsIn[i], i, commParam.windowsOut[i]);
    }
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        const HcclStreamInfo &sinfo = commParam.streamInfo[i];
        HCCL_INFO("HccCommResParamTask.streamInfo[%u] streamId %d, sqId %u, cqId %lu logicCqid %u",
            i,
            sinfo.streamIds,
            sinfo.sqIds,
            sinfo.cqIds,
            sinfo.logicCqids);
    }
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM * 2; i++) { // 2 is number of noIpcNotify
        const HcclSignalInfo &sinfo = commParam.signalInfo.noIpcNotifys[i];
        HCCL_INFO("HccCommResParamTask.noIpcNotifys[%u] resId %lu, addr %p, devId %u, tsId %u, rankId %u", i,
            sinfo.resId, sinfo.addr, sinfo.devId, sinfo.tsId, sinfo.rankId);
    }
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM * 4; i++) { // 4 is number of ipcNotifys
        const HcclSignalInfo &sinfo = commParam.signalInfo.ipcNotifys[i];
        HCCL_INFO("HccCommResParamTask.ipcNotifys[%u] resId %lu, addr %p, devId %u, tsId %u, rankId %u", i,
            sinfo.resId, sinfo.addr, sinfo.devId, sinfo.tsId, sinfo.rankId);
    }
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        const HcclSignalInfo &sinfo = commParam.signalInfo.noIpcEvents[i];
        HCCL_INFO("HccCommResParamTask.noIpcEvents[%u] resId %lu, addr %p, devId %u, tsId %u, rankId %u", i,
            sinfo.resId, sinfo.addr, sinfo.devId, sinfo.tsId, sinfo.rankId);
    }

    for (uint32_t i = 0; i < AICPU_OP_NOTIFY_NUM; i++) {
        const HcclSignalInfo &sinfo = commParam.signalInfo.aicpuOpNotify[i];
        HCCL_INFO("HccCommResParamTask.aicpuOpNotify[%u] resId %lu, addr %p, devId %u, tsId %u, rankId %u", i,
            sinfo.resId, sinfo.addr, sinfo.devId, sinfo.tsId, sinfo.rankId);
    }
    const auto &sigInfo = commParam.signalInfo.aicpuNotify;
    HCCL_INFO("HccCommResParamTask.aicpuNotify resId %lu, addr %p, devId %u, tsId %u, rankId %u", sigInfo.resId,
        sigInfo.addr, sigInfo.devId, sigInfo.tsId, sigInfo.rankId);
    HCCL_INFO("HccCommResParamTask.determinism %u", commParam.config.deterministic);
    HCCL_INFO("HccCommResParamTask.overflowAddr %llx", commParam.overFlowAddr);
    HCCL_INFO("HccCommResParamTask.retryParams: retryEnable %u", commParam.config.retryEnable);
}

void MC2AicpuUtils::PrintHcclCommParamDesc(const HcclCommParamDesc &desc)
{
    auto ctx = AicpuGetComContext();
    if (ctx == nullptr || ctx->logLevel > HCCL_LOG_INFO) {
        return;
    }
    HCCL_INFO("HcclCommParamDesc.version %lu", desc.version);
    HCCL_INFO("HcclCommParamDesc.groupNum %lu", desc.groupNum);
    HCCL_INFO("HcclCommParamDesc.hasFfts %lu", desc.hasFfts);
    HCCL_INFO("HcclCommParamDesc.tilingOff %lu", desc.tilingOff);
    HCCL_INFO("HcclCommParamDesc.isDyn %lu", desc.isDyn);
}

void MC2AicpuUtils::PrintKFCTask(const KFCTask &task)
{
    auto ctx = AicpuGetComContext();
    if (ctx == nullptr || ctx->logLevel > HCCL_LOG_INFO) {
        return;
    }
    HCCL_INFO("KFCTask.inputA Addr %lu", task.inputA);
    HCCL_INFO("KFCTask.outputC Addr %lu", task.outputC);
    HCCL_INFO("KFCTask.commOut Addr %lu", task.commOut);
    HCCL_INFO("KFCTask.context Addr %lu", task.context);
    HCCL_INFO("KFCTask.workSpace Addr %lu", task.workSpace);
}

void MC2AicpuUtils::PrintTilingData(const HcclKFCTilingData &tilingData)
{
    auto ctx = AicpuGetComContext();
    if (ctx == nullptr || ctx->logLevel > HCCL_LOG_INFO) {
        return;
    }
    HCCL_INFO("HcclKFCTilingData.sendOff %lu.", tilingData.sendOff);
    HCCL_INFO("HcclKFCTilingData.recvOff %lu.", tilingData.recvOff);
    HCCL_INFO("HcclKFCTilingData.tailSendOff %lu.", tilingData.tailSendOff);
    HCCL_INFO("HcclKFCTilingData.tailRecvOff %lu.", tilingData.tailRecvOff);
    HCCL_INFO("HcclKFCTilingData.sendCnt %lu.", tilingData.sendCnt);
    HCCL_INFO("HcclKFCTilingData.recvCnt %lu.", tilingData.recvCnt);
    HCCL_INFO("HcclKFCTilingData.tailSendCnt %lu.", tilingData.tailSendCnt);
    HCCL_INFO("HcclKFCTilingData.tailRecvCnt %lu.", tilingData.tailRecvCnt);
    HCCL_INFO("HcclKFCTilingData.totalCnt %lu.", tilingData.totalCnt);
    HCCL_INFO("HcclKFCTilingData.turnNum %u.", tilingData.turnNum);
    HCCL_INFO("HcclKFCTilingData.tailNum %u.", tilingData.tailNum);
    HCCL_INFO("HcclKFCTilingData.stride %u.", tilingData.stride);
    HCCL_INFO("HcclKFCTilingData.workspaceOff %u.", tilingData.workspaceOff);
    HCCL_INFO("HcclKFCTilingData.notifyOff %u.", tilingData.notifyOff);
    HCCL_INFO("HcclKFCTilingData.notifyBeginCnt %u.", tilingData.notifyBeginCnt);
    HCCL_INFO("HcclKFCTilingData.notifyEndCnt %u.", tilingData.notifyEndCnt);
    HCCL_INFO("HcclKFCTilingData.useBufferType %u.", tilingData.useBufferType);
    HCCL_INFO("HcclKFCTilingData.funID %u.", tilingData.funID);
    HCCL_INFO("HcclKFCTilingData.dataType %u.", tilingData.dataType);
    HCCL_INFO("HcclKFCTilingData.groupNum %u.", tilingData.groupNum);
    HCCL_INFO("HcclKFCTilingData.reuseMode %u.", tilingData.reuseMode);
    HCCL_INFO("HcclKFCTilingData.commType %u.", tilingData.commType);
    HCCL_INFO("HcclKFCTilingData.reduceOp %u.", tilingData.reduceOp);
    HCCL_INFO("HcclKFCTilingData.commOrder %u.", tilingData.commOrder);
    HCCL_INFO("HcclKFCTilingData.waitPolicy %u.", tilingData.waitPolicy);
    HCCL_INFO("HcclKFCTilingData.rspPolicy %u.", tilingData.rspPolicy);
    HCCL_INFO("HcclKFCTilingData.exitPolicy %u.", tilingData.exitPolicy);
    HCCL_INFO("HcclKFCTilingData.commAlg %u.", tilingData.commAlg);
    HCCL_INFO("HcclKFCTilingData.taskType %u.", tilingData.taskType);
    HCCL_INFO("HcclKFCTilingData.debugMode %u.", tilingData.debugMode);
    HCCL_INFO("HcclKFCTilingData.stepSize %u.", tilingData.stepSize);
    HCCL_INFO("HcclKFCTilingData.sendArgIndex %u.", tilingData.sendArgIndex);
    HCCL_INFO("HcclKFCTilingData.recvArgIndex %u.", tilingData.recvArgIndex);
    HCCL_INFO("HcclKFCTilingData.commOutArgIndex %u.", tilingData.commOutArgIndex);
    HCCL_INFO("HcclKFCTilingData.hasCommOut %u.", tilingData.hasCommOut);
}

void MC2AicpuUtils::PrintTilingData(const Mc2InitTilingInner &tilingData)
{
    auto ctx = AicpuGetComContext();
    if (ctx == nullptr || ctx->logLevel > HCCL_LOG_INFO) {
        return;
    }
    HCCL_INFO("HcclKFCTilingData.version %lu.", static_cast<u64>(tilingData.version));
    HCCL_INFO("HcclKFCTilingData.mc2HcommCnt %lu.", static_cast<u64>(tilingData.mc2HcommCnt));
    HCCL_INFO("HcclKFCTilingData.debugMode %lu.", static_cast<u64>(tilingData.debugMode));
    HCCL_INFO("HcclKFCTilingData.preparePosition %lu.", static_cast<u64>(tilingData.preparePosition));
    HCCL_INFO("HcclKFCTilingData.queueNum %lu.", static_cast<u64>(tilingData.queueNum));
    HCCL_INFO("HcclKFCTilingData.commBlockNum %lu.", static_cast<u64>(tilingData.commBlockNum));
}

void MC2AicpuUtils::PrintTilingDataError(const Mc2InitTilingInner &tilingData)
{
    HCCL_ERROR("HcclKFCTilingData.version %lu.", static_cast<u64>(tilingData.version));
    HCCL_ERROR("HcclKFCTilingData.mc2HcommCnt %lu.", static_cast<u64>(tilingData.mc2HcommCnt));
    HCCL_ERROR("HcclKFCTilingData.debugMode %lu.", static_cast<u64>(tilingData.debugMode));
    HCCL_ERROR("HcclKFCTilingData.preparePosition %lu.", static_cast<u64>(tilingData.preparePosition));
    HCCL_ERROR("HcclKFCTilingData.queueNum %lu.", static_cast<u64>(tilingData.queueNum));
    HCCL_ERROR("HcclKFCTilingData.commBlockNum %lu.", static_cast<u64>(tilingData.commBlockNum));
}

void MC2AicpuUtils::PrintTilingDataError(const HcclKFCTilingData &tilingData)
{
    HCCL_ERROR("HcclKFCTilingData.sendOff %lu", tilingData.sendOff);
    HCCL_ERROR("HcclKFCTilingData.recvOff %lu", tilingData.recvOff);
    HCCL_ERROR("HcclKFCTilingData.tailSendOff %lu", tilingData.tailSendOff);
    HCCL_ERROR("HcclKFCTilingData.tailRecvOff %lu", tilingData.tailRecvOff);
    HCCL_ERROR("HcclKFCTilingData.sendCnt %lu", tilingData.sendCnt);
    HCCL_ERROR("HcclKFCTilingData.recvCnt %lu", tilingData.recvCnt);
    HCCL_ERROR("HcclKFCTilingData.tailSendCnt %lu", tilingData.tailSendCnt);
    HCCL_ERROR("HcclKFCTilingData.tailRecvCnt %lu", tilingData.tailRecvCnt);
    HCCL_ERROR("HcclKFCTilingData.totalCnt %lu", tilingData.totalCnt);
    HCCL_ERROR("HcclKFCTilingData.turnNum %u", tilingData.turnNum);
    HCCL_ERROR("HcclKFCTilingData.tailNum %u", tilingData.tailNum);
    HCCL_ERROR("HcclKFCTilingData.stride %u", tilingData.stride);
    HCCL_ERROR("HcclKFCTilingData.workspaceOff %u", tilingData.workspaceOff);
    HCCL_ERROR("HcclKFCTilingData.notifyOff %u", tilingData.notifyOff);
    HCCL_ERROR("HcclKFCTilingData.notifyBeginCnt %u", tilingData.notifyBeginCnt);
    HCCL_ERROR("HcclKFCTilingData.notifyEndCnt %u", tilingData.notifyEndCnt);
    HCCL_ERROR("HcclKFCTilingData.useBufferType %u", tilingData.useBufferType);
    HCCL_ERROR("HcclKFCTilingData.funID %u", tilingData.funID);
    HCCL_ERROR("HcclKFCTilingData.dataType %u", tilingData.dataType);
    HCCL_ERROR("HcclKFCTilingData.groupNum %u", tilingData.groupNum);
    HCCL_ERROR("HcclKFCTilingData.reuseMode %u", tilingData.reuseMode);
    HCCL_ERROR("HcclKFCTilingData.commType %u", tilingData.commType);
    HCCL_ERROR("HcclKFCTilingData.reduceOp %u", tilingData.reduceOp);
    HCCL_ERROR("HcclKFCTilingData.commOrder %u", tilingData.commOrder);
    HCCL_ERROR("HcclKFCTilingData.waitPolicy %u", tilingData.waitPolicy);
    HCCL_ERROR("HcclKFCTilingData.rspPolicy %u", tilingData.rspPolicy);
    HCCL_ERROR("HcclKFCTilingData.exitPolicy %u", tilingData.exitPolicy);
    HCCL_ERROR("HcclKFCTilingData.commAlg %u", tilingData.commAlg);
    HCCL_ERROR("HcclKFCTilingData.taskType %u", tilingData.taskType);
    HCCL_ERROR("HcclKFCTilingData.debugMode %u", tilingData.debugMode);
    HCCL_ERROR("HcclKFCTilingData.stepSize %u", tilingData.stepSize);
    HCCL_ERROR("HcclKFCTilingData.sendArgIndex %u", tilingData.sendArgIndex);
    HCCL_ERROR("HcclKFCTilingData.recvArgIndex %u", tilingData.recvArgIndex);
    HCCL_ERROR("HcclKFCTilingData.commOutArgIndex %u", tilingData.commOutArgIndex);
    HCCL_ERROR("HcclKFCTilingData.hasCommOut %u", tilingData.hasCommOut);
}

void MC2AicpuUtils::PrintMC2AicpuContext(const AicpuComContext &ctx)
{
    if (ctx.logLevel > HCCL_LOG_INFO) {
        return;
    }
    HCCL_INFO("AicpuComContext.devId %u", ctx.devId);
    HCCL_INFO("AicpuComContext.ssid %u", ctx.ssid);
    HCCL_INFO("AicpuComContext.rankId %u", ctx.rankId);
    HCCL_INFO("AicpuComContext.rankNum %u", ctx.rankNum);

    HCCL_INFO("AicpuComContext.windowSize %lu", ctx.windowSize);
    HCCL_INFO("AicpuComContext.workSpaceAddr %p", ctx.workSpaceAddr);
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        HCCL_INFO("AicpuComContext.eventIds[%u] %lu", i, ctx.eventIds[i]);

        const auto &si = ctx.streamInfo[i];
        HCCL_INFO("AicpuComContext.streamInfo[%u] streamId %d sqId %d depth %u addr %p", i, si.actualStreamId,
            si.sqId, si.sqDepth, si.sqBaseAddr);

        const auto &noIpcPre = ctx.noIpcPreNotify[i];
        HCCL_INFO("AicpuComContext.noIpcPreNotify[%u] addr %p notifyId %d", i, noIpcPre.address,
            noIpcPre.actualNotifyId);

        const auto &noIpcPost = ctx.noIpcPostNotify[i];
        HCCL_INFO("AicpuComContext.noIpcPostNotify[%u] addr %p notifyId %d", i, noIpcPost.address,
            noIpcPost.actualNotifyId);

        const auto &ipcPreRec = ctx.ipcPreRecordNotify[i];
        HCCL_INFO("AicpuComContext.ipcPreRecordNotify[%u] addr %p notifyId %d", i, ipcPreRec.address,
            ipcPreRec.actualNotifyId);

        const auto &ipcPreWait = ctx.ipcPreWaitNotify[i];
        HCCL_INFO("AicpuComContext.ipcPreWaitNotify[%u] addr %p notifyId %d", i, ipcPreWait.address,
            ipcPreWait.actualNotifyId);

        const auto &ipcPostRec = ctx.ipcPostRecordNotify[i];
        HCCL_INFO("AicpuComContext.ipcPostRecordNotify[%u] addr %p notifyId %d", i, ipcPostRec.address,
            ipcPostRec.actualNotifyId);

        const auto &ipcPostWait = ctx.ipcPostWaitNotify[i];
        HCCL_INFO("AicpuComContext.ipcPostWaitNotify[%u] addr %p notifyId %d", i, ipcPostWait.address,
            ipcPostWait.actualNotifyId);
    }
    HCCL_INFO("AicpuComContext.determinism %u", ctx.determinism);
}

void MC2AicpuUtils::PrintMC2HcclOpResParam(const HcclOpResParam *resParam)
{
    HCCL_INFO("HcclOpResParam.rankId %u", resParam->localUsrRankId);
    HCCL_INFO("HcclOpResParam.rankNum %u", resParam->rankSize);

    HCCL_INFO("HcclOpResParam.windowSize %lu", resParam->winSize);
    HCCL_INFO("HcclOpResParam.workSpaceAddr %p", resParam->mc2WorkSpace.workSpace);
    HCCL_INFO("HcclOpResParam.workSpaceSize %lu", resParam->mc2WorkSpace.workSpaceSize);
    for (uint32_t i = 0; i < resParam->localRes.streamNum; i++) {
        const auto streamInfo = resParam->localRes.streamParam[i].streamInfo;
        HCCL_INFO("HcclOpResParam.streamInfo[%u] streamId %d sqId %u cqId %u logicCqid %u", i, streamInfo.streamIds,
                  streamInfo.sqIds, streamInfo.cqIds, streamInfo.logicCqids);
    }
    const auto mainStreamInfo = resParam->localRes.mainStreamParam.streamInfo;
        HCCL_INFO("HcclOpResParam.mainStreamInfo streamId %d sqId %u cqId %u logicCqid %u", mainStreamInfo.streamIds,
                   mainStreamInfo.sqIds, mainStreamInfo.cqIds, mainStreamInfo.logicCqids);
    for (uint32_t i = 0; i < resParam->localRes.signalNum; i++) {
        const auto &signals = resParam->localRes.localSignals[i];
        HCCL_INFO("HcclOpResParam.localSignals[%u] resId %llx addr %llx devId %u tsId %u rankId %u", i, signals.resId,
            signals.addr, signals.devId, signals.tsId, signals.rankId);
    }

    for (uint32_t i = 0; i < AICPU_OP_NOTIFY_MAX_NUM; i++) {
        const auto &aicpuOpNotify = resParam->localRes.aicpuOpNotify[i];
        HCCL_INFO("HcclOpResParam.aicpuOpNotify[%u] resId %llx addr %llx devId %u tsId %u rankId %u", i, aicpuOpNotify.resId,
            aicpuOpNotify.addr, aicpuOpNotify.devId, aicpuOpNotify.tsId, aicpuOpNotify.rankId);
    }
    HCCL_INFO("HcclOpResParam.determinism %u", resParam->config.deterministic);
}

void MC2AicpuUtils::PrintMC2AicpuContextError(const AicpuComContext &ctx)
{
    HCCL_ERROR("AicpuComContext.devId %u", ctx.devId);
    HCCL_ERROR("AicpuComContext.ssid %u", ctx.ssid);
    HCCL_ERROR("AicpuComContext.rankId %u", ctx.rankId);
    HCCL_ERROR("AicpuComContext.rankNum %u", ctx.rankNum);

    HCCL_ERROR("AicpuComContext.windowSize %lu", ctx.windowSize);
    HCCL_ERROR("AicpuComContext.workSpaceAddr %p", ctx.workSpaceAddr);
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        HCCL_ERROR("AicpuComContext.eventIds[%u] %lu", i, ctx.eventIds[i]);

        const auto &si = ctx.streamInfo[i];
        HCCL_ERROR("AicpuComContext.streamInfo[%u] streamId %d sqId %d depth %u addr %p", i, si.actualStreamId,
            si.sqId, si.sqDepth, si.sqBaseAddr);

        const auto &noIpcPre = ctx.noIpcPreNotify[i];
        HCCL_ERROR("AicpuComContext.noIpcPreNotify[%u] addr %p notifyId %d", i, noIpcPre.address,
            noIpcPre.actualNotifyId);

        const auto &noIpcPost = ctx.noIpcPostNotify[i];
        HCCL_ERROR("AicpuComContext.noIpcPostNotify[%u] addr %p notifyId %d", i, noIpcPost.address,
            noIpcPost.actualNotifyId);

        const auto &ipcPreRec = ctx.ipcPreRecordNotify[i];
        HCCL_ERROR("AicpuComContext.ipcPreRecordNotify[%u] addr %p notifyId %d", i, ipcPreRec.address,
            ipcPreRec.actualNotifyId);

        const auto &ipcPreWait = ctx.ipcPreWaitNotify[i];
        HCCL_ERROR("AicpuComContext.ipcPreWaitNotify[%u] addr %p notifyId %d", i, ipcPreWait.address,
            ipcPreWait.actualNotifyId);

        const auto &ipcPostRec = ctx.ipcPostRecordNotify[i];
        HCCL_ERROR("AicpuComContext.ipcPostRecordNotify[%u] addr %p notifyId %d", i, ipcPostRec.address,
            ipcPostRec.actualNotifyId);

        const auto &ipcPostWait = ctx.ipcPostWaitNotify[i];
        HCCL_ERROR("AicpuComContext.ipcPostWaitNotify[%u] addr %p notifyId %d", i, ipcPostWait.address,
            ipcPostWait.actualNotifyId);
    }
    HCCL_ERROR("AicpuComContext.determinism %u", ctx.determinism);
}

HcclResult MC2AicpuUtils::TraceProfSubmit()
{
    auto ctx = AicpuGetComContext();
    CHK_PTR_NULL(ctx);
    if (dfx::ProfilingManager::GetProfL1State()) {
        CHK_PRT_RET(dfx::ProfilingManager::ReportTaskInfo() != HCCL_SUCCESS, HCCL_ERROR("prof task info failed"),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult MC2AicpuUtils::Getkey(const AicpuComContext &ctx, u32 remoteRankId, const void *userAddr,
    u64 length, u32 &outKey, int32_t keyType)
{
    HCCL_INFO("[MC2AicpuUtils][Getkey] addr[%p] len[%llu]", userAddr, length);
    u64 inAddr = reinterpret_cast<u64>(userAddr);
    MemDetails inputMem = (keyType == LOCAL) ? ctx.ibversData[remoteRankId].localInputMem : ctx.ibversData[remoteRankId].remoteInputMem;
    MemDetails outputMem = (keyType == LOCAL) ? ctx.ibversData[remoteRankId].localOutputMem : ctx.ibversData[remoteRankId].remoteOutputMem;

    u64 inputStartAddr  = inputMem.addr;
    u64 inputCCLSize  = inputMem.size;
    u32 inputKey  = inputMem.key;

    u64 outputStartAddr = outputMem.addr;
    u64 outputCCLSize = outputMem.size;
    u32 outputKey = outputMem.key;
    if (inAddr >= inputStartAddr && inAddr < inputStartAddr + inputCCLSize) {
        outKey = inputKey;
    } else if (inAddr >= outputStartAddr && inAddr <= outputStartAddr + outputCCLSize) {
        outKey = outputKey;
    } else {
        HCCL_ERROR("[MC2AicpuUtils][Getkey]src_ptr=%p is out of range, inputmem src[%p], size[%llu];"
                " outputmem src[%p] size[%llu]",
                userAddr, inputStartAddr, inputCCLSize, outputStartAddr, outputCCLSize);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[MC2AicpuUtils][Getkey] addr[%p] length[%llu] outKey[%u], keyType:[%s]",
        userAddr, length, outKey, (keyType == LOCAL) ? "local" : "remote");

    return HCCL_SUCCESS;
}

std::mutex g_mtxForDoorbell;
HcclResult MC2AicpuUtils::PostSend(const AicpuComContext &ctx, u32 remoteRankId, struct std::vector<hccl::Transport::Buffer> &remoteBuf,
    struct std::vector<hccl::Transport::Buffer> &localBuf, bool isWrite)
{
    if (UNLIKELY(remoteRankId >= ctx.rankNum)) {
        HCCL_ERROR("[AicpuIbverbs][PostSend] remoteRankId %u is out of range, ranknum %u",remoteRankId, ctx.rankNum);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(remoteBuf.size() != localBuf.size(),
        HCCL_ERROR("[AicpuIbverbs][PostSend] remoteBuf list size %u is not equal localBuffer list size %u ",
        remoteBuf.size(), localBuf.size()), HCCL_E_PARA);

    uint32_t len = remoteBuf.size();
    const uint32_t MAX_MEM_NUM = 8;
    CHK_PRT_RET(len > MAX_MEM_NUM,
        HCCL_ERROR("[AicpuIbverbs][PostSend] buffer size is:%u over MAX_MEM_NUM: %u", len, MAX_MEM_NUM), HCCL_E_PARA);

    MemDetails localMems[MAX_MEM_NUM];
    MemDetails remoteMems[MAX_MEM_NUM];
    u32 lkey = 0;
    u32 rkey = 0;
    for (uint32_t index = 0; index < len; index++) {
        u64 remBuffSize = remoteBuf[index].size;
        u64 locBuffSize = localBuf[index].size;
        CHK_PRT_RET(remBuffSize != locBuffSize,
            HCCL_ERROR("[AicpuIbverbs][PostSend] remoteBuf size %u is not equal localBuffer size %u ",
            remBuffSize, locBuffSize), HCCL_E_PARA);
        // 获取WR的lkey和rkey
        CHK_RET(Getkey(ctx, remoteRankId, localBuf[index].addr, locBuffSize, lkey, LOCAL));
        CHK_RET(Getkey(ctx, remoteRankId, remoteBuf[index].addr, remBuffSize, rkey, REMOTE));
        // 设置MemDetails
        localMems[index].addr = reinterpret_cast<u64>(localBuf[index].addr);
        localMems[index].size = locBuffSize;
        localMems[index].key = lkey;

        remoteMems[index].addr = reinterpret_cast<u64>(remoteBuf[index].addr);
        remoteMems[index].size = remBuffSize;
        remoteMems[index].key = rkey;
    }

    u64 db_info = 0;
    u32 memNum = (ctx.ibversData[remoteRankId].qpMode != QPMode::NORMAL) ? 1 : len;
    CHK_RET(LIKELY(isWrite) ?
        hccl::Transport::HcclBatchWrite(ctx.ibversData[remoteRankId], &localMems[0], &remoteMems[0], memNum, db_info) :
        hccl::Transport::HcclBatchRead(ctx.ibversData[remoteRankId], &localMems[0], &remoteMems[0], memNum, db_info));

    if (UNLIKELY(ctx.ibversData[remoteRankId].qpMode != QPMode::NORMAL)) {
        for (u32 i = 1; i < len; i++) {
            CHK_RET(LIKELY(isWrite) ?
                hccl::Transport::HcclBatchWrite(ctx.ibversData[remoteRankId], &localMems[i],
                    &remoteMems[i], 1, db_info) :
                hccl::Transport::HcclBatchRead(ctx.ibversData[remoteRankId], &localMems[i],
                    &remoteMems[i], 1, db_info));
        }
        u64 roceBaseAddr = 0x2000000000ULL;
        u64 roceVfDbCfg0Reg = 0x230ULL;
        u64 chipAddrOffset = 0x80000000000ULL;
        u64 dieAddrOffset = 0x10000000000ULL;
        u64 dbDieIdMask = 0x00ff0000;
        u64 dbDieIdShift = 16; // 16 is dbDieIdShift
        u64 dbAddr = roceBaseAddr + roceVfDbCfg0Reg + chipAddrOffset * ctx.chipId +
            dieAddrOffset * ((ctx.ibversData[remoteRankId].qpInfo.dbIndex & dbDieIdMask) >> dbDieIdShift);
        HCCL_DEBUG("chipId : %llu", ctx.chipId);
        std::lock_guard<std::mutex> lock(g_mtxForDoorbell);
        CHK_RET(DispatcherAicpu::RdmaSend(0, db_info, dbAddr, remoteRankId));
        CHK_RET(DispatcherAicpu::LaunchTask(0));
    }
    return HCCL_SUCCESS;
}

HcclResult MC2AicpuUtils::WaitTaskFinish(AicpuComContext *ctx, bool isWaitTask)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_RET(TraceProfSubmit());
    if (isWaitTask || ctx->retryEnable) {
        ret = TaskOrchestrator::WaitMainStreamFinish(ctx);
        CHK_PRT_RET((ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING), HCCL_ERROR("wait main stream finish failed"), ret);
    }
    return ret;
}

void MC2AicpuUtils::PrintApiBuffer(const void * const buffer, uint64_t totalSize, const std::string &desc)
{
    if (buffer == nullptr) {
        HCCL_DEBUG("buffer is nullptr");
        return;
    }
    HCCL_RUN_INFO("%s, buffer：%p totalSize：%lu", desc.c_str(), buffer, totalSize);
    constexpr uint32_t maxPrintNum = 192U;
    constexpr uint32_t partNum = 64U;
    constexpr uint32_t everyNum = 8U;
    uint32_t cnt = totalSize / sizeof(uint32_t);
    const uint32_t * const cmd = reinterpret_cast<const uint32_t *>(buffer);
    if (cnt <= maxPrintNum) {
        if (cnt < everyNum) {
            for (size_t i = 0UL; i < cnt; i++) {
                HCCL_RUN_INFO("%zu: %08x", i, cmd[i]);
            }
        } else {
            // cnt向下取整到最近的8的倍数
            cnt = cnt / everyNum * everyNum;
            for (size_t i = 0UL; i < cnt; i += everyNum) {
                HCCL_RUN_INFO("%zu: %08x %08x %08x %08x %08x %08x %08x %08x", i,
                    cmd[i], cmd[i + 1U], cmd[i + 2U], cmd[i + 3U], cmd[i + 4U], cmd[i + 5U], cmd[i + 6U], cmd[i + 7U]);
            }
        }
    } else {
        // 打印前64个uint32_t数据
        for (size_t i = 0UL; i < partNum; i += everyNum) {
            HCCL_RUN_INFO("%zu: %08x %08x %08x %08x %08x %08x %08x %08x", i,
                cmd[i], cmd[i + 1U], cmd[i + 2U], cmd[i + 3U], cmd[i + 4U], cmd[i + 5U], cmd[i + 6U], cmd[i + 7U]);
        }
        // 打印中间64个uint32_t数据
        size_t start = (cnt / 2) - (partNum / 2);
        for (size_t i = start; i < start + partNum; i += everyNum) {
            HCCL_RUN_INFO("%zu: %08x %08x %08x %08x %08x %08x %08x %08x", i,
                cmd[i], cmd[i + 1U], cmd[i + 2U], cmd[i + 3U], cmd[i + 4U], cmd[i + 5U], cmd[i + 6U], cmd[i + 7U]);
        }
        // 打印后64个uint32_t数据
        for (size_t i = cnt - partNum; i < cnt; i += everyNum) {
            HCCL_RUN_INFO("%zu: %08x %08x %08x %08x %08x %08x %08x %08x", i,
                cmd[i], cmd[i + 1U], cmd[i + 2U], cmd[i + 3U], cmd[i + 4U], cmd[i + 5U], cmd[i + 6U], cmd[i + 7U]);
        }
    }
}

void MC2AicpuUtils::PrintBuffer(const void * const buffer, uint32_t totalSize, const std::string &desc)
{
    if (buffer == nullptr) {
        HCCL_DEBUG("buffer is nullptr");
        return;
    }
#ifndef RUN_TEST
    constexpr uint32_t maxPrintNum = 128U;

    uint32_t cnt = totalSize / sizeof(uint32_t);
    cnt = std::min(cnt, maxPrintNum);
    const uint32_t * const cmd = reinterpret_cast<const uint32_t *>(buffer);

    for (size_t i = 0UL; i < cnt; i += 8U) {
        HCCL_DEBUG("%p %zu %s: %08x %08x %08x %08x %08x %08x %08x %08x", buffer, i, desc.c_str(), cmd[i],
            cmd[i + 1U], cmd[i + 2U], cmd[i + 3U], cmd[i + 4U], cmd[i + 5U], cmd[i + 6U], cmd[i + 7U]);
    }

    if (cnt > maxPrintNum) {
        for (size_t i = cnt - maxPrintNum; i < cnt - 8U; i += 8U) { // 8 is byte size
            HCCL_DEBUG("%p %zu %s: %08x %08x %08x %08x %08x %08x %08x %08x", buffer, i, desc.c_str(), cmd[i],
                cmd[i + 1U], cmd[i + 2U], cmd[i + 3U], cmd[i + 4U], cmd[i + 5U], cmd[i + 6U], cmd[i + 7U]);
        }
    }
#endif
}

uint32_t MC2AicpuUtils::GenXor(HcclMsg *msg) {
    if (msg == nullptr) {
        return UINT32_MAX;
    }
    DataBlock* block = reinterpret_cast<DataBlock*>(msg);
    uint32_t xorVal = 0;
    for (uint32_t i = 0; i < MC2_API_XORCHECK_LEN; i++) {
        xorVal ^= block->data[i];
    }
    return xorVal;
}

void MC2AicpuUtils::PrintBuffer(AicpuComContext *ctx, const AivAicpuOpParam &msgAddr)
{
    if (ctx == nullptr || ctx->logLevel > HCCL_LOG_DEBUG) {
        return;
    }
#ifdef __aarch64__
    __asm__ __volatile__("dsb ld" : : : "memory");
#endif
#ifdef __amd64__
    __asm__ __volatile__("" : : : "memory");
#endif

    PrintBuffer(reinterpret_cast<void *>(msgAddr.sendBuffer), msgAddr.count * ctx->unitSize, "after copy, send buffer");
    PrintBuffer(reinterpret_cast<void *>(ctx->rankInfo[ctx->rankId].window), msgAddr.count * ctx->unitSize,
        "after copy, window");
    PrintBuffer(reinterpret_cast<void *>(msgAddr.recvBuffer), msgAddr.count * ctx->unitSize, "after copy, recv buffer");
}

int MC2AicpuUtils::GetSendCnt(AicpuComContext *ctx)
{
    if (ctx == nullptr) {
        return 0;
    }
    return static_cast<int>((reinterpret_cast<AivAicpuOpParam *>(ctx->workSpaceAddr + ctx->notifyOff))->sendCnt);
}

int MC2AicpuUtils::GetRecvCnt(AicpuComContext *ctx)
{
    if (ctx == nullptr) {
        return 0;
    }
    return static_cast<int>((reinterpret_cast<AivAicpuOpParam *>(ctx->workSpaceAddr + ctx->notifyOff +
        ctx->notifyBeginCnt * sizeof(uint8_t) * AC_SQE_SIZE))->rcvCnt);
}

void MC2AicpuUtils::PrintAicDebugInfo(AicpuComContext *ctx)
{
    if (ctx == nullptr) {
        return;
    }
    AicDebugInfo *info = reinterpret_cast<AicDebugInfo *>(ctx->workSpaceAddr);
    AicDebugCntInfo *sendCntInfo = reinterpret_cast<AicDebugCntInfo *>(ctx->workSpaceAddr + sizeof(AicDebugInfo));
    AicDebugCntInfo *recvCntInfo =
        reinterpret_cast<AicDebugCntInfo *>(ctx->workSpaceAddr + sizeof(AicDebugInfo) + sizeof(AicDebugCntInfo));

    std::stringstream sendCntSs;
    std::stringstream recvCntSs;
    for (int i = 0; i < static_cast<int>(MAX_DEBUG_CNT); i++) {
        if (sendCntInfo->cnt[i] != 0xff) {
            sendCntSs << "[" << i << "]" << static_cast<int>(sendCntInfo->cnt[i]) << ",";
        }
        if (recvCntInfo->cnt[i] != 0xff) {
            recvCntSs << "[" << i << "]" << static_cast<int>(recvCntInfo->cnt[i]) << ",";
        }
    }

    HCCL_ERROR("AIC %u,%u,%u,%u,%d,%d,%d,%d,%d,%d.", info->opIdx, info->rankM, info->rankK, info->rankN, info->opType,
        info->isTransB, info->isBias, info->isGatherOut, info->tileNum, info->tailNum);
    HCCL_ERROR("AIC %u sendCnt:%s, recvCnt:%s", info->opIdx, sendCntSs.str().c_str(), recvCntSs.str().c_str());
}
bool MC2AicpuUtils::IsDebugModeEquals(const AicpuComContext &ctx, const uint8_t Mode)
{
    return ctx.debugMode == Mode;
}

bool MC2AicpuUtils::NeedRecordTimeTaken(const AicpuComContext &ctx)
{
    return IsDebugModeEquals(ctx, MC2_DEBUG_TIME_TAKEN) || dfx::ProfilingManager::GetProfL1State();
}

void MC2AicpuUtils::PrintAllHcclMsgAreaError(HcclMsgArea *hcclMsgArea, u32 rankSize)
{
    PrintAllHcclMsgArea(hcclMsgArea, rankSize, true);
}

void MC2AicpuUtils::PrintAllHcclMsgAreaEvent(HcclMsgArea *hcclMsgArea, u32 rankSize)
{
    PrintAllHcclMsgArea(hcclMsgArea, rankSize, false);
}

void MC2AicpuUtils::PrintAllHcclMsgAreaError(HcclMsgAreaForMultiQue *hcclMsgArea, u32 rankSize)
{
    PrintAllHcclMsgArea(hcclMsgArea, rankSize, true);
}

void MC2AicpuUtils::PrintAllHcclMsgAreaEvent(HcclMsgAreaForMultiQue *hcclMsgArea, u32 rankSize)
{
    PrintAllHcclMsgArea(hcclMsgArea, rankSize, false);
}

u32 MC2AicpuUtils::GetBlockNum() {
    if (aicpu::GetBlockNum != nullptr) {
        return aicpu::GetBlockNum();
    } else {
        return 1U;
    }
}

u32 MC2AicpuUtils::GetBlockIdx() {
    if (aicpu::GetBlockIdx != nullptr) {
        return aicpu::GetBlockIdx();
    } else {
        return 0U;
    }
}
