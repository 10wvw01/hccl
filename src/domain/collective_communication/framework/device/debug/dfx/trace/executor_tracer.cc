/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <iterator>
#include <functional>
#include <map>
#include "mc2_trace_utils.h"
#include "framework/aicpu_hccl_process.h"
#include "common/aicpu_hccl_common.h"
#include "stream_pub.h"
#include "utils/mc2_aicpu_utils.h"
#include "framework/aicpu_communicator.h"
#include "sal_pub.h"
#include "log_control.h"
#include "cann_error_reporter.h"
#include "executor_tracer.h"

namespace dfx_tracer {
void ExecutorTracer::BackGroundDfx(void *info)
{
    HCCL_RUN_INFO("Start to back ground.");
    // 外部保证info有效
    auto ctx = static_cast<AicpuComContext *>(info);
    hccl::HcclCommAicpu::ResetErrMsgReport(); // 业务重新拉起的场景，重置ErrMesg上报标记位
    while (true) {
        // 停止背景线程
        if (ctx->dfxExtendInfo.commandToBackGroud == dfx::CommandToBackGroud::kStop) {
            HCCL_INFO("Back ground thread returned..");
            break;
        }
        HandleDestroyComm(ctx);
        HandleBackGround(ctx);
        bool isNotStop = false;
        StopBackGround(ctx, isNotStop);
        if (!isNotStop) {
            HCCL_RUN_INFO("stop backGround Thread");
            break;
        }
        HandleReportStatusInComm();
        StopLaunchCommandHandle(ctx);
        KfcCommandHandle(ctx);
        HandleSwitchNic(ctx);
        HandleCqeStatus(ctx);
        HandleResumeChangeLink(ctx);
        usleep(TEN_MILLISECOND_OF_USLEEP);
    }
    (void)dfx::CannErrorReporter::GetInstance().Clear();
}

void ExecutorTracer::HandleDestroyComm(AicpuComContext *const ctx)
{
    std::mutex &commAicpuMapMutex = AicpuHcclProcess::AicpuGetCommMutex();
    std::unique_lock<std::mutex> lock(commAicpuMapMutex);
    std::vector<std::pair<std::string, hccl::HcclCommAicpu *>> aicpuCommInfo;
    (void)AicpuHcclProcess::AicpuGetCommAll(aicpuCommInfo);
    KfcCommand cmd = KfcCommand::kNone;
    for (auto &commInfo : aicpuCommInfo) {
        hccl::HcclCommAicpu *hcclAicpu = commInfo.second;
        if (hcclAicpu->GetCommInfoStatus()) {
            (void) hcclAicpu->GetKfcCommand(cmd);
            if (cmd == KfcCommand::kDestroyComm) {
                auto groupName = hcclAicpu->GetGroupName();
                HCCL_RUN_INFO("[ExecutorTracer][%s]Recv kDestroyComm cmd, group name[%s]", __func__, groupName.c_str());
                hcclAicpu->FlushUtraceInfo();
                KfcExecStatus responseStatus;
                responseStatus.execStatus.kfcStatus = KfcStatus::kDestroyComm;
                // 需要在销毁通信域前返回 kfc status到host，销毁通信域会释放TransferD2H
                s32 ret = hcclAicpu->ResponseBackGroundStatus(responseStatus);
                CHK_PRT_CONT(ret, HCCL_ERROR("[ExecutorTracer][%s]ResponseBackGroundStatus failed, group[%s], ret[%d]",
                    __func__, groupName.c_str(), ret));
                StopKfcThread(ctx, aicpuCommInfo);
                AicpuHcclProcess::AicpuDestoryCommbyGroup(groupName);
                cmd = KfcCommand::kNone;
            }
        }
    }
}

// recv host stop command
void ExecutorTracer::HandleBackGround(AicpuComContext *const ctx)
{
    BackgroundCommand bgCmd;
    if (ctx->commOpenStatus) {
        (void)AicpuHdcUtils::GetBackGroundCommand(ctx->kfcControlTransferH2D, bgCmd);
        if (bgCmd == BackgroundCommand::kStop) {
            HCCL_RUN_INFO("ctx stop back ground");
            StopKfcThread(ctx, {});
            KfcExecStatus bgresponse;
            bgresponse.execStatus.backgroundStatus = BackgroundStatus::kStop;
            ctx->commOpenStatus = false;
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, alreadyInit), false);
            (void)MC2TraceUtils::DestoryHandles();
            (void)AicpuHdcUtils::ResponseBackGroundStatus(ctx->kfcStatusTransferD2H, bgresponse);
            bgCmd = BackgroundCommand::kNone;
        }
    }
}

// stop 主线程
void ExecutorTracer::StopKfcThread(AicpuComContext *const ctx,
                                   std::vector<std::pair<std::string, hccl::HcclCommAicpu *>> aicpuCommInfo)
{
    const uint64_t waitTime =  static_cast<uint64_t>(NSEC_PER_SEC) * 10U;  // 10s
    uint64_t startTime = GetCurCpuTimestamp();
    while (ctx->isRunning) {
        if (!aicpuCommInfo.empty()) {
            for (auto &commInfo : aicpuCommInfo) {
                hccl::HcclCommAicpu *hcclAicpu = commInfo.second;
                dfx::DfxExtendInfo* dfxInfo = hcclAicpu->GetDfxExtendInfo();
                dfxInfo->pollStatus = dfx::PollStatus::kStopAsException;
            }
        } else {
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.pollStatus),
                                       dfx::PollStatus::kStopAsException);
        }

        if ((GetCurCpuTimestamp() - startTime) >= waitTime) {
            HCCL_ERROR("stop kfc thread timeout [10s]");
            break;
        }
    }
}

// stop 背景线程
void ExecutorTracer::StopBackGround(AicpuComContext *const ctx, bool &isNotStop)
{
    if (ctx->commOpenStatus) {
        isNotStop = true;
    } else {
        std::mutex &commAicpuMapMutex = AicpuHcclProcess::AicpuGetCommMutex();
        std::unique_lock<std::mutex> lock(commAicpuMapMutex);
        std::vector<std::pair<std::string, hccl::HcclCommAicpu *>> aicpuCommInfo;
        (void)AicpuHcclProcess::AicpuGetCommAll(aicpuCommInfo);
        for (auto &commInfo : aicpuCommInfo) {
            hccl::HcclCommAicpu *hcclAicpu = commInfo.second;
            if (hcclAicpu->GetCommInfoStatus()) {
                isNotStop = true;
            }
        }
    }
}

void ExecutorTracer::StopBackGroundDfx(void *info)
{
    // 外部保证info有效
    auto ctx = static_cast<AicpuComContext *>(info);
    ctx->dfxExtendInfo.commandToBackGroud = dfx::CommandToBackGroud::kStop;
    HCCL_INFO("Stop back ground thread..");
}

// handle StopLaunch Command
void ExecutorTracer::StopLaunchCommandHandle(AicpuComContext *const ctx)
{
    KfcCommand cmd = KfcCommand::kNone;
    if (ctx->commOpenStatus) {
        if (!ctx->endStopLaunch) {
            (void)AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd);
            if (cmd == KfcCommand::NsStopLaunch) {
                if (!ctx->isOpLaunch) {
                    (void)AicpuHdcUtils::SetOpExecStatus(
                        ctx->kfcStatusTransferD2H, KfcStatus::kStoplaunch, KfcError::kNone, 0);
                    AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), true);
                    HCCL_DEBUG("[NsRecovery][backGround]send in mc2 environment");
                }
            }
        }
    }
    std::mutex &commAicpuMapMutex = AicpuHcclProcess::AicpuGetCommMutex();
    std::unique_lock<std::mutex> lock(commAicpuMapMutex);
    std::vector<std::pair<std::string, hccl::HcclCommAicpu *>> aicpuCommInfo;
    (void)AicpuHcclProcess::AicpuGetCommAll(aicpuCommInfo);
    for (auto &commInfo : aicpuCommInfo) {
        hccl::HcclCommAicpu *hcclAicpu = commInfo.second;
        cmd = KfcCommand::kNone;
        if (hcclAicpu->GetCommInfoStatus()) {
            if (!hcclAicpu->GetNsStopLaunchStatus()) {
                (void)hcclAicpu->BackGroundGetCmd(cmd);
                if (cmd == KfcCommand::NsStopLaunch) {
                    if (!hcclAicpu->BackGroundGetOpStatus()) {
                        (void)hcclAicpu->BackGroundSetStatus(KfcStatus::kStoplaunch);
                        hcclAicpu->SetCommRecoveryFlag(true);
                        hcclAicpu->SetNsStopLaunchStatus(true);
                        HCCL_DEBUG("[NsRecovery][backGround]send in aicpu environment");
                    }
                }
            }
        }
    }
}

// handle StopExec and Clean Command
void ExecutorTracer::KfcCommandHandle(AicpuComContext *const ctx)
{
    if (ctx->commOpenStatus) {
        HandleKfcCommand(ctx);
    }
    std::mutex &commAicpuMapMutex = AicpuHcclProcess::AicpuGetCommMutex();
    std::unique_lock<std::mutex> lock(commAicpuMapMutex);
    std::vector<std::pair<std::string, hccl::HcclCommAicpu *>> aicpuCommInfo;
    (void)AicpuHcclProcess::AicpuGetCommAll(aicpuCommInfo);
    for (auto &commItem : aicpuCommInfo) {
        hccl::HcclCommAicpu *commInfo = commItem.second;
        if (commInfo->GetCommInfoStatus()) {
            if (commInfo->GetCommRecoveryFlag()) {
                HandleAICPUCommand(commInfo);
            }
        }
    }
}

// handle switch nic command
void ExecutorTracer::HandleSwitchNic(AicpuComContext *const ctx)
{
    std::mutex &commAicpuMapMutex = AicpuHcclProcess::AicpuGetCommMutex();
    std::unique_lock<std::mutex> lock(commAicpuMapMutex);
    std::vector<std::pair<std::string, hccl::HcclCommAicpu *>> aicpuCommInfo;
    (void)AicpuHcclProcess::AicpuGetCommAll(aicpuCommInfo);
    for (auto &commItem : aicpuCommInfo) {
        hccl::HcclCommAicpu *hcclAicpu = commItem.second;
        if (hcclAicpu->GetCommInfoStatus()) {
            KfcCommand kfcCmd = KfcCommand::kNone;
            (void) hcclAicpu->BackGroundGetCmd(kfcCmd);
            if (kfcCmd == KfcCommand::kSwitchNic) {
                auto groupName = hcclAicpu->GetGroupName();
                HCCL_RUN_INFO("[ExecutorTracer][%s]group name[%s], aicpu start switch nic",
                    __func__, groupName.c_str());
                HcclResult ret = hcclAicpu->SwitchNic();

                KfcExecStatus switchResp;
                if (ret == HCCL_SUCCESS) {
                    switchResp.execStatus.kfcStatus = KfcStatus::kSwitchSuccess;
                    (void) hcclAicpu->ResponseBackGroundStatus(switchResp);
                } else {
                    switchResp.execStatus.kfcStatus = KfcStatus::kSwitchFail;
                    (void) hcclAicpu->ResponseBackGroundStatus(switchResp);
                }
                HCCL_INFO("[ExecutorTracer][%s]group name[%s], aicpu finish switch nic, ret[%u]",
                    __func__, groupName.c_str(), ret);
            }
        }
    }
}
unsigned int getTrailingZeros(uint8_t num)
{
    uint8_t count = 0;
    while ((num & 1U) == 0) {
        count++;
        num >>= 1;
        if (num == 1U) {
            break;
        }
    }
    return count;
}

void ExecutorTracer::PrintTaskException(const rtLogicCqReport_t &reportOfOne)
{
    const std::vector<std::string> StarsCqeErrorDesc = {
        "task exception",
        "task trap",
        "task timeout",
        "sqe error",
        "resouce conflict error",
        "sq sw status error",
        "warning"
    };
    uint32_t errBit = static_cast<uint32_t>(getTrailingZeros(reportOfOne.errorType));
    const char *const errMsg = errBit < StarsCqeErrorDesc.size() ? StarsCqeErrorDesc[errBit].c_str() : "unknown";
    uint32_t idx = AicpuHcclProcess::GetStreamRankIdx(reportOfOne.streamId);
    SqeInfo sqeInfo;
    (void)SqeContextUtils::QuerySqeInfoByTaskId(idx, reportOfOne.taskId, &sqeInfo);
    HCCL_ERROR("Task run failed of exception, errorType [%u] error msg:[%s] sqe info:[%s]",
        errBit,
        errMsg,
        SqeContextUtils::GetString(sqeInfo).c_str());
}

void ExecutorTracer::HandleCqeStatusByRank(AicpuComContext *const ctx, uint32_t rank)
{
    const HcclComStreamInfo &streamInfo = ctx->streamInfo[rank];
    CqeQueryInput cqeQueryInput;
    SetCqeQueryInput(ctx->devId, streamInfo, cqeQueryInput);
    rtLogicCqReport_t report[AC_SQE_REV_MAX_CNT];
    cqeQueryInput.cqeAddr = reinterpret_cast<uint8_t *>(report);  // 用于存放接收到的cq
    rtLogicCqReport_t cqeException;
    CqeStatus cqeStatus = CqReportRecv(cqeQueryInput, cqeException);
    if (cqeStatus != dfx::CqeStatus::kDefault) {
        (void)MC2TraceUtils::Save();
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.cqeStatus), cqeStatus);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.pollStatus), dfx::PollStatus::kStopAsException);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.cqeException.sqeType), cqeException.sqeType);
        HCCL_ERROR("After send sqe:%d, exception happened on rank %u, cqeStatus[%d], sqetype[%u]",
                   streamInfo.sqId, rank, cqeStatus, cqeException.sqeType);
    }

    if (cqeStatus == dfx::CqeStatus::kCqeException) {
        PrintTaskException(cqeException);
    }
}

void ExecutorTracer::HandleResumeChangeLink(AicpuComContext *const ctx) 
{
    std::mutex &commAicpuMapMutex = AicpuHcclProcess::AicpuGetCommMutex();
    std::unique_lock<std::mutex> lock(commAicpuMapMutex);
    std::vector<std::pair<std::string, hccl::HcclCommAicpu *>> aicpuCommInfo;
    (void)AicpuHcclProcess::AicpuGetCommAll(aicpuCommInfo);
    for (auto &commItem : aicpuCommInfo) {
        hccl::HcclCommAicpu *hcclAicpu = commItem.second;
        if (hcclAicpu == nullptr) {
            HCCL_ERROR("[ExecutorTracer][%s]hcclAicpu is nullptr", __func__);
        }
        if (hcclAicpu != nullptr && hcclAicpu->GetCommInfoStatus()) {
            KfcCommand kfcCmd = KfcCommand::kNone;
            (void) hcclAicpu->BackGroundGetCmd(kfcCmd);
            if (kfcCmd == KfcCommand::NsChangeLink) {
                auto groupName = hcclAicpu->GetGroupName();
                HCCL_INFO("[ExecutorTracer][resume][%s]group name[%s], resume aicpu, start change link",
                    __func__, groupName.c_str());
                HcclResult ret = hcclAicpu->ResumeChangeLink();
                KfcExecStatus resumeResp;
                if (ret == HCCL_SUCCESS) {
                    resumeResp.execStatus.kfcStatus = KfcStatus::kResumeChanged;
                    HCCL_INFO("[ExecutorTracer][resume][%s]group name[%s], resume aicpu, change link, kResumeChanged",__func__, groupName.c_str());
                } else {
                    resumeResp.execStatus.kfcStatus = KfcStatus::kResumeError;
                    HCCL_INFO("[ExecutorTracer][resume][%s]group name[%s], resume aicpu, change link, kResumeError",__func__, groupName.c_str());
                }
                (void) hcclAicpu->ResponseBackGroundStatus(resumeResp);
                HCCL_INFO("[ExecutorTracer][%s]group name[%s], resume process, finish change link, ret[%u]",
                    __func__, groupName.c_str(), ret);
            }
        }
    }  
}

void ExecutorTracer::HandleCqeStatusInComm()
{
    std::mutex &commAicpuMapMutex = AicpuHcclProcess::AicpuGetCommMutex();
    std::unique_lock<std::mutex> lock(commAicpuMapMutex);
    std::vector<std::pair<std::string, hccl::HcclCommAicpu *>> aicpuCommInfo;
    (void)AicpuHcclProcess::AicpuGetCommAll(aicpuCommInfo);

    for (auto &commInfo : aicpuCommInfo) {
        std::vector<hccl::Stream> streams;
        hccl::HcclCommAicpu *hcclAicpu = commInfo.second;

        if (!hcclAicpu->GetCommInfoStatus()) { // 已结束, 不再轮询
            continue;
        }
        dfx::DfxExtendInfo* dfxInfo = hcclAicpu->GetDfxExtendInfo();
        if ((dfxInfo->cqeStatus != dfx::CqeStatus::kDefault) && (dfxInfo->cqeStatus != dfx::CqeStatus::kCqeException)) {
            continue;
        }

        (void)hcclAicpu->GetStreamAll(streams);
        for (hccl::Stream &stream : streams) {
            hcclAicpu->HandleCqeException(stream, false);
        }
    }
}

void ExecutorTracer::HandleReportStatusInComm()
{
    std::mutex &commAicpuMapMutex = AicpuHcclProcess::AicpuGetCommMutex();
    std::unique_lock<std::mutex> lock(commAicpuMapMutex);
    std::vector<std::pair<std::string, hccl::HcclCommAicpu *>> aicpuCommInfo;
    (void)AicpuHcclProcess::AicpuGetCommAll(aicpuCommInfo);

    for (auto &commInfo : aicpuCommInfo) {
        hccl::HcclCommAicpu *hcclAicpu = commInfo.second;

        if (!hcclAicpu || !hcclAicpu->GetCommInfoStatus()) { // 已结束, 不再轮询
            continue;
        }

        u32 deviceId = hcclAicpu->GetDevId();

        std::queue<dfx::ReportStatus> reportStatusQueue;
        (void)hcclAicpu->GetReportStatusQueue(reportStatusQueue);

        while (!reportStatusQueue.empty()) {
            dfx::ReportStatus reportStatus = reportStatusQueue.front();
            HCCL_INFO("Reporting opRetry status[%u] to dp frame, deviceId[%u], report queue size[%u].",
                reportStatus, deviceId, reportStatusQueue.size());
            HcclResult ret = dfx::CannErrorReporter::GetInstance().UpdateSensorNode(deviceId, reportStatus);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("Fail to report reportStatus[%u] to dp frame, status droped, deviceId[%u].",
                    reportStatus, deviceId);
            }
            reportStatusQueue.pop();
        }
    }
}

void ExecutorTracer::HandleCqeStatus(AicpuComContext *const ctx)
{
    HandleCqeStatusInComm();

    if (ctx == nullptr || ctx->alreadyInit == false) {
        return;
    }

    if (ctx->dfxExtendInfo.cqeStatus != dfx::CqeStatus::kDefault) {
        return;
    }

    if (ctx->dfxExtendInfo.kfcStatus != dfx::KfcStatus::kOneStart) {
        return;
    }

    // 遍历每个卡的cqe状态, 如果一个卡有异常，所有的卡的ctx里的轮询状态都会被标记为异常
    // rankNum前面的逻辑保证了是大于0的, 多机场景取1，非多机场景取rankNum
    const u32 streamNum = ctx->multiServerFlag ? 1U : ctx->rankNum;
    for (uint32_t rank = 0U; rank <= streamNum - 1; ++rank) {
        HandleCqeStatusByRank(ctx, rank);
    }
}

void ExecutorTracer::SetCqeQueryInput(const uint32_t devId, const HcclComStreamInfo &streamInfo,
    CqeQueryInput &cqeQueryInput)
{
    cqeQueryInput.devId = devId;
    cqeQueryInput.streamId = streamInfo.actualStreamId;
    cqeQueryInput.sqId = streamInfo.sqId;
    cqeQueryInput.cqId = streamInfo.logicCqId;
    cqeQueryInput.type = static_cast<uint32_t>(DRV_LOGIC_TYPE);
}

void ExecutorTracer::HandleAICPUCommand(hccl::HcclCommAicpu *const commInfo){
    using CommandCall = std::function<void(hccl::HcclCommAicpu *const commInfo)>;
    static std::map<KfcCommand, CommandCall> commandAicpuHandles = {
        {KfcCommand::NsStopExec, AICPUcommandHandles::NsCommStop},
        {KfcCommand::NsClear, AICPUcommandHandles::NsCommClean}};
    KfcCommand cmd = KfcCommand::kNone;
    (void) commInfo->BackGroundGetCmd(cmd);
    auto iter = commandAicpuHandles.find(cmd);
    if (iter == commandAicpuHandles.cend()) {
        return;
    }
    HCCL_DEBUG("Start to run aicpu command %ld", cmd);
    iter->second(commInfo);
}

void ExecutorTracer::HandleKfcCommand(AicpuComContext *const ctx)
{
    using CommandCall = std::function<void(AicpuComContext *const ctx)>;
    static std::map<KfcCommand, CommandCall> commandHandles = {
        {KfcCommand::NsStopExec, KfcCommandHandles::StopFunc},
        {KfcCommand::NsClear, KfcCommandHandles::ClearFunc}};

    KfcCommand cmd = KfcCommand::kNone;
    (void) AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd);
    auto iter = commandHandles.find(cmd);
    if (iter == commandHandles.cend()) {
        return;
    }
    HCCL_DEBUG("Start to run command %ld", cmd);
    iter->second(ctx);
}

void AICPUcommandHandles::NsCommStop(hccl::HcclCommAicpu *const commInfo)
{
    bool streamStatus = commInfo->GetCommInfoStreamStatus();
    if (streamStatus) {
        std::string groupName = commInfo->GetGroupName();
        commInfo->SetCommInfoStreamStatus(false);
        HCCL_RUN_INFO("[NsRecovery][NsCommStop] groupName[%s]", groupName.c_str());
        commInfo->NsCommStop();
    }
}

void AICPUcommandHandles::NsCommClean(hccl::HcclCommAicpu *const commInfo){
    bool streamStatus = commInfo->GetCommInfoStreamStatus();
    if (!streamStatus) {
        std::string groupName = commInfo->GetGroupName();
        commInfo->SetCommInfoStreamStatus(true);
        HCCL_RUN_INFO("[NsRecovery][NsCommClean] groupName[%s]", groupName.c_str());
        commInfo->NsCommClean();
        commInfo->SetCommRecoveryFlag(false);
    }
}

void KfcCommandHandles::StopFunc(AicpuComContext *const ctx)
{
    HCCL_INFO("StopFunc, current clusterId:%d", ctx->clusterId);
    if (ctx->isStopLaunch) {
        // kill 流
        if ((StreamsKill(ctx->devId) != HCCL_SUCCESS) ||
            (DeviceQuery(ctx->devId, ts::APP_ABORT_KILL_FINISH, 0U) != HCCL_SUCCESS)) {
            (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kExec, 0);
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), false);
            return;
        }

        // 停止条件算子
        HcclMsgArea *hcclMsgArea = reinterpret_cast<HcclMsgArea *>(ctx->workSpaceAddr);
        for (uint32_t i = 0; i < AC_MSG_CNT; i++) {
            hcclMsgArea->commitTurnCnt[i].cnt = 0xFF;
        }
        (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kStopExec, KfcError::kNone, 0);
    } else {
        (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, 0);
    }
    HCCL_INFO("StopFunc Finish");
}

void KfcCommandHandles::ClearFunc(AicpuComContext *const ctx)
{
    if (ctx->isStopLaunch) {
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), false);
        // 等待drv任务停止
        if (DeviceQuery(ctx->devId, ts::APP_ABORT_TERMINATE_FINISH, 0U) != HCCL_SUCCESS) {
            (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kExec, 0);
            return;
        }
        HCCL_INFO("ClearFunc, after APP_ABORT_TERMINATE_FINISH");
        // 使能sq,读清cq
        for (u32 i = 0; i < ctx->rankNum; i++) {
            const HcclComStreamInfo &streamInfo = ctx->streamInfo[i];
            HCCL_INFO("ClearFunc, sqid:%d", streamInfo.sqId);
            if (ConfigSqStatusByType(ctx->devId, streamInfo.sqId, DRV_SQCQ_PROP_SQ_DISABLE_TO_ENABLE, 1) !=
                HCCL_SUCCESS) {
                (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kExec, 0);
                return;
            }
            CqeQueryInput cqeQueryInput;
            ExecutorTracer::SetCqeQueryInput(ctx->devId, streamInfo, cqeQueryInput);
            rtLogicCqReport_t report[AC_SQE_REV_MAX_CNT];
            cqeQueryInput.cqeAddr = reinterpret_cast<uint8_t *>(report);  // 用于存放接收到的cq
            rtLogicCqReport_t cqeException;
            (void)CqReportRecv(cqeQueryInput, cqeException);
        }

        // 清理dfxExtendInfo状态
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), dfx::KfcStatus::kDefault);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.cqeStatus), dfx::CqeStatus::kDefault);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.pollStatus), dfx::PollStatus::kDefault);

        // 清理SqeContext
        SqeContextUtils::SyncVariable();
        SqeContextUtils::SaveVariable();
        if (SqeContextUtils::ClearLocalBuff() != HCCL_SUCCESS) {
            (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kExec, 0);
            return;
        }
        SqeContext *sqeContext = GetSqeContext();
        for (u32 i = 0; i < ctx->rankNum; i++) {
            auto &buff = sqeContext->buffPtr[i];
            if ((QuerySqStatusByType(ctx->devId, ctx->streamInfo[i].sqId, DRV_SQCQ_PROP_SQ_TAIL, buff.sqTail) !=
                    HCCL_SUCCESS) ||
                (QuerySqStatusByType(ctx->devId, ctx->streamInfo[i].sqId, DRV_SQCQ_PROP_SQ_HEAD, buff.sqHead) !=
                    HCCL_SUCCESS)) {
                (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kExec, 0);
                return;
            }
            HCCL_INFO("hccl aicpu reset stream buffer, sqid:%d head:%u tail:%u.",
                ctx->streamInfo[i].sqId,
                buff.sqHead,
                buff.sqTail);
        }
        (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kClear, KfcError::kNone, 0);
    } else {
        (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, 0);
    }
    // 清理共享内存
    HcclMsgArea *hcclMsgArea = reinterpret_cast<HcclMsgArea *>(ctx->workSpaceAddr);
    (void)memset_s(hcclMsgArea, sizeof(HcclMsgArea), 0, sizeof(HcclMsgArea));

    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), false);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), false);
    HCCL_INFO("ClearFunc Finish");
}
}  // namespace dfx_tracer
