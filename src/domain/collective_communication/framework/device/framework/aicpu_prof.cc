/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_prof.h"
#include "common/aicpu_hccl_common.h"
#include "profiling_manager_device.h"

thread_local static uint32_t g_curLoopCnt = 0;
thread_local static uint32_t g_profTotalCnt = 0;
thread_local static AicpuComProf g_acprof[AC_MAX_PROF_LOOP];
uint8_t HcclCommProf::debugMode_ = 0;

uint32_t HcclCommProf::GetCurrentLoopCnt() {
    return g_curLoopCnt;
}

AicpuComProf *HcclCommProf::GetCurrentAicpuProf()
{
    return &g_acprof[g_curLoopCnt];
}

void HcclCommProf::SetCurrentProf(uint64_t launchTime)
{
    g_acprof[g_curLoopCnt].tid = syscall(__NR_gettid);
    g_acprof[g_curLoopCnt].launchEntryTime = launchTime;
}

void HcclCommProf::SetKfcTimeLine(KfcTimeLine kfcTimeLine)
{
    if (!HcclCommProf::NeedRecordTimeTaken()) {
        return;
    }
    AicpuComProf *acprof = HcclCommProf::GetCurrentAicpuProf();
    uint32_t recordIndex = acprof->workCnt;
    recordIndex = (recordIndex >= AC_MAX_PROF_COMM_CNT) ? (AC_MAX_PROF_COMM_CNT - 1) : recordIndex;
    switch (kfcTimeLine) {
        case KfcTimeLine::HCC_EXEC_START_TIME:
            acprof->commLoop[recordIndex].hccExecStartTime = GetCurCpuTimestamp(true);
            break;
        case KfcTimeLine::SEND_TASK_START_TIME:
            acprof->commLoop[recordIndex].sendTaskStartTime = GetCurCpuTimestamp(true);
            break;
        case KfcTimeLine::SEND_SQE_FINISH_TIME:
            acprof->commLoop[recordIndex].sendSqeFinishTime = GetCurCpuTimestamp(true);
            break;
        default:
            break;
    }
    return;
}

void HcclCommProf::SetDebugMode(uint8_t debugMode)
{
    debugMode_ = debugMode;
}

bool HcclCommProf::IsDebugModeEquals(const uint8_t mode)
{
    return debugMode_ == mode;
}

bool HcclCommProf::NeedRecordTimeTaken()
{
    return IsDebugModeEquals(MC2_DEBUG_TIME_TAKEN) || dfx::ProfilingManager::GetProfL1State();
}

AicpuComProf *HcclCommProf::GetaicpuProfInst() {
    return &g_acprof[0];
}

uint32_t HcclCommProf::GetProfTotalCnt() {
    return g_profTotalCnt;
}

void HcclCommProf::AddProfLoopCnt(uint32_t addCnt)
{
    if (!HcclCommProf::NeedRecordTimeTaken()) {
        return;
    }
    g_curLoopCnt += addCnt;
    g_curLoopCnt %= AC_MAX_PROF_LOOP;
    HCCL_INFO("g_curLoopCnt set to %u", g_curLoopCnt);
}

void HcclCommProf::AddProfTotalCnt(uint32_t addCnt) {
    g_profTotalCnt += addCnt;
}

void HcclCommProf::AddRcdCnt(uint32_t idx, int32_t sqeCnt)
{
    if (idx >= AC_MAX_PROF_LOOP) {
        HCCL_ERROR("HcclCommProf AddRcdCnt idx %u overflow", idx);
        return;
    }
    g_acprof[idx].workCnt++;
    g_acprof[idx].fillSqeCnt += sqeCnt;
}

void HcclCommProf::OutputProfLog()
{
    if (!HcclCommProf::IsDebugModeEquals(MC2_DEBUG_TIME_TAKEN)) {
        return;
    }
    u32 threshold = AC_MAX_PROF_LOOP - 1;
    if (HcclCommProf::GetCurrentLoopCnt() >= threshold) {
        for (u32 i = 0; i <= threshold; i++) {
            AicpuComProf *acprof = &HcclCommProf::GetaicpuProfInst()[i];
            u32 workRcdCnt = acprof->workCnt > AC_MAX_PROF_COMM_CNT ? AC_MAX_PROF_COMM_CNT : acprof->workCnt;
            HCCL_RUN_INFO("OP %u: clusterID %u, tid %lu, rankId %u, workCnt %u, serverStartTime %lu, waitMsgStartTime "
                          "%lu, rtsqExeEndTime %lu, serverEndTime %lu, StartServer %lu, Finalize %lu, E2E %lu",
                          HcclCommProf::GetProfTotalCnt() + i, acprof->clusterId, acprof->tid, acprof->rankId,
                          acprof->workCnt, acprof->launchEntryTime, acprof->commInitEndTime, acprof->receiveFinalizeTime,
                          acprof->endTime, acprof->commInitEndTime - acprof->launchEntryTime,
                          acprof->endTime - acprof->receiveFinalizeTime, acprof->endTime - acprof->launchEntryTime);

            for (u32 j = 0; j < workRcdCnt; j++) {
                AicpuComProfCommLoop *commRcd = &acprof->commLoop[j];
                HCCL_RUN_INFO("Turn %u: kfcAlgExeStartTime %lu, sendTaskStartTime %lu, sendSqeFinishTime %lu, "
                              "TaskWaitRequest %lu, TaskOrchestration %lu, TaskLaunch %lu, TaskExecute %lu",
                              j + 1, commRcd->hccExecStartTime, commRcd->sendTaskStartTime, commRcd->sendSqeFinishTime,
                              commRcd->hccExecStartTime - acprof->commInitEndTime,
                              commRcd->sendTaskStartTime - commRcd->hccExecStartTime,
                              commRcd->sendSqeFinishTime - commRcd->sendTaskStartTime,
                              acprof->receiveFinalizeTime - commRcd->sendSqeFinishTime);
            }

            acprof->fillSqeCnt = 0;
            acprof->fillSqeTimes = 0;
            acprof->sendSqeBatch = 0;
            acprof->sendSqeTimes = 0;
            acprof->workCnt = 0;
            acprof->traceSubmitTime = 0;
            acprof->traceCtxTime = 0;
            acprof->traceSqeTime = 0;
        }
        HcclCommProf::AddProfTotalCnt(AC_MAX_PROF_LOOP);
    }
}