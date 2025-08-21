/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_base.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "workflow_pub.h"
#include "param_check_pub.h"
#include "rank_consistentcy_checker.h"
#include "externalinput_pub.h"
#include "env_config.h"
#include "detect_connect_anomalies.h"
#include "../common/src/topo/topoinfo_detect.h"
#include "../common/src/topo/topoinfo_ranktable_partition.h"
#include "../common/src/state_guard.h"
#include "sal_pub.h"
#include "profiling_manager_pub.h"
#include "adapter_prof.h"
#include "adapter_qos_pub.h"
#include "adapter_rts_common.h"
#include "device_capacity.h"
#include "mem_host_pub.h"
#include "hcom_common.h"
#include "comm_config_pub.h"
#include "kernel_tiling/kernel_tiling.h"
#include "external/runtime/rt_error_codes.h"
#include "mmpa_api.h"

using namespace std;
using namespace hccl;

HcclResult GetCaptureInfo(aclrtStream stream, rtStreamCaptureStatus &captureStatus, uint32_t &modelId, bool &isCapture)
{
    isCapture = false;
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        HCCL_WARNING("[%s]Stream capture only support opbase mode!", __func__);
        return HCCL_SUCCESS;
    }
    rtModel_t rtModel = nullptr;
    rtError_t ret = rtStreamGetCaptureInfo(stream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture does not support!", __func__);
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(ret != RT_ERROR_NONE,
                    HCCL_ERROR("[%s]rtGet stream get capture status fail. return[%d]", __func__, ret), HCCL_E_RUNTIME);
    }
    if (captureStatus == RT_STREAM_CAPTURE_STATUS_ACTIVE) {
        isCapture = true;
        ret = rtModelGetId(rtModel, &modelId);
        CHK_PRT_RET(ret != RT_ERROR_NONE,
                    HCCL_ERROR("[%s]rtGet stream get capture model id fail. return[%d]", __func__, ret), HCCL_E_RUNTIME);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                         HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();

    bool isCapture;
    rtStreamCaptureStatus captureStatus = rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_NONE;
    uint32_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);

    uint64_t beginTime = hrtMsprofSysCycleTime();

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return all reduce success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
                  std::vector<std::string>({"HcclAllReduce", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
                  std::vector<std::string>({"HcclAllReduce", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
                  std::vector<std::string>({"HcclAllReduce", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));

    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 同通信域同算子复用tag
    const string tag = "AllReduce_" + hcclComm->GetIdentifier();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), count, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp(op), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
                             "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], op[%s], localRank[%u], streamId[%d],"
                             "comm[%p], deviceLogicId[%d]",
                             tag.c_str(), sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
                             localRank, streamId, comm, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));

        std::string logInfo = "Entry-HcclAllReduce: " + std::string(stackLogBuffer) +
                              ", capture status[" + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());
    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());
    CHK_RET_AND_PRINT_IDE(hcclComm->AllReduceOutPlace(tag, sendBuf, recvBuf, count, dataType, op, stream), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLREDUCE, beginTime, count, dataType, tag));

    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclAllReduce:success,take time: " +
                              std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HcclBarrier(HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtStreamCaptureStatus captureStatus = rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_NONE;
    uint32_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    uint64_t beginTime = hrtMsprofSysCycleTime();

    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    // Allreduce入参定义
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 同通信域同算子复用tag
    const string tag = "AllReduce_" + hcclComm->GetIdentifier();

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
                             "tag[%s], sendBuf[%p], recvBuf[%p], count[%d], dataType[%s], op[%s], streamId[%d],"
                             "deviceLogicId[%d]",
                             tag.c_str(), hcclComm->barrierSendBuf, hcclComm->barrierRecvBuf, HCCL_BARRIER_DEFAULT_COUNT,
                             GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclBarrier:" + std::string(stackLogBuffer) +
                              ", capture status[" + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateBarrierMemory(), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(hcclComm->barrierSendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(hcclComm->barrierRecvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->AllReduceOutPlace(tag, hcclComm->barrierSendBuf, hcclComm->barrierRecvBuf,
                                                      HCCL_BARRIER_DEFAULT_COUNT, dataType, op, stream, SyncMode::UNLIMITED_TIMEWAITSYNCMODE),
                          tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLREDUCE, beginTime, HCCL_BARRIER_DEFAULT_COUNT,
                                    dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclBarrier:success,take time: " +
                              std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}
