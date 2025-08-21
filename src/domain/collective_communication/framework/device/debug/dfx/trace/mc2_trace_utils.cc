/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_trace_utils.h"

#include <cmath>
#include <stdio.h>
#include "log.h"
#include "mmpa_api.h"
#include "securec.h"
#include "utrace_api.h"
#include "common/aicpu_hccl_common.h"
#include "common/aicpu_hccl_def.h"
#include "common/sqe_context.h"
#include "utils/mc2_aicpu_utils.h"
#include "adapter_hal_pub.h"

namespace {
constexpr uint32_t TRACE_RING_BUFF_SIZE = 128 * 1024;
constexpr uint32_t MAX_SQE_SUBMIT_NUM = 256U;
constexpr char UTRACE_SO[] = "libutrace.so";

using UtraceCreateWithAttrFunc = TraHandle (*)(TracerType, const char *, const TraceAttr *);
using UtraceGetHandleFunc = TraHandle (*)(TracerType, const char *);
using UtraceSubmitFunc = TraStatus (*)(TraHandle handle, const void *, uint32_t);
using UtraceEventCreateFunc = TraEventHandle (*)(const char *);
using UtraceEventDestroyFunc = void (*)(TraEventHandle);
using UtraceHandleDestroyFunc = void (*)(TraHandle);
using UtraceEventBindTraceFunc = TraStatus (*)(TraEventHandle, TraHandle);
using UtraceEventReportFunc = TraStatus (*)(TraEventHandle);
using UtraceSetGlobalAttrFunc = TraStatus (*)(const TraceGlobalAttr *);
using UtraceStructEntryCreateFunc = TraceStructEntry *(*)(const char *);
using UtraceStructItemFieldSetFunc = void (*)(TraceStructEntry *, const char *, uint8_t, uint8_t, uint64_t);
using UtraceStructItemArraySetFunc = void (*)(TraceStructEntry *, const char *, uint8_t, uint8_t, uint64_t);
using UtraceStructSetAttrFunc = void (*)(TraceStructEntry *, uint8_t, TraceAttr *);
using UtraceStructEntryDestroyFunc = void (*)(TraceStructEntry *);

void *g_soHandle = nullptr;
UtraceCreateWithAttrFunc g_utraceCreateWithAttr = nullptr;
UtraceGetHandleFunc g_utraceGetHandle = nullptr;
UtraceSubmitFunc g_utraceSubmit = nullptr;
UtraceEventCreateFunc g_utraceEventCreate = nullptr;
UtraceEventDestroyFunc g_utraceEventDestroy = nullptr;
UtraceHandleDestroyFunc g_utraceHandleDestroy = nullptr;
UtraceEventBindTraceFunc g_utraceEventBindTrace = nullptr;
UtraceEventReportFunc g_utraceEventReport = nullptr;
UtraceSetGlobalAttrFunc g_utraceSetGlobalAttr = nullptr;
UtraceStructEntryCreateFunc g_utraceStructEntryCreate = nullptr;
UtraceStructItemFieldSetFunc g_utraceStructItemFieldSet = nullptr;
UtraceStructItemArraySetFunc g_utraceStructItemArraySet = nullptr;
UtraceStructSetAttrFunc g_utraceStructSetAttr = nullptr;
UtraceStructEntryDestroyFunc g_utraceStructEntryDestroy = nullptr;

TraEventHandle g_eventHandle = -1;
TraHandle g_traceStrHandle = -1;
TraHandle g_traceTaskAndTilingDataHandle = -1;
TraHandle g_traceAicpuComDataHandle = -1;
TraHandle g_traceMsgInfoHandle = -1;
TraHandle g_traceSqeBatchInfoHandle = -1;
TraceStructEntry *g_traceStrSt = nullptr;
TraceStructEntry *g_traceAicpuComTraceSt = nullptr;
TraceStructEntry *g_traceKFCtaskAndTilingTraceDataSt = nullptr;
TraceStructEntry *g_traceMsgInfoSt = nullptr;
TraceStructEntry *g_traceSqeBatchInfoSt = nullptr;
}

// 当msgNum不为2的幂时，会自动向上取2的幂
uint16_t MC2TraceUtils::GetMsgNum(size_t msgSize)
{
    float logValue = log(TRACE_RING_BUFF_SIZE / (msgSize + 16)) / log(2); // 16 预留字节 2 幂次
    return static_cast<uint16_t>(pow(2, floor(logValue))); // 2 幂次
}

HcclResult MC2TraceUtils::InitTraceStrHandle()
{
    // trace api存在限制: msgNum * msgSize <= 128k
    uint16_t strMsgNum = GetMsgNum(sizeof(TraceStr));
    TraceAttr attr = { 0 };
    attr = { false, strMsgNum, sizeof(TraceStr), nullptr };
    g_traceStrSt = g_utraceStructEntryCreate("TraceStr");
    g_utraceStructItemArraySet(g_traceStrSt, "transmit", TRACE_STRUCT_ARRAY_TYPE_CHAR, TRACE_STRUCT_SHOW_MODE_CHAR,
        sizeof(TraceStr::transmit));
    g_utraceStructSetAttr(g_traceStrSt, 0, &attr);
    g_traceStrHandle = g_utraceCreateWithAttr(TRACER_TYPE_SCHEDULE, "TraceStr", &attr);
    if (g_traceStrHandle < 0) {
        HCCL_ERROR("Create g_traceStrHandle failed, ret:%d", g_traceStrHandle);
        g_utraceStructEntryDestroy(g_traceStrSt);
        return HCCL_E_INTERNAL;
    }
    TraStatus ret = g_utraceEventBindTrace(g_eventHandle, g_traceStrHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("Bind g_traceStrHandle failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::InitTaskAndTilingDataHandle()
{
    uint16_t taskAndTilingDataMsgNum = GetMsgNum(sizeof(KFCtaskAndTilingTraceData));
    TraceAttr attr = { 0 };
    attr = { false, taskAndTilingDataMsgNum, sizeof(KFCtaskAndTilingTraceData), nullptr };
    g_traceKFCtaskAndTilingTraceDataSt = g_utraceStructEntryCreate("KFCtaskAndTilingTraceData");
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "inputA", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "outputC", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "commOut", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "context", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "workSpace", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "tilingData", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "sendOff", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "recvOff", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "tailSendOff", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "tailRecvOff", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "sendCnt", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "recvCnt", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "tailSendCnt", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "tailRecvCnt", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "totalCnt", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "turnNum", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "tailNum", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "stride", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "workspaceOff", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "notifyOff", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "notifyBeginCnt", TRACE_STRUCT_FIELD_TYPE_UINT16,
        TRACE_STRUCT_SHOW_MODE_DEC, 2); // 2 uint16
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "notifyEndCnt", TRACE_STRUCT_FIELD_TYPE_UINT16,
        TRACE_STRUCT_SHOW_MODE_DEC, 2); // 2 uint16
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "useBufferType", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "funID", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "dataType", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "groupNum", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "reuseMode", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "commType", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "reduceOp", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "commOrder", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "waitPolicy", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "rspPolicy", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "exitPolicy", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "commAlg", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "taskType", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "debugMode", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "stepSize", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "sendArgIndex", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "recvArgIndex", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "commOutArgIndex", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "hasCommOut", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "reverse", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceKFCtaskAndTilingTraceDataSt, "reserve2", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructSetAttr(g_traceKFCtaskAndTilingTraceDataSt, 0, &attr);
    g_traceTaskAndTilingDataHandle = g_utraceCreateWithAttr(TRACER_TYPE_SCHEDULE, "KFCtaskAndTilingTraceData", &attr);
    if (g_traceTaskAndTilingDataHandle < 0) {
        HCCL_ERROR("Create g_traceTaskAndTilingDataHandle failed, ret:%d", g_traceTaskAndTilingDataHandle);
        g_utraceStructEntryDestroy(g_traceKFCtaskAndTilingTraceDataSt);
        return HCCL_E_INTERNAL;
    }
    TraStatus ret = g_utraceEventBindTrace(g_eventHandle, g_traceTaskAndTilingDataHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("Bind g_traceTaskAndTilingDataHandle failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::InitAicpuComDataHandle()
{
    uint16_t aicpuComMsgNum = GetMsgNum(sizeof(AicpuComTraceData));
    TraceAttr attr = { 0 };
    attr = { false, aicpuComMsgNum, sizeof(AicpuComTraceData), nullptr };
    g_traceAicpuComTraceSt = g_utraceStructEntryCreate("AicpuComTraceData");
    g_utraceStructItemFieldSet(g_traceAicpuComTraceSt, "devId", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceAicpuComTraceSt, "ssid", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceAicpuComTraceSt, "rankId", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceAicpuComTraceSt, "rankNum", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceAicpuComTraceSt, "windowSize", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_DEC, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceAicpuComTraceSt, "workSpaceAddr", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceAicpuComTraceSt, "kfcNotifyId", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_DEC, 8); // 8 uint64
    g_utraceStructItemArraySet(g_traceAicpuComTraceSt, "eventIds", TRACE_STRUCT_ARRAY_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 128); // 4 uint32 128 AC_MAX_RANK_NUM*uint32
    g_utraceStructItemArraySet(g_traceAicpuComTraceSt, "windowIn", TRACE_STRUCT_ARRAY_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 256); // 8 uint64 256 AC_MAX_RANK_NUM*uint64
    g_utraceStructItemArraySet(g_traceAicpuComTraceSt, "windowOut", TRACE_STRUCT_ARRAY_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 256); // 8 uint64 256 AC_MAX_RANK_NUM*uint64
    g_utraceStructItemArraySet(g_traceAicpuComTraceSt, "actualStreamId", TRACE_STRUCT_ARRAY_TYPE_INT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 128); // 4 uint32 128 AC_MAX_RANK_NUM*uint32
    g_utraceStructItemArraySet(g_traceAicpuComTraceSt, "sqId", TRACE_STRUCT_ARRAY_TYPE_INT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 128); // 4 uint32 128 AC_MAX_RANK_NUM*uint32
    g_utraceStructItemArraySet(g_traceAicpuComTraceSt, "aicpuOpNotifyAddress", TRACE_STRUCT_ARRAY_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 16); // 8 uint64 16 2*uint64
    g_utraceStructItemArraySet(g_traceAicpuComTraceSt, "aicpuOpNotifyActualNotifyId", TRACE_STRUCT_ARRAY_TYPE_INT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 8); // 4 uint32 8 2*uint32
    g_utraceStructItemFieldSet(g_traceAicpuComTraceSt, "clusterId", TRACE_STRUCT_FIELD_TYPE_INT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructSetAttr(g_traceAicpuComTraceSt, 0, &attr);
    g_traceAicpuComDataHandle = g_utraceCreateWithAttr(TRACER_TYPE_SCHEDULE, "AicpuComTraceData", &attr);
    if (g_traceAicpuComDataHandle < 0) {
        HCCL_ERROR("Create g_traceAicpuComDataHandle failed, ret:%d", g_traceAicpuComDataHandle);
        g_utraceStructEntryDestroy(g_traceAicpuComTraceSt);
        return HCCL_E_INTERNAL;
    }
    TraStatus ret = g_utraceEventBindTrace(g_eventHandle, g_traceAicpuComDataHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("Bind g_traceAicpuComDataHandle failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::InitMsgInfoHandle()
{
    uint16_t msgInfoNum = GetMsgNum(sizeof(AivAicpuOpParam));
    TraceAttr attr = { 0 };
    attr = { false, msgInfoNum, sizeof(AivAicpuOpParam), nullptr };
    g_traceMsgInfoSt = g_utraceStructEntryCreate("AivAicpuOpParam");
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "commType", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "opType", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "sendBuffer", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "recvBuffer", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "count", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_DEC, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "strideLen", TRACE_STRUCT_FIELD_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_DEC, 8); // 8 uint64
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "hcclDataType", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "valid", TRACE_STRUCT_FIELD_TYPE_UINT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "isLast", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "funID", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "sendCnt", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "rcvCnt", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "everyTurnRsp", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "everyTurnWait", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "totalTurnCnt", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "useBufferType", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_utraceStructItemFieldSet(g_traceMsgInfoSt, "winOffset", TRACE_STRUCT_ARRAY_TYPE_UINT64,
        TRACE_STRUCT_SHOW_MODE_DEC, 8); // 8 uint64
    g_utraceStructItemArraySet(g_traceMsgInfoSt, "res", TRACE_STRUCT_ARRAY_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 2); // 1 uint8 2 2*uint8
    g_utraceStructSetAttr(g_traceMsgInfoSt, 0, &attr);
    g_traceMsgInfoHandle = g_utraceCreateWithAttr(TRACER_TYPE_SCHEDULE, "AivAicpuOpParam", &attr);
    if (g_traceMsgInfoHandle < 0) {
        HCCL_ERROR("Create g_traceMsgInfoHandle failed, ret:%d", g_traceMsgInfoHandle);
        g_utraceStructEntryDestroy(g_traceMsgInfoSt);
        return HCCL_E_INTERNAL;
    }
    TraStatus ret = g_utraceEventBindTrace(g_eventHandle, g_traceMsgInfoHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("Bind g_traceMsgInfoHandle failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::InitSqeBatchInfoHandle()
{
    uint16_t sqeBatchInfoMsgNum = GetMsgNum(sizeof(SqeBatchInfo));
    TraceAttr attr = { 0 };
    attr = { false, sqeBatchInfoMsgNum, sizeof(SqeBatchInfo), nullptr };
    g_traceSqeBatchInfoSt = g_utraceStructEntryCreate("SqeBatchInfo");
    for (uint32_t i = 0; i < MAX_SQE_BATCH_SIZE; i++) {
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "addr1High", TRACE_STRUCT_FIELD_TYPE_UINT32,
            TRACE_STRUCT_SHOW_MODE_HEX, 4); // 4 uint32
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "addr1Low", TRACE_STRUCT_FIELD_TYPE_UINT32,
            TRACE_STRUCT_SHOW_MODE_HEX, 4); // 4 uint32
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "addr2High", TRACE_STRUCT_FIELD_TYPE_UINT32,
            TRACE_STRUCT_SHOW_MODE_HEX, 4); // 4 uint32
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "addr2Low", TRACE_STRUCT_FIELD_TYPE_UINT32,
            TRACE_STRUCT_SHOW_MODE_HEX, 4); // 4 uint32
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "sqeHeadIdx", TRACE_STRUCT_FIELD_TYPE_UINT32,
            TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "notifyId", TRACE_STRUCT_FIELD_TYPE_UINT32,
            TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "length", TRACE_STRUCT_FIELD_TYPE_UINT32,
            TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "partId", TRACE_STRUCT_FIELD_TYPE_UINT32,
            TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "remoteRank", TRACE_STRUCT_FIELD_TYPE_UINT32,
            TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "dataType", TRACE_STRUCT_FIELD_TYPE_UINT32,
            TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "streamId", TRACE_STRUCT_FIELD_TYPE_UINT16,
            TRACE_STRUCT_SHOW_MODE_DEC, 2); // 2 uint16
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "eventId", TRACE_STRUCT_FIELD_TYPE_UINT16,
            TRACE_STRUCT_SHOW_MODE_DEC, 2); // 2 uint16
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "taskId", TRACE_STRUCT_FIELD_TYPE_UINT16,
            TRACE_STRUCT_SHOW_MODE_DEC, 2); // 2 uint16
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "condValue", TRACE_STRUCT_FIELD_TYPE_UINT16,
            TRACE_STRUCT_SHOW_MODE_DEC, 2); // 2 uint16
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "isLast", TRACE_STRUCT_FIELD_TYPE_UINT8,
            TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "opCode", TRACE_STRUCT_FIELD_TYPE_UINT8,
            TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "sqeNum", TRACE_STRUCT_FIELD_TYPE_UINT8,
            TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "type", TRACE_STRUCT_FIELD_TYPE_UINT8,
            TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "subType", TRACE_STRUCT_FIELD_TYPE_UINT8,
            TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
        g_utraceStructItemFieldSet(g_traceSqeBatchInfoSt, "valid", TRACE_STRUCT_FIELD_TYPE_UINT8,
            TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
        g_utraceStructItemArraySet(g_traceSqeBatchInfoSt, "reverse", TRACE_STRUCT_ARRAY_TYPE_UINT8,
            TRACE_STRUCT_SHOW_MODE_DEC, 10); // 1 uint8 10 10*uint8
    }
    g_utraceStructSetAttr(g_traceSqeBatchInfoSt, 0, &attr);
    g_traceSqeBatchInfoHandle = g_utraceCreateWithAttr(TRACER_TYPE_SCHEDULE, "SqeBatchInfo", &attr);
    if (g_traceSqeBatchInfoHandle < 0) {
        HCCL_ERROR("Create g_traceSqeBatchInfoHandle failed, ret:%d", g_traceSqeBatchInfoHandle);
        g_utraceStructEntryDestroy(g_traceSqeBatchInfoSt);
        return HCCL_E_INTERNAL;
    }
    TraStatus ret = g_utraceEventBindTrace(g_eventHandle, g_traceSqeBatchInfoHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("Bind g_traceSqeBatchInfoHandle failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::InitTraceHandle()
{
    g_eventHandle = g_utraceEventCreate("mc2_dfx");
    if (g_eventHandle < 0) {
        HCCL_ERROR("Create mc2_dfx event handle failed");
        return HCCL_E_INTERNAL;
    }
    CHK_RET(InitTaskAndTilingDataHandle());
    CHK_RET(InitAicpuComDataHandle());
    CHK_RET(InitTraceStrHandle());
    CHK_RET(InitMsgInfoHandle());
    CHK_RET(InitSqeBatchInfoHandle());
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::InitFuncHandle()
{
    g_utraceCreateWithAttr = reinterpret_cast<UtraceCreateWithAttrFunc>(mmDlsym(g_soHandle, "UtraceCreateWithAttr"));
    CHK_PRT_RET(g_utraceCreateWithAttr == nullptr, HCCL_ERROR("Get g_utraceCreateWithAttr error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_utraceGetHandle = reinterpret_cast<UtraceGetHandleFunc>(mmDlsym(g_soHandle, "UtraceGetHandle"));
    CHK_PRT_RET(g_utraceGetHandle == nullptr, HCCL_ERROR("Get g_utraceGetHandle error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_utraceSubmit = reinterpret_cast<UtraceSubmitFunc>(mmDlsym(g_soHandle, "UtraceSubmit"));
    CHK_PRT_RET(g_utraceSubmit == nullptr, HCCL_ERROR("Get g_utraceSubmit error: %s", mmDlerror()), HCCL_E_SYSCALL);
    g_utraceEventCreate = reinterpret_cast<UtraceEventCreateFunc>(mmDlsym(g_soHandle, "UtraceEventCreate"));
    CHK_PRT_RET(g_utraceEventCreate == nullptr, HCCL_ERROR("Get g_utraceEventCreate error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_utraceEventDestroy = reinterpret_cast<UtraceEventDestroyFunc>(mmDlsym(g_soHandle, "UtraceEventDestroy"));
    CHK_PRT_RET(g_utraceEventDestroy == nullptr, HCCL_ERROR("Get g_utraceEventDestroy error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_utraceHandleDestroy = reinterpret_cast<UtraceHandleDestroyFunc>(mmDlsym(g_soHandle, "UtraceDestroy"));
    CHK_PRT_RET(g_utraceHandleDestroy == nullptr, HCCL_ERROR("Get g_utraceHandleDestroy error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_utraceEventBindTrace = reinterpret_cast<UtraceEventBindTraceFunc>(mmDlsym(g_soHandle, "UtraceEventBindTrace"));
    CHK_PRT_RET(g_utraceEventBindTrace == nullptr, HCCL_ERROR("Get g_utraceEventBindTrace error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_utraceEventReport = reinterpret_cast<UtraceEventReportFunc>(mmDlsym(g_soHandle, "UtraceEventReport"));
    CHK_PRT_RET(g_utraceEventReport == nullptr, HCCL_ERROR("Get g_utraceEventReport error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_utraceSetGlobalAttr = reinterpret_cast<UtraceSetGlobalAttrFunc>(mmDlsym(g_soHandle, "UtraceSetGlobalAttr"));
    CHK_PRT_RET(g_utraceSetGlobalAttr == nullptr, HCCL_ERROR("Get g_utraceSetGlobalAttr error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_utraceStructEntryCreate =
        reinterpret_cast<UtraceStructEntryCreateFunc>(mmDlsym(g_soHandle, "UtraceStructEntryCreate"));
    CHK_PRT_RET(g_utraceStructEntryCreate == nullptr,
        HCCL_ERROR("Get g_utraceStructEntryCreate error: %s", mmDlerror()), HCCL_E_SYSCALL);
    g_utraceStructItemFieldSet =
        reinterpret_cast<UtraceStructItemFieldSetFunc>(mmDlsym(g_soHandle, "UtraceStructItemFieldSet"));
    CHK_PRT_RET(g_utraceStructItemFieldSet == nullptr,
        HCCL_ERROR("Get g_utraceStructItemFieldSet error: %s", mmDlerror()), HCCL_E_SYSCALL);
    g_utraceStructItemArraySet =
        reinterpret_cast<UtraceStructItemArraySetFunc>(mmDlsym(g_soHandle, "UtraceStructItemArraySet"));
    CHK_PRT_RET(g_utraceStructItemArraySet == nullptr,
        HCCL_ERROR("Get g_utraceStructItemArraySet error: %s", mmDlerror()), HCCL_E_SYSCALL);
    g_utraceStructSetAttr = reinterpret_cast<UtraceStructSetAttrFunc>(mmDlsym(g_soHandle, "UtraceStructSetAttr"));
    CHK_PRT_RET(g_utraceStructSetAttr == nullptr, HCCL_ERROR("Get g_utraceStructSetAttr error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_utraceStructEntryDestroy =
        reinterpret_cast<UtraceStructEntryDestroyFunc>(mmDlsym(g_soHandle, "UtraceStructEntryDestroy"));
    CHK_PRT_RET(g_utraceStructEntryDestroy == nullptr,
        HCCL_ERROR("Get g_utraceStructEntryDestroy error: %s", mmDlerror()), HCCL_E_SYSCALL);
    return HCCL_SUCCESS;
}

__attribute__ ((constructor)) void DlopenTraceSo()
{
    if (g_soHandle == nullptr) {
        g_soHandle = mmDlopen(UTRACE_SO, RTLD_NOW | RTLD_GLOBAL);
        HCCL_INFO("mmDlopen success");
    }
}

HcclResult MC2TraceUtils::Init()
{
    if (g_soHandle == nullptr) {
        HCCL_WARNING("Cannot find so file: %s", UTRACE_SO);
        return HCCL_SUCCESS;
    }
    CHK_RET(InitFuncHandle());
    CHK_RET(InitTraceHandle());
    unsigned int hostpid = 0;
    unsigned int cpType = DEVDRV_PROCESS_CPTYPE_MAX;
    CHK_RET(HrtHalDrvQueryProcessHostPid(getpid(), nullptr, nullptr, &hostpid, &cpType));

    TraceGlobalAttr traceGlobalAttr = { 0 };
    traceGlobalAttr.saveMode = 1;
    traceGlobalAttr.deviceId = AicpuGetComContext()->devId;
    traceGlobalAttr.pid = hostpid;
    TraStatus ret = g_utraceSetGlobalAttr(&traceGlobalAttr);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("UtraceSetGlobalAttrFunc failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

template <typename T> HcclResult MC2TraceUtils::Submit(const T * const traceData)
{
    if (g_soHandle == nullptr || g_utraceGetHandle == nullptr || g_utraceSubmit == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(traceData);
    std::string typeName;
    if (std::is_same<T, KFCtaskAndTilingTraceData>::value) {
        typeName = "KFCtaskAndTilingTraceData";
    } else if (std::is_same<T, AicpuComTraceData>::value) {
        typeName = "AicpuComTraceData";
    } else if (std::is_same<T, AivAicpuOpParam>::value) {
        typeName = "AivAicpuOpParam";
    } else if (std::is_same<T, SqeBatchInfo>::value) {
        typeName = "SqeBatchInfo";
    } else {
        HCCL_ERROR("find typename: %s failed", typeid(T).name());
        return HCCL_E_INTERNAL;
    }

    auto traceHandle = g_utraceGetHandle(TRACER_TYPE_SCHEDULE, typeName.c_str());
    if (traceHandle < 0) {
        HCCL_ERROR("getHandle %s failed, ret:%d", typeName.c_str(), traceHandle);
        return HCCL_E_INTERNAL;
    }
    auto traceRet = g_utraceSubmit(traceHandle, reinterpret_cast<const void *>(traceData), sizeof(T));
    if (traceRet != TRACE_SUCCESS) {
        HCCL_ERROR("submit %s failed, ret:%d", typeName.c_str(), traceRet);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::Submit(AicpuComContext *ctx)
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(ctx);
    uint64_t startUsec = GetCurCpuTimestamp();
    AicpuComTraceData comTraceData;
    comTraceData.devId = ctx->devId;
    comTraceData.ssid = ctx->ssid;
    comTraceData.rankId = ctx->rankId;
    comTraceData.rankNum = ctx->rankNum;
    comTraceData.windowSize = ctx->windowSize;
    comTraceData.workSpaceAddr = ctx->workSpaceAddr;
    comTraceData.kfcNotifyId = ctx->kfcNotifyId;
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        comTraceData.eventIds[i] = ctx->eventIds[i];
        comTraceData.windowIn[i] = ctx->rankInfo[i].window;
        comTraceData.windowOut[i] = ctx->rankInfo[i].windowOut;
        comTraceData.actualStreamId[i] = ctx->streamInfo[i].actualStreamId;
        comTraceData.sqId[i] = ctx->streamInfo[i].sqId;
    }
    for (uint32_t i = 0; i < 2; i++) { // 2 aicpuOpNotify size
        comTraceData.aicpuOpNotifyAddress[i] = ctx->aicpuOpNotify[i].address;
        comTraceData.aicpuOpNotifyActualNotifyId[i] = ctx->aicpuOpNotify[i].actualNotifyId;
    }
    comTraceData.clusterId = ctx->clusterId;
    CHK_RET(Submit<AicpuComTraceData>(&comTraceData));
    if (MC2AicpuUtils::IsDebugModeEquals(*ctx, MC2_DEBUG_TIME_TAKEN)) {
        ctx->acprof[g_proxLoopCnt].traceCtxTime += GetCurCpuTimestamp() - startUsec;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::Submit(const KFCTask * const task, const HcclKFCTilingData * const tilingData)
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(task);
    CHK_PTR_NULL(tilingData);
    uint64_t startUsec = GetCurCpuTimestamp();
    KFCtaskAndTilingTraceData taskAndTilingData;
    error_t ret = memcpy_s(&taskAndTilingData, sizeof(KFCtaskAndTilingTraceData), task, sizeof(KFCTask));
    if (ret != EOK) {
        HCCL_ERROR("task memcpy_s fail ret:%d", ret);
        return HCCL_E_MEMORY;
    }
    ret = memcpy_s(reinterpret_cast<uint8_t *>(&taskAndTilingData) + sizeof(KFCTask),
        sizeof(KFCtaskAndTilingTraceData) - sizeof(KFCTask), tilingData, sizeof(HcclKFCTilingData));
    if (ret != EOK) {
        HCCL_ERROR("tilingData memcpy_s fail ret:%d", ret);
        return HCCL_E_MEMORY;
    }
    CHK_RET(Submit<KFCtaskAndTilingTraceData>(&taskAndTilingData));
    auto ctx = AicpuGetComContext();
    if (MC2AicpuUtils::IsDebugModeEquals(*ctx, MC2_DEBUG_TIME_TAKEN)) {
        ctx->acprof[g_proxLoopCnt].traceSubmitTime += GetCurCpuTimestamp() - startUsec;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::Submit(const std::string &traceStr)
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    return Submit(traceStr.c_str());
}

HcclResult MC2TraceUtils::Submit(const char *traceStr)
{
    if (g_soHandle == nullptr || g_utraceSubmit == nullptr) {
        return HCCL_SUCCESS;
    }
    uint64_t startUsec = GetCurCpuTimestamp();
    uint32_t strSize = strlen(traceStr);
    uint32_t posSize = 0U;
    TraceStr singleTraceStr;
    while (posSize < strSize) {
        uint32_t traceStrLen = std::min<uint32_t>(strSize - posSize, sizeof(singleTraceStr.transmit) - 1);
        CHK_SAFETY_FUNC_RET(
            memset_s(reinterpret_cast<void *>(&singleTraceStr.transmit), sizeof(singleTraceStr.transmit), 0,
            sizeof(singleTraceStr.transmit)));
        error_t ret = memcpy_s(singleTraceStr.transmit, traceStrLen, traceStr + posSize, traceStrLen);
        if (ret != EOK) {
            HCCL_ERROR("traceStr memcpy_s fail ret:%d", ret);
            return HCCL_E_MEMORY;
        }
        posSize += traceStrLen;
        auto traceStrRet =
            g_utraceSubmit(g_traceStrHandle, reinterpret_cast<const void *>(&singleTraceStr), sizeof(singleTraceStr));
        if (traceStrRet != TRACE_SUCCESS) {
            HCCL_ERROR("submittraceStr failed, ret:%d", traceStrRet);
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO("Submit string: %s", singleTraceStr.transmit);
    }
    auto ctx = AicpuGetComContext();
    if (MC2AicpuUtils::IsDebugModeEquals(*ctx, MC2_DEBUG_TIME_TAKEN)) {
        ctx->acprof[g_proxLoopCnt].traceSubmitTime += GetCurCpuTimestamp() - startUsec;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::SubmitBatchSqeInfo()
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    uint64_t startUsec = GetCurCpuTimestamp();
    SqeContext *context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    for (uint32_t streamId = 0U; streamId < AC_MAX_RANK_NUM; streamId++) {
        auto &buff = context->buffPtr[streamId];
        HCCL_INFO("sqTail: %u, sqHead: %u,  sqeCnt: %u, tailSqeTaskId: %u, tailSqeIdx: %u",
            buff.sqTail, buff.sqHead, buff.sqeCnt, buff.tailSqeTaskId, buff.tailSqeIdx);
        auto submitNum = std::min<uint32_t>(buff.tailSqeIdx - buff.sqeCnt, MAX_SQE_SUBMIT_NUM);
        for (uint32_t i = 0U; i <= submitNum / MAX_SQE_BATCH_SIZE; i++) {
            SqeBatchInfo sqeBatchInfo;
            for (uint32_t j = 0U; j < MAX_SQE_BATCH_SIZE; j++) {
                const auto idx = i * MAX_SQE_BATCH_SIZE + j;
                if (idx >= submitNum) {
                    break;
                }
                if (SqeContextUtils::QuerySqeInfoByTaskId(streamId, buff.tailSqeTaskId - submitNum + idx,
                    &sqeBatchInfo.sqeInfos[j]) != HCCL_SUCCESS) {
                    HCCL_WARNING("Query sqe info failed, stream:%u, id:%u", streamId, idx);
                    break;
                }
            }
            CHK_RET(Submit<SqeBatchInfo>(&sqeBatchInfo));
        }
    }
    auto ctx = AicpuGetComContext();
    if (MC2AicpuUtils::IsDebugModeEquals(*ctx, MC2_DEBUG_TIME_TAKEN)) {
        ctx->acprof[g_proxLoopCnt].traceSqeTime += GetCurCpuTimestamp() - startUsec;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::Save()
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    TraStatus ret = g_utraceEventReport(g_eventHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("TraceEventReport failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::DestoryHandles()
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    g_utraceStructEntryDestroy(g_traceStrSt);
    g_utraceStructEntryDestroy(g_traceAicpuComTraceSt);
    g_utraceStructEntryDestroy(g_traceKFCtaskAndTilingTraceDataSt);
    g_utraceStructEntryDestroy(g_traceMsgInfoSt);
    g_utraceStructEntryDestroy(g_traceSqeBatchInfoSt);
    g_utraceHandleDestroy(g_traceStrHandle);
    g_utraceHandleDestroy(g_traceTaskAndTilingDataHandle);
    g_utraceHandleDestroy(g_traceAicpuComDataHandle);
    g_utraceHandleDestroy(g_traceMsgInfoHandle);
    g_utraceHandleDestroy(g_traceSqeBatchInfoHandle);
    g_utraceEventDestroy(g_eventHandle);

    return HCCL_SUCCESS;
}

__attribute__ ((destructor)) void Finalize()
{
    if (g_soHandle != nullptr) {
        (void)mmDlclose(g_soHandle);
    }
    g_soHandle = nullptr;
}

// 显式实例化
template HcclResult MC2TraceUtils::Submit(const SqeBatchInfo * const);