/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "securec.h"
#include "hccl_common.h"
#include "common/aicpu_hccl_common.h"
#include "common/sqe_context.h"
#include "mc2_trace_utils.h"
#include "profiling_manager_device.h"
#include "utils/mc2_aicpu_utils.h"
#include "framework/aicpu_rpc_server.h"
#include "log.h"
#include "utils/aicpu_hdc_utils.h"
#include "aicpu_operator_pub.h"
#include "sal_pub.h"
#include "hccl_types.h"
#include "aicpu_hccl_dispatcher.h"
#include "adapter_hal_pub.h"

using namespace hccl;

HcclResult DispatcherAicpu::SignalWait(u16 streamId, u16 notifyId, bool innerChip, bool preNotify)
{
    auto ctx = AicpuGetComContext();
    AicpuComSignalInfo *notifyInfo = innerChip ?
        (preNotify ? &ctx->noIpcPreNotify[notifyId] : &ctx->noIpcPostNotify[notifyId]) :
        (preNotify ? &ctx->ipcPreWaitNotify[notifyId] : &ctx->ipcPostWaitNotify[notifyId]);
    return SignalWaitWithNotify(streamId, notifyId, innerChip, notifyInfo);
}

HcclResult DispatcherAicpu::AicpuUnfoldSignalWait(u16 streamId, u16 notifyId, bool innerChip)
{
    auto ctx = AicpuGetComContext();
    AicpuComSignalInfo *notifyInfo = &ctx->aicpuOpNotify[notifyId];
    return SignalWaitWithNotify(streamId, notifyId, innerChip, notifyInfo);
}

HcclResult DispatcherAicpu::SignalWaitWithNotify(u16 streamId, u16 notifyId, bool innerChip,
    AicpuComSignalInfo *notifyInfo)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo *streamInfo = &ctx->streamInfo[streamId];
    uint8_t *sqeBuffer = nullptr;
    uint8_t *sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(SqeContextUtils::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));
    if (innerChip || (ctx->devType != DevType::DEV_TYPE_310P1 && ctx->devType != DevType::DEV_TYPE_310P3)) {
        AicpuAddOneNotifyWaitSqe addOneNotifyWaitSqe = AicpuGetAddOneNotifyWaitSqe();
        if (addOneNotifyWaitSqe == nullptr) {
            HCCL_ERROR("AicpuAddOneNotifyWaitSqe is null");
            return HCCL_SUCCESS;
        }
        if (ctx->debugMode == MC2_DEBUG_NOTIFY_WAIT_TIMEOUT) {
            addOneNotifyWaitSqe(streamInfo->actualStreamId, taskId, INVALID_U64, sqeBuffer, sqeTypeAddr,
                ctx->dfxExtendInfo.dfxTimeOutConfig);
        } else {
            addOneNotifyWaitSqe(streamInfo->actualStreamId, taskId, notifyInfo->actualNotifyId, sqeBuffer, sqeTypeAddr,
                ctx->dfxExtendInfo.dfxTimeOutConfig);
        }

        if (innerChip) {
            CHK_RET(SqeContextUtils::RecordAddInfo(streamId, ctx->rankId));
        } else {
            CHK_RET(SqeContextUtils::RecordAddInfo(streamId, notifyId));
        }
    } else {
        u32 notifyRevisedOffset = 15U; // eventid偏移15位后为1
        u32 notifyGetEventId = 0x3FFU; // 取低15位
        if ((static_cast<u32>(notifyInfo->actualNotifyId) >> notifyRevisedOffset) != 0) {
            AicpuAddOneEventWaitSqe addOneEventWaitSqe = AicpuGetAddOneEventWaitSqe();
            if (addOneEventWaitSqe == nullptr) {
                HCCL_ERROR("addOneEventWaitSqe is null");
                return HCCL_SUCCESS;
            }
            addOneEventWaitSqe(streamInfo->actualStreamId,
                (static_cast<u32>(notifyInfo->actualNotifyId) & notifyGetEventId), taskId, sqeBuffer, sqeTypeAddr);
            CHK_RET(SqeContextUtils::RecordAddInfo(streamId, notifyId));

            uint8_t *sqeBuffer1 = nullptr;
            uint8_t *sqeTypeAddr1 = nullptr;
            CHK_RET(SqeContextUtils::GetNextSqeBufferAddr(streamId, sqeBuffer1, sqeTypeAddr1, taskId));

            AicpuAddOneEventResetSqe addOneEventResetSqe = AicpuGetAddOneEventResetSqe();
            if (addOneEventResetSqe == nullptr) {
                HCCL_ERROR("addOneEventResetSqe is null");
                return HCCL_SUCCESS;
            }
            addOneEventResetSqe(streamInfo->actualStreamId,
                (static_cast<u32>(notifyInfo->actualNotifyId) & notifyGetEventId), taskId, streamId, 0,
                notifyInfo->address, sqeBuffer1, sqeTypeAddr1);
            CHK_RET(SqeContextUtils::RecordAddInfo(streamId, notifyId));
        } else {
            HCCL_WARNING("SignalWait id is not event, please check %d", notifyInfo->actualNotifyId);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult DispatcherAicpu::SignalRecord(u16 streamId, u16 notifyId, bool innerChip, bool preNotify)
{
    auto ctx = AicpuGetComContext();
    AicpuComSignalInfo *notifyInfo = innerChip ?
        (preNotify ? &ctx->noIpcPreNotify[notifyId] : &ctx->noIpcPostNotify[notifyId]) :
        (preNotify ? &ctx->ipcPreRecordNotify[notifyId] : &ctx->ipcPostRecordNotify[notifyId]);
    return SignalRecordWithNotify(streamId, notifyId, innerChip, notifyInfo);
}

HcclResult DispatcherAicpu::AicpuUnfoldSignalRecord(u16 streamId, u16 notifyId, bool innerChip)
{
    auto ctx = AicpuGetComContext();
    AicpuComSignalInfo *notifyInfo = &ctx->aicpuOpNotify[notifyId];
    return SignalRecordWithNotify(streamId, notifyId, innerChip, notifyInfo);
}

HcclResult DispatcherAicpu::SignalRecordWithNotify(u16 streamId, u16 notifyId, bool innerChip,
    AicpuComSignalInfo *notifyInfo)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo *streamInfo = &ctx->streamInfo[streamId];
    uint8_t *sqeBuffer = nullptr;
    uint8_t *sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(SqeContextUtils::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    if (innerChip) {
        AicpuAddOneRecordSqe addOneRecordSqe = AicpuGetAddOneRecordSqe();
        if (addOneRecordSqe == nullptr) {
            HCCL_ERROR("AicpuAddOneRecordSqe is null");
            return HCCL_SUCCESS;
        }
        addOneRecordSqe(streamInfo->actualStreamId, taskId, notifyInfo->actualNotifyId, sqeBuffer, sqeTypeAddr);
        CHK_RET(SqeContextUtils::RecordAddInfo(streamId, ctx->rankId));
    } else {
        AicpuAddOneWriteValueRecordSqe addOneWriteValueRecordSqe = AicpuGetAddOneWriteValueRecordSqe();
        if (addOneWriteValueRecordSqe == nullptr) {
            HCCL_ERROR("AicpuAddOneWriteValueRecordSqe is null");
            return HCCL_SUCCESS;
        }
        addOneWriteValueRecordSqe(streamInfo->actualStreamId, taskId, notifyInfo->address, sqeBuffer, sqeTypeAddr);
        CHK_RET(SqeContextUtils::RecordAddInfo(streamId, notifyId));
    }
    return HCCL_SUCCESS;
}

HcclResult DispatcherAicpu::CopyData(u16 streamId, void *src, void *dst, u32 len, HcclDataType dataType,
    HcclReduceOp reduceOp, u32 remoteRank)
{
    if (len == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo *streamInfo = &ctx->streamInfo[streamId];

    rtDataType_t rtDataType = DT_MAP_TABLE[dataType];
    rtRecudeKind_t rtReduceOp = RK_MAP_TABLE[reduceOp];

    uint8_t *sqeBuffer = nullptr;
    uint8_t *sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(SqeContextUtils::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    AicpuAddOneMemcpySqe addOneMemcpySqe = AicpuGetAddOneMemcpySqe();
    if (addOneMemcpySqe == nullptr) {
        HCCL_ERROR("addOneMemcpySqe is null");
        return HCCL_SUCCESS;
    }
    if (ctx->debugMode == MC2_DEBUG_SDMA_ERROR) {
        src = nullptr;
    }
    addOneMemcpySqe(streamInfo->actualStreamId, taskId, src, len, rtDataType, rtReduceOp, dst, 0, ctx->ssid, ctx->devId,
        ctx->overflowAddr, static_cast<uint8_t>(LinkType::LINK_RESERVED), sqeBuffer, sqeTypeAddr);
    CHK_RET(SqeContextUtils::RecordAddInfo(streamId, (remoteRank << 16) + static_cast<uint32_t>(dataType)));  // 16 bit
    return HCCL_SUCCESS;
}

HcclResult DispatcherAicpu::CopyData(uint16_t streamId, u64 src, u64 dst, uint32_t len, HcclDataType dataType,
    HcclReduceOp reduceOp, u32 remoteRank)
{
    return CopyData(streamId, reinterpret_cast<void *>(src), reinterpret_cast<void *>(dst), len, dataType,
                    reduceOp, remoteRank);
}

HcclResult DispatcherAicpu::LaunchTask(uint32_t streamId)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo *streamInfo = &ctx->streamInfo[streamId];
    auto &sqeContextBuffer = GetSqeContext()->buffPtr[streamId];
    const auto cnt = sqeContextBuffer.sqeCnt;
    if (cnt == 0U) {
        HCCL_DEBUG("no sqe, rankid:%u, streamId:%d, sqId:%u", streamId, streamInfo->actualStreamId, streamInfo->sqId);
        return HCCL_SUCCESS;
    }
    auto &head = sqeContextBuffer.sqHead;
    auto &tail = sqeContextBuffer.sqTail;
    u32 newTail = (tail + cnt) % streamInfo->sqDepth;
    HCCL_INFO("Before send sqe:%d cnt:%u head:%u curtail:%u newTail:%u", streamInfo->sqId, cnt, head, tail, newTail);

    u64 startUsec = GetCurCpuTimestamp();
    while ((tail < head ? streamInfo->sqDepth : 0U) + tail - head + cnt >= streamInfo->sqDepth) { // 存在回绕
        CHK_RET(QuerySqStatusByType(ctx->devId, streamInfo->sqId, DRV_SQCQ_PROP_SQ_HEAD, head));
        if (GetCurCpuTimestamp() - startUsec > NSEC_PER_SEC * ctx->dfxExtendInfo.dfxTimeOutConfig.sqFullWaitTimeOut) {
            HCCL_ERROR("Rtsq full, timeout %lus. cur head:%u, sqId:%d",
                       ctx->dfxExtendInfo.dfxTimeOutConfig.sqFullWaitTimeOut,
                       head,
                       streamInfo->sqId);
            return HCCL_E_INTERNAL;
        }
    }

    auto memcpyFunc = [&](uint32_t dst, uint32_t dstMax, uint32_t src, uint32_t length) -> HcclResult {
        HCCL_DEBUG("Memcpy rank:%u , dst:%u, dstMax:%u, src:%u, length:%u", streamId, dst, dstMax, src, length);
        if (length == 0U) {
            return HCCL_SUCCESS;
        }
        errno_t ret = memcpy_s(reinterpret_cast<uint8_t *>(streamInfo->sqBaseAddr) + dst * AC_SQE_SIZE,
            dstMax * AC_SQE_SIZE, sqeContextBuffer.localBuff + src * AC_SQE_SIZE, length * AC_SQE_SIZE);
        if (ret != EOK) {
            HCCL_ERROR("Memcpy ret %d, dst:%u, dstMax:%u, src:%u, length:%u", ret, dst, dstMax, src, length);
            return HCCL_E_MEMORY;
        }
        return HCCL_SUCCESS;
    };
    uint32_t left = streamInfo->sqDepth - tail;                     // sqeAddr 剩余空间
    const auto tailSqeIdx = sqeContextBuffer.tailSqeIdx;
    HCCL_INFO("cpy sqe, left:%u, tailSqeId:%u, cnt:%u", left, tailSqeIdx, cnt);
    if (cnt <= left) { // 剩余buffer放得下新增sqe
        CHK_RET(memcpyFunc(tail, left, tailSqeIdx - cnt, cnt));
    } else {
        CHK_RET(memcpyFunc(tail, left, tailSqeIdx - cnt, left));
        CHK_RET(memcpyFunc(0, streamInfo->sqDepth, tailSqeIdx - cnt + left, cnt - left));
    }
    CHK_RET(ConfigSqStatusByType(ctx->devId, streamInfo->sqId, DRV_SQCQ_PROP_SQ_TAIL, newTail));

    tail = newTail;
    HCCL_INFO("After send sqe:%d, sqe_num:%u, curHead:%u, curtail:%u, sqeCnt:%u, tailSqeIdx:%u", streamInfo->sqId, cnt,
        head, tail, sqeContextBuffer.sqeCnt, sqeContextBuffer.tailSqeIdx);
    sqeContextBuffer.sqeCnt = 0;
    // StartMC2MaintenanceThread函数如果为空，说明是老的驱动包，为了解决老的驱动包可能的异常cq占满物理cq队列的情况，
    // 我们这里使用物理cq查询接口来清理队列； 如果是新的驱动包，会在StartMC2MaintenanceThread线程中使用logic cq进行查询并解析
    if (!IsSupportStartMC2MaintenanceThread() &&
        (ctx->devType != DevType::DEV_TYPE_310P1 && ctx->devType != DevType::DEV_TYPE_310P3)) {
        CqeQueryInput cqeQueryInput;
        cqeQueryInput.devId = ctx->devId;
        cqeQueryInput.streamId = streamInfo->actualStreamId;
        cqeQueryInput.sqId = streamInfo->sqId;
        cqeQueryInput.cqId = streamInfo->sqId;  // 使用sqid替代cqid，只有在sq cq成对申请，sqid cqid一样时才可以
        cqeQueryInput.type = static_cast<uint32_t>(DRV_NORMAL_TYPE);
        uint8_t tmpAddr[MAX_REPORT_CNT * 16];  // 16 cqe byte size
        cqeQueryInput.cqeAddr = tmpAddr;
        HCCL_DEBUG("Start to call cq report with [%s]", cqeQueryInput.ToString().c_str());
        rtLogicCqReport_t cqeException;
        (void)CqReportRecv(cqeQueryInput, cqeException);
    }
    return HCCL_SUCCESS;
}

HcclResult DispatcherAicpu::AddCcoreWait(uint16_t streamId, u64 waitAddr, uint32_t turnNum, bool isLast)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo *streamInfo = &ctx->streamInfo[streamId];

    uint8_t *sqeBuffer = nullptr;
    uint8_t *sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(SqeContextUtils::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    HCCL_INFO("[SQE]Add ccore wait addr %p, workSpaceAddr %p, notifyOff %u, turnNum %u, streamId=%u, isLast=%d",
        waitAddr, ctx->workSpaceAddr, ctx->notifyOff, turnNum, streamInfo->actualStreamId, isLast);
    if (ctx->debugMode == MC2_DEBUG_COMMIT_TIMEOUT) {
        ctx->turnValue[turnNum] = 0xFF;
    }
    AddOneWaitStartSqe(streamInfo->actualStreamId, taskId, waitAddr, reinterpret_cast<u64>(&ctx->turnValue[turnNum]),
        isLast, reinterpret_cast<rtStarsCcoreWaitStartSqe_t *>(sqeBuffer), sqeTypeAddr);
    CHK_RET(SqeContextUtils::RecordAddInfo(streamId, (turnNum << 16) + static_cast<uint32_t>(isLast)));  // 16 bit
    return HCCL_SUCCESS;
}

HcclResult DispatcherAicpu::AddWaitStartTaskOnMainStream(u16 streamId)
{
    auto ctx = AicpuGetComContext();
    // 保持和AIC的间消息长度一致，每隔64字节(sizeof(u8)*AC_SQE_SIZE)写一个地址。
    u64 waitAddr = 0;
    uint32_t turnNum = 0;
    bool isLast = 0;
    if (ctx->preparePosition == TASK_PREPARE_KERNEL) {
        waitAddr = ctx->workSpaceAddr + offsetof(HcclMsgArea, commitTurnCnt) +
                   ctx->msgPosForKernel * sizeof(TurnCnt) + offsetof(TurnCnt, cnt);

        turnNum = ctx->curTurnCntForKernel;
        isLast = ctx->curTurnCntForKernel >= ctx->totalTurnCntForKernel;
        HCCL_INFO("aicpu kernel mode, curTurnCnt %u, totalTurnCnt %u", ctx->curTurnCntForKernel,
            ctx->totalTurnCntForKernel);
    } else {
        waitAddr = ctx->workSpaceAddr + ctx->notifyOff + offsetof(AivAicpuOpParam, sendCnt);
        turnNum = (ctx->curTurnCnt + 1);
        isLast = (ctx->curTurnCnt + 1 >= ctx->totalTurnCnt);
    }
    return AddCcoreWait(streamId, waitAddr, turnNum, isLast);
}

HcclResult DispatcherAicpu::AddCcoreNotify(uint16_t streamId, uint32_t turnNum)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo *streamInfo = &ctx->streamInfo[streamId];
    uint8_t *sqeBuffer = nullptr;
    uint8_t *sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(SqeContextUtils::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    u64 recordAddr = 0;
    if (ctx->preparePosition == TASK_PREPARE_KERNEL) {
        recordAddr = ctx->workSpaceAddr + offsetof(HcclMsgArea, finishedTurnCnt) +
            ctx->msgPosForKernel * sizeof(TurnCnt) + offsetof(TurnCnt, cnt);
    } else {
        recordAddr =
            ctx->workSpaceAddr + ctx->notifyOff + ctx->notifyBeginCnt * AC_SQE_SIZE + offsetof(AivAicpuOpParam, rcvCnt);
    }
    HCCL_INFO("[SQE]Add ccore notify recordAddr %p, workSpaceAddr %p, notifyOff %u, notifyBeginCnt %u,"
        "streamId=%u, curTurnCnt %u, turnNum %u, preparePosition %u, msgPos %u",
        recordAddr, ctx->workSpaceAddr, ctx->notifyOff, ctx->notifyBeginCnt, streamInfo->actualStreamId,
        ctx->curTurnCnt, turnNum, ctx->preparePosition, ctx->msgPosForKernel);

    if (ctx->debugMode == MC2_DEBUG_AICORE_WAIT_TIMEOUT) {
        ctx->turnValue[turnNum] = 0;
    }
    AddOneWriteValueStartSqe(streamInfo->actualStreamId, taskId, recordAddr,
        reinterpret_cast<u64>(&ctx->turnValue[turnNum]), reinterpret_cast<rtStarsCcoreWriteValueSqe_t *>(sqeBuffer),
        sqeTypeAddr);
    CHK_RET(SqeContextUtils::RecordAddInfo(streamId, turnNum));
    return HCCL_SUCCESS;
}

HcclResult DispatcherAicpu::AddExecEndTaskOnMainStream(u16 streamId)
{
    auto ctx = AicpuGetComContext();
    uint32_t turnNum = ctx->preparePosition == TASK_PREPARE_KERNEL ? ctx->curTurnCntForKernel : ctx->curTurnCnt;
    return AddCcoreNotify(streamId, turnNum);
}

HcclResult DispatcherAicpu::AddAllEndTaskOnMainStream(u16 streamId)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo *streamInfo = &ctx->streamInfo[streamId];
    uint8_t *sqeBuffer = nullptr;
    uint8_t *sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(SqeContextUtils::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    AicpuAddOneRecordSqe addOneRecordSqe = AicpuGetAddOneRecordSqe();
    if (addOneRecordSqe == nullptr) {
        HCCL_ERROR("AicpuAddOneRecordSqe is null");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[SQE]Add all end task kfcNotifyId %lu, streamId %d", ctx->kfcNotifyId, streamInfo->actualStreamId);
    addOneRecordSqe(streamInfo->actualStreamId, taskId, ctx->kfcNotifyId, sqeBuffer, sqeTypeAddr);
    CHK_RET(SqeContextUtils::RecordAddInfo(streamId, ctx->rankId));
    return HCCL_SUCCESS;
}

HcclResult DispatcherAicpu::RdmaSend(uint16_t streamId, u64 dbInfo, u64 dbAddr, u32 userRank)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo *streamInfo = &ctx->streamInfo[streamId];
    uint8_t *sqeBuffer = nullptr;
    uint8_t *sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(SqeContextUtils::GetNextSqeBufferAddr(streamId, sqeBuffer, sqeTypeAddr, taskId));

    AicpuAddOneRdmaDbSendSqe AddOneRdmaDbSendSqe = AicpuGetAddOneRdmaDbSendSqe();
    if (AddOneRdmaDbSendSqe == nullptr) {
        HCCL_ERROR("[DispatcherAiCpu][RdmaSend] AddOneRdmaDbSendSqe is null");
        return HCCL_E_PTR;
    }
    AddOneRdmaDbSendSqe(streamInfo->actualStreamId, taskId, dbInfo, dbAddr,
        0, static_cast<uint8_t>(hccl::RdmaType::RDMA_TYPE_RESERVED), sqeBuffer, sqeTypeAddr);

    HCCL_INFO("[DispatcherAiCpu][RdmaSend] Call RdmaSend. para: rankId[%u] "
        "taskId[%u], streamId[%u]", userRank, taskId, streamInfo->actualStreamId);

    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::DoPreSync()
{
    // 15 sqe on main, 35 sqe on sub
    CHK_RET(MainSubPreSync());

    CHK_RET(IpcPreSync());

    CHK_RET(MainSubPostSync());

    CHK_RET(MainSubPreSync());

    HCCL_INFO("[SQE]Do pre sync on main stream 21 tasks, sub stream 35 tasks");
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::DoPostSync()
{
    // 8 sqe on main, 21 sqe on sub
    CHK_RET(IpcPostSync());

    CHK_RET(MainSubPostSync());

    HCCL_INFO("[SQE]Do post sync on main stream 7 tasks, sub stream 21 tasks");
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpySnd2Win(void *sndAddr, u64 dataSize, u64 sndOffset, u64 winOffset,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;

    AicpuComRankInfo *rankInfo = &ctx->rankInfo[rankId];
    void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + sndOffset);
    void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + winOffset);

    CHK_RET(DispatcherAicpu::CopyData(rankId, src, dst, dataSize, dataType, opType, rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpyRcv2Win(void *rcvAddr, u64 dataSize, u64 rcvOffset, u64 winOffset,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;

    AicpuComRankInfo *rankInfo = &ctx->rankInfo[rankId];
    void *src = static_cast<void *>(static_cast<s8 *>(rcvAddr) + rcvOffset);
    void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + winOffset);

    CHK_RET(DispatcherAicpu::CopyData(rankId, src, dst, dataSize, dataType, opType, rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2Win(u64 *dataSize, u64 *winOffsets, HcclReduceOp opType, u64 sendOff,
    HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    AicpuComRankInfo *selfRankInfo = &ctx->rankInfo[ctx->rankId];
    u64 offset = (winOffsets == nullptr) ? 0 : winOffsets[ctx->rankId];
    void *selfWindow = reinterpret_cast<void *>(static_cast<const uintptr_t>(selfRankInfo->window));
    void *dst = static_cast<void *>(static_cast<s8 *>(selfWindow) + sendOff + offset);
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            void *otherRankWindow = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window));
            void *src = static_cast<void *>(static_cast<s8 *>(otherRankWindow) + sendOff + offset);
            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize[ctx->rankId], dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2Win(const std::vector<u64> &dataSizes, u64 sendOff,
    const std::vector<u64> &winOffsets, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    AicpuComRankInfo *selfRankInfo = &ctx->rankInfo[ctx->rankId];
    u64 offset = winOffsets.empty() ? 0 : winOffsets[ctx->rankId];
    void *selfWindow = reinterpret_cast<void *>(static_cast<const uintptr_t>(selfRankInfo->window));
    void *dst = static_cast<void *>(static_cast<s8 *>(selfWindow) + sendOff + offset);
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            void *otherRankWindow = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window));
            void *src = static_cast<void *>(static_cast<s8 *>(otherRankWindow) + sendOff + offset);
            u64 dataSize = dataSizes.empty() ? 0 : dataSizes[ctx->rankId];
            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);

    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2WinEx(u32 mainRankId, u64 dataSize, u64 winOffset, HcclReduceOp opType,
    HcclDataType dataType, u32 maxStreamNum)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;

    AicpuComRankInfo *mainRankInfo = &ctx->rankInfo[mainRankId];
    AicpuComRankInfo *rankInfo = &ctx->rankInfo[rankId];
    void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + winOffset);
    void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(mainRankInfo->window) + winOffset);

    CHK_RET(DispatcherAicpu::CopyData(rankId % maxStreamNum, src, dst, dataSize, dataType, opType, mainRankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpySnd2WinEx(u32 mainRankId, void *sndAddr, u64 dataSize, u64 sndOffset, u64 winOffset,
    HcclReduceOp opType, HcclDataType dataType, u32 maxStreamNum)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;

    AicpuComRankInfo *rankInfo = &ctx->rankInfo[mainRankId];
    void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + sndOffset);
    void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + winOffset);

    CHK_RET(DispatcherAicpu::CopyData(rankId % maxStreamNum, src, dst, dataSize, dataType, opType, mainRankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpyWin2Rcv(void *rcvAddr, u64 dataSize, u64 winOffset, u64 rcvOffset,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;
    AicpuComRankInfo *rankInfo = &ctx->rankInfo[rankId];

    void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + winOffset);
    void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + rcvOffset);

    CHK_RET(DispatcherAicpu::CopyData(rankId, src, dst, dataSize, dataType, opType, rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpyWin2RcvEx1(void *rcvAddr, u64 dataSize, u64 rcvOffset, u64 winOffset,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;
    AicpuComRankInfo *rankInfo = &ctx->rankInfo[rankId];
    void *window = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window));
    void *src = static_cast<void *>(static_cast<s8 *>(window) + winOffset + rcvOffset);
    void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + rcvOffset);

    CHK_RET(DispatcherAicpu::CopyData(rankId, src, dst, dataSize, dataType, opType, rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpySnd2WinEx1(void *sndAddr, u64 dataSize, u64 sndOffset, u64 winOffset,
    HcclReduceOp opType, HcclDataType dataType, u32 maxStreamNum)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();

    AicpuComRankInfo *rankInfo = &ctx->rankInfo[ctx->rankId];
    void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + sndOffset);
    void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + winOffset);

    CHK_RET(DispatcherAicpu::CopyData(ctx->rankId % maxStreamNum, src, dst, dataSize, dataType, opType, ctx->rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpySnd2RcvEx(void *sndAddr, void *rcvAddr, u64 sndOffsets, u64 rcvOffsets,
    u64 dataSize, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 maxStreamNum = ctx->rankNum;

    void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + sndOffsets);
    void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + rcvOffsets);
    CHK_RET(DispatcherAicpu::CopyData(ctx->rankId % maxStreamNum, src, dst, dataSize, dataType, opType, ctx->rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpyWin2RcvEx(u32 mainRankId, void *rcvAddr, u64 dataSize, u64 winOffset, u64 rcvOffset,
    HcclReduceOp opType, HcclDataType dataType, u32 maxStreamNum)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;
    AicpuComRankInfo *rankInfo = &ctx->rankInfo[mainRankId];

    void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + winOffset);
    void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + rcvOffset);

    CHK_RET(DispatcherAicpu::CopyData(rankId % maxStreamNum, src, dst, dataSize, dataType, opType, mainRankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpySnd2Rcv(void *sndAddr, void *rcvAddr, u64 sndOffsets, u64 rcvOffsets, u64 dataSize,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 maxStreamNum = ctx->rankNum;

    void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + sndOffsets);
    void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + rcvOffsets);
    CHK_RET(DispatcherAicpu::CopyData(ctx->rankId % maxStreamNum, src, dst, dataSize, dataType, opType, ctx->rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2Win(void *sndAddr, u64 dataSize, u64 *sndOffsets, u64 *winOffsets,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (sndOffsets == nullptr) ? 0 : sndOffsets[index];
            void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + srcOffset);
            u64 dstOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + dstOffset);
            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2Win(void *sndAddr, u64 dataSize, u64 srcOffset, u64 dstOffset,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + srcOffset);
            void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + dstOffset);
            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2Win(void *sndAddr, const std::vector<u64> &dataSizes,
    const std::vector<u64> &sndOffsets, u64 *winOffsets, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = sndOffsets.empty() ? 0 : sndOffsets[index];
            void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + srcOffset);
            u64 dstOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + dstOffset);

            CHK_RET(
                DispatcherAicpu::CopyData(index, src, dst, dataSizes.empty() ? 0 : dataSizes[index],
                                          dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2Win(void *sndAddr, u64 dataSize, u64 *sndOffsets, u64 winOffsets,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (sndOffsets == nullptr) ? 0 : sndOffsets[index];
            void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + srcOffset);
            void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + winOffsets);
            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2Win(void *sndAddr, u64 *dataSize, u64 *sndOffsets, u64 *winOffsets,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (sndOffsets == nullptr) ? 0 : sndOffsets[index];
            void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + srcOffset);
            u64 dstOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + dstOffset);
            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize[index], dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

// 将本端 snd 发送至对端 window
HcclResult TaskOrchestrator::IpcCpySnd2WinP2P(void *sndAddr, u32 dstRank, u64 dataSize, u64 sndOffsets, u64 winOffsets,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 selfRank = ctx->rankId;

    void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + sndOffsets);
    void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(ctx->rankInfo[dstRank].window) + winOffsets);
    // 下发到主流上
    CHK_RET(DispatcherAicpu::CopyData(selfRank, src, dst, dataSize, dataType, opType, dstRank));
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

// 从对端window拷贝到本端window
HcclResult TaskOrchestrator::IpcCpyWin2WinP2P(u32 srcRank, u64 dataSize, u64 srcOffsets, u64 dstOffsets,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 selfRank = ctx->rankId;

    void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(ctx->rankInfo[srcRank].window) + srcOffsets);
    void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(ctx->rankInfo[selfRank].window) + dstOffsets);
    CHK_RET(DispatcherAicpu::CopyData(srcRank, src, dst, dataSize, dataType, opType, srcRank));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2Rcv(void *rcvAddr, u64 dataSize, u64 *winOffsets, u64 *rcvOffsets,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + srcOffset);
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + dstOffset);

            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvEx(void *rcvAddr, u64 dataSize, u64 *rcvOffsets, u64 winOffset,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void *window = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window));
            void *src = static_cast<void *>(static_cast<s8 *>(window) + winOffset + dstOffset);
            void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + dstOffset);

            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvEx(void *rcvAddr, u64 *dataSize, u64 *rcvOffsets, u64 winOffset,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void *window = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window));
            void *src = static_cast<void *>(static_cast<s8 *>(window) + winOffset + dstOffset);
            void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + dstOffset);

            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize[index], dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvEx(void *rcvAddr, const std::vector<u64> &dataSizes,
    const std::vector<u64> &rcvOffsets, u64 recvOff, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 dstOffset = rcvOffsets.empty() ? 0 : rcvOffsets[index];
            void *window = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window));
            void *src = static_cast<void *>(static_cast<s8 *>(window) + recvOff + dstOffset);
            void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + dstOffset);
            u64 dataSize = dataSizes.empty() ? 0 : dataSizes[index];
            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2Rcv(void *rcvAddr, const std::vector<u64> &dataSizes, u64 *winOffsets,
    const std::vector<u64> &rcvOffsets, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + srcOffset);
            u64 dstOffset = rcvOffsets.empty() ? 0 : rcvOffsets[index];
            void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + dstOffset);

            CHK_RET(
                DispatcherAicpu::CopyData(index, src, dst, dataSizes.empty() ? 0 : dataSizes[index],
                                          dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2Rcv(void *rcvAddr, u64 dataSize, u64 winOffsets, u64 *rcvOffsets,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + winOffsets);
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + dstOffset);

            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2Rcv(void *rcvAddr, u64 *dataSize, u64 *winOffsets, u64 *rcvOffsets,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + srcOffset);
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + dstOffset);

            CHK_RET(DispatcherAicpu::CopyData(index, src, dst, dataSize[index], dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvP2P(void *rcvAddr, u32 srcRank, u64 dataSize, u64 srcOffset, u64 dstOffset,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();

    void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(ctx->rankInfo[srcRank].window) + srcOffset);
    void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + dstOffset);
    CHK_RET(DispatcherAicpu::CopyData(srcRank, src, dst, dataSize, dataType, opType, srcRank));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvP2PMainStream(void *rcvAddr, u32 srcRank, u64 dataSize, u64 srcOffset,
    u64 dstOffset, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 selfRank = ctx->rankId;

    void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(ctx->rankInfo[srcRank].window) + srcOffset);
    void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + dstOffset);
    CHK_RET(DispatcherAicpu::CopyData(selfRank, src, dst, dataSize, dataType, opType, srcRank));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2WinEx(void *sndAddr, u64 dataSize, u64 *sndOffsets, u64 *winOffsets,
    HcclReduceOp opType, HcclDataType dataType, u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 streamId = 0;
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            streamId = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (sndOffsets == nullptr) ? 0 : sndOffsets[index];
            void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + srcOffset);
            u64 dstOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + dstOffset);

            CHK_RET(DispatcherAicpu::CopyData(streamId, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvEx(void *rcvAddr, u64 dataSize, u64 *winOffsets, u64 *rcvOffsets,
    HcclReduceOp opType, HcclDataType dataType, u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 streamId = 0;
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            streamId = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + srcOffset);
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + dstOffset);

            CHK_RET(DispatcherAicpu::CopyData(streamId, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2WinSliceEx(void *sndAddr, std::vector<Slice> &dataSlice, u64 *winOffsets,
    HcclReduceOp opType, HcclDataType dataType, u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 streamId = 0;
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            streamId = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = dataSlice[index].offset;
            void *src = static_cast<void *>(static_cast<s8 *>(sndAddr) + srcOffset);
            u64 dstOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + dstOffset);

            CHK_RET(DispatcherAicpu::CopyData(streamId, src, dst, dataSlice[index].size, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvSliceEx(void *rcvAddr, std::vector<Slice> &dataSlice, u64 *winOffsets,
    HcclReduceOp opType, HcclDataType dataType, u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 streamId = 0;
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            streamId = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            AicpuComRankInfo *rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + srcOffset);
            u64 dstOffset = dataSlice[index].offset;
            void *dst = static_cast<void *>(static_cast<s8 *>(rcvAddr) + dstOffset);

            CHK_RET(DispatcherAicpu::CopyData(streamId, src, dst, dataSlice[index].size, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfLocalReduce(u64 dataSize, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;
    AicpuComRankInfo *rankInfo = &ctx->rankInfo[rankId];
    void *src = nullptr;
    void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window));
    u64 srcOffset = 0LU;
    u64 cpySize = 0LU;

    u32 rankNum = ctx->rankNum;
    u32 power = static_cast<u32>(log2(rankNum));
    u32 rankPower = static_cast<u32>(pow(2, power));
    if (rankPower < rankNum) {
        srcOffset = rankPower * dataSize;
        cpySize = (rankNum - rankPower) * dataSize;
        HCCL_DEBUG("SelfLocalReduce: rankNum %u, power %u, rankPower %u, srcOffset %lu, cpySize %lu", rankNum, power,
            rankPower, srcOffset, cpySize);
        src = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + srcOffset);
        CHK_RET(DispatcherAicpu::CopyData(rankId, src, dst, cpySize, dataType, opType, rankId));
    }

    for (u32 round = 0u; round < power; round++) {
        u32 sliceNum = rankPower / static_cast<u32>(pow(2, round + 1));
        srcOffset = sliceNum * dataSize;
        cpySize = srcOffset;
        HCCL_DEBUG("SelfLocalReduce: sliceNum %u, rankNum %u, power %u, rankPower %u, srcOffset %lu", sliceNum, rankNum,
            power, rankPower, srcOffset);
        src = reinterpret_cast<void *>(static_cast<const uintptr_t>(rankInfo->window) + srcOffset);
        CHK_RET(DispatcherAicpu::CopyData(rankId, src, dst, cpySize, dataType, opType, rankId));
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::LaunchTasks()
{
    auto ctx = AicpuGetComContext();
    return LaunchTasksEx(0, ctx->rankNum - 1, ctx->rankNum);
}

HcclResult TaskOrchestrator::LaunchTasksEx(u32 subStart, u32 subEnd, u32 maxStreamNum)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = GetCurCpuTimestamp();
    auto ctx = AicpuGetComContext();
    if (MC2AicpuUtils::NeedRecordTimeTaken(*ctx)) {
        RECORD_PROF_TIME(sendTaskStartTime);
    }
    u32 activeRank = ctx->rankId % maxStreamNum;

    /* 两阶段模式，主流待正式执行时再下 */
    /* 一阶段第一次，可以先下主流 */
    if (ctx->directlySendMainSteramSqe) {
        CHK_PRT_RET(ActiveRecordMain(activeRank) != HCCL_SUCCESS,
            HCCL_ERROR("launch task failed, sqid:%u", activeRank),
            HCCL_E_INTERNAL);
    }

    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != activeRank) {
            if (MC2AicpuUtils::NeedRecordTimeTaken(*ctx)) {
                ctx->acprof[g_proxLoopCnt].fillSqeCnt += GetSqeContext()->buffPtr[index].sqeCnt;
            }
            CHK_PRT_RET(DispatcherAicpu::LaunchTask(index) != HCCL_SUCCESS,
                HCCL_ERROR("launch task failed, sqid:%u", index), HCCL_E_INTERNAL);
        }
    }
    HCCL_INFO("LaunchTasksEx sqeBufferLocal, subStart=%u, subEnd=%u", subStart, subEnd);

    if (MC2AicpuUtils::NeedRecordTimeTaken(*ctx)) {
        const u64 endTime = GetCurCpuTimestamp();
        ctx->acprof[g_proxLoopCnt].sendSqeTimes += endTime - startTime;
        ctx->acprof[g_proxLoopCnt].sendSqeBatch += ctx->rankNum;
        RECORD_PROF_TIME(sendSqeFinishTime);
    }
    return HCCL_SUCCESS;
}

// 主流notify从流 从流wait主流
HcclResult TaskOrchestrator::MainSubPreSync()
{
    auto ctx = AicpuGetComContext();
    return MainSubPreSync(ctx->rankId, 0U, ctx->rankNum - 1U, ctx->rankNum);
}

HcclResult TaskOrchestrator::MainSubPreSync(const uint32_t subStream)
{
    auto ctx = AicpuGetComContext();
    return MainSubPreSync(ctx->rankId, subStream, subStream, ctx->rankNum);
}

HcclResult TaskOrchestrator::MainSubPreSync(uint32_t mainStream, uint32_t subStart, uint32_t subEnd, uint32_t maxStream)
{
    if (maxStream == 0U) {
        HCCL_ERROR("Max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != mainStream) {
            CHK_RET(DispatcherAicpu::SignalRecord(mainStream % maxStream, index, DispatcherAicpu::NO_IPC,
                DispatcherAicpu::PRE_SYNC));
            CHK_RET(DispatcherAicpu::SignalWait(index % maxStream, index, DispatcherAicpu::NO_IPC,
                DispatcherAicpu::PRE_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

// 从流notify主流 主流wait从流
HcclResult TaskOrchestrator::MainSubPostSync()
{
    auto ctx = AicpuGetComContext();
    return MainSubPostSync(ctx->rankId, 0U, ctx->rankNum - 1U, ctx->rankNum);
}

HcclResult TaskOrchestrator::MainSubPostSync(const uint32_t subStream)
{
    auto ctx = AicpuGetComContext();
    return MainSubPostSync(ctx->rankId, subStream, subStream, ctx->rankNum);
}

HcclResult TaskOrchestrator::MainSubPostSync(uint32_t mainStream, uint32_t subStart, uint32_t subEnd,
    uint32_t maxStream)
{
    if (maxStream == 0U) {
        HCCL_ERROR("Max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    for (uint32_t index = subStart; index <= subEnd; index++) {
        if (index != mainStream) {
            CHK_RET(DispatcherAicpu::SignalRecord(index % maxStream, index, DispatcherAicpu::NO_IPC,
                DispatcherAicpu::POST_SYNC));
            CHK_RET(DispatcherAicpu::SignalWait(mainStream % maxStream, index, DispatcherAicpu::NO_IPC,
                DispatcherAicpu::POST_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPreSync()
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            CHK_RET(DispatcherAicpu::SignalRecord(index, index, DispatcherAicpu::IPC, DispatcherAicpu::PRE_SYNC));
            CHK_RET(DispatcherAicpu::SignalWait(index, index, DispatcherAicpu::IPC, DispatcherAicpu::PRE_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPreRecordEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(DispatcherAicpu::SignalRecord(stream_id, index, DispatcherAicpu::IPC, DispatcherAicpu::PRE_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPreWaitEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(DispatcherAicpu::SignalWait(stream_id, index, DispatcherAicpu::IPC, DispatcherAicpu::PRE_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPreSyncEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(DispatcherAicpu::SignalRecord(stream_id, index, DispatcherAicpu::IPC, DispatcherAicpu::PRE_SYNC));
            CHK_RET(DispatcherAicpu::SignalWait(stream_id, index, DispatcherAicpu::IPC, DispatcherAicpu::PRE_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPreSyncOnMainStream()
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            CHK_RET(DispatcherAicpu::SignalRecord(ctx->rankId, index, DispatcherAicpu::IPC, DispatcherAicpu::PRE_SYNC));
            CHK_RET(DispatcherAicpu::SignalWait(ctx->rankId, index, DispatcherAicpu::IPC, DispatcherAicpu::PRE_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPostSyncOnMainStream()
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            CHK_RET(
                DispatcherAicpu::SignalRecord(ctx->rankId, index, DispatcherAicpu::IPC, DispatcherAicpu::POST_SYNC));
            CHK_RET(DispatcherAicpu::SignalWait(ctx->rankId, index, DispatcherAicpu::IPC, DispatcherAicpu::POST_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPostSync()
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            CHK_RET(DispatcherAicpu::SignalRecord(index, index, DispatcherAicpu::IPC, DispatcherAicpu::POST_SYNC));
            CHK_RET(DispatcherAicpu::SignalWait(index, index, DispatcherAicpu::IPC, DispatcherAicpu::POST_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPostRecordEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(DispatcherAicpu::SignalRecord(stream_id, index, DispatcherAicpu::IPC, DispatcherAicpu::POST_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPostWaitEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(DispatcherAicpu::SignalWait(stream_id, index, DispatcherAicpu::IPC, DispatcherAicpu::POST_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPostSyncEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(DispatcherAicpu::SignalRecord(stream_id, index, DispatcherAicpu::IPC, DispatcherAicpu::POST_SYNC));
            CHK_RET(DispatcherAicpu::SignalWait(stream_id, index, DispatcherAicpu::IPC, DispatcherAicpu::POST_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::ActiveRecordMain(u16 sqId)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo *streamInfo = &ctx->streamInfo[sqId];
    HCCL_DEBUG("ActiveStream rankId:%d, devId:%d, sqId:%lu, sqeCnt:%d",
        sqId,
        ctx->devId,
        streamInfo->sqId,
        GetSqeContext()->buffPtr[sqId].sqeCnt);
    if (GetSqeContext()->buffPtr[sqId].sqeCnt == 0U) {
        return HCCL_SUCCESS;
    }
    if (MC2AicpuUtils::NeedRecordTimeTaken(*ctx)) {
        ctx->acprof[g_proxLoopCnt].fillSqeCnt += GetSqeContext()->buffPtr[sqId].sqeCnt;
    }
    CHK_PRT_RET(DispatcherAicpu::LaunchTask(sqId) != HCCL_SUCCESS,
        HCCL_ERROR("Launch task failed, sqid:%u", sqId),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::WaitMainStreamFinish(AicpuComContext *ctx)
{
    s32 sqId = ctx->streamInfo[ctx->rankId].sqId;
    HCCL_INFO("Start WaitMainStreamFinish..devId = %d rankId:%u, sqid:%d", ctx->devId, ctx->rankId, sqId);

    auto ret = WaitFinishWhileLoop(ctx);
    if (ret != HCCL_SUCCESS) {
        if (ret != HCCL_E_SUSPENDING) {
            HCCL_ERROR("WaitFinishWhileLoop failed, determinism %u, ret %u.", ctx->determinism, ret);
        }
        return ret;
    }
    HCCL_INFO("End WaitMainStreamFinish..devId = %d rankid:%u, sqid:%d", ctx->devId, ctx->rankId, sqId);

    return HCCL_SUCCESS;
}

bool TaskOrchestrator::IsTaskExceptionForHccs(AicpuComContext *ctx)
{
    if (ctx->dfxExtendInfo.cqeStatus != dfx::CqeStatus::kCqeException) {
        return false;
    }

    // NOTE: 需要task exception补全dfx能力，定位故障task的remote rank; 目前暂不具备识别是否跨片的能力，默认失败的task均为跨片操作。
    if (ctx->dfxExtendInfo.cqeException.sqeType == RT_STARS_SQE_TYPE_WRITE_VALUE ||
        ctx->dfxExtendInfo.cqeException.sqeType == RT_STARS_SQE_TYPE_SDMA) {
        return true;
    }
    return false;
}

HcclResult TaskOrchestrator::WaitFinishWhileLoop(AicpuComContext *ctx)
{
    static uint32_t logHead = UINT32_MAX;
    static uint32_t logTail = UINT32_MAX;
    const uint64_t startUsec = GetCurCpuTimestamp();

    int32_t sqId = ctx->streamInfo[ctx->rankId].sqId;
    uint32_t sqHead = 0;
    uint32_t sqTail = 0;
    CHK_RET(QuerySqStatusByType(ctx->devId, sqId, DRV_SQCQ_PROP_SQ_TAIL, sqTail));
    uint32_t loopCnt = 0;
    ctx->sendCntRecord[1] = MC2AicpuUtils::GetSendCnt(ctx); // 1 记录下发完任务后的sendCnt
    ctx->recvCntRecord[1] = MC2AicpuUtils::GetRecvCnt(ctx); // 1 记录下发完任务后的recvCnt
    do {
        if (ctx->dfxExtendInfo.pollStatus == dfx::PollStatus::kStopAsException) {
            if (IsTaskExceptionForHccs(ctx)) {
                HCCL_WARNING("hccl aicpu stop wait task exec finish, for task exception.");
                return HCCL_E_SUSPENDING;
            } else {
                HCCL_ERROR("hccl aicpu exec failed, for task exception.");
                return HCCL_E_INTERNAL;
            }
        }

        KfcCommand cmd = KfcCommand::kNone;
        CHK_RET(AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd));
        if (cmd == KfcCommand::kStopLaunch) {
            HCCL_WARNING("hccl aicpu stop wait finish, for recv stop launch cmd");
            return HCCL_E_SUSPENDING;
        } else if ((cmd == KfcCommand::NsStopLaunch) && (ctx->commOpenStatus == true) && (ctx->endStopLaunch == false)) {
            HCCL_WARNING("N second stop Launch for recv stop launch cmd.");
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), true);
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), true);
            return HCCL_E_SUSPENDING;
        } else if (cmd == KfcCommand::kDestroyComm) {
            HCCL_WARNING("hccl aicpu stop wait finish, for recv destroy comm cmd");
            return HCCL_E_SUSPENDING;
        } else if (cmd == KfcCommand::kExit) {
            HCCL_ERROR("hccl aicpu stop wait finish, for recv exit cmd.");
            return HCCL_E_INTERNAL;
        }

        CHK_RET(QuerySqStatusByType(ctx->devId, sqId, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
        if (loopCnt > 10000) { // 10000 is max loop cnt
            uint32_t overflowFlag = 0;
            OverflowAddrCheck(ctx, overflowFlag, sqHead, sqTail);
            loopCnt = 0;
            if (logHead != sqHead || logTail != sqTail) {
                logHead = sqHead;
                logTail = sqTail;
                HCCL_INFO("Current state. devId:%u sqid:%d, head:%u, tail:%u", ctx->devId, sqId, sqHead, sqTail);
            }
            CHK_RET(WorkSpacePrint(ctx));
        }
        CHK_RET(CheckTaskTimeout(ctx, startUsec));
        HCCL_INFO("Current state. loopCnt:%u, devId:%u sqid:%d, head:%u, tail:%u", loopCnt, ctx->devId, sqId, sqHead, sqTail);
        loopCnt++;
    } while (sqHead != sqTail);
    return HCCL_SUCCESS;
}

void TaskOrchestrator::PrintTimeOutSqInfo(AicpuComContext *ctx, u64 timeThreshold)
{
    uint32_t status = 0U;
    int32_t sqId = ctx->streamInfo[ctx->rankId].sqId;
    auto ret = QuerySqStatusByType(ctx->devId, sqId, DRV_SQCQ_PROP_SQ_CQE_STATUS, status);
    if (ret != 0) {
        HCCL_ERROR("QuerySqStatusByType status failed. ret = %u sqid:%d", ret, sqId);
    }
    for (uint32_t i = 0U; i < ctx->rankNum; i++) {
        uint32_t sqHead = 0U;
        uint32_t sqTail = 0U;
        (void)QuerySqStatus(ctx->devId, ctx->streamInfo[i].sqId, sqHead, sqTail);
        SqeInfo sqeInfo;
        auto headRet = SqeContextUtils::QuerySqeInfoByHead(i, sqHead, &sqeInfo);
        if (headRet != HCCL_SUCCESS) {
            HCCL_ERROR("QuerySqeInfoByHead status failed. ret = %u sqHead:%d", headRet, sqHead);
            continue;
        }
        HCCL_ERROR("KFC timeout..[%lu]s, commId %s, stream %u sqid %d head %u tail %u. SqeInfo:%s",
            timeThreshold, ctx->hcomId, i, ctx->streamInfo[i].sqId, sqHead, sqTail,
            SqeContextUtils::GetString(sqeInfo).c_str());
    }
}

HcclResult TaskOrchestrator::CheckTaskTimeout(AicpuComContext *ctx, uint64_t startUsec)
{
    const uint64_t sqeTimeoutSec  = ctx->dfxExtendInfo.dfxTimeOutConfig.sqeWaitTimeOut;
    if (GetCurCpuTimestamp() - startUsec > static_cast<uint64_t>(NSEC_PER_SEC) * sqeTimeoutSec ) {
        PrintTimeOutSqInfo(ctx, sqeTimeoutSec);
        CHK_RET(MC2TraceUtils::Save());
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), dfx::KfcStatus::kTimeOut);
        return HCCL_E_TIMEOUT;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::WorkSpacePrint(AicpuComContext *ctx)
{
    static int staticSndCnt = -1;
    uint64_t waitAddr = ctx->workSpaceAddr + ctx->notifyOff;
    int sndCnt = static_cast<int>((reinterpret_cast<AivAicpuOpParam *>(waitAddr))->sendCnt);
    if (staticSndCnt != sndCnt) {
        staticSndCnt = sndCnt;
        std::stringstream recordLog;
        recordLog << "waitAddr:0x" << std::hex << waitAddr << ", sendCnt:" << std::dec << sndCnt;
        HCCL_INFO("%s", recordLog.str().c_str());
        CHK_RET(MC2TraceUtils::Submit(recordLog.str().c_str()));
    }

    static int staticRcvCnt = -1;
    uint64_t recordAddr = ctx->workSpaceAddr + ctx->notifyOff + ctx->notifyBeginCnt * sizeof(uint8_t) * AC_SQE_SIZE;
    int rcvCnt = static_cast<int>((reinterpret_cast<AivAicpuOpParam *>(recordAddr))->rcvCnt);
    if (staticRcvCnt != rcvCnt) {
        staticRcvCnt = rcvCnt;
        std::stringstream recordLog;
        recordLog << "recordAddr:0x" << std::hex << recordAddr << ", rcvCnt:" << std::dec << rcvCnt;
        HCCL_INFO("%s", recordLog.str().c_str());
        CHK_RET(MC2TraceUtils::Submit(recordLog.str().c_str()));
    }
    return HCCL_SUCCESS;
}

void TaskOrchestrator::OverflowAddrCheck(AicpuComContext *ctx, uint32_t &overflowFlag, uint32_t sqHead, uint32_t sqTail)
{
    if (ctx->devType != DevType::DEV_TYPE_310P1 && ctx->devType != DevType::DEV_TYPE_310P3) {
        return;
    }

    if (ctx->overflowAddr == 0) {
        return;
    }

    uint32_t overflowValTmp = *reinterpret_cast<uint32_t *>(ctx->overflowAddr);
    if ((overflowFlag == 0) && ((overflowValTmp & 0x11) == 0x11)) { // 与runtime对齐，溢出时会给该地址里填写0x11
        HCCL_WARNING("data is overflow, sqHead cur head:%u tail:%u, overflowVal:%u overflowValTmp:%u", sqHead, sqTail,
            overflowFlag, overflowValTmp);
        overflowFlag = 1;
    }
}

HcclResult TaskOrchestrator::AddBarrier(uint32_t mainStream, uint32_t rankId, uint32_t rankNum)
{
    const uint32_t preRankId = (rankId + rankNum - 1U) % rankNum;
    const uint32_t postRankId = (rankId + 1U) % rankNum;
    // 片间同步 notify后卡 wait前卡
    auto ret = DispatcherAicpu::SignalRecord(mainStream, postRankId, DispatcherAicpu::IPC, DispatcherAicpu::PRE_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Add notify post rank failed"), ret);
    ret = DispatcherAicpu::SignalWait(mainStream, preRankId, DispatcherAicpu::IPC, DispatcherAicpu::PRE_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Add wait pre rank failed"), ret);
    // 片间同步 notify前卡 wait后卡
    ret = DispatcherAicpu::SignalRecord(mainStream, preRankId, DispatcherAicpu::IPC, DispatcherAicpu::POST_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Add notify pre rank failed"), ret);
    ret = DispatcherAicpu::SignalWait(mainStream, postRankId, DispatcherAicpu::IPC, DispatcherAicpu::POST_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Add wait post rank failed"), ret);
    return ret;
}