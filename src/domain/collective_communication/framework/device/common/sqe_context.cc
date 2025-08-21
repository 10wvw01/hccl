/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/sqe_context.h"

#include <sstream>
#include <unordered_map>

#include "common/aicpu_hccl_common.h"
#include "utils/mc2_aicpu_utils.h"
#include "securec.h"

struct SqeContextVariable {
    int32_t lastClusterId = -1;
    SqeLocalRingBuffer *variablePtr = nullptr;
};

static SqeLocalRingBuffer g_ringBuffer[CLUSTER_CNT][AC_MAX_RANK_NUM];
static SqeContext g_sqeContext[CLUSTER_CNT];
static SqeContextVariable g_sqeVariable;

SqeContext *GetSqeContext()
{
    return &g_sqeContext[MC2AicpuUtils::GetCurClusterId()];
}

namespace {
void ParseNotifySqe(const uint8_t *sqeLocal, uint32_t addInfo, SqeInfo *info)
{
    auto sqe = reinterpret_cast<const rtStarsNotifySqeV1_t *>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->notifyId = sqe->notify_id;
    info->remoteRank = addInfo;
}
void ParseWriteValueSqe(const uint8_t *sqeLocal, uint32_t addInfo, SqeInfo *info)
{
    auto sqe = reinterpret_cast<const rtStarsWriteValueSqe_t *>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->subType = sqe->sub_type;
    info->eventId = sqe->res7;
    info->addr1High = sqe->write_addr_high;
    info->addr1Low = sqe->write_addr_low;
    info->remoteRank = addInfo;
    info->length = sqe->rdmaWrLenth; // rdma wr len
    info->taskRelated.rdmaType = sqe->rdmaType; // rdma type
}
void ParseFlipPlaceHolderSqe(const uint8_t *sqeLocal, uint32_t addInfo, SqeInfo *info)
{
    auto sqe = reinterpret_cast<const rtStarsPlaceHolderSqe_t *>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->length = sqe->u.flip_task_info.flipNumReport;
    info->remoteRank = addInfo;
}
void ParseEventSqe(const uint8_t *sqeLocal, uint32_t /* addInfo */, SqeInfo *info)
{
    auto sqe = reinterpret_cast<const rtStarsEventSqe_t *>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->eventId = sqe->eventId;
}
void ParseMemcpyAsyncSqe(const uint8_t *sqeLocal, uint32_t addInfo, SqeInfo *info)
{
    auto sqe = reinterpret_cast<const rtStarsMemcpyAsyncSqe_t *>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->opCode = sqe->opcode;
    info->length = sqe->length;
    info->addr1High = sqe->src_addr_high;
    info->addr1Low = sqe->src_addr_low;
    info->addr2High = sqe->dst_addr_high;
    info->addr2Low = sqe->dst_addr_low;
    info->partId = sqe->partid;
    info->remoteRank = addInfo >> 16; // 16 bit
    info->dataType = static_cast<uint16_t>(addInfo);
    info->taskRelated.linkType = static_cast<uint8_t>(sqe->linkType); // linkType
}
void ParseCcoreWaitStartSqe(const uint8_t *sqeLocal, uint32_t addInfo, SqeInfo *info)
{
    auto sqe = reinterpret_cast<const rtStarsCcoreWaitStartSqe_t *>(sqeLocal);
    info->type = sqe->sqeHeader.type;
    info->streamId = sqe->sqeHeader.rtStreamId;
    info->taskId = sqe->sqeHeader.taskId;
    info->addr1High = sqe->ldrImm2.immdAddrHigh;
    info->addr1Low = sqe->ldrImm2.immdAddrLow;
    info->condValue = addInfo >> 16; // 16 bit
    info->isLast = addInfo & 1;
}
void ParseCcoreWriteValueSqe(const uint8_t *sqeLocal, uint32_t addInfo, SqeInfo *info)
{
    auto sqe = reinterpret_cast<const rtStarsCcoreWriteValueSqe_t *>(sqeLocal);
    info->type = sqe->sqeHeader.type;
    info->streamId = sqe->sqeHeader.rtStreamId;
    info->taskId = sqe->sqeHeader.taskId;
    info->addr1High = sqe->lhwi1.immd;
    info->addr1Low = sqe->llwi1.immdHigh;
    info->addr2High = sqe->llwi1.immdLow;
    info->condValue = addInfo;
}
void ParseNotifySqeV2(const uint8_t *sqeLocal, uint32_t /* addInfo */, SqeInfo *info)
{
    auto sqe = reinterpret_cast<const rtStarsNotifySqeV2_t *>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rt_stream_id;
    info->taskId = sqe->header.task_id;
    info->notifyId = sqe->notify_id;
}
void ParseWriteValueSqeV2(const uint8_t *sqeLocal, uint32_t /* addInfo */, SqeInfo *info)
{
    auto sqe = reinterpret_cast<const rtStarsWriteValueSqeV2_t *>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rt_stream_id;
    info->taskId = sqe->header.task_id;
    info->addr1High = sqe->reg_addr_high;
    info->addr1Low = sqe->reg_addr_low;
}
void ParseEventSqeV2(const uint8_t *sqeLocal, uint32_t /* addInfo */, SqeInfo *info)
{
    auto sqe = reinterpret_cast<const rtStarsEventSqeV2_t *>(sqeLocal);
    info->type = sqe->type;
    info->streamId = sqe->rt_stream_id;
    info->taskId = sqe->task_id;
    info->eventId = sqe->event_id;
}
void ParseMemcpyAsyncSqeV2(const uint8_t *sqeLocal, uint32_t /* addInfo */, SqeInfo *info)
{
    auto sqe = reinterpret_cast<const rtStarsMemcpyAsyncSqeV2_t *>(sqeLocal);
    info->type = sqe->type;
    info->streamId = sqe->rt_stream_id;
    info->taskId = sqe->task_id;
    info->opCode = sqe->opcode;
    info->length = sqe->length;
    info->addr1High = sqe->src_addr_high;
    info->addr1Low = sqe->src_addr_low;
    info->addr2High = sqe->dst_addr_high;
    info->addr2Low = sqe->dst_addr_low;
}
}

void SqeContextUtils::InitSqeContext()
{
    for (uint32_t i = 0U; i < CLUSTER_CNT; i++) {
        SqeContext *context = &g_sqeContext[i];
        context->buffPtr = g_ringBuffer[i];
        (void)memset_s(context->buffPtr, sizeof(SqeLocalRingBuffer[AC_MAX_RANK_NUM]), 0,
            sizeof(SqeLocalRingBuffer[AC_MAX_RANK_NUM]));
        context->clusterId = i;
    }
}

void SqeContextUtils::SyncVariable()
{
    SqeContext *context = GetSqeContext();
    HCCL_DEBUG("SyncCtxVariable, cur clusterId %d, lastClusterId %d", context->clusterId, g_sqeVariable.lastClusterId);
    if (context->clusterId == g_sqeVariable.lastClusterId) {
        return;
    }
    if (g_sqeVariable.lastClusterId < 0 || g_sqeVariable.lastClusterId >= CLUSTER_CNT) {
        HCCL_DEBUG("SyncCtxVariable, invalid lastClusterId = %d", g_sqeVariable.lastClusterId);
        return;
    }
    context->buffPtr = g_sqeVariable.variablePtr;
}

void SqeContextUtils::SaveVariable()
{
    SqeContext *context = GetSqeContext();
    HCCL_DEBUG("Save sqe context variable, cur clusterId = %d", context->clusterId);
    g_sqeVariable.lastClusterId = context->clusterId;
    g_sqeVariable.variablePtr = context->buffPtr;
}

HcclResult SqeContextUtils::GetNextSqeBufferAddr(uint32_t streamId, uint8_t *&sqeBufferAddr, uint8_t *&sqeTypeAddr,
    uint16_t &taskId)
{
    CHK_PRT_RET((streamId >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[SqeContextUtils][GetNextSqeBufferAddr]Invalid streamId[%u] >= %u", streamId, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    SqeContext *context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    auto &buff = context->buffPtr[streamId];
    if (buff.tailSqeIdx >= AC_SQE_MAX_CNT) {
        HCCL_WARNING("Sqe cnt is overflow, need revise buff content, current streamid: %u", streamId);
        HCCL_INFO("buffer modify before ==> sqTail: %u, sqHead: %u,  sqeCnt: %u, tailSqeTaskId: %u, tailSqeIdx: %u",
            buff.sqTail, buff.sqHead, buff.sqeCnt, buff.tailSqeTaskId, buff.tailSqeIdx);
        CHK_RET(MC2AicpuUtils::TraceProfSubmit());
        CHK_RET(SqeContextUtils::ModifyBuffer(streamId));
        HCCL_INFO("buffer modify after ==> sqTail: %u, sqHead: %u,  sqeCnt: %u, tailSqeTaskId: %u, tailSqeIdx: %u",
            buff.sqTail, buff.sqHead, buff.sqeCnt, buff.tailSqeTaskId, buff.tailSqeIdx);
    }
    // nextTaskId=0的时候下发PlaceHolder
    if (UNLIKELY(buff.tailSqeTaskId == 0 && buff.filpNum != 0)) {
        CHK_RET(AddFlipTask(streamId));
    }

    buff.profTimestap[buff.tailSqeIdx] = GetCurCpuTimestamp(true);
    sqeBufferAddr = buff.localBuff + buff.tailSqeIdx * AC_SQE_SIZE;
    sqeTypeAddr = &buff.sqeType[buff.tailSqeIdx];
    taskId = buff.tailSqeTaskId;
    HCCL_DEBUG("Get stream:%u next idx:%u, taskId:%u", streamId, buff.tailSqeIdx, taskId);
    if (buff.tailSqeTaskId == UINT16_MAX) {
        buff.filpNum++;
        HCCL_WARNING("Sqe context cur taskId is uint16_max");
    }
    buff.tailSqeTaskId++;
    buff.tailSqeIdx++;
    buff.sqeCnt++;
    return HCCL_SUCCESS;
}

HcclResult SqeContextUtils::AddFlipTask(uint32_t streamId)
{
    if (!dfx::ProfilingManager::GetProfL0State()) {
        return HCCL_SUCCESS;
    }
    SqeContext *context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    auto &buff = context->buffPtr[streamId];
    uint16_t filpNum = buff.filpNum;
    uint16_t taskId = buff.tailSqeTaskId;
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo *streamInfo = &ctx->streamInfo[streamId];
    CHK_RET(dfx::ProfilingManager::ReportFilpTask(streamInfo->actualStreamId, taskId, filpNum));

    buff.profTimestap[buff.tailSqeIdx] = GetCurCpuTimestamp(true);
    uint8_t *sqeBufferAddr = buff.localBuff + buff.tailSqeIdx * AC_SQE_SIZE;
    uint8_t *sqeTypeAddr  = &buff.sqeType[buff.tailSqeIdx];
    AicpuAddOneFlipPlaceHolderSqe addOneFlipPlaceHolderSqe = AicpuGetAddOneFlipPlaceHolderSqe();
    if (addOneFlipPlaceHolderSqe == nullptr) {
        HCCL_WARNING("AicpuAddOneFlipPlaceHolderSqe is null");
        return HCCL_SUCCESS;
    }
    addOneFlipPlaceHolderSqe(streamInfo->actualStreamId, filpNum, taskId, sqeBufferAddr, sqeTypeAddr);
    buff.tailSqeTaskId++;
    buff.tailSqeIdx++;
    buff.sqeCnt++;

    HCCL_INFO("[SqeContextUtils][AddFlipTask] Call AddFlipTask. para: taskId[%u], streamId[%u], filpNum[%u]]", taskId,
        streamInfo->actualStreamId, filpNum);

    return HCCL_SUCCESS;
}

HcclResult SqeContextUtils::RecordAddInfo(uint32_t streamId, uint32_t addInfo)
{
    CHK_PRT_RET((streamId >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[SqeContextUtils][RecordAddInfo]Invalid streamId[%u] >= %u", streamId, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    SqeContext *context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    auto &buff = context->buffPtr[streamId];
    CHK_PRT_RET(((buff.tailSqeIdx == 0) || (buff.tailSqeIdx > AC_SQE_MAX_CNT)),
        HCCL_ERROR("[SqeContextUtils][RecordAddInfo]Invalid tailSqeIdx[%u]", buff.tailSqeIdx),
        HCCL_E_PARA);
    buff.addInfo[buff.tailSqeIdx - 1] = addInfo;
    return HCCL_SUCCESS;
}

HcclResult SqeContextUtils::QuerySqeInfoByHead(uint32_t streamId, uint32_t sqHead, SqeInfo *info)
{
    CHK_PRT_RET((streamId >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[SqeContextUtils][QuerySqeInfoByHead]Invalid streamId[%u] >= %u", streamId, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    CHK_PTR_NULL(info);
    SqeContext *context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    auto &buff = context->buffPtr[streamId];
    const uint32_t sqDepth = AicpuGetComContext()->streamInfo[streamId].sqDepth;
    uint32_t sqUnexecuted = (buff.sqTail + sqDepth - sqHead) % sqDepth;
    if (buff.tailSqeIdx < sqUnexecuted) {
        HCCL_WARNING("tail sqe idx %u is less then sq unexecuted num %u", buff.tailSqeIdx, sqUnexecuted);
        return HCCL_E_INTERNAL;
    }
    uint16_t idx = buff.tailSqeIdx - sqUnexecuted;
    HCCL_INFO("Query streamId:%u, sqeIdx:%u, actual idx:%u, type:%u", streamId, sqHead, idx, buff.sqeType[idx]);
    info->sqeHeadIdx = sqHead;
    return QuerySqeInfo(buff.localBuff + idx * AC_SQE_SIZE, buff.sqeType[idx], buff.addInfo[idx], info);
}

HcclResult SqeContextUtils::QuerySqeInfoByTaskId(uint32_t streamId, uint16_t taskId, SqeInfo *info)
{
    CHK_PRT_RET((streamId >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[SqeContextUtils][QuerySqeInfoByTaskId]Invalid streamId[%u] >= %u", streamId, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    CHK_PTR_NULL(info);
    SqeContext *context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    auto &buff = context->buffPtr[streamId];
    uint16_t tailRemain = buff.tailSqeTaskId - taskId;
    const uint32_t sqDepth = AicpuGetComContext()->streamInfo[streamId].sqDepth;
    uint32_t sqHeadIdx = (buff.sqTail + sqDepth - tailRemain) % sqDepth;
    if (buff.tailSqeIdx < tailRemain) {
        HCCL_WARNING("tail sqe idx %u is less then tail remain num %u", buff.tailSqeIdx, tailRemain);
        return HCCL_E_INTERNAL;
    }
    uint16_t idx = buff.tailSqeIdx - tailRemain;
    HCCL_INFO("Query streamId:%u, sqeIdx:%u, actual idx:%u, type:%u", streamId, sqHeadIdx, idx, buff.sqeType[idx]);
    info->sqeHeadIdx = sqHeadIdx;
    return QuerySqeInfo(buff.localBuff + idx * AC_SQE_SIZE, buff.sqeType[idx], buff.addInfo[idx], info);
}

HcclResult SqeContextUtils::QuerySqeInfo(const uint8_t *sqeLocal, uint8_t sqeType, uint32_t addInfo, SqeInfo *info)
{
    static const std::unordered_map<uint8_t, void (*)(const uint8_t *, uint32_t, SqeInfo *)> funcMap = {
        { SqeType::NOTIFY_SQE, ParseNotifySqe },
        { SqeType::WRITE_VALUE_SQE, ParseWriteValueSqe },
        { SqeType::EVENT_SQE, ParseEventSqe },
        { SqeType::MEMCPY_ASYNC_SQE, ParseMemcpyAsyncSqe },
        { SqeType::CCORE_WAIT_START_SQE, ParseCcoreWaitStartSqe },
        { SqeType::CCORE_WRITE_VALUE_SQE, ParseCcoreWriteValueSqe },
        { SqeType::NOTIFY_SQE_V2, ParseNotifySqeV2 },
        { SqeType::WRITE_VALUE_SQE_V2, ParseWriteValueSqeV2 },
        { SqeType::EVENT_SQE_V2, ParseEventSqeV2 },
        { SqeType::MEMCPY_ASYNC_SQE_V2, ParseMemcpyAsyncSqeV2 },
        { SqeType::RDMA_DB_SEND_SQE, ParseWriteValueSqe },
        { SqeType::FLIP_PLACEHOLDER_SQE, ParseFlipPlaceHolderSqe}
    };
    auto it = funcMap.find(sqeType);
    if (it == funcMap.cend()) {
        HCCL_WARNING("sqetype:%u is unsupported", sqeType);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(info);
    (it->second)(sqeLocal, addInfo, info);
    info->valid = 1;
    return HCCL_SUCCESS;
}

HcclResult SqeContextUtils::ClearCurBuff(uint32_t streamid, uint32_t leftBound)
{
    CHK_PRT_RET((streamid >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[SqeContextUtils][ClearCurBuff]Invalid streamId[%u] >= %u", streamid, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    SqeContext *context = GetSqeContext();
    auto &buff = context->buffPtr[streamid];
    HCCL_INFO(
        "leftBound:%u, buff.sqeCnt:%u, buff.sqHead:%u, buff.sqTail:%u, buff.tailSqeIdx:%u, buff.tailSqeTaskId:%u",
        leftBound, buff.sqeCnt, buff.sqHead, buff.sqTail, buff.tailSqeIdx, buff.tailSqeTaskId);
    if (memset_s(buff.localBuff + leftBound * AC_SQE_SIZE, sizeof(buff.localBuff) - leftBound * AC_SQE_SIZE, 0,
        (buff.tailSqeIdx - leftBound) * AC_SQE_SIZE) != EOK) {
        return HCCL_E_MEMORY;
    }
    if (memset_s(buff.sqeType + leftBound, sizeof(buff.sqeType) - leftBound, 0, buff.tailSqeIdx - leftBound) != EOK) {
        return HCCL_E_MEMORY;
    }
    if (memset_s(buff.addInfo + leftBound, sizeof(buff.addInfo) - leftBound, 0, buff.tailSqeIdx - leftBound) != EOK) {
        return HCCL_E_MEMORY;
    }
    buff.sqeCnt = 0;
    buff.tailSqeIdx = 0;
    AicpuGetComContext()->profilingExtendInfo.lastSqeIdxs[streamid] = 0;
    return HCCL_SUCCESS;
}

HcclResult SqeContextUtils::ModifyBuffer(uint32_t streamid)
{
    CHK_PRT_RET((streamid >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[SqeContextUtils][ModifyBuffer]Invalid streamId[%u] >= %u", streamid, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    // 未下发的sqe移到前面
    SqeContext *context = GetSqeContext();
    auto &buff = context->buffPtr[streamid];
    uint32_t cnt = buff.sqeCnt;
    uint32_t leftSrc = buff.tailSqeIdx - buff.sqeCnt;
    HCCL_DEBUG("buff.sqeCnt:%d, buff.sqHead:%u, buff.sqTail:%u, buff.tailSqeIdx:%u, buff.tailSqeTaskId:%u", buff.sqeCnt,
        buff.sqHead, buff.sqTail, buff.tailSqeIdx, buff.tailSqeTaskId);
    if (memmove_s(buff.localBuff, sizeof(buff.localBuff), buff.localBuff + leftSrc * AC_SQE_SIZE, cnt * AC_SQE_SIZE) !=
        EOK) {
        return HCCL_E_MEMORY;
    }
    if (memmove_s(buff.sqeType, sizeof(buff.sqeType), buff.sqeType + leftSrc, cnt) != EOK) {
        return HCCL_E_MEMORY;
    }
    if (memmove_s(buff.addInfo, sizeof(buff.addInfo), buff.addInfo + leftSrc, cnt) != EOK) {
        return HCCL_E_MEMORY;
    }
    // 队列后面已经拷贝到rtsq上的sqe清除掉
    CHK_RET(ClearCurBuff(streamid, cnt));
    // 更新index和sqeCnt
    buff.tailSqeIdx = cnt;
    buff.sqeCnt = cnt;
    AicpuGetComContext()->profilingExtendInfo.lastSqeIdxs[streamid] = cnt;
    return HCCL_SUCCESS;
}

HcclResult SqeContextUtils::ClearLocalBuff()
{
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        CHK_RET(ClearCurBuff(i));
    }
    return HCCL_SUCCESS;
}

std::string SqeContextUtils::RtsqTaskTypeToStr(uint8_t type)
{
    switch(type) {
        case RT_STARS_SQE_TYPE_NOTIFY_WAIT:
            return "NOTIFY WAIT";
        case RT_STARS_SQE_TYPE_NOTIFY_RECORD:
            return "NOTIFY RECORD";
        case RT_STARS_SQE_TYPE_WRITE_VALUE:
            return "WRITE VALUE";
        case RT_STARS_SQE_TYPE_SDMA:
            return "SDMA";
        case RT_STARS_SQE_TYPE_COND:
            return "COND";
        default:
            return std::to_string(type);
    }
}

std::string SqeContextUtils::GetString(const SqeInfo &sqeInfo)
{
    std::stringstream ss;
    ss << "SqeInfo ";
    ss << "sqeIdx:" << sqeInfo.sqeHeadIdx << ",";
    ss << "type:" << RtsqTaskTypeToStr(sqeInfo.type) << ",";
    ss << "subType:" << static_cast<uint16_t>(sqeInfo.subType) << ",";
    ss << "streamId:" << sqeInfo.streamId << ",";
    ss << "taskId:" << sqeInfo.taskId << ",";
    ss << "notifyId:" << sqeInfo.notifyId << ",";
    ss << "eventId:" << sqeInfo.eventId << ",";
    ss << "partId:" << sqeInfo.partId << ",";
    ss << "length:" << sqeInfo.length << ",";
    ss << "condValue:" << sqeInfo.condValue << ",";
    ss << "isLast:" << static_cast<uint16_t>(sqeInfo.isLast) << ",";
    ss << "opCode:" << static_cast<uint16_t>(sqeInfo.opCode) << ",";
    ss << "sqeNum:" << static_cast<uint16_t>(sqeInfo.sqeNum) << ",";
    ss << "valid:" << static_cast<uint16_t>(sqeInfo.valid) << ",";
    ss << "addr1High:0x" << std::hex << sqeInfo.addr1High << ",";
    ss << "addr1Low:0x" << std::hex << sqeInfo.addr1Low << ",";
    ss << "addr2High:0x" << std::hex << sqeInfo.addr2High << ",";
    ss << "addr2Low:0x" << std::hex << sqeInfo.addr2Low << ".";
    return ss.str();
}