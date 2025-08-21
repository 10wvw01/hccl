/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/aicpu_rpc_serverv2.h"
#include <numeric>
#include "common/aicpu_hccl_common.h"
#include "utils/mc2_aicpu_utils.h"
#include "hccl_common.h"
#include "framework/aicpu_prof.h"
#include "framework/aicpu_hccl_process.h"
#include "securec.h"
#include "log_control.h"

static constexpr uint16_t TURN_LEFT_SHIFT_BIT = 16;

HcclResult AicpuRpcServerV2::Init(const HcclMC2WorkSpace &workspaceInfo, const Mc2InitTilingInner *tilingData)
{
    // 为提升效率，workspace 必须512 对齐
    u64 addr = workspaceInfo.workSpace;
    if (addr & 0x1ff) {
        addr = (addr & (~((uint64_t)0x1ff))) + 0x200;
    }
    Reset();
    blockNum_ = MC2AicpuUtils::GetBlockNum();
    CHK_PRT_RET(blockNum_ == 0U, HCCL_ERROR("Invalid block number."), HCCL_E_INTERNAL);
    HCCL_INFO("Align hcclmsgarea from %p to %p, block number %u, currrent block idx %u.",
        workspaceInfo.workSpace, addr, blockNum_, MC2AicpuUtils::GetBlockIdx());
    if (tilingData != nullptr && tilingData->queueNum > 0U) {
        totalQueueNum_ = tilingData->commBlockNum * tilingData->queueNum;
        CHK_PRT_RET(totalQueueNum_ > LOCAL_STREAM_MAX_NUM || blockNum_ > MAX_AICPU_BLOCK_DIM,
                    HCCL_ERROR("Invalid para, comm block %u, aicpu block %u, queue number %u.",
                               tilingData->commBlockNum, blockNum_, tilingData->queueNum), HCCL_E_INTERNAL);
        hcclMultiMsgArea_ = reinterpret_cast<HcclMsgAreaForMultiQue *>(addr);
        hcclMsgArea_ = nullptr;
    } else {
        hcclMsgArea_ = reinterpret_cast<HcclMsgArea *>(addr);
        hcclMultiMsgArea_ = nullptr;
        totalQueueNum_ = 0U;
    }
    turnNumAddr_ = addr + sizeof(HcclMsgArea);
    if (turnNumAddr_ + sizeof(u32) * TILING_TURN_MAX * AC_MAX_RANK_NUM_V2 >
        workspaceInfo.workSpace + workspaceInfo.workSpaceSize) {
        HCCL_ERROR("Turn number addr %#llx, space for turn number is %lu, the space after workspace %#llx will "
            "be overwrited.", turnNumAddr_, sizeof(u32) * TILING_TURN_MAX * AC_MAX_RANK_NUM_V2,
            workspaceInfo.workSpace + workspaceInfo.workSpaceSize);
        return HCCL_E_INTERNAL;
    }
    uint32_t *turnNums = reinterpret_cast<uint32_t *>(turnNumAddr_);
    std::iota(&turnNums[0], &turnNums[TILING_TURN_MAX * AC_MAX_RANK_NUM_V2], 0);
    tilingBaseAddr_ = reinterpret_cast<u64>(tilingData);
    return HCCL_SUCCESS;
}

void AicpuRpcServerV2::GetLocalQueueRange(u32 &start, u32 &end)
{
    if (blockNum_ == 0U || totalQueueNum_ == 0U) {
        start = end = 0U;
        return;
    }
    const u32 base = totalQueueNum_ / blockNum_;
    const u32 remainder = totalQueueNum_ % blockNum_;
    const u32 blockIdx = MC2AicpuUtils::GetBlockIdx();
    if (blockIdx < remainder) {
        start = blockIdx * base + blockIdx;
        end = start + base;
    } else {
        start = blockIdx * base + remainder;
        end = start + base - 1U;
    }
}

void AicpuRpcServerV2::Reset()
{
    (void)memset_s(msgPos_, sizeof(msgPos_), 0, sizeof(msgPos_));
    msgPosForKernel_ = 0;
    for (int8_t i = 0; i < MC2_HCCL_MAX_HANDLE_ID; i++) {
        handleIdMsgPosition_[i] = -1;
    }
    (void)memset_s(isFinalize_, sizeof(isFinalize_), 0, sizeof(isFinalize_));
    (void)memset_s(barrierFlags_, sizeof(barrierFlags_), 0, sizeof(barrierFlags_));
    (void)memset_s(barrierFinishCnt_, sizeof(barrierFinishCnt_), 0, sizeof(barrierFinishCnt_));
    const u64 ts = GetCurCpuTimestamp();
    for (u32 i = 0U; i < MAX_QUE_NUM; ++i) {
        prepareTime_[i] = ts;
    }
    eventPrintTurn_ = 1U;
}

void AicpuRpcServerV2::SetMsgHandlePos(uint32_t msgPos, HcclHandle handleId) {
    if (handleId >= MC2_HCCL_MAX_HANDLE_ID || handleId < 0) {
        return;
    }
    handleIdMsgPosition_[handleId] = msgPos;
}

int32_t AicpuRpcServerV2::GetMsgHandlePos(HcclHandle handleId) {
    if (handleId >= MC2_HCCL_MAX_HANDLE_ID || handleId < 0) {
        HCCL_ERROR("[GetMsgHandlePos] invalid handleId %d", handleId);
        return -1;
    }
    if (handleIdMsgPosition_[handleId] < 0) {
        HCCL_WARNING("[GetMsgHandlePos] invalid handleIdMsgPosition %d", handleIdMsgPosition_[handleId]);
        return -1;
    }
    return handleIdMsgPosition_[handleId];
}

bool AicpuRpcServerV2::IsPrintLog() const {
    return isPrintLog_;
}

bool AicpuRpcServerV2::GetIsFinalize(u32 queueId) {
    if (queueId < MAX_QUE_NUM) {
        return isFinalize_[queueId];
    }
    u32 start = 0U;
    u32 end = 0U;
    GetLocalQueueRange(start, end);
    for (u32 i = start; i <= end; ++i) {
        if (!isFinalize_[i]) {
            return false;
        }
    }
    return true;
}

void AicpuRpcServerV2::SetIsFinalize(u32 queueId, bool finalize) {
    isFinalize_[queueId] = finalize;
}

uint32_t AicpuRpcServerV2::GetMsgPos(u32 queueId) {
    return msgPos_[queueId];
}

void AicpuRpcServerV2::SetMsgPos(u32 queueId, u32 pos) {
    msgPos_[queueId] = pos;
}

HcclMsgExt* AicpuRpcServerV2::GetHcclMsgExtPtr() {
    return msgExt_.get();
}

HcclMsgArea* AicpuRpcServerV2::GetHcclMsgArea(void) {
    return hcclMsgArea_;
}

void AicpuRpcServerV2::GetMsgWorkSpace(HcclMsg *msgLists[], u32 &msgListCnt) {
    if (hcclMsgArea_ != nullptr) {
        msgListCnt = 1U;
        msgLists[0] = hcclMsgArea_->sendMsgList;
    } else {
        u32 start = 0U;
        u32 end = 0U;
        GetLocalQueueRange(start, end);
        msgListCnt = end - start + 1U;
        for (u32 i = 0U; i < msgListCnt; ++i) {
            msgLists[i] = hcclMultiMsgArea_->sendMsgList[i + start];
        }
    }
}

uint64_t AicpuRpcServerV2::GetFinishAddr(int32_t idx) {
    if (idx >= static_cast<int32_t>(AC_MSG_CNT) || hcclMsgArea_ == nullptr) {
        HCCL_ERROR("idx %d exceed AC_MSG_CNT or hcclMsgArea_ is not initialized.", idx);
        return 0;
    }
    return reinterpret_cast<uint64_t>(&(hcclMsgArea_->finishedTurnCnt[idx].cnt));
}

uint64_t AicpuRpcServerV2::GenXorForMsgExt(int32_t idx, u32 rankSize) {
    if (hcclMsgArea_ == nullptr) {
        return 0UL;
    }
    uint64_t xorVal = 0U;
    for (uint32_t i = 0U; i < rankSize; ++i) {
        xorVal ^= hcclMsgArea_->paramExtMsgList[idx].sendCounts[i];
        xorVal ^= hcclMsgArea_->paramExtMsgList[idx].sendOffset[i];
        xorVal ^= hcclMsgArea_->paramExtMsgList[idx].recvCounts[i];
        xorVal ^= hcclMsgArea_->paramExtMsgList[idx].recvOffset[i];
    }
    xorVal ^= hcclMsgArea_->paramExtMsgList[idx].valid;
    return xorVal;
}

uint64_t AicpuRpcServerV2::GetCommitareaAddr(int32_t idx) {
    if (idx >= static_cast<int32_t>(AC_MSG_CNT) || hcclMsgArea_ == nullptr) {
        HCCL_ERROR("idx %d exceed AC_MSG_CNT or hcclMsgArea_ is not initialized.", idx);
        return 0;
    }
    return reinterpret_cast<uint64_t>(&(hcclMsgArea_->commitTurnCnt[idx].cnt));
}

HcclResult AicpuRpcServerV2::AddFlipTask(HcclDispatcher dispatcherPtr, hccl::Stream *stream)
{
    if (!dfx::ProfilingManager::GetProfL0State()) {
        return HCCL_SUCCESS;
    }
    hccl::HcclSqeContext *sqeCtx = stream->GetSqeContextPtr();
    CHK_PTR_NULL(sqeCtx);
    hccl::SqeRingBuffer &buff = sqeCtx->buffer;
    // nextTaskId=0的时候下发PlaceHolder
    if (UNLIKELY(buff.tailSqeTaskId == 0 && buff.filpNum != 0)) {
        CHK_RET(AddRetryPreamble(dispatcherPtr, *stream));
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuRpcServerV2::AddCcoreWait(HcclDispatcher dispatcherPtr, u64 waitAddr, uint32_t turnNum,
                                          hccl::Stream *stream, bool isLast) // client commit wait
{
    uint8_t *sqeBuffer = nullptr;
    uint8_t *sqeTypeAddr = nullptr;
    uint8_t *sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;

    CHK_RET(AddFlipTask(dispatcherPtr, stream));
    CHK_RET(stream->GetNextSqeBufferAddr(sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
    const HcclComStreamInfo &streamInfo = stream->GetHcclStreamInfo();
    if (HcclCommProf::IsDebugModeEquals(MC2_DEBUG_COMMIT_TIMEOUT)) {
        uint32_t *turnNums = reinterpret_cast<uint32_t *>(turnNumAddr_);
        turnNums[turnNum] = 0xFF;
    }
    AddOneWaitStartSqe(streamInfo.actualStreamId, taskId, waitAddr, turnNumAddr_ + turnNum * sizeof(u32),
        isLast, reinterpret_cast<rtStarsCcoreWaitStartSqe_t *>(sqeBuffer), sqeTypeAddr);
    hccl::HcclSqeContext* sqeCtx = stream->GetSqeContextPtr();
    if (sqeCtx == nullptr) {
        HCCL_ERROR("AddCcoreWait sqeCtx is nullptr");
        return HCCL_E_INTERNAL;
    }
    sqeCtx->buffer.addInfo[taskId % hccl::HCCL_SQE_MAX_CNT] =
        ((turnNum << TURN_LEFT_SHIFT_BIT) + static_cast<uint32_t>(isLast));
    return HCCL_SUCCESS;
}

HcclResult AicpuRpcServerV2::AddCcoreNotify(HcclDispatcher dispatcherPtr, u64 recordAddr, uint32_t turnNum,
                                            hccl::Stream *stream) // client finish notify
{
    uint8_t *sqeBuffer = nullptr;
    uint8_t *sqeTypeAddr = nullptr;
    uint8_t *sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;

    CHK_RET(AddFlipTask(dispatcherPtr, stream));
    CHK_RET(stream->GetNextSqeBufferAddr(sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
    const HcclComStreamInfo &streamInfo = stream->GetHcclStreamInfo();
    if (HcclCommProf::IsDebugModeEquals(MC2_DEBUG_AICORE_WAIT_TIMEOUT)) {
        uint32_t *turnNums = reinterpret_cast<uint32_t *>(turnNumAddr_);
        turnNums[turnNum] = 0;
    }
    AddOneWriteValueStartSqe(streamInfo.actualStreamId, taskId, recordAddr,
        turnNumAddr_ + turnNum * sizeof(u32), reinterpret_cast<rtStarsCcoreWriteValueSqe_t *>(sqeBuffer),
        sqeTypeAddr);
    hccl::HcclSqeContext* sqeCtx = stream->GetSqeContextPtr();
    if (sqeCtx == nullptr) {
        HCCL_ERROR("AddCcoreNotify sqeCtx is nullptr");
        return HCCL_E_INTERNAL;
    }
    sqeCtx->buffer.addInfo[taskId % hccl::HCCL_SQE_MAX_CNT] = turnNum;
    return HCCL_SUCCESS;
}

uint64_t AicpuRpcServerV2::GetFinishAddrByHandleId(HcclHandle handleId)
{
    int32_t msgPos = GetMsgHandlePos(handleId);
    if (msgPos < 0) {
        return 0;
    }
    return GetFinishAddr(msgPos);
}

void AicpuRpcServerV2::SetMsgRepeatCnt(u8 repeatCnt)
{
    repeatCnt_[msgPos_[0U]] = (totalStep_ == 0U ? repeatCnt : repeatCnt * totalStep_);
}

int32_t AicpuRpcServerV2::GetMsgRepeatCnt(HcclHandle handleId)
{
    int32_t msgPos = GetMsgHandlePos(handleId);
    if (msgPos < 0) {
        return -1;
    }
    return repeatCnt_[msgPos];
}

HcclResult AicpuRpcServerV2::ProcessExpectPrepareMsg(uint8_t seqNum, uint8_t expectId)
{
    // 当前无翻转场景，只需考虑单个通信域提前Finializa场景
    if (seqNum == 0 && expectId > 0) {
        return HCCL_SUCCESS;
    }
    if (seqNum < expectId) {
        HCCL_ERROR("curMsg seqNum %d is smaller than expect %d ignore.", seqNum, expectId);
        return HCCL_E_INTERNAL;
    }
    if (totalQueueNum_ == 0U && seqNum > expectId) {
        HCCL_INFO("curMsg seqNum %d is bigger than expect %d ignore.", seqNum, expectId);
        return HCCL_E_UNAVAIL;
    }
    return HCCL_SUCCESS;
}

void AicpuRpcServerV2::SetNeedRetryFlag(bool needRetryFlag)
{
    needReProcess_ = needRetryFlag;
}

bool AicpuRpcServerV2::ReadValidMsgExtArea(int32_t idx, u32 rankSize)
{
#ifdef __aarch64__
        __asm__ __volatile__("dsb ld" : : : "memory");
#endif
#ifdef __amd64__
        __asm__ __volatile__("" : : : "memory");
#endif
    if (hcclMsgArea_ == nullptr || hcclMsgArea_->paramExtMsgList[idx].valid != static_cast<u64>(AC_MSG_VALID_MASK)) {
        return false;
    }
    uint64_t msgExtXorCheck = GenXorForMsgExt(idx, rankSize);
    static uint32_t msgExtXorCheckTurn = 0;
    if (msgExtXorCheckTurn % MC2_API_XORCHECK_PRINT_NUM == 0 &&
        msgExtXorCheck != hcclMsgArea_->paramExtMsgList[idx].xorCheck) {
        HCCL_WARNING("data is modified! modified_xor:%llu, origin_xor:%llu.", msgExtXorCheck,
            hcclMsgArea_->paramExtMsgList[idx].xorCheck);
        msgExtXorCheckTurn++;
        return false;
    }
    HCCL_INFO("hcclMsgArea xorCheck[%llu]", hcclMsgArea_->paramExtMsgList[idx].xorCheck);
    CHK_PRT_RET(memcpy_s(msgExt_->sendCounts, sizeof(uint64_t) * rankSize,
        hcclMsgArea_->paramExtMsgList[idx].sendCounts, sizeof(uint64_t) * rankSize) != EOK,
        HCCL_WARNING("memcpy_s sendCounts failed."), false);
    CHK_PRT_RET(memcpy_s(msgExt_->sendOffset, sizeof(uint64_t) * rankSize,
        hcclMsgArea_->paramExtMsgList[idx].sendOffset, sizeof(uint64_t) * rankSize) != EOK,
        HCCL_WARNING("memcpy_s sendOffset failed."), false);
    CHK_PRT_RET(memcpy_s(msgExt_->recvCounts, sizeof(uint64_t) * rankSize,
        hcclMsgArea_->paramExtMsgList[idx].recvCounts, sizeof(uint64_t) * rankSize) != EOK,
        HCCL_WARNING("memcpy_s recvCounts failed."), false);
    CHK_PRT_RET(memcpy_s(msgExt_->recvOffset, sizeof(uint64_t) * rankSize,
        hcclMsgArea_->paramExtMsgList[idx].recvOffset, sizeof(uint64_t) * rankSize) != EOK,
        HCCL_WARNING("memcpy_s recvOffset failed."), false);
    CHK_PRT_RET(memcpy_s(msgExt_->reserved, sizeof(uint64_t) * HCCL_MSG_EXT_CHECK_LEN,
        hcclMsgArea_->paramExtMsgList[idx].reserved, sizeof(uint64_t) * HCCL_MSG_EXT_CHECK_LEN) != EOK,
        HCCL_WARNING("memcpy_s reserved failed."), false);

#ifdef __aarch64__
        __asm__ __volatile__("dsb ld" : : : "memory");
#endif
#ifdef __amd64__
        __asm__ __volatile__("" : : : "memory");
#endif
    hcclMsgArea_->paramExtMsgList[idx].valid = static_cast<u64>(~AC_MSG_VALID_MASK);
#ifdef __aarch64__
        __asm__ __volatile__("dsb st" : : : "memory");
#endif
    HCCL_INFO("reset paramExtMsgList valid value %lu", hcclMsgArea_->paramExtMsgList[idx].valid);
    return true;
}

bool AicpuRpcServerV2::IsExceedLimit(AicpuComType commType, u32 rankSize)
{
    if (rankSize > AC_MAX_RANK_NUM_V2 && (commType == HCCL_CMD_ALLTOALLV || commType == HCCL_CMD_ALLTOALL)) {
        return true;
    }
    return false;
}

bool AicpuRpcServerV2::ReadValidMsg(HcclMsg *rMsg, HcclMsg *msg, bool needReProcess, uint32_t msgPos, u32 rankSize)
{
#ifdef __aarch64__
    __asm__ __volatile__("dsb ld" : : : "memory");
#endif
#ifdef __amd64__
    __asm__ __volatile__("" : : : "memory");
#endif
    // 重处理消息
    if (needReProcess) {
        *rMsg = *msg;
        return true;
    }
    if (msg->valid != AC_MSG_VALID_MASK) {
        return false;
    }
    memcpy_s(rMsg, sizeof(HcclMsg), msg, sizeof(HcclMsg));
    uint32_t msgXorCheck = MC2AicpuUtils::GenXor(rMsg);
    static uint32_t msgXorCheckTurn = 0;
    if (UNLIKELY(msgXorCheck != rMsg->xorCheck && msgXorCheckTurn % MC2_API_XORCHECK_PRINT_NUM == 0)) {
        rMsg->PrintMsg("Rcv Msg");
        msg->PrintMsg("Rcv Msg");
        HCCL_WARNING("data is modified! modified_xor:%u, origin_xor:%u.", msgXorCheck, rMsg->xorCheck);
        msgXorCheckTurn++;
        return false;
    }
    if (UNLIKELY(IsExceedLimit(rMsg->commType, rankSize))) {
        return false;
    }
    if (UNLIKELY(rMsg->commType == HCCL_CMD_ALLTOALLV && !ReadValidMsgExtArea(msgPos, rankSize))) {
        return false;
    }
    msg->valid = ~AC_MSG_VALID_MASK;
    if (UNLIKELY(HcclCommProf::IsDebugModeEquals(MC2_DEBUG_PREPARE_TIMEOUT) && (rMsg->commType != HCCL_CMD_FINALIZE))) {
        return false;
    }
    if (UNLIKELY(HcclCommProf::IsDebugModeEquals(MC2_DEBUG_FINALIZE_TIMEOUT) && (rMsg->commType == HCCL_CMD_FINALIZE))) {
        return false;
    }
    HCCL_INFO("reset valid value 0x%x", msg->valid);
    return true;
}

bool AicpuRpcServerV2::ReadAddrMsg(HcclMsg *hcclMsg, HcclMsg *msgList, u32 queueIdx, u32 msgPos, u32 rankSize)
{
    bool ret = ReadValidMsg(hcclMsg, &(msgList[msgPos]), needReProcess_, msgPos, rankSize);
    isPrintLog_ = false;
    if (LIKELY(ret)) {
        HCCL_DEBUG("read valid msg msgPos %u commType %u", msgPos, static_cast<uint32_t>(hcclMsg->commType));
        PrintMsg(hcclMsg, msgPos, rankSize);
        // Prepare 成功，打印耗时
        u64 prepareTime = GetCurCpuTimestamp();
        if (eventPrintTurn_ > 1) {
            HCCL_RUN_INFO("[AicpuRpcServerV2][ReadAddrMsg] Read HcclMsg[%u] cost %llu",
                          msgPos, prepareTime - prepareTime_[queueIdx]);
        } else {
            HCCL_INFO("[AicpuRpcServerV2][ReadAddrMsg] Read HcclMsg[%u] cost %llu",
                      msgPos, prepareTime - prepareTime_[queueIdx]);
        }
        prepareTime_[queueIdx] = prepareTime;
        eventPrintTurn_ = 1;
    } else if (GetCurCpuTimestamp() - prepareTime_[queueIdx] >
               static_cast<unsigned long long>(NSEC_PER_SEC) * MC2_API_MSG_TIMEOUT * eventPrintTurn_) {
        // Prepare 等待 20s
        HCCL_RUN_WARNING("[AicpuRpcServerV2][ReadAddrMsg] ReadValidMsg[%u] timeout %lus", msgPos,
                         MC2_API_MSG_TIMEOUT * eventPrintTurn_);
        eventPrintTurn_ *= 2;  // 2 is print event log times
        LogControl logControl(false, true);
        PrintAllHcclMsgArea(rankSize);
        isPrintLog_ = true;
    }
    return ret;
}

// reset消息区msgPos的commitTurnId
HcclResult AicpuRpcServerV2::ResetCommitTaskAdd(HcclDispatcher dispatcherPtr, hccl::Stream *stream)
{
    // reset函数复用 AddCcoreWait,turnNum保证条件算子恒成立
    for (uint32_t i = 0; i < static_cast<uint32_t>(msgPos_[0U]); i++) {
        uint64_t waitAddr = GetCommitareaAddr(i);
        CHK_RET(AddCcoreWait(dispatcherPtr, waitAddr, 0, stream, true));
    }
    return HCCL_SUCCESS;
}

void AicpuRpcServerV2::SetMsgPosForKernel(uint32_t msgPos)
{
    msgPosForKernel_ = msgPos;
}
uint32_t AicpuRpcServerV2::GetMsgPosForKernel(void)
{
    return msgPosForKernel_;
}

void AicpuRpcServerV2::WriteFinishWhenAllFinalize()
{
    if (hcclMsgArea_ == nullptr) {
        return;
    }
    uint32_t msgPos = GetMsgPos();
    hcclMsgArea_->finishedTurnCnt[msgPos].cnt = MC2_API_TurnCnt_VALID;  // 1234567899999999999 用于校验的非法值
    HCCL_INFO("Post finishedTurnCnt[%u].cnt = %lu.", msgPos, hcclMsgArea_->finishedTurnCnt[msgPos].cnt);
    #ifdef __aarch64__
    __asm__ __volatile__("dsb st" : : : "memory");
    #endif
}

void AicpuRpcServerV2::WriteRestartFlag()
{
    if (hcclMsgArea_ != nullptr) {
        for (uint32_t i = 0; i < AC_MSG_CNT; i++) {
            hcclMsgArea_->sendMsgList[i].valid = ~AC_MSG_VALID_MASK;
            hcclMsgArea_->commitTurnCnt[i].cnt = 0;
            hcclMsgArea_->finishedTurnCnt[i].cnt = 0;
        }
        hcclMsgArea_->controlMsg.restart = 1;
    } else {
        for (uint32_t i = 0; i < MAX_QUE_NUM; i++) {
            for (uint32_t j = 0; j < AC_MSG_CNT; j++) {
                hcclMultiMsgArea_->sendMsgList[i][j].valid = ~AC_MSG_VALID_MASK;
            }
        }
        hcclMultiMsgArea_->controlMsg.restart = 1;
    }
#ifdef __aarch64__
    __asm__ __volatile__("dsb st" : : : "memory");
#endif
}

void AicpuRpcServerV2::PrintAllHcclMsgArea(u32 rankSize)
{
    if (hcclMsgArea_ != nullptr) {
        MC2AicpuUtils::PrintAllHcclMsgAreaError(hcclMsgArea_, rankSize);
    } else if (MC2AicpuUtils::GetBlockIdx() == 0U) {
        MC2AicpuUtils::PrintAllHcclMsgAreaError(hcclMultiMsgArea_, rankSize);
    }
}

void AicpuRpcServerV2::PrintMsg(HcclMsg *hcclMsg, uint32_t msgPos, u32 rankSize)
{
    if (HcclCommProf::IsDebugModeEquals(MC2_DEBUG_PRINT_MSG)) {
        hcclMsg->PrintMsg("ReadAddrMsg msgPos " + std::to_string(msgPos), true);
        if (hcclMsgArea_ != nullptr) {
            MC2AicpuUtils::PrintAllHcclMsgAreaEvent(hcclMsgArea_, rankSize);
        } else {
            MC2AicpuUtils::PrintAllHcclMsgAreaEvent(hcclMultiMsgArea_, rankSize);
        }
    } else {
        hcclMsg->PrintMsg("ReadAddrMsg msgPos " + std::to_string(msgPos));
    }
    if (HcclCommProf::IsDebugModeEquals(MC2_DEBUG_PRINT_BUFF)) {
        if (hcclMsg->version >= MC2_MSG_VERSION_FOR_TILING_API) {
            MC2AicpuUtils::PrintApiBuffer(reinterpret_cast<void *>(reinterpret_cast<HcclMsgV1 *>(hcclMsg)->sendBuffer),
                                          DataUnitSize(reinterpret_cast<HcclMsgV1 *>(hcclMsg)->hcclDataType) *
                                          reinterpret_cast<HcclMsgV1 *>(hcclMsg)->dataCnt,
                                          "before communication sendBuffer "+ std::to_string(msgPos));
        } else {
            MC2AicpuUtils::PrintApiBuffer(reinterpret_cast<void *>(hcclMsg->sendBuffer),
                                          DataUnitSize(hcclMsg->hcclDataType) * hcclMsg->dataCnt,
                                          "before communication sendBuffer "+ std::to_string(msgPos));
        }
    }
}

void AicpuRpcServerV2::PrintAllHcclMsgAreaData()
{
    if (hcclMsgArea_ == nullptr) {
        return;
    }
    for (uint32_t i = 0; i < AC_MSG_CNT; ++i) {
        HcclMsg *hcclMsg = &(hcclMsgArea_->sendMsgList[i]);
        if (hcclMsg->commType == HCCL_CMD_INVALID || hcclMsg->commType >= HCCL_CMD_MAX) {
            continue;
        }
        if (hcclMsg->version >= MC2_MSG_VERSION_FOR_TILING_API) {
            MC2AicpuUtils::PrintApiBuffer(reinterpret_cast<void *>(reinterpret_cast<HcclMsgV1 *>(hcclMsg)->sendBuffer),
                                          DataUnitSize(reinterpret_cast<HcclMsgV1 *>(hcclMsg)->hcclDataType) *
                                          reinterpret_cast<HcclMsgV1 *>(hcclMsg)->dataCnt,
                                          "after communication sendBuffer "+ std::to_string(i));
            MC2AicpuUtils::PrintApiBuffer(reinterpret_cast<void *>(reinterpret_cast<HcclMsgV1 *>(hcclMsg)->recvBuffer),
                                          DataUnitSize(reinterpret_cast<HcclMsgV1 *>(hcclMsg)->hcclDataType) *
                                          reinterpret_cast<HcclMsgV1 *>(hcclMsg)->dataCnt,
                                          "after communication recvBuffer "+ std::to_string(i));
        } else {
            MC2AicpuUtils::PrintApiBuffer(reinterpret_cast<void *>(hcclMsg->sendBuffer),
                                          DataUnitSize(hcclMsg->hcclDataType) * hcclMsg->dataCnt,
                                          "after communication sendBuffer "+ std::to_string(i));
            MC2AicpuUtils::PrintApiBuffer(reinterpret_cast<void *>(hcclMsg->recvBuffer),
                                          DataUnitSize(hcclMsg->hcclDataType) * hcclMsg->dataCnt,
                                          "after communication recvBuffer "+ std::to_string(i));
        }
    }
}