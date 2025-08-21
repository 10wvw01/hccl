/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/aicpu_rpc_server.h"
#include "algorithm/aicpu_hccl_dispatcher.h"
#include "common/aicpu_hccl_common.h"
#include "utils/mc2_aicpu_utils.h"
#include "utils/aicpu_hdc_utils.h"
#include "securec.h"
#include "log_control.h"

void AicpuRpcServer::Init(u64 workSpaceAddr, uint32_t notifyOff, uint16_t notifyBeginCnt, KFCTask *taskParam)
{
    tilingData_ = reinterpret_cast<HcclKFCTilingData *>(taskParam->tilingData);

    // 为提升效率，workspace 必须512 对齐
    u64 addr = workSpaceAddr;
    HCCL_DEBUG("AicpuRpcServer::Init addr:%u", addr);

    // 规划每个AIV的消息接收地址， 总计使用： MAX_AIV_NUM * AC_MSG_CNT
    if (static_cast<TASK_PREPARE_POSITION>(tilingData_->preparePosition) == TASK_PREPARE_HOST) {
        msgBody_ = reinterpret_cast<RpcMsgBody *>(addr);
        msgBody_->msgRcvArea[0][0].res[0] = 0U;
        msgSndWorkArea_ = reinterpret_cast<AivAicpuOpParam *>(workSpaceAddr + notifyOff);
        msgRcvRspArea_ = reinterpret_cast<AivAicpuOpParam *>(workSpaceAddr + notifyOff + notifyBeginCnt * sizeof(u8) *
                        AC_SQE_SIZE);
        for (uint32_t i = 0; i < AC_MAX_AIV; i++) {
            rcvMsgPos_[i] = 0;
            sndMsgPos_[i] = 0;
            aivState_[i] = 0;
        }
    } else {
        hcclMsgArea_ = reinterpret_cast<HcclMsgArea *>(addr);
    }

    genTaskNum_ = 0;
    genTaskParam_ = taskParam;
}

void AicpuRpcServer::Init(u64 workSpaceAddr)
{
    hcclMsgArea_ = reinterpret_cast<HcclMsgArea *>(workSpaceAddr);
    genTaskNum_ = 0;
    genTaskParam_ = nullptr;
    tilingData_ = nullptr;
}

bool AicpuRpcServer::PostMsg(uint32_t curTurnCnt) const
{
    AivAicpuOpParam *msg = msgRcvRspArea_;
    msg->rcvCnt = curTurnCnt;
    msg->valid = AC_MSG_VALID_MASK;
    msg->PrintMsg("Snd");

#ifdef __aarch64__
    __asm__ __volatile__("dsb st" : : : "memory");
#endif

    return true;
}

void AicpuRpcServer::WriteFinishWhenAllFinalize(uint32_t msgPos)
{
    hcclMsgArea_->finishedTurnCnt[msgPos].cnt = MC2_API_TurnCnt_VALID;  // 1234567899999999999 用于校验的非法值
    HCCL_INFO("Post finishedTurnCnt[%u].cnt = %lu.", msgPos, hcclMsgArea_->finishedTurnCnt[msgPos].cnt);
    #ifdef __aarch64__
    __asm__ __volatile__("dsb st" : : : "memory");
    #endif
}

void AicpuRpcServer::WriteTurnCnt(uint32_t msgPos)
{
    hcclMsgArea_->commitTurnCnt[msgPos].cnt = 0;
    hcclMsgArea_->finishedTurnCnt[msgPos].cnt += 1;
    HCCL_INFO("Post position %u commitTurnCnt cnt = %lu, finishedTurnCnt cnt = %lu.",
        msgPos, hcclMsgArea_->commitTurnCnt[msgPos].cnt, hcclMsgArea_->finishedTurnCnt[msgPos].cnt);
    #ifdef __aarch64__
    __asm__ __volatile__("dsb st" : : : "memory");
    #endif
}

inline std::string AicpuRpcServer::GetMsgTypeString(uint8_t msgType)
{
    if (msgType == RANK_ADDR) {
        return "Addr";
    }
    if (msgType == RANK_WORK) {
        return "work";
    }
    if (msgType == RANK_ADD_AND_WORK) {
        return "Addr&work";
    }
    if (msgType == RANK_TAIL_TIME) {
        return "EndNotify";
    }
    return "unkown";
}

#pragma GCC push_options
#pragma GCC optimize("O0")
bool AicpuRpcServer::RcvMsg(AivAicpuOpParam *rMsg, uint32_t aivID, uint8_t msgType)
{
    if (rMsg == nullptr) {
        return false;
    }
    auto pos = rcvMsgPos_[aivID];
    auto msg = &msgBody_->msgRcvArea[aivID][pos];
    if (NeedAutoGenMsg()) {
        HCCL_DEBUG("RcvMsg by task param:%d/%d", genTaskNum_ + 1, tilingData_->turnNum);
        GenMsgByTaskParam(rMsg);
        genTaskNum_++;
    } else {
        HCCL_DEBUG("RcvMsg on msg:%p, aivId:%d, pos:%d", msg, aivID, pos);

#ifdef __aarch64__
        __asm__ __volatile__("dsb ld" : : : "memory");
#endif
#ifdef __amd64__
        __asm__ __volatile__("" : : : "memory");
#endif

        do {
        } while (!ReadValidMsg(rMsg, msg, msgType, false));
    }

    msg->PrintMsg(GetMsgTypeString(msgType));

    if (rMsg->isLast) {
        aivState_[aivID] = 1;
    }

    pos = (pos + 1) % AC_MSG_CNT;
    rcvMsgPos_[aivID] = pos;

    return true;
}

template <typename T>
bool AicpuRpcServer::ReadValidMsg(T *rMsg, T *msg, uint8_t msgType, bool reset)
{
    (void)msgType;
    if (msg->valid != AC_MSG_VALID_MASK) {
        return false;
    }
    *rMsg = *msg;
    if (reset) {
        msg->valid = ~AC_MSG_VALID_MASK;
    }
#ifdef __aarch64__
        __asm__ __volatile__("dsb st" : : : "memory");
#endif
    HCCL_INFO("reset valid value %u", msg->valid);
    return true;
}

bool AicpuRpcServer::ReadApiValidMsg(HcclMsg *rMsg, HcclMsg *msg, bool reset)
{
#ifdef __aarch64__
        __asm__ __volatile__("dsb ld" : : : "memory");
#endif
#ifdef __amd64__
        __asm__ __volatile__("" : : : "memory");
#endif
    if (msg->valid != AC_MSG_VALID_MASK) {
        CHK_RET(MC2AicpuUtils::TraceProfSubmit());
        return false;
    }
    memcpy_s(rMsg, sizeof(HcclMsg), msg, sizeof(HcclMsg));
    uint32_t modifiedXor = MC2AicpuUtils::GenXor(rMsg);
    static uint32_t xorCheckNum = 0;
    if (xorCheckNum % MC2_API_XORCHECK_PRINT_NUM == 0 && modifiedXor != rMsg->xorCheck) {
        HCCL_WARNING("data is modified! rMsg:%s msg:%s, modifiedXor:%u, origin_xor:%u.",
            rMsg->MsgToStr().c_str(), msg->MsgToStr().c_str(), modifiedXor, rMsg->xorCheck);
        xorCheckNum++;
        return false;
    }
#ifdef __aarch64__
        __asm__ __volatile__("dsb ld" : : : "memory");
#endif
#ifdef __amd64__
        __asm__ __volatile__("" : : : "memory");
#endif

    if (memcmp(rMsg, msg, sizeof(HcclMsg)) != 0) {
        HCCL_ERROR("Check msg equal fail, rMsg:%s msg:%s", rMsg->MsgToStr().c_str(), msg->MsgToStr().c_str());
        return false;
    }

    if (reset) {
        msg->valid = ~AC_MSG_VALID_MASK;
    }
    auto ctx = AicpuGetComContext();
    if ((ctx->debugMode == MC2_DEBUG_PREPARE_TIMEOUT) && (rMsg->commType != HCCL_CMD_FINALIZE)) {
        return false;
    }

    if ((ctx->debugMode == MC2_DEBUG_FINALIZE_TIMEOUT) && (rMsg->commType == HCCL_CMD_FINALIZE)) {
        return false;
    }
#ifdef __aarch64__
        __asm__ __volatile__("dsb st" : : : "memory");
#endif
    HCCL_INFO("reset valid value %u", msg->valid);
    return true;
}

#pragma GCC pop_options

bool AicpuRpcServer::ReadAddrMsg(AivAicpuOpParam *rMsg, uint32_t aivID)
{
    (void)aivID;
    GenMsgByTaskParam(rMsg);
    return true;
}

bool AicpuRpcServer::ReadWorkMsg(AivAicpuOpParam *rMsg, uint32_t aivID, uint32_t curTurnCnt)
{
    (void)aivID;
    return ReadValidMsg(rMsg, msgSndWorkArea_, RANK_WORK, false) && (curTurnCnt <= rMsg->sendCnt);
}

bool AicpuRpcServer::CheckRcvWorkMsg(AivAicpuOpParam *rMsg, uint32_t aivID, uint32_t curTurnCnt)
{
    (void)aivID;
#ifdef __aarch64__
        __asm__ __volatile__("dsb ld" : : : "memory");
#endif
#ifdef __amd64__
        __asm__ __volatile__("" : : : "memory");
#endif
    HCCL_INFO("CheckRcvWorkMsg, curTurnCnt %u", curTurnCnt);
    rMsg->PrintMsg(GetMsgTypeString(RANK_MSG_TYPE::RANK_WORK));
    uint32_t loopCnt = 0;
    u64 startUsec = GetCurCpuTimestamp();
    do {
        /************调测使用，正式交付的时候删除************/
        if (loopCnt > 10000) {  // 10000 is max loop cnt
            loopCnt = 0;
            // 打印所有流的sq状态
            HCCL_INFO("current states %s Msg %p[sendCnt:%d, valid:%d, curTurnCnt %u",
                GetMsgTypeString(RANK_MSG_TYPE::RANK_WORK).c_str(), msgSndWorkArea_, msgSndWorkArea_->sendCnt,
                msgSndWorkArea_->valid, curTurnCnt);
        }

        if (GetCurCpuTimestamp() - startUsec > static_cast<unsigned long long>(NSEC_PER_SEC) * 6) {  // 6 is over time
            HCCL_ERROR("ReadValidMsg timeout 6s... ");
            break;
        }
        loopCnt++;
        /************************************************/
    } while (!(ReadValidMsg(rMsg, msgSndWorkArea_, RANK_MSG_TYPE::RANK_WORK, false) && (curTurnCnt <= rMsg->sendCnt)));

    rMsg->PrintMsg(GetMsgTypeString(RANK_MSG_TYPE::RANK_WORK));
    return  true;
}

bool AicpuRpcServer::CheckRcvAddrMsg(AivAicpuOpParam *rMsg, uint32_t aivID)
{
    HCCL_INFO("RcvMsg by task param %u", tilingData_->turnNum);
    GenMsgByTaskParam(rMsg);
    genTaskNum_++;

    if (rMsg->isLast != 0) {
        aivState_[aivID] = 1;
    }
    HCCL_INFO("CheckRcvAddrMsg, genTaskNum %u", genTaskNum_);
    rMsg->PrintMsg(GetMsgTypeString(RANK_MSG_TYPE::RANK_ADDR));

    return true;
}

bool AicpuRpcServer::CheckRcvAddrMsg(HcclMsg *hcclMsg, uint32_t msgPos)
{
    if (!ReadApiValidMsg(hcclMsg, &(hcclMsgArea_->sendMsgList[msgPos]), false)) {
        return false;
    }
    hcclMsg->PrintMsg("CheckRcvAddrMsg hcclMsg");
    return true;
}

bool AicpuRpcServer::ReadAddrMsg(HcclMsg *hcclMsg, uint32_t msgPos)
{
    if (hcclMsg == nullptr) {
        HCCL_ERROR("Msg is nullptr");
        return false;
    }
    auto ctx = AicpuGetComContext();
    if (ctx == nullptr) {
        HCCL_ERROR("Get ctx is nullptr");
        return false;
    }
    uint32_t loopCnt = 0;
    u64 startUsec = GetCurCpuTimestamp();
    bool timeoutFlag = false;
    u8 eventPrintTurn = 1;   // 标记 Event日志的打印
    do {
        if (ctx->dfxExtendInfo.pollStatus == dfx::PollStatus::kStopAsException) {
            HCCL_ERROR("hccl aicpu exec failed, for exception.");
            return false;
        }

        KfcCommand cmd = KfcCommand::kNone;
        CHK_RET(AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd));
        if ((cmd == KfcCommand::NsStopLaunch) && (ctx->commOpenStatus) && (!ctx->endStopLaunch)) {
            HCCL_WARNING("N second stop Launch for recv stop launch cmd.");
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), true);
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), true);
            return false;
        }
        if (loopCnt > 10000) {  // 10000 is max loop cnt
            loopCnt = 0;
            // 打印所有流的sq状态
            HCCL_INFO("current states %s Msg %p, msgPos %u",
                GetMsgTypeString(RANK_MSG_TYPE::RANK_ADDR).c_str(), &(hcclMsgArea_->sendMsgList[msgPos]), msgPos);
        }
        if (GetCurCpuTimestamp() - startUsec >
            static_cast<unsigned long long>(NSEC_PER_SEC) * MC2_API_MSG_TIMEOUT * eventPrintTurn) {
            HCCL_RUN_WARNING("[AicpuRpcServer][ReadAddrMsg] ReadValidMsg[%u] timeout %lus",
                             msgPos, MC2_API_MSG_TIMEOUT * eventPrintTurn);
            LogControl logControl(false, true);
            PrintAllHcclMsgArea();
            TaskOrchestrator::PrintTimeOutSqInfo(ctx, MC2_API_MSG_TIMEOUT * eventPrintTurn);
            eventPrintTurn *= 2;    // 2 is print event log times
        }

#ifdef CCL_LLT
        if (!timeoutFlag && msgPos != 0 &&
            GetCurCpuTimestamp() - startUsec > static_cast<unsigned long long>(NSEC_PER_SEC)) {
#else
        if (!timeoutFlag && msgPos != 0 &&
            GetCurCpuTimestamp() - startUsec > static_cast<unsigned long long>(NSEC_PER_SEC) * dfx::kKfcTimeOut) {
#endif
            HCCL_ERROR("ReadValidMsg timeout %lus... ", dfx::kKfcTimeOut);
            PrintAllHcclMsgArea();
            TaskOrchestrator::PrintTimeOutSqInfo(ctx, dfx::kKfcTimeOut);
            timeoutFlag = true;
            return false;
        }

#ifdef CCL_LLT
        if (GetCurCpuTimestamp() - startUsec > static_cast<unsigned long long>(NSEC_PER_SEC)) {
#else
        if (GetCurCpuTimestamp() - startUsec > static_cast<unsigned long long>(NSEC_PER_SEC) * dfx::kKfcTimeOut) {
#endif
            HCCL_ERROR("ReadValidMsg timeout %ds... ", dfx::kKfcTimeOut);
            return false;
        }
        loopCnt++;
    } while (!(ReadApiValidMsg(hcclMsg, &(hcclMsgArea_->sendMsgList[msgPos]), true)));

    // 打印读消息的时间
    if (eventPrintTurn > 1) {
        HCCL_RUN_INFO("[AicpuRpcServer][ReadAddrMsg] Read HcclMsg[%u] cost %llu",
                      msgPos, GetCurCpuTimestamp() - startUsec);
    } else {
        HCCL_INFO("[AicpuRpcServer][ReadAddrMsg] Read HcclMsg[%u] cost %llu", msgPos, GetCurCpuTimestamp() - startUsec);
    }

    PrintMsg(hcclMsg, msgPos);
    return true;
}

void AicpuRpcServer::HcclMsg2AicAicpuOpParam(CommonHcclMsg *hcclMsg, AivAicpuOpParam *opMsg)
{
    Mc2CcTilingInner *innerTiling = reinterpret_cast<Mc2CcTilingInner *>(hcclMsg->ccOpTilingData);
    AicpuComContext *ctx = AicpuGetComContext();
    if (tilingData_ == nullptr && innerTiling == nullptr) {
        HCCL_ERROR("Invalid tiling data, please check opType or other fields.");
        return;
    }
    opMsg->commType = hcclMsg->commType;
    opMsg->opType = hcclMsg->opType;
    opMsg->sendBuffer = hcclMsg->sendBuffer;
    opMsg->recvBuffer = hcclMsg->recvBuffer;
    opMsg->winOffset = 0U;
    opMsg->count = hcclMsg->commType == HCCL_CMD_REDUCE_SCATTER ? hcclMsg->dataCnt * ctx->rankNum : hcclMsg->dataCnt;
    opMsg->hcclDataType = hcclMsg->hcclDataType;
    opMsg->isLast = 0U;
    opMsg->sendCnt = 0x34;
    opMsg->rcvCnt = 0x12;
    opMsg->valid = hcclMsg->valid;
    opMsg->everyTurnRsp = hcclMsg->everyTurnRsp;
    opMsg->everyTurnWait = hcclMsg->everyTurnWait;
    opMsg->strideLen = static_cast<u64>(hcclMsg->strideCount);

    if (tilingData_ != nullptr) {
        opMsg->funID = tilingData_->funID;
        opMsg->totalTurnCnt = tilingData_->turnNum;
        opMsg->useBufferType = tilingData_->useBufferType;
    } else {
        opMsg->useBufferType = innerTiling->skipBufferWindowCopy;
        ctx->skipLocalDataCopy = innerTiling->skipLocalRankCopy;
    }
    if (ctx->gatherOut == 0U && opMsg->commType == HCCL_CMD_ALLGATHER) {
        ctx->gatherOut = opMsg->recvBuffer;
    }
    HCCL_DEBUG("useBufferType:%u, recvBuffer[%#llx], gatherOut[%#llx], commType[%d].",
              opMsg->useBufferType, opMsg->recvBuffer, ctx->gatherOut, opMsg->commType);
    // 不需要gather out，就不需要拷贝本卡数据。需要gather out时，如果是aic负责拷贝本卡数据，reduceOp设为1即可
    if (opMsg->commType == HCCL_CMD_ALLGATHER && opMsg->opType != HCCL_REDUCE_PROD) {
        opMsg->opType = ctx->skipLocalDataCopy ? HCCL_REDUCE_PROD : HCCL_REDUCE_SUM;
    }
    opMsg->PrintMsg("CheckRcvAddrMsg opMsg");
}

bool AicpuRpcServer::CheckAivIsEnd(uint32_t aivId) { return (aivState_[aivId] == 1); }

bool AicpuRpcServer::NeedAutoGenMsg() { return genTaskParam_ != nullptr && genTaskNum_ < tilingData_->turnNum; }

bool AicpuRpcServer::GenMsgIsLastMsg() { return (genTaskNum_ + 1 == tilingData_->turnNum); }

uint8_t AicpuRpcServer::GetWaitPolicy() { return (tilingData_->waitPolicy); }

uint8_t AicpuRpcServer::GetTaskType() const { return (tilingData_->taskType); }

uint8_t AicpuRpcServer::GetRspPolicy() { return (tilingData_->rspPolicy); }

uint8_t AicpuRpcServer::GetGenTaskNum() { return genTaskNum_; }

TASK_PREPARE_POSITION AicpuRpcServer::GetPreparePosition() const
{
    return static_cast<TASK_PREPARE_POSITION>(tilingData_->preparePosition);
}

void AicpuRpcServer::GenMsgByTaskParam(AivAicpuOpParam *outMsg)
{
    outMsg->commType = static_cast<AicpuComType>(tilingData_->commType);
    outMsg->opType = static_cast<HcclReduceOp>(tilingData_->reduceOp);

    switch (outMsg->commType) {
        case HCCL_CMD_ALLGATHER: {
            CalcAllgatherBuffer(outMsg);

            // 不需要gather out，就不需要拷贝本卡数据。需要gather out时，如果是aic负责拷贝本卡数据，reduceOp设为1即可
            if (outMsg->opType != HCCL_REDUCE_PROD){
                outMsg->opType = tilingData_->hasCommOut? HCCL_REDUCE_SUM : HCCL_REDUCE_PROD;
            }
            break;
        }
        case HCCL_CMD_ALLREDUCE: {
            CalcAllreduceBuffer(outMsg);
            break;
        }
        case HCCL_CMD_REDUCE_SCATTER: {
            CalcReduceScatterBuffer(outMsg);
            break;
        }
        default: {
            HCCL_ERROR("commType [%d] is not supported.", outMsg->commType);
            break;
        }
    }

    outMsg->count = genTaskNum_ < tilingData_->turnNum - tilingData_->tailNum ?
        tilingData_->sendCnt : tilingData_->tailSendCnt;
    outMsg->hcclDataType = static_cast<HcclDataType>(tilingData_->dataType);

    outMsg->isLast = GenMsgIsLastMsg() ? 1 : 0;
    outMsg->funID = tilingData_->funID;
    outMsg->totalTurnCnt = tilingData_->turnNum;
    outMsg->sendCnt = 0x34;
    outMsg->rcvCnt = 0x12;
    outMsg->valid = AC_MSG_VALID_MASK;
    outMsg->everyTurnRsp = tilingData_->rspPolicy;
    outMsg->everyTurnWait = tilingData_->waitPolicy;
    outMsg->strideLen = static_cast<u64>(tilingData_->stride);
    outMsg->useBufferType = tilingData_->useBufferType;
}

u64 AicpuRpcServer::GetSendOff() const
{
    if (tilingData_->commAlg == COMM_ALG_DOUBLE_RING || tilingData_->commAlg == COMM_ALG_SWITCH_WING) {
        return 0UL;
    }
    const u64 headNum = tilingData_->turnNum - tilingData_->tailNum;
    if (genTaskNum_ <= headNum) {
        return genTaskNum_ * tilingData_->sendOff;
    }
    return headNum * tilingData_->sendOff + (genTaskNum_ - headNum) * tilingData_->tailSendOff;
}

u64 AicpuRpcServer::GetRecvOff() const
{
    if (tilingData_->commAlg == COMM_ALG_DOUBLE_RING || tilingData_->commAlg == COMM_ALG_SWITCH_WING) {
        return 0UL;
    }
    const u64 headNum = tilingData_->turnNum - tilingData_->tailNum;
    if (genTaskNum_ <= headNum) {
        return genTaskNum_ * tilingData_->recvOff;
    }
    return headNum * tilingData_->recvOff + (genTaskNum_ - headNum) * tilingData_->tailRecvOff;
}

void AicpuRpcServer::CalcAllgatherBuffer(AivAicpuOpParam *outMsg) const
{
    const auto recvOff = GetRecvOff();
    outMsg->sendBuffer = genTaskParam_->inputA + GetSendOff();
    if (!tilingData_->useBufferType) {
        outMsg->recvBuffer = genTaskParam_->commOut + recvOff;
    } else {
        outMsg->recvBuffer = genTaskParam_->workSpace + tilingData_->workspaceOff + recvOff;
    }
}

void AicpuRpcServer::CalcAllreduceBuffer(AivAicpuOpParam *outMsg) const
{
    const auto sendOff = GetSendOff();
    const auto recvOff = GetRecvOff();
    u64 sendBuffer = 0UL;
    u64 recvBuffer = 0UL;
    if (tilingData_->commOrder == 0) { // 通信在前 或 aicpu通信展开(单allreduce)
        sendBuffer = genTaskParam_->inputA + sendOff;
        if (!tilingData_->useBufferType) {
            recvBuffer = genTaskParam_->commOut + recvOff;
        } else {
            recvBuffer = genTaskParam_->workSpace + tilingData_->workspaceOff + recvOff;
        }
    } else {
        sendBuffer = genTaskParam_->outputC + sendOff;
        recvBuffer = genTaskParam_->outputC + recvOff;
    }

    outMsg->sendBuffer = sendBuffer;
    outMsg->recvBuffer = recvBuffer;
    outMsg->winOffset = sendOff;
}

void AicpuRpcServer::CalcReduceScatterBuffer(AivAicpuOpParam *outMsg) const
{
    const auto sendOff = GetSendOff();
    const auto recvOff = GetRecvOff();
    u64 sendBuffer = 0UL;
    u64 recvBuffer = 0UL;
    if (tilingData_->commOrder == 0) { // aicpu通信展开(单reducescatter)
        sendBuffer = genTaskParam_->inputA + sendOff;
        if (!tilingData_->useBufferType) {
            recvBuffer = genTaskParam_->commOut + recvOff;
        } else {
            recvBuffer = genTaskParam_->workSpace + tilingData_->workspaceOff + recvOff;
        }
    } else {
        sendBuffer = genTaskParam_->workSpace + tilingData_->workspaceOff + sendOff;
        recvBuffer = genTaskParam_->outputC + recvOff;
    }
    outMsg->sendBuffer = sendBuffer;
    outMsg->recvBuffer = recvBuffer;
}

void AicpuRpcServer::ClearWorkMsg() const
{
    msgSndWorkArea_->sendCnt = 0;
    msgSndWorkArea_->valid = 0;
}

void AicpuRpcServer::PrintAllHcclMsgArea()
{
    const auto ctx = AicpuGetComContext();
    if (ctx == nullptr) {
        return;
    }
    MC2AicpuUtils::PrintAllHcclMsgAreaError(hcclMsgArea_, ctx->rankNum);
}

void AicpuRpcServer::PrintMsg(HcclMsg *hcclMsg, uint32_t msgPos)
{
    const auto ctx = AicpuGetComContext();
    if (ctx->debugMode == MC2_DEBUG_PRINT_MSG) {
        hcclMsg->PrintMsg("ReadAddrMsg msgPos " + std::to_string(msgPos), true);
        MC2AicpuUtils::PrintAllHcclMsgAreaEvent(hcclMsgArea_, ctx->rankNum);
    } else {
        hcclMsg->PrintMsg("ReadAddrMsg msgPos " + std::to_string(msgPos));
    }

    if (ctx->debugMode == MC2_DEBUG_PRINT_BUFF) {
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

void AicpuRpcServer::PrintAllHcclMsgAreaData()
{
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
