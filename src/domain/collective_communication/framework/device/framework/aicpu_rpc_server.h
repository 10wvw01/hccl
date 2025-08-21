/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_RPC_SERVER_H__
#define __AICPU_RPC_SERVER_H__

#include "common/aicpu_hccl_def.h"

constexpr uint32_t AC_MAX_AIV = 64U;  // 最多有64个AIV

enum RANK_MSG_TYPE {
    RANK_ADDR = 1,
    RANK_WORK = 2,
    RANK_ADD_AND_WORK = 3,
    RANK_TAIL_TIME = 6,
    RANK_MSG_END
};

class AicpuRpcServer {
public:
    AicpuRpcServer() = default;
    ~AicpuRpcServer() = default;

    void Init(u64 workSpaceAddr, uint32_t notifyOff, uint16_t notifyBeginCnt, KFCTask *taskParam);
    void Init(u64 workSpaceAddr);
    bool RcvMsg(AivAicpuOpParam *rMsg, uint32_t aivID, uint8_t msgType);
    bool CheckRcvAddrMsg(AivAicpuOpParam *rMsg, uint32_t aivID);
    bool CheckRcvAddrMsg(HcclMsg *hcclMsg, uint32_t msgPos);
    bool CheckRcvWorkMsg(AivAicpuOpParam *rMsg, uint32_t aivID, uint32_t curTurnCnt);
    bool PostMsg(AivAicpuOpParam *rMsg, uint32_t aivID);
    bool ReadWorkMsg(AivAicpuOpParam *rMsg, uint32_t aivID, uint32_t curTurnCnt);
    bool ReadAddrMsg(AivAicpuOpParam *rMsg, uint32_t aivID);
    bool ReadAddrMsg(HcclMsg *hcclMsg, uint32_t msgPos);
    bool ReadApiValidMsg(HcclMsg *rMsg, HcclMsg *msg, bool reset);
    bool PostMsg(uint32_t curTurnCnt) const;
    bool CheckAivIsEnd(uint32_t aivId);
    bool NeedAutoGenMsg();
    uint8_t GetWaitPolicy();
    uint8_t GetTaskType() const;
    uint8_t GetRspPolicy();
    uint8_t GetGenTaskNum();
    TASK_PREPARE_POSITION GetPreparePosition() const;
    void ClearWorkMsg() const;
    void WriteFinishWhenAllFinalize(uint32_t msgPos);
    void HcclMsg2AicAicpuOpParam(CommonHcclMsg *hcclMsg, AivAicpuOpParam *opMsg);
    void WriteTurnCnt(uint32_t msgPos);
    void PrintAllHcclMsgArea();
    void PrintAllHcclMsgAreaData();
    void PrintMsg(HcclMsg *hcclMsg, uint32_t msgPos);

private:
    template <typename T>
    bool ReadValidMsg(T *rMsg, T *msg, uint8_t msgType, bool reset);
    bool GenMsgIsLastMsg();
    std::string GetMsgTypeString(uint8_t msgType);
    void GenMsgByTaskParam(AivAicpuOpParam *outMsg);
    u64 GetSendOff() const;
    u64 GetRecvOff() const;
    void CalcAllgatherBuffer(AivAicpuOpParam *outMsg) const;
    void CalcAllreduceBuffer(AivAicpuOpParam *outMsg) const;
    void CalcReduceScatterBuffer(AivAicpuOpParam *outMsg) const;

private:
    struct RpcMsgBody {
        // Rank* aiv * MsgSize * sizeof(消息)
        AivAicpuOpParam msgRcvArea[AC_MAX_AIV][AC_MSG_CNT];
        AivAicpuOpParam msgSndArea[AC_MAX_AIV][AC_MSG_CNT];
    };
    RpcMsgBody *msgBody_ = nullptr;

    HcclMsgArea *hcclMsgArea_ = nullptr;

    AivAicpuOpParam *msgSndWorkArea_ = nullptr;
    AivAicpuOpParam *msgRcvRspArea_ = nullptr;
    // 记录每个消息队列的位置
    uint32_t rcvMsgPos_[AC_MAX_AIV] = {0};  // 接收队列的初始化位置
    uint32_t sndMsgPos_[AC_MAX_AIV] = {0};
    uint32_t aivState_[AC_MAX_AIV] = {0};

    uint64_t genTaskNum_;
    KFCTask *genTaskParam_;
    HcclKFCTilingData *tilingData_;
};

#endif  // __AICPU_RPC_SERVER_HPP__
