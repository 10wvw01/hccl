/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_RPC_SERVERV2_H__
#define __AICPU_RPC_SERVERV2_H__

#include "common/aicpu_hccl_def.h"
#include "stream_pub.h"

class AicpuRpcServerV2 {
public:
    AicpuRpcServerV2() = default;
    ~AicpuRpcServerV2() = default;
    HcclResult Init(const HcclMC2WorkSpace &workspaceInfo, const Mc2InitTilingInner *tilingData = nullptr);
    void Reset();
    HcclMsgArea* GetHcclMsgArea(void);
    void GetMsgWorkSpace(HcclMsg *msgLists[], u32 &msgListCnt);
    uint64_t GetFinishAddr(int32_t idx);
    uint64_t GetCommitareaAddr(int32_t idx);
    uint64_t GetFinishAddrByHandleId(HcclHandle handleId);
    bool GetIsFinalize(u32 queueId = MAX_QUE_NUM);
    void SetIsFinalize(u32 queueId, bool finalize);
    HcclResult AddCcoreNotify(HcclDispatcher dispatcherPtr, u64 recordAddr, uint32_t turnNum, hccl::Stream *stream);
    HcclResult AddCcoreWait(HcclDispatcher dispatcherPtr, u64 waitAddr, uint32_t turnNum, hccl::Stream *stream,
                            bool isLast);
    HcclResult AddFlipTask(HcclDispatcher dispatcherPtr, hccl::Stream *stream);
    HcclResult ResetCommitTaskAdd(HcclDispatcher dispatcherPtr, hccl::Stream *stream);
    void WriteFinishWhenAllFinalize();
    void WriteRestartFlag();
    uint32_t GetMsgPos(u32 queueId = 0U);
    void SetMsgPos(u32 queueId, u32 pos);
    bool IsPrintLog() const;
    void SetMsgRepeatCnt(u8 repeatCnt);
    int32_t GetMsgRepeatCnt(HcclHandle handleId);
    int32_t GetMsgHandlePos(HcclHandle handleId);
    void PrintAllHcclMsgArea(u32 rankSize);
    void PrintAllHcclMsgAreaData();
    void PrintMsg(HcclMsg *hcclMsg, uint32_t msgPos, u32 rankSize);
    void SetMsgPosForKernel(uint32_t msgPos);
    uint32_t GetMsgPosForKernel(void);
    void SetMsgHandlePos(uint32_t msgPos, HcclHandle handleId);
    void SetNeedRetryFlag(bool needRetryFlag);
    bool ReadAddrMsg(HcclMsg *hcclMsg, HcclMsg *msgList, u32 queueIdx, u32 msgPos, u32 rankSize);
    bool IsExceedLimit(AicpuComType commType, u32 rankSize);
    HcclMsgExt* GetHcclMsgExtPtr();
    HcclResult ProcessExpectPrepareMsg(uint8_t seqNum, uint8_t expectId);

public:
    void SetStepSize(u8 stepSize) { curStepSize_ = stepSize; };
    void SetTotalStep(u16 totalStep) { totalStep_ = totalStep; };
    u16 GetStepSize() const { return curStepSize_; }
    u64 GetTurnNumAddr() { return turnNumAddr_; }
    u32 GetTotalQueueNum() const { return totalQueueNum_; }
    BarrierInfo *GetBarrierInfoByGroupIdx(u32 idx) { return barrierFlags_[idx]; }
    void ClearBarrierStatus() { (void)memset_s(barrierFlags_, sizeof(barrierFlags_), 0U, sizeof(barrierFlags_)); }
    u32 *GetBarrierFinishCnts() { return barrierFinishCnt_; }
    u64 GetTilingBaseAddr() const { return tilingBaseAddr_; }
    void GetLocalQueueRange(u32 &start, u32 &end);

private:
    bool ReadValidMsg(HcclMsg *rMsg, HcclMsg *msg, bool needReProcess, uint32_t msgPos, u32 rankSize);
    uint64_t GenXorForMsgExt(int32_t idx, u32 rankSize);
    bool ReadValidMsgExtArea(int32_t idx, u32 rankSize);
    void GenMsgByTaskParam(AivAicpuOpParam *outMsg);

private:
    uint64_t workSpace_ = 0;
    HcclMsgArea *hcclMsgArea_ = nullptr;
    HcclMsgAreaForMultiQue *hcclMultiMsgArea_ = nullptr;
    uint32_t repeatCnt_[AC_MSG_CNT];
    int8_t handleIdMsgPosition_[MC2_HCCL_MAX_HANDLE_ID];
    uint64_t streamId_;
    uint32_t msgPosForKernel_;
    uint32_t msgPos_[MAX_QUE_NUM];
    bool needReProcess_ = false;
    bool isFinalize_[MAX_QUE_NUM];
    std::shared_ptr<HcclMsgExt> msgExt_ = std::make_shared<HcclMsgExt>();
    u64 prepareTime_[MAX_QUE_NUM]; // 记录 Prepare 消息的时间
    u8 eventPrintTurn_; // 记录打印event的turn
    bool isPrintLog_ = false;
    u8 curStepSize_ = 0U;
    u16 totalStep_ = 0U;
    u64 turnNumAddr_;
    u32 blockNum_ = 1U;
    u32 totalQueueNum_ = 0U;
    BarrierInfo barrierFlags_[MAX_COMM_CTX_NUM][MAX_QUE_NUM];
    u32 barrierFinishCnt_[MAX_AICPU_BLOCK_DIM];
    u64 tilingBaseAddr_;
};

#endif  // __AICPU_RPC_SERVERV2_H__