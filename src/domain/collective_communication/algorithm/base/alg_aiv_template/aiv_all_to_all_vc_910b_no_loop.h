/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base.h"

using namespace AscendC;

class AivAll2AllVCNoLoop910B : public AivCommBase {
public:
    __aicore__ inline AivAll2AllVCNoLoop910B() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, int32_t tag, ExtraArgs &extraArgs);
};

template<typename T>
__aicore__ inline void AivAll2AllVCNoLoop910B::Process(GM_ADDR input, GM_ADDR output, int32_t tag,
    ExtraArgs &extraArgs)
{
    uint32_t targetRank = (block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx); // 0-2*rankSize

    // 内存准备
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;
    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[targetRank]);
    tag = tag << TAG_MOVE_LEFT_BITS;
    if (block_idx < rankSize_) { // 前rankSize个aiv负责userin->cclin
        uint64_t localSendOffset = 0;
        for (uint32_t i = 0; i < block_idx; i++) {
            localSendOffset += extraArgs.sendCountMatrix[rank_ * rankSize_ + i];
        }
        uint64_t localSendCount = extraArgs.sendCountMatrix[rank_ * rankSize_ + block_idx];

        CpGM2GMWithFlagWrap(cclGMSelf + localSendOffset, inputGM + localSendOffset, localSendCount, block_idx, 16, tag);
    } else { // 后rankSize个aiv负责cclother->usrout
        // 读对端数据前确认对端已进入本算子
        uint64_t remoteSendOffset = 0; // 远端ccl发送给本端output的数据偏移，远端卡号为block_idx，可能为本rank
        for (uint32_t i = 0; i < rank_; i++) {
            remoteSendOffset += extraArgs.sendCountMatrix[targetRank * rankSize_ + i];
        }

        uint64_t localRecvOffset = 0; // 本端output接收远端ccl的数据偏移，目标远端卡号为block_idx，可能为本rank
        for (uint32_t i = 0; i < targetRank; i++) {
            localRecvOffset += extraArgs.sendCountMatrix[i * rankSize_ + rank_];
        }

        // 远端ccl发送给本端output的数据量，远端可能为本rank
        uint64_t remoteSendCount = extraArgs.sendCountMatrix[targetRank * rankSize_ + rank_];
        uint64_t remoteSendSize = remoteSendCount * sizeof(T);

        uint64_t processedBatchCount = 0;

        while (true) {
            if (processedBatchCount >= CeilDiv(remoteSendSize, UB_DB_DATA_BATCH_SIZE)) {
                break;
            }

            int32_t localFlag = CountWait(targetRank, rank_);
            if (localFlag <= tag){
                continue;
            }
            uint64_t preparedBatchCount = localFlag - tag;

            if (preparedBatchCount <= 0 || processedBatchCount >= preparedBatchCount) {
                continue;
            }

            uint64_t curSize = (preparedBatchCount - processedBatchCount) * UB_DB_DATA_BATCH_SIZE;
            if (preparedBatchCount * UB_DB_DATA_BATCH_SIZE > remoteSendSize) {
                curSize = remoteSendSize - processedBatchCount * UB_DB_DATA_BATCH_SIZE;
            }

            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

            uint64_t curProcessedOffset = processedBatchCount * UB_DB_DATA_BATCH_SIZE / sizeof(T);
            CpGM2GM(outputGM + localRecvOffset + curProcessedOffset, cclGMOther + remoteSendOffset + curProcessedOffset,
                curSize / sizeof(T));

            processedBatchCount = preparedBatchCount;
        }

        // 通知对端，自己已经把对端的那片数据拉回来了
        PipeBarrier<PIPE_ALL>();
        Record(tag, targetRank, AivNotifyType::DataSignal);
        PipeBarrier<PIPE_ALL>();
        
        // 确认对端已经将对应的数据拉走
        Wait(tag, targetRank, AivNotifyType::DataSignal);
    }
}

template<typename T>
__aicore__ inline void aiv_all_to_all_vc_910b_no_loop(EXTERN_KERNEL_ARGS_DEF)
{
    AivAll2AllVCNoLoop910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    op.Process<T>(input, output, tag, extraArgs);
    op.TailCounter();
}
