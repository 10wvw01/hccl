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

class AivAllReduceBig910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceBig910B() {}

    template<typename T>
    __aicore__ inline void ReduceWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther, uint64_t count,
        int32_t dstRank, int32_t tag);

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, int32_t maxCount);
};

template<typename T>
__aicore__ inline void AivAllReduceBig910B::ReduceWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther,
    uint64_t count, int32_t dstRank, int32_t tag)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);
    
    while (true) {
        if (processedBatchCount >= CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE)) {
            break;
        }

        uint64_t localFlagValue = CountWait(rank_, rank_);
        uint64_t RemoteFlagValue = CountWait(dstRank, rank_);
        if (localFlagValue <= tag || RemoteFlagValue <= tag) {
            continue;
        }

        uint64_t preparedBatchCount = (localFlagValue <= RemoteFlagValue) ? 
            (localFlagValue - tag) : (RemoteFlagValue - tag);
        if (processedBatchCount >= preparedBatchCount) {
            continue;
        }

        uint64_t curSize = (preparedBatchCount - processedBatchCount) * UB_DB_DATA_BATCH_SIZE;
        if (preparedBatchCount * UB_DB_DATA_BATCH_SIZE > avgSizePerSlice) {
            curSize = avgSizePerSlice - processedBatchCount * UB_DB_DATA_BATCH_SIZE;
        }

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

        uint64_t curProcessedOffset = processedBatchCount * UB_DB_DATA_BATCH_SIZE / sizeof(T);
        CpGM2GM(cclGmSelf + curProcessedOffset, cclGmOther + curProcessedOffset, curSize / sizeof(T), true, reduceOp_);

        processedBatchCount = preparedBatchCount;
    }

    return;
}

template<typename T>
__aicore__ inline void AivAllReduceBig910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, int32_t maxCount)
{
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
    uint64_t avgLengthPerRank = CeilDiv(len, rankSize_);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerRank, padCount) * padCount; // 32B对齐
    uint64_t loopCount = maxCount/rankSize_;
    //loopCount = loopCount/ padCount * padCount; // 32B对齐

    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    uint64_t count = 0;

    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = (block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx); // 0-2*rankSize_
    

    __gm__ T *inputGm = (__gm__ T *)input;
    __gm__ T *outputGm = (__gm__ T *)output;
    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[targetRank]);

    if (block_idx < blockNumPerGroup) {
        uint64_t gmOffset = targetRank * loopCount;
        count = CalActualCount(targetRank, sliceCount, avgLengthPerSlice, tailLength);
        // 做localcopy, 写偏移16 FLAG_SIZE
        CpGM2GMWithFlagWrap(cclGmSelf + gmOffset, inputGm + targetRank * avgLengthPerSlice, count, block_idx, 8, tag);
    } else if (targetRank != rank_) {
        uint64_t gmOffset = rank_ * loopCount;
        count = CalActualCount(rank_, sliceCount, avgLengthPerSlice, tailLength);

        // 做reduce, 检查偏移16 FLAG_SIZE
        ReduceWithFlagWrap(cclGmSelf + gmOffset, cclGmOther + gmOffset, count, targetRank, tag);
    }

    pipe_barrier(PIPE_ALL);

    if (block_idx >= blockNumPerGroup) {
        if (targetRank!=rank_){
            RecordNv1(tag, rank_);
        }
        return;
    }
    pipe_barrier(PIPE_ALL);
    
    if (block_idx == rank_) {
        // check 本端aiv 所有reduce结果是否完成
        Wait1vN(tag * (rankSize_ - 1), CommPattern::intraRank);

        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID2);

        // 告诉别人自己已经加完所有卡了
        Record1vN(tag, CommPattern::interRank);

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    } else {

    // 每个aiv读相应对端的flag
      WaitNv1(tag, targetRank);
    }

    pipe_barrier(PIPE_ALL);

    // 3. 每个aiv再把rankSize张卡上其他位置的数据搬运到本卡的对应位置
    uint64_t gmOffset = block_idx * avgLengthPerSlice;
    count = CalActualCount(block_idx, sliceCount, avgLengthPerSlice, tailLength);

    CpGM2GM(outputGm + gmOffset, cclGmOther + block_idx * loopCount, count);
    pipe_barrier(PIPE_ALL);

    // 通知对端，自己已经把对端的那片数据拉回来了
    Record(tag, targetRank, AivNotifyType::DataSignal);
    pipe_barrier(PIPE_ALL);
    
    // 确认对端已经将对应的数据拉走
    Wait(tag, targetRank, AivNotifyType::DataSignal);
    pipe_barrier(PIPE_ALL);
    
    RecordNv1(tag, rank_);
    if (block_idx ==rank_) {
        Wait1vN(tag * rankSize_, CommPattern::intraRank);
    }    

    return;
}

template<typename T>
__aicore__ inline void aiv_all_reduce_910b_bigdata(KERNEL_ARGS_DEF)
{
    AivAllReduceBig910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    if (bufferSize > MaxBufferSize) {
        bufferSize = MaxBufferSize;
    }
    uint64_t maxCountPerLoop = bufferSize /(rankSize * UB_ALIGN_SIZE) * (rankSize * UB_ALIGN_SIZE) /sizeof(T);
    uint64_t countLeft = len;

    GM_ADDR curInput = input;
    GM_ADDR curOutput = output;
    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);
    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);
        
        // 执行kernel
        op.Process<T>(curInput, curOutput, curCount, curTag, maxCountPerLoop);

        countLeft -= curCount;
        curInput += curSize;
        curOutput += curSize;
        curTag += curSize / rankSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
    op.TailCounter();

    return;
}
