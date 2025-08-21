/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "aiv_communication_base.h"
 
using namespace AscendC;
 
class AivReduceScatterDeterMid910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterDeterMid910B() {}
 
    template<typename T>
    __aicore__ inline void CPGM2GMAccordingFlag(__gm__ T *cclGMSelf, __gm__ T *cclGMOther,
        uint64_t count, __gm__ int32_t* ctrlFlagGMSelf, __gm__ int32_t* ctrlFlagGMOther, int32_t tag, uint64_t ff=0);
 
    __aicore__ inline void EndSync(int32_t tag);

    __aicore__ inline int64_t GetDeterministicRankOffset(int64_t x);
 
     template<typename T>
    __aicore__ inline void SumByPairs(__gm__ T *cclGMSelf, int64_t x, int64_t count, int32_t tag, int64_t flagOffsetBase);
 
    template<typename T>
    __aicore__ inline void ReduceWithFlagWrap(__gm__ T *cclGMSelf, __gm__ T *cclGMOther, uint64_t count, int32_t tag,
        __gm__ int32_t* flagCntDoneBase, __gm__ int32_t* flagCntSelf,__gm__ int32_t* flagCntDoneSelf);
 
    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize);
};

__aicore__ inline void AivReduceScatterDeterMid910B::EndSync(int32_t tag)
{
    uint32_t targetRank = block_idx % rankSize_;

    int64_t flagOffsetBasic = seperateOffset + BASE_FLAG_OFFSET * AIV_REDUCE_SCATTER_DETER_910B_MIDDATA;
    uint32_t flagOffset = (((tag % 2 == 0) ? 3 : 9) * rankSize_ * FLAG_SIZE) + flagOffsetBasic;

    if (block_idx < rankSize_) {
        if (targetRank != rank_) {
            PipeBarrier<PIPE_ALL>();
            SetSignalValue((__gm__ int32_t *)(GM_OUT[targetRank] + flagOffset + rank_ * FLAG_SIZE), localSetTensor, tag);
            WaitSignalValue((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + targetRank * FLAG_SIZE), localCheckTensor, tag);
            PipeBarrier<PIPE_ALL>();
            SetSignalValue((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + targetRank * FLAG_SIZE), localSetTensor, 0);
        }
    }
}

// 根据ctrlFlagGMOther的值，从cclGMOther拷贝到cclGMSelf
// 如果ff大于0，则更新ctrlFlagGMSelf
template<typename T>
__aicore__ inline void AivReduceScatterDeterMid910B::CPGM2GMAccordingFlag(__gm__ T *cclGMSelf, __gm__ T *cclGMOther,
    uint64_t count, __gm__ int32_t* ctrlFlagGMSelf, __gm__ int32_t* ctrlFlagGMOther, int32_t tag, uint64_t ff)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);
    uint64_t maxBatchCount = CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE);
 
    while (true) {
        if (processedBatchCount >= maxBatchCount) {
            break;
        }
 
        LocalTensor<int32_t> localFlagX = flagInQue.AllocTensor<int32_t>();

        uint64_t localFlagValueX = GetSignalValue(ctrlFlagGMOther, localFlagX);

        flagInQue.FreeTensor(localFlagX);
 
        if (localFlagValueX <= tag) {
            continue;
        }
 
        uint64_t preparedBatchCount = localFlagValueX - tag;
        if (processedBatchCount >= preparedBatchCount) {
            continue;
        }
 
        uint64_t curSize = (preparedBatchCount - processedBatchCount) * UB_DB_DATA_BATCH_SIZE;
        if (preparedBatchCount * UB_DB_DATA_BATCH_SIZE > avgSizePerSlice) {
            curSize = avgSizePerSlice - processedBatchCount * UB_DB_DATA_BATCH_SIZE;
        } 
 
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
 
        //  搬运数据
        uint64_t curProcessedOffset = processedBatchCount * UB_DB_DATA_BATCH_SIZE / sizeof(T);
        CpGM2GM(cclGMSelf + curProcessedOffset, cclGMOther + curProcessedOffset, curSize / sizeof(T));
 
        // 设置已经搬运的数据量
        processedBatchCount = preparedBatchCount;
        if(ff > 0 && (processedBatchCount % ff ==0 || processedBatchCount >=maxBatchCount)) {
            SyncFunc<HardEvent::MTE3_S>();
            SetSignalValue(ctrlFlagGMSelf, localSetTensor, processedBatchCount + tag);
        }
    }
}
 
__aicore__ inline int64_t AivReduceScatterDeterMid910B::GetDeterministicRankOffset(int64_t x)
{
    int64_t tmp = 1;
    while(!(x & 1)) {
        x >>= 1;
        tmp <<= 1;
    }
    return tmp;
}
 
template<typename T>
__aicore__ inline void AivReduceScatterDeterMid910B::SumByPairs(__gm__ T *cclGMSelf, int64_t x, int64_t count,
 int32_t tag, int64_t flagOffsetBase)
{
    int64_t multiple = GetDeterministicRankOffset(x);
    int64_t target = x - multiple;
 
    int64_t flagOffsetSelf = 0;
    int64_t flagOffsetTarget = 0;
    int64_t flagOffset2st = flagOffsetBase + (rankSize_ + x) * FLAG_SIZE;
    int64_t flagOffset3st = flagOffsetBase + (2*rankSize_ + x) * FLAG_SIZE;
    if (x & 1) {
        if (target == 0) {
            flagOffsetTarget = flagOffsetBase + (2*rankSize_ + target) * FLAG_SIZE;
        }else{
            flagOffsetTarget = flagOffsetBase + (rankSize_ + target) * FLAG_SIZE;
        }
        flagOffsetSelf = flagOffset2st; 
    } else {
        flagOffsetTarget = flagOffsetBase + (2*rankSize_ + target + multiple / DOUBLE) * FLAG_SIZE;
        int64_t multipleTemp  = multiple / DOUBLE;
        while (x + multipleTemp >= rankSize_) {
            multipleTemp /= DOUBLE;
        }
        if (multipleTemp <= 0) {
            flagOffsetSelf = flagOffset2st; 
        }else{
            flagOffsetSelf = flagOffset3st + multipleTemp * FLAG_SIZE;
        }
    }
    __gm__ int32_t *flagCntDonePre = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetTarget);
    __gm__ int32_t *flagCntSelf = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetSelf);
    __gm__ int32_t *flagCntDoneSelf = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset3st);
    ReduceWithFlagWrap(cclGMSelf + target * count, cclGMSelf + x * count, count, tag, flagCntDonePre, flagCntSelf, flagCntDoneSelf);
}
 
template<typename T>
__aicore__ inline void AivReduceScatterDeterMid910B::ReduceWithFlagWrap(__gm__ T *cclGMSelf, __gm__ T *cclGMOther, 
    uint64_t count, int32_t tag, __gm__ int32_t* flagCntDoneBase, __gm__ int32_t* flagCntSelf, __gm__ int32_t* flagCntDoneSelf)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);
    uint64_t maxBatchCount = CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE);
    
    while (true) {
        if (processedBatchCount >= maxBatchCount) {
            break;
        }

        LocalTensor<int32_t> localFlag = flagInQue.AllocTensor<int32_t>();
        LocalTensor<int32_t> localFlagY = flagInQue.AllocTensor<int32_t>();

        uint64_t localFlagValue = GetSignalValue(flagCntDoneBase, localFlag);
        uint64_t localFlagYValue = GetSignalValue(flagCntSelf, localFlagY);
 
        flagInQue.FreeTensor(localFlag);
        flagInQue.FreeTensor(localFlagY);
 
        if (localFlagValue <= tag || localFlagYValue <= tag) {
            continue;
        }
 
        uint64_t preparedBatchCount = (localFlagValue <= localFlagYValue) ? localFlagValue : localFlagYValue;
        preparedBatchCount -= tag;
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
        CpGM2GM(cclGMSelf + curProcessedOffset, cclGMOther + curProcessedOffset, curSize / sizeof(T), true, reduceOp_);
 
        processedBatchCount = preparedBatchCount;
        if(processedBatchCount % 8 ==0 || processedBatchCount >= maxBatchCount) {
            SyncFunc<HardEvent::MTE3_S>();
            SetSignalValue(flagCntDoneSelf, localSetTensor, processedBatchCount + tag);
        }
    }
}
 
template<typename T>
__aicore__ inline void AivReduceScatterDeterMid910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize)
{
    int64_t count = len;
    int64_t allCount = count*rankSize_;
    int64_t blockNumPerGroup = rankSize_;
    int64_t x = block_idx % blockNumPerGroup;
    int64_t flagOffsetBasic = seperateOffset + BASE_FLAG_OFFSET * AIV_REDUCE_SCATTER_DETER_910B_MIDDATA;

    uint32_t flagOffsetBase = ((tag % 2 == 0) ? 0 : 6 * rankSize_ * FLAG_SIZE) + flagOffsetBasic;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;
    
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_] + dataOffset);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[x] + dataOffset);
    __gm__ T *outputGM = (__gm__ T *)output;
    
    int64_t flagOffset1stCount = flagOffsetBase + (x) * FLAG_SIZE;
    int64_t flagOffset2stCount = flagOffsetBase + (rankSize_ + x) * FLAG_SIZE;
    int64_t flagOffset3stCount = flagOffsetBase + (2 * rankSize_ + x) * FLAG_SIZE;
    int64_t flagOffsetCheck = flagOffsetBase + (4*rankSize_ ) * FLAG_SIZE;
 
    // 第一组 先从input拷贝到cclbuffer
    if (block_idx < blockNumPerGroup) {
        CpGM2GMWithFlagWrap(cclGMSelf + x * count, inputGM + x * count, count, (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset1stCount), 8, tag);
    } 
    // 第二组 拷贝cclbuffer前半部分到cllbuffer后半部分
    else if (blockNumPerGroup<=block_idx && block_idx < 2*blockNumPerGroup) {
        __gm__ int32_t *flagCntDoneOtner = (__gm__ int32_t *)(GM_OUT[x] + flagOffsetBase + (rank_)* FLAG_SIZE);
        __gm__ int32_t *flagCntSelf = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset2stCount);
        if (x == 0) {
            // 更新到3st区域
            flagCntSelf = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset3stCount);
        }
 
        CPGM2GMAccordingFlag(cclGMSelf + allCount + x * count, cclGMOther + rank_ * count, count,
            flagCntSelf, flagCntDoneOtner, tag, 8);
    }
    // 第三组 进行reduce操作
    else{
        if (x == 0) {
            return;
        }
        int64_t lastOpCore = 0; 
        if (rankSize_ >= DETERMINISTIC_RANKSIZE) {
            SumByPairs(cclGMSelf + allCount, x, count, tag, flagOffsetBase);
            lastOpCore = rankSize_ > DETERMINISTIC_RANKSIZE ? DETERMINISTIC_RANKSIZE : DOUBLE;
        } else {
            __gm__ int32_t *flagCntDonePre = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset3stCount - FLAG_SIZE);
            __gm__ int32_t *flagCntSelf = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset2stCount);
            __gm__ int32_t *flagCntDoneSelf = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset3stCount);
            
            ReduceWithFlagWrap(cclGMSelf + allCount, cclGMSelf + allCount + x * count, count, tag, 
                flagCntDonePre, flagCntSelf, flagCntDoneSelf);
            lastOpCore = rankSize_ - 1;
        }
        if (x == lastOpCore) {
            SetSignalValue((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetBase + flagOffsetCheck), localSetTensor, tag);
        }       
    }
 
    // 第2组搬运cllbuffer到output
    if (blockNumPerGroup <= block_idx && block_idx < DOUBLE * blockNumPerGroup) {
        int32_t lastOpCore = rankSize_ - 1;
        if (rankSize_ >= DETERMINISTIC_RANKSIZE) {
            lastOpCore = rankSize_ > DETERMINISTIC_RANKSIZE ? DETERMINISTIC_RANKSIZE : DOUBLE;
        }
        __gm__ int32_t *ctrlFlagGMDone = (__gm__ int32_t *)(GM_OUT[x] + flagOffsetBase + (DOUBLE * rankSize_ + lastOpCore) * FLAG_SIZE);
        WaitSignalValue((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetBase + flagOffsetCheck), localCheckTensor, tag);
        int64_t copyLen = CeilDiv(count, rankSize_);
        int64_t needCopy = copyLen;
        if (x == rankSize_ - 1) {
            needCopy = count - (rankSize_ - 1) * copyLen;
        }

        PipeBarrier<PIPE_ALL>();
        CpGM2GM(outputGM + x * copyLen, cclGMSelf + allCount + x * copyLen, needCopy);
    }
}
 
template<typename T>
__aicore__ inline void aiv_reduce_scatter_deter_910b_middata(KERNEL_ARGS_DEF)
{
    AivReduceScatterDeterMid910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    int32_t curTag = (tag << 15);
    op.Process<T>(input, output, len, curTag, bufferSize);
    if (tag == 1000) {
        op.EndSync(tag);
    }
    op.TailCounter();
}