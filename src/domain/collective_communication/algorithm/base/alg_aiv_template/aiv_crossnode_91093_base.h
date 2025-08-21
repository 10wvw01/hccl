/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_CROSSNODE_91093_BASE_H
#define AIV_CROSSNODE_91093_BASE_H

#include "aiv_communication_base.h"

using namespace AscendC;

class AivCrossNode91093Base {
public:
    __aicore__ inline AivCrossNode91093Base() {}

    __aicore__ inline void Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize); // sync的init

    __aicore__ inline void Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize,
        bool useDoubleBuffer); // ALL2ALL的init

    template<typename T>
    __aicore__ inline void Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize,
        uint64_t perRankBufferCount, uint64_t len, uint32_t reduceOp, bool useDoubleBuffer); // AG、RS单算子的init

    template<typename T>
    __aicore__ inline void Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize,
        uint64_t len, uint32_t reduceOp, bool useDoubleBuffer); // AG、RS图模式的init

    __aicore__ inline void InitOffset(); // 初始化offset

    __aicore__ inline void InitSetCheckClearArgsTensor();
    
    __aicore__ inline void CalcNumTargetsAndTargetRanks();

    template<typename T>
    __aicore__ inline void SetAtomicOp(uint32_t atomicOp);

    __aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b);

    __aicore__ inline uint64_t CalActualCount(uint32_t sliceIdx, uint64_t sliceCount, uint64_t avgLengthPerSlice,
        uint64_t tailLength);

    __aicore__ inline void CalCountAndBlockOffset(uint64_t len, uint32_t blockNumPerGroup, uint32_t blockIdxInGroup, 
        uint32_t padCount, uint64_t &count, uint64_t &blockOffset);

    template<typename T>
    __aicore__ inline void DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
        const uint32_t calCount);

    template<typename T>
    __aicore__ inline void DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
        const uint32_t calCount);

    template<typename T>
    __aicore__ inline void CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, bool atomic = false,
        uint32_t atomicOp = 0);

    template<HardEvent event> 
    __aicore__ inline void SyncFunc();
   
    __aicore__ inline void SingleRecordBatchWaitCoreLevel(int32_t curTag, bool isTheSingleCore,
        AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void BatchRecordSingleWaitCoreLevel(int32_t curTag, bool isTheSingleCore,
        AivNotifyType notifyType = AivNotifyType::ACK);
    
    __aicore__ inline void SingleRecordBatchWait(int32_t curTag, GM_ADDR* buffersOut, bool isTheSingleCore,
        AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void BatchRecordWait(int32_t curTag, GM_ADDR* buffersOut,
        AivNotifyType notifyType = AivNotifyType::ACK);
    
    __aicore__ inline void Record(uint32_t tag, GM_ADDR waitAddr, AivNotifyType notifyType);

    __aicore__ inline void Record1vN(uint32_t tag, CommPattern pattern,
        AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void RecordNv1(uint32_t tag, GM_ADDR waitAddr, bool ifCoreLevel,
        AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void Wait(uint32_t tag, int32_t recordRank,
        AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void WaitNv1(uint32_t tag, GM_ADDR recordAddr, bool ifCoreLevel,
        AivNotifyType notifyType = AivNotifyType::ACK);
    
    __aicore__ inline void Wait1vN(uint32_t tag, CommPattern pattern, bool ifClear = true,
        AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void InitOpCounter(GM_ADDR headCountMem, GM_ADDR tailCountMem, GM_ADDR addOneMem, 
        uint32_t counterMemSize, bool isEnableCounter)
    {
        headCountMem_ = headCountMem;
        tailCountMem_ = tailCountMem;
        addOneMem_ = addOneMem;
        counterMemSize_ = counterMemSize;
        isEnableCounter_ = isEnableCounter;
    }

    __aicore__ inline void HeadCounter()
    {
        if (block_idx == 0 && isEnableCounter_) {
            CpGM2GM((__gm__ int32_t*)headCountMem_, (__gm__ int32_t*)addOneMem_, counterMemSize_ / sizeof(int32_t), true,
                HcclReduceOp::HCCL_REDUCE_SUM);
        }
    }

    __aicore__ inline void TailCounter()
    {
        if (block_idx == 0 && isEnableCounter_) {
            CpGM2GM((__gm__ int32_t*)tailCountMem_, (__gm__ int32_t*)addOneMem_, counterMemSize_ / sizeof(int32_t), true,
                HcclReduceOp::HCCL_REDUCE_SUM);
        }
    }

    uint32_t baseFlagOffset_ = 0;
    GM_ADDR flagAddrSelf_;
    uint32_t rank_;
    uint32_t rankSize_;
    uint32_t reduceOp_;
    uint32_t usedBlockNum_;
    bool useDoubleBuffer_;

    TPipe pipe;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue;
    TBuf<> localFlagBuf;
    LocalTensor<int32_t> localSetTensor;
    LocalTensor<int32_t> localCheckTensor;
    LocalTensor<int32_t> localClearTensor;
    TBuf<> bufferArgsBuf;
    LocalTensor<uint64_t> bufferArgsTensor; // buffer地址GM-UB
    TBuf<> offsetArgsBuf;
    LocalTensor<uint64_t> offsetArgsTensor; // count参数UB-GM，类似做allgather

    // 每个aiv核的数据搬运参数，用于多核并行优化方案
    uint32_t numTargets; // 每个aiv需要顺序与几个对端通信，ranksize太大时，aiv不够用，需要多次
    uint32_t targetRanks[MAX_TARGET_NUM] = {}; // 最多768/48 = 16 次（一次代表服务48张卡）
    uint32_t blockNumPerGroup = 1; // 多少个aiv服务一个rank
    uint32_t blockIdxInGroup = 0; // 同一组中的aiv编号
    uint64_t countMid; // 中间轮一个aiv负责搬运的数据量（一轮代表一次ccl buffer装满）
    uint64_t countTail; // 尾轮一个aiv负责搬运的数据量
    uint64_t blockOffsetMid; // 数据块offset，区分中间轮和尾轮
    uint64_t blockOffsetTail;
    uint32_t flagOffsetInGroup; // 标志位offset，不区分中间轮和尾轮
    uint64_t blockOffset; // 数据块offset，不区分中间轮和尾轮
    uint64_t countPerCore; // 每个核负责的数据块大小，不区分中间轮和尾轮

    // 维测相关
    GM_ADDR headCountMem_;
    GM_ADDR tailCountMem_;
    GM_ADDR addOneMem_;
    uint32_t counterMemSize_;
    bool isEnableCounter_;

    uint32_t localOffset;
    uint32_t multiOffset;
    uint32_t pingpongOffset;
    uint32_t countOffset;
};

__aicore__ inline uint64_t AivCrossNode91093Base::CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t AivCrossNode91093Base::CalActualCount(uint32_t sliceIdx, uint64_t sliceCount,
    uint64_t avgLengthPerSlice, uint64_t tailLength)
{
    if (sliceIdx == sliceCount - 1) {
        return tailLength;
    } else if (sliceIdx < sliceCount - 1) {
        return avgLengthPerSlice;
    } else {
        return 0;
    }
}

__aicore__ inline void AivCrossNode91093Base::CalCountAndBlockOffset(uint64_t len, uint32_t blockNumPerGroup, 
    uint32_t blockIdxInGroup, uint32_t padCount, uint64_t &count, uint64_t &blockOffset)
{
    uint64_t avgLengthPerBlock = CeilDiv(len, blockNumPerGroup);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice; // 多核并行搬数据，最后一核搬运的数据量
    count = CalActualCount(blockIdxInGroup, sliceCount, avgLengthPerSlice, tailLength);
    blockOffset = blockIdxInGroup * avgLengthPerSlice;
}

__aicore__ inline void AivCrossNode91093Base::CalcNumTargetsAndTargetRanks()
{
    // 计算本core的numTargets和targetsList
    // 前concurrentSize/2个aiv负责与左边rank号的通信，后concurrentSize/2个负责与右边rank号的通信
    uint32_t halfConcurrent = usedBlockNum_ / 2; // usedBlockNum_需要为偶数
    numTargets = (rankSize_ - 1) / usedBlockNum_; // 除去本rank，可能需要补上一个
    uint32_t tailRankSize = (rankSize_ - 1) % usedBlockNum_;
    uint32_t leftTailRankSize = 0;
    uint32_t rightTailRankSize = 0;
    if (tailRankSize > 0) {
        if (tailRankSize <= halfConcurrent) {
            leftTailRankSize = tailRankSize;
        } else {
            leftTailRankSize = halfConcurrent;
            rightTailRankSize = tailRankSize - halfConcurrent;
        }
        if (block_idx < halfConcurrent && (halfConcurrent - block_idx) <= leftTailRankSize) {
            numTargets += 1;
        }
        if (block_idx >= halfConcurrent && (block_idx - halfConcurrent + 1) <= rightTailRankSize) {
            numTargets += 1;
        }
    }

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank;
        if (block_idx < halfConcurrent) {
            targetRank = (rank_ + rankSize_ - (halfConcurrent - block_idx) - i * halfConcurrent) % rankSize_; // left
        } else {
            targetRank = (rank_ + (block_idx - halfConcurrent + 1) + i * halfConcurrent) % rankSize_; // right
        }
        targetRanks[i] = targetRank;
    }
}

__aicore__ inline void AivCrossNode91093Base::InitSetCheckClearArgsTensor() 
{
    pipe.InitBuffer(localFlagBuf, UB_FLAG_SIZE * FLAG_BUF_NUM);
    localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, 0);
    localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, UB_FLAG_SIZE);
    localClearTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, UB_FLAG_SIZE * IDX_2);
    localClearTensor.SetValue(0, 0);
    pipe.InitBuffer(bufferArgsBuf, UB_FLAG_SIZE * MAX_TARGET_NUM);
    bufferArgsTensor = bufferArgsBuf.Get<uint64_t>();
    pipe.InitBuffer(inOutQue, DOUBLE, UB_DB_DATA_BATCH_SIZE);
}

// sync的init
__aicore__ inline void AivCrossNode91093Base::Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize)
{
    flagAddrSelf_ = buffOut0;

    rank_ = rank;
    rankSize_ = rankSize;
    usedBlockNum_ = block_num - block_num % DOUBLE; // 确保为偶数

    InitSetCheckClearArgsTensor();
    CalcNumTargetsAndTargetRanks();
    InitOffset();
}

// ALL2ALL的init
__aicore__ inline void AivCrossNode91093Base::Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize,
    bool useDoubleBuffer)
{
    flagAddrSelf_ = buffOut0;

    rank_ = rank;
    rankSize_ = rankSize;
    useDoubleBuffer_ = useDoubleBuffer;
    usedBlockNum_ = block_num;
    
    InitSetCheckClearArgsTensor();
    pipe.InitBuffer(offsetArgsBuf, UB_FLAG_SIZE * MAX_TARGET_NUM);
    offsetArgsTensor = offsetArgsBuf.Get<uint64_t>();

    CalcNumTargetsAndTargetRanks();
    InitOffset();
}

__aicore__ inline void AivCrossNode91093Base::InitOffset()
{
    blockIdxInGroup = block_idx % blockNumPerGroup;
    int32_t notifyArea = MAX_RANK_SIZE_A3;
    if (rankSize_ * blockNumPerGroup < MAX_RANK_SIZE_A3) {
        notifyArea = rankSize_ * blockNumPerGroup;
    }
    localOffset = (notifyArea * FLAG_BUF_NUM) * FLAG_SIZE;
    multiOffset = MAX_BLOCK_DIM * DOUBLE * FLAG_SIZE+ localOffset;
}

// AG、RS单算子的Init
template<typename T>
__aicore__ inline void AivCrossNode91093Base::Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize,
    uint64_t perRankBufferCount, uint64_t len, uint32_t reduceOp, bool useDoubleBuffer)
{
    flagAddrSelf_ = buffOut0;

    rank_ = rank;
    rankSize_ = rankSize;
    reduceOp_ = reduceOp;
    useDoubleBuffer_ = useDoubleBuffer;
    usedBlockNum_ = block_num;

    InitSetCheckClearArgsTensor();

    // 以下根据不同情况，计算每个aiv核的数据搬运参数
    // 当rankSize大于总aiv核数的一半时，使用1个aiv服务一个对端，需要多次通信
    if (rankSize > HALF_MAX_BLOCK_DIM) {
        CalcNumTargetsAndTargetRanks();

        blockNumPerGroup = 1;
        if (len <= perRankBufferCount) { // ccl够用，只需要搬一轮的情况
            countMid = 0;
            countTail = len;
        } else if (len % perRankBufferCount == 0) { // ccl不够用，要搬多轮的情况1: 能整除
            countMid = perRankBufferCount;
            countTail = perRankBufferCount;
        } else { // ccl不够用，要搬多轮的情况2: 不能整除
            countMid = perRankBufferCount;
            countTail = len % perRankBufferCount;
        }
        blockOffsetMid = 0;
        blockOffsetTail = 0;
        flagOffsetInGroup = 0;
        countPerCore = len;
        blockOffset = 0;
    
    // 当rankSize小于等于总aiv核数的一半时，根据ranksize和数据量大小选择使用多个aiv服务一个对端（多核并行），只需一次通信
    } else {
        numTargets = 1;
        blockNumPerGroup = block_num / rankSize_; // 多少个aiv服务一个rank
        targetRanks[0] = block_idx / blockNumPerGroup;

        uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
        blockIdxInGroup = block_idx % blockNumPerGroup;

        if (len <= perRankBufferCount) { // ccl够用，只需要搬一轮的情况
            countMid = 0;
            blockOffsetMid = 0;
            CalCountAndBlockOffset(len, blockNumPerGroup, blockIdxInGroup, padCount, countTail, blockOffsetTail);
        } else if (len % perRankBufferCount == 0) { // ccl不够用，要搬多轮的情况1: 能整除
            CalCountAndBlockOffset(perRankBufferCount, blockNumPerGroup, blockIdxInGroup, padCount, countMid, blockOffsetMid);
            countTail = countMid;
            blockOffsetTail = blockOffsetMid;
        } else { // ccl不够用，要搬多轮的情况2: 不能整除
            CalCountAndBlockOffset(perRankBufferCount, blockNumPerGroup, blockIdxInGroup, padCount, countMid, blockOffsetMid);
            uint64_t remainLen = len % perRankBufferCount;
            CalCountAndBlockOffset(remainLen, blockNumPerGroup, blockIdxInGroup, padCount, countTail, blockOffsetTail);
        }
        flagOffsetInGroup = blockIdxInGroup * FLAG_SIZE;
        CalCountAndBlockOffset(len, blockNumPerGroup, blockIdxInGroup, padCount, countPerCore, blockOffset);
    }
    InitOffset();
}

// AG、RS图模式的Init
template<typename T>
__aicore__ inline void AivCrossNode91093Base::Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize,
    uint64_t len, uint32_t reduceOp, bool useDoubleBuffer)
{
    flagAddrSelf_ = buffOut0;

    rank_ = rank;
    rankSize_ = rankSize;
    reduceOp_ = reduceOp;
    useDoubleBuffer_ = useDoubleBuffer;
    usedBlockNum_ = block_num;

    InitSetCheckClearArgsTensor();

    // 以下根据不同情况，计算每个aiv核的数据搬运参数
    // 当rankSize大于总aiv核数的一半时，使用1个aiv服务一个对端，需要多次通信
    if (rankSize > HALF_MAX_BLOCK_DIM) {
        CalcNumTargetsAndTargetRanks();

        blockNumPerGroup = 1;
        flagOffsetInGroup = 0;
        countPerCore = len;
        blockOffset = 0;
    
    // 当rankSize小于等于总aiv核数的一半时，根据ranksize和数据量大小选择使用多个aiv服务一个对端（多核并行），只需一次通信
    } else {
        numTargets = 1;
        blockNumPerGroup = block_num / rankSize_; // 多少个aiv服务一个rank
        targetRanks[0] = block_idx / blockNumPerGroup;

        uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
        blockIdxInGroup = block_idx % blockNumPerGroup;

        flagOffsetInGroup = blockIdxInGroup * FLAG_SIZE;
        CalCountAndBlockOffset(len, blockNumPerGroup, blockIdxInGroup, padCount, countPerCore, blockOffset);
    }
    InitOffset();
}

template<typename T>
__aicore__ inline void AivCrossNode91093Base::SetAtomicOp(uint32_t atomicOp)
{
    switch (atomicOp) {
        case HcclReduceOp::HCCL_REDUCE_SUM:
            SetAtomicAdd<T>(); break;
        case HcclReduceOp::HCCL_REDUCE_MAX:
            SetAtomicMax<T>(); break;
        case HcclReduceOp::HCCL_REDUCE_MIN:
            SetAtomicMin<T>(); break;
        default:
            SetAtomicNone(); break;
    }
}

template<typename T>
__aicore__ inline void AivCrossNode91093Base::DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstLocal, srcGlobal, calCount);
    } else {
        // 结构体DataCopyExtParams最后一个参数是rsv保留位
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 1, 0};
        DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
    }
}

template<typename T>
__aicore__ inline void AivCrossNode91093Base::DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstGlobal, srcLocal, calCount);
    } else {
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    }
}

template<typename T>
__aicore__ inline void AivCrossNode91093Base::CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, bool atomic,
    uint32_t atomicOp)
{
    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);
    
    if (atomic) {
        SetAtomicOp<T>(atomicOp);
    }

    uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
    if (useDoubleBuffer_) {
        maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    }

    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);

        count -= curCount;
        curOffset += curCount;
    }

    if (atomic) {
        SetAtomicNone();
    }
    return;
}

template<HardEvent event> 
__aicore__ inline void AivCrossNode91093Base::SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

__aicore__ inline void AivCrossNode91093Base::SingleRecordBatchWaitCoreLevel(int32_t curTag, bool isTheSingleCore,
    AivNotifyType notifyType)
{
    if (isTheSingleCore) {
        Record1vN(curTag, CommPattern::intraRank, notifyType);
    // 其他核去读该flag，达到核间同步的目的
    } else {
        WaitNv1(curTag, flagAddrSelf_,true, notifyType);
    }
}

__aicore__ inline void AivCrossNode91093Base::BatchRecordSingleWaitCoreLevel(int32_t curTag, bool isTheSingleCore,
    AivNotifyType notifyType)
{
    // 负责localcopy的核去查该flag，等所有其他核已经完成写（原子累加）
    if (isTheSingleCore) {
        Wait1vN(curTag * (rankSize_ - 1), CommPattern::intraRank);
    // 其他核去写该flag，做原子累加达到核间同步的目的
    } else {   
        RecordNv1(curTag, flagAddrSelf_, true);
    }
}

__aicore__ inline void AivCrossNode91093Base::SingleRecordBatchWait(int32_t curTag, GM_ADDR* buffersOut, bool isTheSingleCore,
    AivNotifyType notifyType)
{
    if (isTheSingleCore) {
        Record1vN(curTag, CommPattern::interRank, notifyType);
    }
    for (uint32_t i = 0; i < numTargets; i++) {
        WaitNv1(curTag, buffersOut[i], false, notifyType);
    }
}

__aicore__ inline void AivCrossNode91093Base::BatchRecordWait(int32_t curTag, GM_ADDR* buffersOut, AivNotifyType notifyType)
{
    // 写所有对端的flag          
    for (uint32_t i = 0; i < numTargets; i++) {
       Record(curTag, buffersOut[i], notifyType);
    }
    // 读自己的所有flag
    for (uint32_t i = 0; i < numTargets; i++) {
       Wait(curTag, targetRanks[i], notifyType);
    }
}

__aicore__ inline void AivCrossNode91093Base::Record(uint32_t tag, GM_ADDR waitAddr, AivNotifyType notifyType)
{
    int32_t recordOffset = (blockIdxInGroup * rankSize_ * FLAG_BUF_NUM + int32_t(notifyType) * rankSize_ + rank_ ) * FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(waitAddr + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void AivCrossNode91093Base::Record1vN(uint32_t tag, CommPattern pattern, AivNotifyType notifyType)
{
    int32_t recordOffset = multiOffset + (int32_t(pattern) * 2 * blockNumPerGroup +
        int32_t(notifyType) * blockNumPerGroup + blockIdxInGroup) * ATOMIC_FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(flagAddrSelf_ + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void AivCrossNode91093Base::RecordNv1(uint32_t tag, GM_ADDR waitAddr, bool ifCoreLevel, AivNotifyType notifyType)
{
    int32_t recordOffset = multiOffset + 2 * 2 * blockNumPerGroup * ATOMIC_FLAG_SIZE +
        (int32_t(ifCoreLevel) * blockNumPerGroup * 2 + int32_t(notifyType) * blockNumPerGroup
        + blockIdxInGroup) * ATOMIC_FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(waitAddr + recordOffset);
    AddSignalValue(ctrlFlagGM, localSetTensor, tag);
}


__aicore__ inline void AivCrossNode91093Base::Wait(uint32_t tag, int32_t recordRank, AivNotifyType notifyType)
{
    int32_t waitOffset = (blockIdxInGroup * rankSize_ * FLAG_BUF_NUM + int32_t(notifyType) * rankSize_ + recordRank) * FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(flagAddrSelf_ + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
}

__aicore__ inline void AivCrossNode91093Base::WaitNv1(uint32_t tag, GM_ADDR recordAddr, bool ifCoreLevel, AivNotifyType notifyType)
{
    int32_t waitOffset = multiOffset + (int32_t(ifCoreLevel) * blockNumPerGroup * 2 +
        int32_t(notifyType) * blockNumPerGroup + blockIdxInGroup) * ATOMIC_FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(recordAddr + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
}
//是否直接清零
__aicore__ inline void AivCrossNode91093Base::Wait1vN(uint32_t tag, CommPattern pattern, bool ifClear, AivNotifyType notifyType)
{
    int32_t waitOffset = multiOffset + 2 * 2 * blockNumPerGroup * ATOMIC_FLAG_SIZE +
        (int32_t(pattern) * blockNumPerGroup * 2 +
        int32_t(notifyType) * blockNumPerGroup + blockIdxInGroup) * ATOMIC_FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(flagAddrSelf_ + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
    PipeBarrier<PIPE_ALL>();
    if (ifClear) {
        SetSignalValue(ctrlFlagGM, localSetTensor, 0);
    }
}


#endif  /* AIV_CROSSNODE_91093_BASE_H */