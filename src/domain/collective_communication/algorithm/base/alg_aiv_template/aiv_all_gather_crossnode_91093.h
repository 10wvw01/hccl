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
#include "aiv_crossnode_91093_base.h"

using namespace AscendC;

class AivAllGatherCrossNode91093 : public AivCrossNode91093Base {
public:
    __aicore__ inline AivAllGatherCrossNode91093() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input,
        GM_ADDR output, int32_t tag, uint64_t bufferCount, uint64_t len);
};

template<typename T>
__aicore__ inline void AivAllGatherCrossNode91093::Process(GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr,
    GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t bufferCount, uint64_t len)
{
    // 内存准备
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;
    __gm__ T *cclGMSelf = (__gm__ T *)buffIn0;

    GlobalTensor<uint64_t> bufferArgsGT;
    __gm__ uint64_t *buffersGmAddr = (__gm__ uint64_t *)(commInfoAddr);
    bufferArgsGT.SetGlobalBuffer(buffersGmAddr, FLAG_SIZE * rankSize_ / sizeof(uint64_t));

    // 准备参数，buffer地址和最大收发count
    GM_ADDR buffersIn[MAX_TARGET_NUM] = {};
    GM_ADDR buffersOut[MAX_TARGET_NUM] = {};

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank = targetRanks[i];
        DataCopy(bufferArgsTensor[i * 4], bufferArgsGT[2 * targetRank], 4); // buffersIn buffersOut
    }
    uint32_t bufferLoopNum = (len + bufferCount - 1) / bufferCount;

    SyncFunc<HardEvent::MTE2_S>();

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t curIdx = i * 4;
        buffersIn[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx));
        buffersOut[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx + 1));
    }

    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);
    uint64_t curOffset = 0;
    uint64_t curCount;
    uint64_t curBlockOffset;

    // Case1：当做不了multi-core并行搬运数据时(rankSize过大)，使用最后一个aiv做localcopy
    // Case2：当能做multi-core并行时，使用targetrank为本rank的aivs做localcopy
    bool isFirstLocalCopyCores = (rankSize_ > HALF_MAX_BLOCK_DIM && block_idx == block_num - 1) || (rankSize_ <= HALF_MAX_BLOCK_DIM && targetRanks[0] == rank_);
    
    // 只有Case1需要做最后的localcopy；对于Case2，它的localcopy已经夹在ccl->outout中了
    bool isSecondLocalCopyCore = (rankSize_ > HALF_MAX_BLOCK_DIM && block_idx == block_num - 1);

    for (uint32_t loop = 0; loop < bufferLoopNum; loop++) {
        if (loop == bufferLoopNum - 1) { // 最后一轮ccl填充
            curCount = countTail;
            curBlockOffset = blockOffsetTail;
        } else {
            curCount = countMid;
            curBlockOffset = blockOffsetMid;
        }

        PipeBarrier<PIPE_ALL>();

        if (isFirstLocalCopyCores) {
            CpGM2GM(cclGMSelf + curBlockOffset, inputGM + curOffset + curBlockOffset, curCount);
            PipeBarrier<PIPE_ALL>();
        }
        
        // 首次卡间同步，多等一（Case1/2目标核做完localcopy后告知其他卡所有remotecopy的核它完成了）
        SingleRecordBatchWait(curTag, buffersOut, isFirstLocalCopyCores);

        PipeBarrier<PIPE_ALL>();

        // 读对端ccl到usrout
        for (uint32_t i = 0; i < numTargets; i++) {
            __gm__ T *cclGMOther = (__gm__ T *)(buffersIn[i]);

            uint64_t localRecvOffset = len * targetRanks[i];
            CpGM2GM(outputGM + localRecvOffset + curOffset + curBlockOffset, cclGMOther + curBlockOffset, curCount);
        }

        PipeBarrier<PIPE_ALL>();

        // 结尾卡间同步，多等多（所有卡等待其他卡的remotecopy完成）
        BatchRecordWait(curTag, buffersOut, AivNotifyType::DataSignal);

        if (loop != bufferLoopNum - 1) {
            // 卡内核间同步，避免下一轮last core做localcopy时抢跑
            BatchRecordSingleWaitCoreLevel(curTag, isFirstLocalCopyCores);

            curTag += 1;
            curOffset += bufferCount;
        }
    }

    if (isSecondLocalCopyCore) {
        CpGM2GM(outputGM + rank_ * len, inputGM, len);
    }
}

template<typename T>
__aicore__ inline void aiv_all_gather_crossnode_91093(KERNEL_ARGS_DEF)
{
    AivAllGatherCrossNode91093 op;

    // 每张卡的CCLBuffer大小为bufferSize; bufferSize中能装下的数据个数为bufferCount
    uint64_t bufferCount = bufferSize / sizeof(T);
    
    op.Init<T>(buffOut0, rank, rankSize, bufferCount, len, reduceOp, true);
    op.InitOpCounter(headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter);
    op.HeadCounter();
    op.Process<T>(buffIn0, buffOut0, buffOut1, input, output, tag, bufferCount, len);
    op.TailCounter();
}