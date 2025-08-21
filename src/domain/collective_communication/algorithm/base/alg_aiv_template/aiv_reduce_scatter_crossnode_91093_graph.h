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

class AivReduceScatterCrossNodeGraph91093 : public AivCrossNode91093Base {
public:
    __aicore__ inline AivReduceScatterCrossNodeGraph91093() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
        uint64_t len);
};

template<typename T>
__aicore__ inline void AivReduceScatterCrossNodeGraph91093::Process(GM_ADDR buffOut0, GM_ADDR commInfoAddr,
    GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t len)
{
    // 内存准备
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;

    GlobalTensor<uint64_t> bufferArgsGT;
    __gm__ uint64_t *buffersGmAddr = (__gm__ uint64_t *)(commInfoAddr);
    bufferArgsGT.SetGlobalBuffer(buffersGmAddr, FLAG_SIZE * rankSize_ / sizeof(uint64_t));

    // 准备参数，buffer地址
    GM_ADDR buffersIn[MAX_TARGET_NUM] = {};
    GM_ADDR buffersOut[MAX_TARGET_NUM] = {};

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank = targetRanks[i];
        DataCopy(bufferArgsTensor[i * 4], bufferArgsGT[2 * targetRank], 4); // buffersIn buffersOut
    }

    SyncFunc<HardEvent::MTE2_S>();

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t curIdx = i * 4;
        buffersIn[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx));
        buffersOut[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx + 1));
    }

    // Case1：当做不了multi-core并行搬运数据时(rankSize过大)，使用最后一个aiv做localcopy
    // Case2：当能做multi-core并行时，使用targetrank为本rank的aivs做localcopy
    bool isLocalCopyCores = (rankSize_ > HALF_MAX_BLOCK_DIM && block_idx == block_num - 1) || (rankSize_ <= HALF_MAX_BLOCK_DIM && targetRanks[0] == rank_);

    // RS需要先保证input->output完成，再做remote copy进行原子累加
    if (isLocalCopyCores) {
        CpGM2GM(outputGM + blockOffset, inputGM + rank_ * len + blockOffset, countPerCore);
        PipeBarrier<PIPE_ALL>();
    }

    // localcopy后的卡内核间同步，多等一（Case1/2目标核做完localcopy后告知本卡其他核）
    SingleRecordBatchWaitCoreLevel(tag, isLocalCopyCores);

    PipeBarrier<PIPE_ALL>();

    // 首次卡间同步
    BatchRecordWait(tag, buffersOut);

    PipeBarrier<PIPE_ALL>();

    // 读对端userin到usrout
    for (uint32_t i = 0; i < numTargets && targetRanks[i] != rank_; i++) {
        __gm__ T *inputGMOther = (__gm__ T *)(buffersIn[i]);
        CpGM2GM(outputGM + blockOffset, inputGMOther + rank_ * len + blockOffset, countPerCore, true, reduceOp_);
    }

    PipeBarrier<PIPE_ALL>();

    // 结尾卡间同步
    BatchRecordWait(tag, buffersOut, AivNotifyType::DataSignal);
}

template<typename T>
__aicore__ inline void aiv_reduce_scatter_crossnode_91093_graph(KERNEL_ARGS_DEF)
{
    AivReduceScatterCrossNodeGraph91093 op;
    op.Init<T>(buffOut0, rank, rankSize, len, reduceOp, true);
    op.InitOpCounter(headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter);
    op.HeadCounter();
    op.Process<T>(buffOut0, buffOut1, input, output, tag, len);
    op.TailCounter();
}