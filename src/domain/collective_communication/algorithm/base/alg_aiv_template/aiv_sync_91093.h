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
#include "aiv_crossnode_91093_base.h"

using namespace AscendC;

class AivSync91093 : public AivCrossNode91093Base {
public:
    __aicore__ inline AivSync91093() {}
    __aicore__ inline void Process(GM_ADDR buffOut0, GM_ADDR commInfoAddr, int32_t tag);

    __aicore__ inline void BatchRecordWaitWithClear(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag);
};

__aicore__ inline void AivSync91093::BatchRecordWaitWithClear(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag)
{
    // tx
    localSetTensor.SetValue(0, curTag);
    GlobalTensor<int32_t> globalTag;
    SyncFunc<HardEvent::S_MTE3>();
    for (uint32_t i = 0; i < numTargets; i++) {
        GM_ADDR flagAddrOther = buffersOut[i];
        globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrOther + flagOffset + rank_ * FLAG_SIZE),
            UB_FLAG_PAD_COUNT);
        DataCopy(globalTag, localSetTensor, UB_FLAG_PAD_COUNT);
    }
    // rx and clear
    for (uint32_t i = 0; i < numTargets; i++) {
        globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrSelf_ + flagOffset + targetRanks[i] * FLAG_SIZE),
            UB_FLAG_PAD_COUNT);
        while (true) {
            DataCopy(localCheckTensor, globalTag, UB_FLAG_PAD_COUNT);
            SyncFunc<HardEvent::MTE2_S>();
            if (localCheckTensor.GetValue(0) == curTag) {
                break;
            }
        }
        DataCopy(globalTag, localClearTensor, UB_FLAG_PAD_COUNT); //清零
    }
}

__aicore__ inline void AivSync91093::Process(GM_ADDR buffOut0, GM_ADDR commInfoAddr, int32_t tag)
{
    if (block_idx >= usedBlockNum_) {
        return;
    }
    
    uint32_t flagOffset = SYNC_BUFFER_OFFSET;
    flagOffset += ((tag % AIV_PING_PONG_FACTOR_TWO == 0) ? 0 : rankSize_ * FLAG_SIZE);

    GlobalTensor<uint64_t> bufferArgsGT;
    __gm__ uint64_t *buffersGmAddr = (__gm__ uint64_t *)(commInfoAddr);
    bufferArgsGT.SetGlobalBuffer(buffersGmAddr, FLAG_SIZE * rankSize_ / sizeof(uint64_t));

    // 准备参数，buffer地址
    GM_ADDR buffersOut[MAX_TARGET_NUM] = {};

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank = targetRanks[i];
        DataCopy(bufferArgsTensor[i * 4], bufferArgsGT[2 * targetRank], UB_ADDRESS_PAD_COUNT); // buffersIn buffersOut
    }

    SyncFunc<HardEvent::MTE2_S>();

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t curIdx = i * 4;
        buffersOut[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx + 1));
    }

    PipeBarrier<PIPE_ALL>();

    BatchRecordWaitWithClear(buffersOut, flagOffset, tag);
}

__aicore__ inline void aiv_sync_91093_inner(KERNEL_ARGS_DEF)
{
    AivSync91093 op;
    op.Init(buffOut0, rank, rankSize);
    op.Process(buffOut0, buffOut1, tag);
}
