/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_hccl_dmy_cal_allreduce.h"
#include "common/aicpu_hccl_def.h"

HcclResult AicpuHcclDmyCalAllreduce::RunAlgorithm(HcclReduceOp opType, void *sendBuffer, void *recvBuffer,
    u64 dataCount, HcclDataType dataType, u64 strideLen, AivAicpuOpParam *)
{
    CHK_PTR_NULL(ctx_);
    HcclResult ret = HCCL_SUCCESS;
#ifdef RUN_TEST
    ctx_->windowSize = ALL_REDUCE_THRESHOLD_UT * ctx_->unitSize;
    if (ctx_->commLen < ALL_REDUCE_THRESHOLD_UT) {
#else
    if (ctx_->commLen < ALL_REDUCE_THRESHOLD) {
#endif
        ret = RunAllReduceAL(opType, sendBuffer, recvBuffer, dataCount, dataType);
    } else {
        const u64 tailCount = dataCount % (HCCL_COPY_ALIGN / ctx_->unitSize);
        const u64 tileCount = dataCount - tailCount;
        CHK_RET(RunAllReduceRLA(opType, sendBuffer, recvBuffer, tileCount, dataType));
        const u64 offset = tileCount * ctx_->unitSize;
        CHK_RET(RunAllReduceAL(opType, static_cast<u8 *>(sendBuffer) + offset,
            static_cast<u8 *>(recvBuffer) + offset, tailCount, dataType));
    }
    return ret;
}

HcclResult AicpuHcclDmyCalAllreduce::RunAllReduceAL(HcclReduceOp opType, void *sendBuffer, void *recvBuffer,
                                                    u64 dataCount, HcclDataType dataType) {
    if (dataCount == 0UL) {
        return HCCL_SUCCESS;
    }
    u32 unitSize = ctx_->unitSize;
    u8 *curInputPtr = static_cast<u8 *>(sendBuffer);
    u8 *curOutputPtr = static_cast<u8 *>(recvBuffer);

    u64 windowSlices[AC_MAX_RANK_NUM] = {0};
    u64 curCount = dataCount;
    u64 curSize = curCount * unitSize;

    HCCL_INFO("RunDmyCalAllreduceAL:curInputPtr[%p], curOutputPtr[%p], curCount[%llu], curSize[%llu]",
                     curInputPtr, curOutputPtr, curCount, curSize);
    
    for (u32 i = 0; i < ctx_->rankNum; i++) {
        windowSlices[i] = i * curSize;
    }

    // 1. 片内数据拷贝 snd->win
    TaskOrchestrator::SelfCpySnd2Win(curInputPtr, curSize, 0, windowSlices[ctx_->rankId], HCCL_REDUCE_RESERVED,
                                     dataType);

    // 2. 前同步
    TaskOrchestrator::DoPreSync();

    // 3. 跨片SDMA，send->对端win
    TaskOrchestrator::IpcCpySnd2Win(curInputPtr, curSize, static_cast<u64>(0), windowSlices[ctx_->rankId],
                                    HCCL_REDUCE_RESERVED, dataType);

    // 4. 后同步
    TaskOrchestrator::DoPostSync();

    // 5. 折半计算
    TaskOrchestrator::SelfLocalReduce(curSize, opType, dataType);

    // 6. 前同步
    TaskOrchestrator::DoPreSync();

    // 7. 片内数据拷贝 本端win->当前rcv buff
    TaskOrchestrator::SelfCpyWin2Rcv(curOutputPtr, curSize, 0, 0, HCCL_REDUCE_RESERVED, dataType);

    // 8. 后同步
    TaskOrchestrator::DoPostSync();

    TaskOrchestrator::LaunchTasks();
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclDmyCalAllreduce::RunAllReduceRLA(HcclReduceOp opType, void *sendBuffer, void *recvBuffer,
                                                    u64 dataCount, HcclDataType dataType) const {
    u64 windowSize = ctx_->windowSize;
    u32 unitSize = ctx_->unitSize;
    u64 maxCountPerLoop = windowSize / unitSize;

    u8 *curInputPtr = static_cast<u8 *>(sendBuffer);
    u8 *curOutputPtr = static_cast<u8 *>(recvBuffer);
    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 countLeft = dataCount;
    u64 windowSlices[AC_MAX_RANK_NUM] = {0};

    u32 loopIdx = 0;
    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize;
        u64 sliceSize = curSize / ctx_->rankNum;

        HCCL_INFO("DmyCalAllreduceRLA:loop %u, curInputPtr[%p], curOutputPtr[%p], curCount[%llu], curSize[%llu]",
                         loopIdx++, curInputPtr, curOutputPtr, curCount, curSize);
        
        for (u32 i = 0; i < ctx_->rankNum; i++) {
            windowSlices[i] = i * sliceSize;
        }

        // 1. 片内数据拷贝 snd->win
        TaskOrchestrator::SelfCpySnd2Win(curInputPtr, sliceSize, windowSlices[ctx_->rankId],
                                         windowSlices[ctx_->rankId], HCCL_REDUCE_RESERVED, dataType);
        // 2. 前同步
        TaskOrchestrator::DoPreSync();

        // 3. 跨片SDMA，send->其他window
        TaskOrchestrator::IpcCpySnd2Win(curInputPtr, sliceSize, windowSlices, windowSlices[ctx_->rankId],
                                        HCCL_REDUCE_RESERVED, dataType);

        // 4. 后同步
        TaskOrchestrator::DoPostSync();

        // 5. 折半计算
        TaskOrchestrator::SelfLocalReduce(sliceSize, opType, dataType);

        // 6. 片内数据拷贝 本端win->当前rcv buff
        TaskOrchestrator::SelfCpyWin2Rcv(curOutputPtr, sliceSize, 0, windowSlices[ctx_->rankId], HCCL_REDUCE_RESERVED,
                                         dataType);

        // 7. 前同步
        TaskOrchestrator::DoPreSync();

        // 8. 跨片SDMA send->其他window
        TaskOrchestrator::IpcCpyWin2Rcv(curOutputPtr, sliceSize, nullptr, windowSlices, HCCL_REDUCE_RESERVED, dataType);

        // 9. 后同步
        TaskOrchestrator::DoPostSync();

        TaskOrchestrator::LaunchTasks();

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}