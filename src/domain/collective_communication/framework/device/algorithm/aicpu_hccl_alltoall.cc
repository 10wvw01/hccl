/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_hccl_alltoall.h"

#include "common/aicpu_hccl_common.h"
#include "framework/aicpu_rpc_server.h"

HcclResult AicpuHcclAllToAll::RunAlgorithm(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
                                           HcclDataType dataType, u64 strideLen, AivAicpuOpParam *nextTask)
{
    CHK_PTR_NULL(ctx_);
    if (ctx_->useBufferType == MC2_BUFFER_TYPE_WINDOW_IN) {
        return RunAllToAllWinIn(recvBuffer, dataCount, dataType, strideLen);
    } else {
        return RunAllToAll(sendBuffer, recvBuffer, dataCount, dataType, strideLen);
    }
}

HcclResult AicpuHcclAllToAll::RunAllToAllWinIn(void *recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideLen)
{
    u8 *curOutputPtr = static_cast<u8 *>(recvBuffer);
    u64 averageSize = dataCount * unitSize_; // 单位 byte
    u64 displs[AC_MAX_RANK_NUM] = {0};

    for (u32 i = 0; i < rankNum_; i++) {
        displs[i] = i * strideLen * unitSize_;
    }
    u64 winOffset = ctx_->winOffset + displs[rankId_];
    HCCL_INFO("dataCount %lu, strideLen %lu, winOffset %lu, unitSize_ %u", dataCount, strideLen, winOffset, unitSize_);

    // 1. 前同步
    CHK_RET(TaskOrchestrator::DoPreSync());

    // 2. 片内win->rcv, skipLocalDataCopy为true表示不需要本卡数据。
    if (!ctx_->skipLocalDataCopy) {
        CHK_RET(TaskOrchestrator::SelfCpyWin2Rcv(curOutputPtr, averageSize, winOffset, displs[rankId_],
                                                 HCCL_REDUCE_RESERVED, dataType));
    }

    // 3. 跨片SDMA，其他win->当前rcv buff
    CHK_RET(
        TaskOrchestrator::IpcCpyWin2Rcv(curOutputPtr, averageSize, winOffset, displs, HCCL_REDUCE_RESERVED, dataType));

    // 4. 后同步
    CHK_RET(TaskOrchestrator::DoPostSync());

    CHK_RET(TaskOrchestrator::LaunchTasks());

    return HCCL_SUCCESS;
}

HcclResult AicpuHcclAllToAll::RunAllToAll(void *sendBuffer, void *recvBuffer, u64 dataCount, HcclDataType dataType,
                                          u64 strideLen)
{
    u64 windowSize = ctx_->windowSize;
    u64 maxCountPerLoop = windowSize / unitSize_; // 中转内存单次最多能够接受的output count

    u8 *curInputPtr = static_cast<u8 *>(sendBuffer);
    u8 *curOutputPtr = static_cast<u8 *>(recvBuffer);
    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 countLeft = dataCount * rankNum_;

    u64 displs[AC_MAX_RANK_NUM] = {0};
    for (u32 i = 0; i < rankNum_; i++) {
        displs[i] = i * strideLen * unitSize_;
    }
    HCCL_INFO("windowSize %lu, maxCountPerLoop %lu, strideLen %lu, unitSize_ %u", windowSize, maxCountPerLoop,
              strideLen, unitSize_);

    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize_; // 单位 byte
        u64 averageSize = curSize / rankNum_;
        u64 winOffsets[AC_MAX_RANK_NUM] = {0};
        for (u32 i = 0; i < rankNum_; i++) {
            winOffsets[i] = i * averageSize;
        }
        HCCL_INFO("inputOffset %lu, outputOffset %lu, countLeft %lu, averageSize %lu", inputOffset,
                  outputOffset, countLeft, averageSize);

        // 1. 片内数据拷贝 snd->win
        for (u32 i = 0; i < rankNum_; i++) {
            if ((ctx_->skipLocalDataCopy) && (i == rankId_)) {
                continue;
            }

            CHK_RET(TaskOrchestrator::SelfCpySnd2Win(curInputPtr, averageSize, displs[i], winOffsets[i],
                                                     HCCL_REDUCE_RESERVED, dataType));
        }
             
        // 2. 前同步
        CHK_RET(TaskOrchestrator::DoPreSync());

        // 3. 片内win->rcv, skipLocalDataCopy为true表示不需要本卡数据。
        if (!ctx_->skipLocalDataCopy) {
            CHK_RET(TaskOrchestrator::SelfCpyWin2Rcv(curOutputPtr, averageSize, winOffsets[rankId_], displs[rankId_],
                                                     HCCL_REDUCE_RESERVED, dataType));
        }

        // 4. 跨片SDMA，其他win->当前rcv buff
        CHK_RET(TaskOrchestrator::IpcCpyWin2Rcv(curOutputPtr, averageSize, winOffsets[rankId_], displs,
                                                HCCL_REDUCE_RESERVED, dataType));

        // 5. 后同步
        CHK_RET(TaskOrchestrator::DoPostSync());

        CHK_RET(TaskOrchestrator::LaunchTasks());

        countLeft -= curCount;
        inputOffset = averageSize;
        outputOffset = averageSize;
    }

    return HCCL_SUCCESS;
}
