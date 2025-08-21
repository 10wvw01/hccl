/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_ALLGATHER_H__
#define __AICPU_HCCL_ALLGATHER_H__

#include "aicpu_hccl_algorithm.h"

class AicpuHcclAllgather : public AicpuHcclAlgorithm {
public:
    explicit AicpuHcclAllgather(AicpuComContext *ctx) : AicpuHcclAlgorithm(ctx) {}
    ~AicpuHcclAllgather() override = default;

    HcclResult RunAlgorithm(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
                            HcclDataType dataType, u64 strideLen = 0, AivAicpuOpParam *nextTask = nullptr) override;
private:
    HcclResult RunAllGatherv(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
                             HcclDataType dataType, u64 strideCnt);
    HcclResult RunAllGathervMC(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
                               HcclDataType dataType, u64 strideCnt, AivAicpuOpParam *nextTask);

    HcclResult GenRingTask(HcclReduceOp opType, u64 sndAddr, u64 rcvAddr, u64 gatherSize, HcclDataType dataType,
        uint32_t streamId, bool isClockwise, uint32_t step, bool isWindowLast) const;
    HcclResult RunDoubleRingAllGather(HcclReduceOp opType, u64 sendBuffer,
        u64 recvBuffer, u64 dataCount, HcclDataType dataType) const;

    u64 GetWindowOffset(u32 curTurnCnt, u64 curSize, u64 strideCnt, u64 recvBuffer);

    static bool isMCFirstCall;
};

#endif