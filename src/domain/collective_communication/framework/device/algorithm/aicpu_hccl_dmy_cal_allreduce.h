/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_DMY_CAL_ALLREDUCE_H__
#define __AICPU_HCCL_DMY_CAL_ALLREDUCE_H__

#include "aicpu_hccl_algorithm.h"

class AicpuHcclDmyCalAllreduce : public AicpuHcclAlgorithm {
public:
    explicit AicpuHcclDmyCalAllreduce(AicpuComContext *ctx) : AicpuHcclAlgorithm(ctx) {}
    ~AicpuHcclDmyCalAllreduce() override = default;
    // determinacy calculate 确定性计算
    HcclResult RunAlgorithm(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
        HcclDataType dataType, u64 strideLen = 0, AivAicpuOpParam *nextTask = nullptr) override;
private:
    // allgather + localreduce
    HcclResult RunAllReduceAL(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
        HcclDataType dataType);
    // reducescatter + localreduce + allgather
    HcclResult RunAllReduceRLA(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
        HcclDataType dataType) const;
};


#endif