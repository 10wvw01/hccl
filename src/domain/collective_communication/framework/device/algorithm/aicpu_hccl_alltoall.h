/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_ALLTOALL_H__
#define __AICPU_HCCL_ALLTOALL_H__

#include "aicpu_hccl_algorithm.h"

class AicpuHcclAllToAll : public AicpuHcclAlgorithm {
public:
    explicit AicpuHcclAllToAll(AicpuComContext *ctx) : AicpuHcclAlgorithm(ctx) {}
    ~AicpuHcclAllToAll() override = default;

    HcclResult RunAlgorithm(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
                            HcclDataType dataType, u64 strideLen = 0, AivAicpuOpParam *nextTask = nullptr) override;

private:
    HcclResult RunAllToAllWinIn(void *recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideLen);

    HcclResult RunAllToAll(void *sendBuffer, void *recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideLen);
};

#endif
