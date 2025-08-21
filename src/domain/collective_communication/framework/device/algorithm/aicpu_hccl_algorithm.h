/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_ALGORITHM_H__
#define __AICPU_HCCL_ALGORITHM_H__

#include <memory>

#include "aicpu_hccl_dispatcher.h"

constexpr uint32_t RING_NUM = 2U;

class AicpuHcclAlgorithm {
public:
    explicit AicpuHcclAlgorithm(AicpuComContext *ctx)
    {
        ctx_ = ctx;
        rankId_ = ctx_ != nullptr ? ctx_->rankId : 0U;
        rankNum_ = ctx_ != nullptr ? ctx_->rankNum : 0U;
        unitSize_ = ctx_ != nullptr ? ctx_->unitSize : 0U;
        turn_ = ctx_ != nullptr ? ctx_->curTurnCnt : 0U;
    }
    virtual ~AicpuHcclAlgorithm() = default;

    virtual HcclResult RunAlgorithm(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
        HcclDataType dataType, u64 strideLen = 0, AivAicpuOpParam *nextTask = nullptr) = 0;

protected:
    AicpuComContext *ctx_ { nullptr };
    uint32_t rankId_ { 0U };
    uint32_t rankNum_ { 0U };
    uint32_t unitSize_ { 0U };
    uint32_t turn_ { 0U };
};

#endif