/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "jetty_strategy.h"

#include "log.h"

namespace ops_hccl {

// 地址对齐常量，与 examples/05_custom_ops_allgather 保持一致
constexpr u64 HCCL_MIN_SLICE_ALIGN = 128;

HcclResult SingleJettyStrategy::Decide(u64 sliceSize, u32 dataTypeSize, JettyDecision &decision)
{
    (void)dataTypeSize;
    decision.jettyNum = 1;
    decision.sliceSizePerJetty = sliceSize;
    decision.lastSliceSizePerJetty = sliceSize;
    return HCCL_SUCCESS;
}

HcclResult MultiJettyStrategy::Decide(u64 sliceSize, u32 dataTypeSize, JettyDecision &decision)
{
    decision.jettyNum = jettyNum_;
    if (jettyNum_ <= 1 || dataTypeSize == 0) {
        decision.sliceSizePerJetty = sliceSize;
        decision.lastSliceSizePerJetty = sliceSize;
        return HCCL_SUCCESS;
    }

    // 按 HCCL_MIN_SLICE_ALIGN 对齐切分（迁移自 ccu_temp_all_gather_nhr_1D_multi_jetty_mem2mem.cc:174-177）
    const u64 dataCount = sliceSize / dataTypeSize;
    const u64 alignInElements = HCCL_MIN_SLICE_ALIGN / dataTypeSize;
    if (alignInElements == 0) {
        HCCL_ERROR("[MultiJettyStrategy::Decide] dataTypeSize[%u] > HCCL_MIN_SLICE_ALIGN[%llu], cannot align",
                   dataTypeSize, HCCL_MIN_SLICE_ALIGN);
        return HCCL_E_PARA;
    }
    const u64 sliceCountPerJetty = (dataCount / jettyNum_ / alignInElements) * alignInElements;
    const u64 lastCountSizePerJetty = dataCount - sliceCountPerJetty * (jettyNum_ - 1);

    decision.sliceSizePerJetty = sliceCountPerJetty * dataTypeSize;
    decision.lastSliceSizePerJetty = lastCountSizePerJetty * dataTypeSize;

    HCCL_DEBUG("[MultiJettyStrategy::Decide] jettyNum=%u sliceSizePerJetty=%llu lastSliceSizePerJetty=%llu",
               decision.jettyNum, decision.sliceSizePerJetty, decision.lastSliceSizePerJetty);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
