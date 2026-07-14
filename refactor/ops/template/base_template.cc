/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_template.h"

#include "log.h"
#include "base_engine.h"
#include "engine/aicpu/aicpu_engine.h"

namespace ops_hccl {

BaseTemplate::~BaseTemplate()
{
    // engine_ 由 unique_ptr 自动释放（若后续改为智能指针）；
    // 当前为裸指针，按原有逻辑释放。
    delete engine_;
}

BaseEngine &BaseTemplate::GetEngine()
{
    if (!engine_) {
        switch (engineType_) {
            case HcclAlgEngineType::AICPU:
                engine_ = new AiCpuEngine();
                break;
            default:
                HCCL_ERROR("[BaseTemplate][GetEngine] unsupported engineType[%d]",
                           static_cast<int>(engineType_));
                break;
        }
    }
    return *engine_;
}

}  // namespace ops_hccl
