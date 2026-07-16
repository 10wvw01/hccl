/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_launch.h"
#include "log.h"

namespace ops_hccl {
int32_t HcclLaunchDPUKernel(uint64_t ptr, int32_t size)
{
    if ((ptr == 0) || (size <= 0)) {
        HCCL_ERROR("%s get nullptr or error size", __func__);
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    // DPU kernel launch 需要依赖 template 注册框架（InsAlgTemplateRegistry），
    // 当前重构尚未引入该框架，保持桩实现。
    HCCL_ERROR("[HcclLaunchDPUKernel] DPU kernel launch not implemented in refactor");
    return static_cast<int32_t>(HCCL_E_NOT_SUPPORT);
}
}
