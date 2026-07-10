/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res_dl.h"
#include "ccu_launch_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

CcuResult HcommCcuGetMemToken(uint64_t srcVa, uint64_t size, uint64_t *tokenInfo)
{
    if (g_AsccommCcuGetMemToken == nullptr) {
        fprintf(stderr, "[HcclWrapper] %s not resolved\n", __func__);
        return CCU_E_UNAVAIL;
    }
    return g_AsccommCcuGetMemToken(srcVa, size, tokenInfo);
}

// 初始化
void CcuResDlInit(void* libHcommHandle) {
    (void)libHcommHandle;
}
