/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef AIV_AG_SUPERKERNEL_H
#define AIV_AG_SUPERKERNEL_H
 
#include "aiv_communication_base.h"
#include "aiv_all_gather_91093_smalldata.h"
 
extern "C" __aicore__ void sk_allgather(SUPERKERNEL_LITE_ARGS_DEF) {
    SUPERKERNEL_LITE_ARGS_EXTRACT;
    return sk_all_gather_91093_smalldata(SUPERKERNEL_ARGS_CALL);
}
 
 
#endif  /* AIV_AG_SUPERKERNEL_H */