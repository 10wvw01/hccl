/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DFX_QUEUE_READ_KERNEL_H
#define HCCL_DFX_QUEUE_READ_KERNEL_H

#include "kernel_operator.h"

using namespace AscendC;

#define EXPORT_AIV_META_INFO(kernel_name) \
static const struct FunLevelKType kernel_name##_kernel_type_section __attribute__ \
((used, section (".ascend.meta." #kernel_name))) \
= {{F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}}

constexpr uint32_t HCCL_DFX_QUEUE_READ_MAX_BYTES = 256U;
constexpr uint32_t HCCL_DFX_QUEUE_READ_SUCCESS = 0U;
constexpr uint32_t HCCL_DFX_QUEUE_READ_INVALID_PARAM = 1U;

struct HcclDfxQueueReadArgs {
    uint64_t srcAddr;
    uint64_t resultAddr;
    uint32_t size;
    uint32_t reserved;
};

struct HcclDfxQueueReadResult {
    uint32_t ret;
    uint32_t size;
    uint8_t data[HCCL_DFX_QUEUE_READ_MAX_BYTES];
};

extern "C" __global__ __aicore__ void hccl_dfx_queue_read_kernel(HcclDfxQueueReadArgs args)
{
    if (GetBlockIdx() != 0) {
        return;
    }

    __gm__ HcclDfxQueueReadResult *result =
        reinterpret_cast<__gm__ HcclDfxQueueReadResult *>(args.resultAddr);
    if (result == nullptr) {
        return;
    }

    result->ret = HCCL_DFX_QUEUE_READ_INVALID_PARAM;
    result->size = 0;
    if (args.srcAddr == 0 || args.size == 0 || args.size > HCCL_DFX_QUEUE_READ_MAX_BYTES) {
        return;
    }

    __gm__ uint8_t *src = reinterpret_cast<__gm__ uint8_t *>(args.srcAddr);
    for (uint32_t idx = 0; idx < args.size; ++idx) {
        result->data[idx] = src[idx];
    }
    result->size = args.size;
    result->ret = HCCL_DFX_QUEUE_READ_SUCCESS;
}

EXPORT_AIV_META_INFO(hccl_dfx_queue_read_kernel);

#endif // HCCL_DFX_QUEUE_READ_KERNEL_H
