/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#include "hccl_op_utils.h"

#include <cstring>

namespace hccl {
HcclDataType GeDataTypeToHccl(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:
            return HCCL_DATA_TYPE_FP32;
        case ge::DT_FLOAT16:
            return HCCL_DATA_TYPE_FP16;
        case ge::DT_INT8:
            return HCCL_DATA_TYPE_INT8;
        case ge::DT_INT16:
            return HCCL_DATA_TYPE_INT16;
        case ge::DT_INT32:
            return HCCL_DATA_TYPE_INT32;
        case ge::DT_INT64:
            return HCCL_DATA_TYPE_INT64;
        case ge::DT_UINT8:
            return HCCL_DATA_TYPE_UINT8;
        case ge::DT_UINT16:
            return HCCL_DATA_TYPE_UINT16;
        case ge::DT_UINT32:
            return HCCL_DATA_TYPE_UINT32;
        case ge::DT_UINT64:
            return HCCL_DATA_TYPE_UINT64;
        case ge::DT_DOUBLE:
            return HCCL_DATA_TYPE_FP64;
        case ge::DT_BF16:
            return HCCL_DATA_TYPE_BFP16;
        default:
            return HCCL_DATA_TYPE_RESERVED;
    }
}

HcclReduceOp StringToReduceOp(const char *reduction)
{
    if (reduction == nullptr) {
        return HCCL_REDUCE_RESERVED;
    }
    if (strcmp(reduction, "sum") == 0) {
        return HCCL_REDUCE_SUM;
    }
    if (strcmp(reduction, "prod") == 0) {
        return HCCL_REDUCE_PROD;
    }
    if (strcmp(reduction, "max") == 0) {
        return HCCL_REDUCE_MAX;
    }
    if (strcmp(reduction, "min") == 0) {
        return HCCL_REDUCE_MIN;
    }
    return HCCL_REDUCE_RESERVED;
}
}  // namespace hccl
