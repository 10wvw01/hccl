/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#include "hccl_custom_op.h"

namespace hccl {
ge::graphStatus HcclCustomOpBase::Execute(gert::EagerOpExecutionContext *ctx)
{
    ctx_ = ctx;
    HCCL_GE_CHK_RET(ExtractParams());
    HCCL_GE_CHK_RET(GetCommunicator());
    HCCL_GE_CHK_RET(GetOptions());
    HCCL_GE_CHK_RET(SelectAlgorithm());
    HCCL_GE_CHK_RET(CalcResources());
    HCCL_GE_CHK_RET(LaunchHcclOp());
    HCCL_GE_CHK_RET(HandleOutput());
    return ge::GRAPH_SUCCESS;
}
}  // namespace hccl
