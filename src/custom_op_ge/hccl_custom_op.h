/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCCL_CUSTOM_OP_H
#define HCCL_CUSTOM_OP_GE_HCCL_CUSTOM_OP_H

#include "graph/custom_op.h"

namespace hccl {
/**
 * HCCL custom op 基类。
 * 所有 HCCL 通信算子（AllReduce/AllGather/Broadcast/...）的 GE EagerExecuteOp 适配层
 * 继承此基类，共享 Execute 流程模板（ExtractParams → GetCommunicator → ... → HandleOutput）。
 */
class HcclCustomOpBase : public ge::EagerExecuteOp {
 public:
  ~HcclCustomOpBase() override = default;

  ge::graphStatus Execute(gert::EagerOpExecutionContext *ctx) override;
};
}  // namespace hccl

#endif  // HCCL_CUSTOM_OP_GE_HCCL_CUSTOM_OP_H
