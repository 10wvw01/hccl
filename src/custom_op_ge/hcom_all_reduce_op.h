/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCOM_ALL_REDUCE_OP_H
#define HCCL_CUSTOM_OP_GE_HCOM_ALL_REDUCE_OP_H

#include "hccl_custom_op.h"

namespace hccl {
/**
 * HcomAllReduce 的 custom op 适配层。
 * 继承 HcclCustomOpBase，后续按需 override 各流程步骤。
 */
class HcclAllReduceOp : public HcclCustomOpBase {
 public:
  ~HcclAllReduceOp() override = default;
};
}  // namespace hccl

#endif  // HCCL_CUSTOM_OP_GE_HCOM_ALL_REDUCE_OP_H
