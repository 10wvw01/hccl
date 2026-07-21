/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCCL_CUSTOM_OP_H
#define HCCL_CUSTOM_OP_GE_HCCL_CUSTOM_OP_H

#include "graph/custom_op.h"

namespace hccl {

#define HCCL_GE_CHK_RET(stmt)                     \
    do {                                          \
        ge::graphStatus _ret = (stmt);            \
        if (_ret != ge::GRAPH_SUCCESS) {          \
            return _ret;                          \
        }                                         \
    } while (0)

class HcclCustomOpBase : public ge::EagerExecuteOp, public ge::ShapeInferOp {
 public:
  ~HcclCustomOpBase() override = default;

  ge::graphStatus Execute(gert::EagerOpExecutionContext *ctx) final;

 protected:
  gert::EagerOpExecutionContext *ctx_ = nullptr;

  virtual ge::graphStatus ExtractParams() = 0;
  virtual ge::graphStatus GetCommunicator() { return ge::GRAPH_SUCCESS; }
  virtual ge::graphStatus GetOptions() { return ge::GRAPH_SUCCESS; }
  virtual ge::graphStatus SelectAlgorithm() { return ge::GRAPH_SUCCESS; }
  virtual ge::graphStatus CalcResources() { return ge::GRAPH_SUCCESS; }
  virtual ge::graphStatus LaunchHcclOp() { return ge::GRAPH_SUCCESS; }
  virtual ge::graphStatus HandleOutput() { return ge::GRAPH_SUCCESS; }
};
}  // namespace hccl

#endif  // HCCL_CUSTOM_OP_GE_HCCL_CUSTOM_OP_H
