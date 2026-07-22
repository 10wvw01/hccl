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
#include "hccl/hccl_types.h"

namespace hccl {

#define HCCL_GE_CHK_RET(stmt)                     \
    do {                                          \
        ge::graphStatus _ret = (stmt);            \
        if (_ret != ge::GRAPH_SUCCESS) {          \
            return _ret;                          \
        }                                         \
    } while (0)

struct HcclOpState {
    gert::EagerOpExecutionContext *ctx = nullptr;
    HcclComm comm = nullptr;
    const char *group = nullptr;
    void *inputPtr = nullptr;
    void *outputPtr = nullptr;
    uint64_t count = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    void *stream = nullptr;
    void *scratchMem = nullptr;
    uint64_t scratchMemSize = 0;
    HcclReduceOp reduceOp = HCCL_REDUCE_RESERVED;
    uint32_t streamNum = 0;
    bool ifAiv = false;
    uint64_t cclBuffSize = 0;
};

class HcclCustomOpBase : public ge::EagerExecuteOp, public ge::ShapeInferOp {
 public:
  ~HcclCustomOpBase() override = default;

  ge::graphStatus Execute(gert::EagerOpExecutionContext *ctx) final;

 protected:
  virtual ge::graphStatus ExtractParams(HcclOpState &st) = 0;
  static ge::graphStatus GetCommunicator(HcclOpState &st);
  virtual ge::graphStatus CalcResources(HcclOpState &st) { return ge::GRAPH_SUCCESS; }
  virtual ge::graphStatus LaunchHcclOp(HcclOpState &st) { return ge::GRAPH_SUCCESS; }
  virtual ge::graphStatus HandleOutput(HcclOpState &st) { return ge::GRAPH_SUCCESS; }
};
}  // namespace hccl

#endif  // HCCL_CUSTOM_OP_GE_HCCL_CUSTOM_OP_H
