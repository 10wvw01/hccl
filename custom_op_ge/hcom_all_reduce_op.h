/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCOM_ALL_REDUCE_OP_H
#define HCCL_CUSTOM_OP_GE_HCOM_ALL_REDUCE_OP_H

#include "hccl_custom_op.h"
#include <cstdint>

class HcomAllReduceOp : public hccl::HcclCustomOpBase {
 public:
  ~HcomAllReduceOp() override;

  static constexpr size_t INPUT_INDEX = 0;
  static constexpr size_t OUTPUT_INDEX = 0;
  static constexpr size_t ATTR_REDUCTION = 0;
  static constexpr size_t ATTR_GROUP = 1;
  static constexpr size_t ATTR_FUSION = 2;
  static constexpr size_t ATTR_FUSION_ID = 3;

 protected:
  ge::graphStatus ExtractParams(hccl::HcclOpState &st) override;
  ge::graphStatus CalcResources(hccl::HcclOpState &st) override;
  ge::graphStatus LaunchHcclOp(hccl::HcclOpState &st) override;
  ge::graphStatus InferShape(gert::InferShapeContext *ctx) override;
  ge::graphStatus InferDataType(gert::InferDataTypeContext *ctx) override;

 private:
  ge::graphStatus LaunchDirect(hccl::HcclOpState &st);
  ge::graphStatus LaunchLoop(hccl::HcclOpState &st);
  ge::graphStatus CreateIndirectCCLbuf();
  ge::graphStatus CleanCracks(void *baseAddr, uint64_t inputOffset);
  ge::graphStatus RefreshInputAddr(hccl::HcclOpState &st, uint64_t inputOffset, uint64_t curSize);
  ge::graphStatus RefreshOutputAddr(hccl::HcclOpState &st, uint64_t outputOffset, uint64_t curSize);

  void *indirectInCCLbuf_ = nullptr;
  void *indirectOutCCLbuf_ = nullptr;
  bool indirectBufInited_ = false;
};

#endif  // HCCL_CUSTOM_OP_GE_HCOM_ALL_REDUCE_OP_H
