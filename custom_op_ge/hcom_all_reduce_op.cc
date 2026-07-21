/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#include "hcom_all_reduce_op.h"

#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/shape.h"
#include "hccl_op_utils.h"
#include "log.h"

ge::graphStatus HcomAllReduce::ExtractParams()
{
    const gert::Tensor *inputTensor = ctx_->GetInputTensor(INPUT_INDEX);
    if (inputTensor == nullptr) {
        HCCL_ERROR("HcomAllReduce::ExtractParams: input tensor is null.");
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = ctx_->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("HcomAllReduce::ExtractParams: attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const char *reduction = attrs->GetStr(ATTR_REDUCTION);
    const char *group = attrs->GetStr(ATTR_GROUP);
    const int64_t *fusion = attrs->GetInt(ATTR_FUSION);
    const int64_t *fusionId = attrs->GetInt(ATTR_FUSION_ID);

    HcclDataType hcclDataType = hccl::GeDataTypeToHccl(inputTensor->GetDataType());
    if (hcclDataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR("HcomAllReduce::ExtractParams: unsupported data type %d.",
                   static_cast<int>(inputTensor->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    HcclReduceOp hcclReduceOp = hccl::StringToReduceOp(reduction);
    if (hcclReduceOp == HCCL_REDUCE_RESERVED) {
        HCCL_ERROR("HcomAllReduce::ExtractParams: unsupported reduction '%s'.",
                   reduction ? reduction : "(null)");
        return ge::GRAPH_FAILED;
    }

    gert::Tensor *outputTensor =
        ctx_->MallocOutputTensor(OUTPUT_INDEX, inputTensor->GetShape(), inputTensor->GetFormat(), inputTensor->GetDataType());
    if (outputTensor == nullptr) {
        HCCL_ERROR("HcomAllReduce::ExtractParams: failed to allocate output tensor.");
        return ge::GRAPH_FAILED;
    }

    params_.inputPtr = const_cast<void *>(inputTensor->GetAddr());
    params_.outputPtr = outputTensor->GetAddr();
    params_.count = static_cast<uint64_t>(inputTensor->GetShapeSize());
    params_.dataType = hcclDataType;
    params_.reduceOp = hcclReduceOp;
    params_.group = group;
    params_.stream = ctx_->GetStream();

    HCCL_INFO("HcomAllReduce::ExtractParams: input=%p output=%p count=%lu dataType=%d reduceOp=%d group=%s stream=%p "
              "fusion=%ld fusionId=%ld.",
              params_.inputPtr, params_.outputPtr, params_.count, params_.dataType, params_.reduceOp,
              params_.group ? params_.group : "(null)", params_.stream, fusion ? *fusion : -1,
              fusionId ? *fusionId : -1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduce::InferShape(gert::InferShapeContext *ctx)
{
    const gert::Shape *inputShape = ctx->GetInputShape(INPUT_INDEX);
    if (inputShape == nullptr) {
        HCCL_ERROR("HcomAllReduce::InferShape: input shape is null.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape *outputShape = ctx->GetOutputShape(OUTPUT_INDEX);
    if (outputShape == nullptr) {
        HCCL_ERROR("HcomAllReduce::InferShape: output shape is null.");
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduce::InferDataType(gert::InferDataTypeContext *ctx)
{
    ge::DataType dataType = ctx->GetInputDataType(INPUT_INDEX);
    if (dataType == ge::DT_UNDEFINED) {
        HCCL_ERROR("HcomAllReduce::InferDataType: input data type is undefined.");
        return ge::GRAPH_FAILED;
    }
    return ctx->SetOutputDataType(OUTPUT_INDEX, dataType);
}

REG_AUTO_MAPPING_OP(HcomAllReduce); 
