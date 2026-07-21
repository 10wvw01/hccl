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
#include "hccl/hccl_comm.h"
#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "log.h"

extern "C" {
HcclResult HcclCreateOpParamGraphMode(void **opParam);
HcclResult HcclDestroyOpParamGraphMode(void *opParam);
HcclResult HcclSetOpParamGraphModeOpType(void *opParam, const char *opType);
HcclResult HcclSetOpParamGraphModeDataCount(void *opParam, const uint64_t *dataCount);
HcclResult HcclSetOpParamGraphModeRankSize(void *opParam, const uint32_t *rankSize);
HcclResult HcclSetOpParamGraphModeHCCLBufferSize(void *opParam, const uint64_t *hcclBufferSize);
HcclResult HcclSetOpParamGraphModeDataType(void *opParam, HcclDataType dataType);
HcclResult HcclSetAivSelectOpParamGraphMode(void *opParam, uint32_t aivCoreLimit);
HcclResult HcclSelectAlgGraphMode(const char *group, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                                  HcclCMDType opType, uint32_t aivCoreLimit, bool *ifAiv, char *algName);
HcclResult HcclCalcOpResOfflineGraphMode(void *opParam, uint64_t *opMemSize, uint32_t *streamNum,
                                         uint32_t *taskNum, uint32_t *aivCoreNum);
}

ge::graphStatus HcomAllReduceOp::ExtractParams(hccl::HcclOpState &st)
{
    const gert::Tensor *inputTensor = st.ctx->GetInputTensor(INPUT_INDEX);
    if (inputTensor == nullptr) {
        HCCL_ERROR("HcomAllReduceOp::ExtractParams: input tensor is null.");
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("HcomAllReduceOp::ExtractParams: attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const char *reduction = attrs->GetStr(ATTR_REDUCTION);
    const char *group = attrs->GetStr(ATTR_GROUP);
    const int64_t *fusion = attrs->GetInt(ATTR_FUSION);
    const int64_t *fusionId = attrs->GetInt(ATTR_FUSION_ID);

    st.dataType = hccl::GeDataTypeToHccl(inputTensor->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR("HcomAllReduceOp::ExtractParams: unsupported data type %d.",
                   static_cast<int>(inputTensor->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    st.reduceOp = hccl::StringToReduceOp(reduction);
    if (st.reduceOp == HCCL_REDUCE_RESERVED) {
        HCCL_ERROR("HcomAllReduceOp::ExtractParams: unsupported reduction '%s'.",
                   reduction ? reduction : "(null)");
        return ge::GRAPH_FAILED;
    }

    gert::Tensor *outputTensor =
        st.ctx->MallocOutputTensor(OUTPUT_INDEX, inputTensor->GetShape(), inputTensor->GetFormat(), inputTensor->GetDataType());
    if (outputTensor == nullptr) {
        HCCL_ERROR("HcomAllReduceOp::ExtractParams: failed to allocate output tensor.");
        return ge::GRAPH_FAILED;
    }

    st.inputPtr = const_cast<void *>(inputTensor->GetAddr());
    st.outputPtr = outputTensor->GetAddr();
    st.count = static_cast<uint64_t>(inputTensor->GetShapeSize());
    st.group = group;
    st.stream = st.ctx->GetStream();

    HCCL_INFO("HcomAllReduceOp::ExtractParams: input=%p output=%p count=%lu dataType=%d reduceOp=%d group=%s stream=%p "
              "fusion=%ld fusionId=%ld.",
              st.inputPtr, st.outputPtr, st.count, st.dataType, st.reduceOp,
              st.group ? st.group : "(null)", st.stream, fusion ? *fusion : -1,
              fusionId ? *fusionId : -1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::CalcResources(hccl::HcclOpState &st)
{
    uint32_t rankSize = 0;
    HcclResult ret = HcclGetRankSize(st.comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CalcResources: HcclGetRankSize failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    uint64_t cclBuffSize = 0;
    ret = HcomGetCommCCLBufferSize(st.group, cclBuffSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CalcResources: HcomGetCommCCLBufferSize failed for group '%s', ret=%d.",
                   st.group, static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    void *opParam = nullptr;
    ret = HcclCreateOpParamGraphMode(&opParam);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CalcResources: HcclCreateOpParamGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HcclSetAivSelectOpParamGraphMode(opParam, 0);
    HcclSetOpParamGraphModeOpType(opParam, HCCL_KERNEL_OP_TYPE_ALLREDUCE.c_str());
    HcclSetOpParamGraphModeDataType(opParam, st.dataType);
    HcclSetOpParamGraphModeRankSize(opParam, &rankSize);
    HcclSetOpParamGraphModeDataCount(opParam, &st.count);
    HcclSetOpParamGraphModeHCCLBufferSize(opParam, &cclBuffSize);

    uint64_t opMemSize = 0;
    uint32_t taskNum = 0;
    uint32_t aivCoreNum = 0;
    ret = HcclCalcOpResOfflineGraphMode(opParam, &opMemSize, &st.streamNum, &taskNum, &aivCoreNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CalcResources: HcclCalcOpResOfflineGraphMode failed, ret=%d.", static_cast<int>(ret));
        HcclDestroyOpParamGraphMode(opParam);
        return ge::GRAPH_FAILED;
    }

    char algName[ALG_NAME_MAX_LEN] = {0};
    ret = HcclSelectAlgGraphMode(st.group, st.count, st.dataType, st.reduceOp, HCCL_CMD_ALLREDUCE, 0, &st.ifAiv, algName);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CalcResources: HcclSelectAlgGraphMode failed, ret=%d.", static_cast<int>(ret));
        HcclDestroyOpParamGraphMode(opParam);
        return ge::GRAPH_FAILED;
    }

    if (st.ifAiv) {
        constexpr uint32_t AIV_WORKSPACE_MEM_SIZE = 512;
        constexpr uint32_t AIV_TASK_NUM = 3;
        st.streamNum = 0;
        opMemSize = AIV_WORKSPACE_MEM_SIZE;
        taskNum = AIV_TASK_NUM;
    }

    HcclDestroyOpParamGraphMode(opParam);

    st.scratchMemSize = opMemSize;

    HCCL_INFO("HcomAllReduceOp::CalcResources: rankSize=%u scratchMemSize=%lu streamNum=%u taskNum=%u ifAiv=%d.",
              rankSize, st.scratchMemSize, st.streamNum, taskNum, st.ifAiv);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::InferShape(gert::InferShapeContext *ctx)
{
    const gert::Shape *inputShape = ctx->GetInputShape(INPUT_INDEX);
    if (inputShape == nullptr) {
        HCCL_ERROR("HcomAllReduceOp::InferShape: input shape is null.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape *outputShape = ctx->GetOutputShape(OUTPUT_INDEX);
    if (outputShape == nullptr) {
        HCCL_ERROR("HcomAllReduceOp::InferShape: output shape is null.");
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::InferDataType(gert::InferDataTypeContext *ctx)
{
    ge::DataType dataType = ctx->GetInputDataType(INPUT_INDEX);
    if (dataType == ge::DT_UNDEFINED) {
        HCCL_ERROR("HcomAllReduceOp::InferDataType: input data type is undefined.");
        return ge::GRAPH_FAILED;
    }
    return ctx->SetOutputDataType(OUTPUT_INDEX, dataType);
}

// TODO: 等 GE 提供支持指定 op type 字符串的注册宏后替换此手动注册
static ge::BaseCustomOp *CreateHcomAllReduceOp() { return new HcomAllReduceOp(); }
static const ge::CustomOpCreatorRegister g_hcom_all_reduce_register(
    HCCL_KERNEL_OP_TYPE_ALLREDUCE.c_str(), CreateHcomAllReduceOp);
