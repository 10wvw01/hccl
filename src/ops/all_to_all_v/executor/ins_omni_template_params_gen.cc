/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_omni_template_params_gen.h"
#include "log.h"

namespace ops_hccl {
namespace omni {

HcclResult InsOmniTemplateParamsGenerator::StartInstruction(
    const OmniSendRecvInfo& instructionInfo,
    const OpParam& param,
    const AlgResourceCtxSerializable& resCtx,
    const OmniParamGenConfig& config)
{
    // 参数验证
    if (currentInstruction_ != nullptr) {
        HCCL_ERROR("[InsOmniTemplateParamsGenerator][StartInstruction] Another instruction is already in progress");
        return HCCL_E_PARA;
    }

    if (config.dataSize == 0 || config.dataTypeSize == 0) {
        HCCL_ERROR("[InsOmniTemplateParamsGenerator][StartInstruction] Invalid config: dataSize=%lu, dataTypeSize=%lu",
                   config.dataSize, config.dataTypeSize);
        return HCCL_E_PARA;
    }

    if (config.sliceNum == 0) {
        HCCL_ERROR("[InsOmniTemplateParamsGenerator][StartInstruction] Invalid config: sliceNum=%lu",
                   config.sliceNum);
        return HCCL_E_PARA;
    }

    currentInstruction_ = &instructionInfo;
    currentParam_ = &param;
    currentResCtx_ = &resCtx;
    currentConfig_ = config;
    currentIndex_ = 0;
    isLast_ = false;

    // 初始化参数
    HcclResult ret = InitializeParams();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[InsOmniTemplateParamsGenerator][StartInstruction] InitializeParams failed: %d", ret);
        return ret;
    }

    HCCL_INFO("[InsOmniTemplateParamsGenerator][StartInstruction] Starting instruction type: %d",
              instructionInfo.optype);
    return HCCL_SUCCESS;
}

bool InsOmniTemplateParamsGenerator::HasNext() const
{
    return !isLast_;
}

HcclResult InsOmniTemplateParamsGenerator::GetNext(TemplateDataParams& params)
{
    if (isLast_) {
        HCCL_ERROR("[InsOmniTemplateParamsGenerator][GetNext] No more runs available");
        return HCCL_E_PARA;
    }

    if (currentInstruction_ == nullptr) {
        HCCL_ERROR("[InsOmniTemplateParamsGenerator][GetNext] No instruction in progress, call StartInstruction first");
        return HCCL_E_PARA;
    }

    // 根据当前执行索引更新参数
    HcclResult ret = UpdateParamsForCurrentRun();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[InsOmniTemplateParamsGenerator][GetNext] UpdateParamsForCurrentRun failed: %d", ret);
        return ret;
    }

    // 返回当前参数
    params = params_;

    HCCL_INFO("[InsOmniTemplateParamsGenerator][GetNext] Generated params for run %lu, isLast: %d",
              currentIndex_ + 1, isLast_);

    currentIndex_++;
    return HCCL_SUCCESS;
}

HcclResult InsOmniTemplateParamsGenerator::InitializeParams()
{
    // 参数检查
    if (currentParam_ == nullptr || currentResCtx_ == nullptr) {
        HCCL_ERROR("[InitializeParams] Invalid state: currentParam_ or currentResCtx_ is null");
        return HCCL_E_INTERNAL;
    }

    // 检查除零情况
    if (currentConfig_.dataTypeSize == 0) {
        HCCL_ERROR("[InitializeParams] Invalid dataTypeSize: %lu", currentConfig_.dataTypeSize);
        return HCCL_E_PARA;
    }

    if (currentConfig_.sliceNum == 0) {
        HCCL_ERROR("[InitializeParams] Invalid sliceNum: %lu", currentConfig_.sliceNum);
        return HCCL_E_PARA;
    }

    // 设置基本缓冲信息
    params_.buffInfo.inputPtr = currentParam_->inputPtr;
    params_.buffInfo.outputPtr = currentParam_->outputPtr;
    params_.buffInfo.hcclBuff = currentResCtx_->cclMem;
    params_.buffInfo.inBuffBaseOff = 0;
    params_.buffInfo.outBuffBaseOff = 0;
    params_.buffInfo.hcclBuffBaseOff = 0;
    params_.buffInfo.inBuffType = BufferType::INPUT;
    params_.buffInfo.outBuffType = BufferType::OUTPUT;

    // 设置数据参数
    params_.count = currentConfig_.dataSize / currentConfig_.dataTypeSize;
    params_.sliceSize = currentConfig_.dataSize / currentConfig_.sliceNum;
    params_.dataType = currentConfig_.dataType;

    // 设置重复参数
    params_.repeatNum = 1;
    params_.inputRepeatStride = 0;
    params_.outputRepeatStride = 0;

    HCCL_INFO("[InitializeParams] dataSize=%lu, sliceNum=%lu, sliceSize=%lu, count=%lu",
              currentConfig_.dataSize, currentConfig_.sliceNum, params_.sliceSize, params_.count);
    return HCCL_SUCCESS;
}

HcclResult InsOmniTemplateParamsGenerator::UpdateParamsForCurrentRun()
{
    if (!currentInstruction_) {
        HCCL_ERROR("[UpdateParamsForCurrentRun] No current instruction");
        return HCCL_E_PARA;
    }

    // 根据指令类型设置切片相关参数
    if (!currentInstruction_->srcSliceInfo.empty() || !currentInstruction_->dstSliceInfo.empty()) {
        // 检查除零错误
        if (currentConfig_.sliceNum == 0) {
            HCCL_ERROR("[UpdateParamsForCurrentRun] Invalid sliceNum for stride calculation: %lu",
                       currentConfig_.sliceNum);
            return HCCL_E_PARA;
        }

        // 如果有切片信息，可以设置切片偏移
        params_.inputSliceStride = currentConfig_.dataSize / currentConfig_.sliceNum;
        params_.outputSliceStride = currentConfig_.dataSize / currentConfig_.sliceNum;

        HCCL_INFO("[UpdateParamsForCurrentRun] inputSliceStride=%lu, outputSliceStride=%lu",
                  params_.inputSliceStride, params_.outputSliceStride);
    }

    // 根据操作类型可能需要设置其他参数
    switch (currentInstruction_->optype) {
        case OP_GROUP_BROAD_CAST:
        case OP_GROUP_REDUCE:
            // 组操作可能需要特殊的参数设置
            HCCL_INFO("[UpdateParamsForCurrentRun] Setting up for group operation, run %lu",
                     currentIndex_ + 1);
            break;

        case OP_LOCAL_REDUCE:
        case OP_SEND_RECV_WRITE_REDUCE:
        case OP_SEND_WRITE_REDUCE:
        case OP_RECV_WRITE_REDUCE:
        case OP_SEND_RECV_READ_REDUCE:
        case OP_SEND_READ_REDUCE:
        case OP_RECV_READ_REDUCE:
            // 归约操作可能需要设置归约类型
            HCCL_INFO("[UpdateParamsForCurrentRun] Setting up for reduce operation");
            break;

        default:
            // 其他操作使用默认参数
            break;
    }
    isLast_ = true;
    return HCCL_SUCCESS;
}

} // namespace omni
} // namespace ops_hccl