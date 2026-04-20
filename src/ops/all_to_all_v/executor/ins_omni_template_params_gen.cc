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

TemplateDataParams InsOmniTemplateParamsGenerator::GenerateForSync(
    const OmniSyncInfo& syncInfo,
    const OpParam& param,
    const AlgResourceCtxSerializable& resCtx,
    const OmniParamGenConfig& config)
{
    HCCL_INFO("[InsOmniTemplateParamsGenerator][GenerateForSync] Generating template params for sync operation type: %d",
              syncInfo.optype);

    TemplateDataParams params;

    // 填充通用参数
    PopulateCommonParams(params, param, resCtx, config);

    // 同步操作通常不需要额外的切片参数
    HCCL_INFO("[InsOmniTemplateParamsGenerator][GenerateForSync] Template params generated");
    return params;
}

TemplateDataParams InsOmniTemplateParamsGenerator::GenerateForInstruction(
    const OmniSendRecvInfo& instructionInfo,
    const OpParam& param,
    const AlgResourceCtxSerializable& resCtx,
    const OmniParamGenConfig& config)
{
    HCCL_INFO("[InsOmniTemplateParamsGenerator][GenerateForInstruction] Generating template params for instruction operation type: %d",
              instructionInfo.optype);

    TemplateDataParams params;

    // 填充通用参数
    PopulateCommonParams(params, param, resCtx, config);

    // 根据指令类型设置切片参数
    SetupSliceParamsForInstruction(params, instructionInfo, config);

    HCCL_INFO("[InsOmniTemplateParamsGenerator][GenerateForInstruction] Template params generated");
    return params;
}

void InsOmniTemplateParamsGenerator::PopulateCommonParams(
    TemplateDataParams& params,
    const OpParam& param,
    const AlgResourceCtxSerializable& resCtx,
    const OmniParamGenConfig& config)
{
    // 设置基本缓冲信息
    params.buffInfo.inputPtr = param.inputPtr;
    params.buffInfo.outputPtr = param.outputPtr;
    params.buffInfo.hcclBuff = resCtx.cclMem;
    params.buffInfo.inBuffBaseOff = 0;
    params.buffInfo.outBuffBaseOff = 0;
    params.buffInfo.hcclBuffBaseOff = 0;
    params.buffInfo.inBuffType = BufferType::INPUT;
    params.buffInfo.outBuffType = BufferType::OUTPUT;

    // 设置数据参数
    params.count = config.dataSize / config.dataTypeSize;
    params.sliceSize = config.dataSize / config.sliceNum;
    params.dataType = config.dataType;

    // 设置重复参数
    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;

    HCCL_INFO("[PopulateCommonParams] dataSize=%lu, sliceNum=%lu, sliceSize=%lu, count=%lu",
              config.dataSize, config.sliceNum, params.sliceSize, params.count);
}

void InsOmniTemplateParamsGenerator::SetupSliceParamsForInstruction(
    TemplateDataParams& params,
    const OmniSendRecvInfo& instructionInfo,
    const OmniParamGenConfig& config)
{
    // 根据指令类型设置切片相关参数
    if (!instructionInfo.srcSliceInfo.empty() || !instructionInfo.dstSliceInfo.empty()) {
        // 如果有切片信息，可以设置切片偏移
        params.inputSliceStride = config.dataSize / config.sliceNum;
        params.outputSliceStride = config.dataSize / config.sliceNum;

        HCCL_INFO("[SetupSliceParamsForInstruction] inputSliceStride=%lu, outputSliceStride=%lu",
                  params.inputSliceStride, params.outputSliceStride);
    }

    // 根据操作类型可能需要设置其他参数
    switch (instructionInfo.optype) {
        case OP_GROUP_BROAD_CAST:
        case OP_GROUP_REDUCE:
            // 组操作可能需要特殊的参数设置
            HCCL_INFO("[SetupSliceParamsForInstruction] Setting up for group operation");
            break;

        case OP_LOCAL_REDUCE:
        case OP_SEND_RECV_WRITE_REDUCE:
        case OP_SEND_WRITE_REDUCE:
        case OP_RECV_WRITE_REDUCE:
        case OP_SEND_RECV_READ_REDUCE:
        case OP_SEND_READ_REDUCE:
        case OP_RECV_READ_REDUCE:
            // 归约操作可能需要设置归约类型
            HCCL_INFO("[SetupSliceParamsForInstruction] Setting up for reduce operation");
            break;

        default:
            // 其他操作使用默认参数
            break;
    }
}

} // namespace omni
} // namespace ops_hccl