/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ORDER_PRESERVED_COMMON_H
#define HCCL_ORDER_PRESERVED_COMMON_H

#include "alg_param.h"
#include "alg_env_config.h"
#include "log.h"

namespace ops_hccl {

constexpr u32 MIN_STRICT_RANK_NUM_ORDER_PRESERVED = 2;
constexpr u32 MAX_RANK_NUM_FOR_ORDER_PRESERVED = 32;

struct OrderPreservedBaseParams {
    u32 myRank;
    u32 rankSize;
    DevType devType;
    u64 dataCount;
    u32 dataTypeSize;
    u64 dataSize;
    HcclDataType dataType;
    HcclReduceOp reduceOp;
    u64 maxTmpMemSize;
};

inline OrderPreservedBaseParams InitOrderPreservedBaseParams(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    OrderPreservedBaseParams params;
    params.myRank = resCtx.topoInfo.userRank;
    params.rankSize = resCtx.topoInfo.userRankSize;
    params.devType = resCtx.topoInfo.deviceType;
    params.dataCount = param.DataDes.count;
    params.dataTypeSize = SIZE_TABLE[param.DataDes.dataType];
    params.dataSize = params.dataCount * params.dataTypeSize;
    params.dataType = param.DataDes.dataType;
    params.reduceOp = param.reduceType;
    params.maxTmpMemSize = resCtx.cclMem.size;
    return params;
}

// 规约保序支持的算子与数据类型组合：sum 支持 fp16/fp32/bfp16/fp64，prod 仅支持 fp64
inline bool IsOrderPreservedCombinationSupported(HcclDataType dataType, HcclReduceOp reduceType)
{
    if (reduceType == HcclReduceOp::HCCL_REDUCE_SUM) {
        return dataType == HcclDataType::HCCL_DATA_TYPE_FP16 ||
            dataType == HcclDataType::HCCL_DATA_TYPE_FP32 ||
            dataType == HcclDataType::HCCL_DATA_TYPE_BFP16 ||
            dataType == HcclDataType::HCCL_DATA_TYPE_FP64;
    }
    if (reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
        return dataType == HcclDataType::HCCL_DATA_TYPE_FP64;
    }
    return false;
}

// 是否处于保序意图：strict 已开启且 rankSize 超过触发阈值
inline bool IsOrderPreserveIntentActive(u32 rankSize)
{
    return (GetExternalInputHcclDeterministic() == static_cast<u8>(DeterministicEnableLevel::DETERMINISTIC_STRICT))
        && (rankSize > MIN_STRICT_RANK_NUM_ORDER_PRESERVED);
}

// 是否需要保序模式：保序意图且组合支持。CCU_MS/CCU_SCHED/AIV 等 guard 复用，命中即回退到 AICPU
inline bool IsNeedStrictModeForOrderPreserved(const OpParam& opParam, u32 rankSize)
{
    return IsOrderPreserveIntentActive(rankSize)
        && IsOrderPreservedCombinationSupported(opParam.DataDes.dataType, opParam.reduceType);
}

// 保序算子选择：组合不支持或 rankSize 超限时报错并返回 false，匹配则设置 selectAlgName 并返回 true
inline bool TrySelectOrderPreservedAlgo(const OpParam& opParam, u32 rankSize,
    const char* algoName, std::string& selectAlgName)
{
    if (!IsOrderPreservedCombinationSupported(opParam.DataDes.dataType, opParam.reduceType)) {
        HCCL_ERROR("[OrderPreserve] DETERMINISTIC_STRICT enabled but reduceOp[%d]+dataType[%d] not supported "
            "for order-preserve (supported: sum+fp16/fp32/bfp16/fp64, prod+fp64). Refuse to execute.",
            opParam.reduceType, opParam.DataDes.dataType);
        return false;
    }
    if (rankSize > MAX_RANK_NUM_FOR_ORDER_PRESERVED) {
        HCCL_ERROR("[OrderPreserve] OrderPreserved mode not supported for rankSize[%u] > %u.",
            rankSize, MAX_RANK_NUM_FOR_ORDER_PRESERVED);
        return false;
    }
    selectAlgName = algoName;
    HCCL_INFO("[OrderPreserve] DETERMINISTIC_STRICT mode, select [%s]", algoName);
    return true;
}

} // namespace ops_hccl

#endif // HCCL_ORDER_PRESERVED_COMMON_H