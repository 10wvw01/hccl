/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {

// 全局 AICPU ReduceScatter 算法表（定义在 algorithm/reduce_scatter/algorithm_reduce_scatter_aicpu.cc），
// 以 HcclAicpuReduceScatterAlgoType 枚举值为数组下标。extern 声明位于 hccl_algorithm.h。

SelectorStatus ReduceScatterAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, HcclAlgorithm &alg) const
{
    HCCL_DEBUG("[ReduceScatterAutoSelector][%s] start, topoInfo topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    HCCL_INFO("[ReduceScatterAutoSelector][SelectAicpuAlgo] topoLevelNums=[%d], deviceNumPerModule=[%d], "
              "level0Topo=[%d]",
              topoInfo->topoLevelNums, topoInfo->deviceNumPerModule, topoInfo->level0Topo);
    HcclAicpuReduceScatterAlgoType selectAlgEnum =
        HcclAicpuReduceScatterAlgoType::AICPU_REDUCESCATTER_ALGO_TYPE_COUNT;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->Level1Nhr) {
            selectAlgEnum = HcclAicpuReduceScatterAlgoType::AICPU_REDUCESCATTER_NHR;
            HCCL_INFO("[ReduceScatterAutoSelector] Level1Nhr=true, select [%d]", static_cast<int>(selectAlgEnum));
        } else if (topoInfo->Level0Nhr) {
            selectAlgEnum = HcclAicpuReduceScatterAlgoType::AICPU_REDUCESCATTER_NHR;
        } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            selectAlgEnum = HcclAicpuReduceScatterAlgoType::AICPU_REDUCESCATTER_NHR;
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgEnum = HcclAicpuReduceScatterAlgoType::AICPU_REDUCESCATTER_MESH1D;
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgEnum = HcclAicpuReduceScatterAlgoType::AICPU_REDUCESCATTER_NHR;
        } else {
            HCCL_ERROR("[ReduceScatterAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgEnum = HcclAicpuReduceScatterAlgoType::AICPU_REDUCESCATTER_MESH1D;
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            selectAlgEnum = HcclAicpuReduceScatterAlgoType::AICPU_REDUCESCATTER_NHR;
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgEnum = HcclAicpuReduceScatterAlgoType::AICPU_REDUCESCATTER_NHR;
        } else {
            HCCL_ERROR("[ReduceScatterAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_DEBUG("[ReduceScatterAutoSelector][%s] Algo match[%d]", __func__, static_cast<int>(selectAlgEnum));
    alg = g_aicpuReduceScatterAlgoMap[static_cast<size_t>(selectAlgEnum)];
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails *topoInfo,
    const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    HcclAlgorithm &alg) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)alg;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectDPUAlgo(const TopoInfoWithNetLayerDetails *topoInfo,
    const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    HcclAlgorithm &alg) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)alg;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails *topoInfo,
    const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    HcclAlgorithm &alg) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)alg;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails *topoInfo,
    const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    HcclAlgorithm &alg) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)alg;
    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, 18, ReduceScatterAutoSelector);

// 显式定义析构，强制编译器在本 .cc emit vtable（否则 vtable 是 undefined 符号，
// 跨 .so dlopen 时 dynamic linker 找不到）
ReduceScatterAutoSelector::~ReduceScatterAutoSelector() = default;

}  // ops_hccl
