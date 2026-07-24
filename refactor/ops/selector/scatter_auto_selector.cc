/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "scatter_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {

// 触发 Parallel 的最小单卡数据量（与 src scatter_auto_selector 阈值保持一致）
constexpr u64 SCATTER_AICPU_1D_MIN_DATA_SIZE = 4 * 1024 * 1024;                  // 4MB
constexpr u64 SCATTER_AICPU_1D_TWO_LEVER_DATA_SIZE_THRESHOLD = 1 * 1024 * 1024 * 1024; // 1GB 触发 Concurrent

// 全局 AICPU Scatter 算法表（定义在 algorithm/scatter/algorithm_scatter_aicpu.cc），
// 以 HcclAicpuScatterAlgoType 枚举值为数组下标。extern 声明位于 hccl_algorithm.h。

SelectorStatus ScatterAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, HcclAlgorithm &alg) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[ScatterAutoSelector][%s] start, topoInfo topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);

    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    HcclAicpuScatterAlgoType selectAlgEnum = HcclAicpuScatterAlgoType::AICPU_SCATTER_ALGO_TYPE_COUNT;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            selectAlgEnum = HcclAicpuScatterAlgoType::AICPU_SCATTER_NHR;
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            // 多级 MESH_1D 拓扑：大数据量走 Parallel（两阶段并行），小数据量走 Mesh1D
            if (dataSize > SCATTER_AICPU_1D_MIN_DATA_SIZE) {
                selectAlgEnum = HcclAicpuScatterAlgoType::AICPU_SCATTER_PARALLEL_MESH1D_NHR;
            } else {
                selectAlgEnum = HcclAicpuScatterAlgoType::AICPU_SCATTER_MESH1D;
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            HCCL_WARNING("[ScatterAutoSelector] level0Shape[%d] is not supported yet for levelNum > 1.",
                         topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING("[ScatterAutoSelector] topo not match for aicpu algo");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            // 单级 MESH_1D：两层网络大数据量切到 Concurrent
            if (IsTwoLevelNetLayer(topoInfo) &&
                dataSize * topoInfo->userRankSize > SCATTER_AICPU_1D_TWO_LEVER_DATA_SIZE_THRESHOLD) {
                selectAlgEnum = HcclAicpuScatterAlgoType::AICPU_SCATTER_CONCURRENT_MESH1D_NHR;
            } else {
                selectAlgEnum = HcclAicpuScatterAlgoType::AICPU_SCATTER_MESH1D;
            }
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                // MESH_1D 即可链接所有卡，使用 MESH_1D 算法
                selectAlgEnum = HcclAicpuScatterAlgoType::AICPU_SCATTER_MESH1D;
            } else if (topoInfo->level0PcieMix) {
                selectAlgEnum = HcclAicpuScatterAlgoType::AICPU_SCATTER_PARALLEL_MESH1D_NHR_PCIE;
            } else {
                selectAlgEnum = HcclAicpuScatterAlgoType::AICPU_SCATTER_PARALLEL_MESH1D_NHR_UBX;
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            if (topoInfo->level0PcieMix) {
                selectAlgEnum = HcclAicpuScatterAlgoType::AICPU_SCATTER_NHR;
            } else {
                selectAlgEnum = HcclAicpuScatterAlgoType::AICPU_SCATTER_MESH1D;
            }
        } else {
            HCCL_WARNING("[ScatterAutoSelector] topo not match for aicpu algo");
            return SelectorStatus::NOT_MATCH;
        }
    }

    HCCL_INFO("[ScatterAutoSelector][%s] Algo match[%d]", __func__, static_cast<int>(selectAlgEnum));
    alg = g_aicpuScatterAlgoMap[static_cast<size_t>(selectAlgEnum)];
    return SelectorStatus::MATCH;
}

SelectorStatus ScatterAutoSelector::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails *topoInfo,
    const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    HcclAlgorithm &alg) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)alg;
    HCCL_WARNING("[ScatterAutoSelector][%s] not supported yet for ccu_schedule mode.", __func__);
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus ScatterAutoSelector::SelectDPUAlgo(const TopoInfoWithNetLayerDetails *topoInfo,
    const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    HcclAlgorithm &alg) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)alg;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus ScatterAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails *topoInfo,
    const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    HcclAlgorithm &alg) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)alg;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus ScatterAutoSelector::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails *topoInfo,
    const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    HcclAlgorithm &alg) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)alg;
    HCCL_WARNING("[ScatterAutoSelector][%s] not supported yet for ccu_ms mode.", __func__);
    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_SCATTER, 18, ScatterAutoSelector);

// 显式定义析构，强制编译器在本 .cc emit vtable（否则 vtable 是 undefined 符号，
// 跨 .so dlopen 时 dynamic linker 找不到）
ScatterAutoSelector::~ScatterAutoSelector() = default;

}  // namespace ops_hccl
