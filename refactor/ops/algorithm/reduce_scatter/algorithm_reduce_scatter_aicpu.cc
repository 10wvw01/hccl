/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_algorithm.h"
#include "reduce_scatter_template_desc.h"
#include "topo_match_1d.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {

/**
 * 构造 AICPU ReduceScatter 算法描述对象。
 * 统一填充 hcclCmdType=HCCL_CMD_REDUCE_SCATTER、engineType=AICPU。
 * algoExecDesc 由调用方传入；不传则默认构造为空。
 */
static HcclAlgorithm MakeAicpuReduceScatterAlgo(
    std::shared_ptr<TopoMatchBase> topoMatch, AlgoExecDesc algoExecDesc = AlgoExecDesc{})
{
    HcclAlgorithm algo;
    algo.hcclCmdType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    algo.engineType = HcclAlgEngineType::AICPU;
    algo.topoMatch = std::move(topoMatch);
    algo.algoExecDesc = algoExecDesc;
    return algo;
}

/**
 * 构造 AICPU ReduceScatter NHR 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXEC_V2(..., InsReduceScatterNHR,
 *   InsV2ReduceScatterSoleExecutor, TopoMatch1D, InsTempReduceScatterNHR)。
 * execPolicy=SEQUENCE，children 仅一层 TemplateExecDesc（subCommIndex=0），dataSplitRatio=1。
 */
static AlgoExecDesc MakeAicpuReduceScatterNhrAlgoExecDesc()
{
    TemplateDesc templateDesc = g_reduceScatterTemplateDescMap[static_cast<size_t>(
        HcclReduceScatterTemplateDescType::REDUCESCATTER_TEMPLATE_NHR_SINGLE_JETTY)];
    TemplateExecDesc templateExecDesc{templateDesc, SUB_COMM_INDEX_INTRA};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {templateExecDesc};
    algoExecDesc.dataSplitRatio = {1};
    return algoExecDesc;
}

/**
 * 构造 AICPU ReduceScatter Mesh1D 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXEC_V2(..., InsReduceScatterMesh1D,
 *   InsV2ReduceScatterSoleExecutor, TopoMatch1D, InsTempReduceScatterMesh1D)。
 * execPolicy=SEQUENCE，children 仅一个 TemplateExecDesc 为 Mesh1D（FULLMESH），dataSplitRatio=1。
 */
static AlgoExecDesc MakeAicpuReduceScatterMesh1DAlgoExecDesc()
{
    TemplateDesc templateDesc = g_reduceScatterTemplateDescMap[static_cast<size_t>(
        HcclReduceScatterTemplateDescType::REDUCESCATTER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateExecDesc templateExecDesc{templateDesc, SUB_COMM_INDEX_INTRA};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {templateExecDesc};
    algoExecDesc.dataSplitRatio = {1};
    return algoExecDesc;
}

/**
 * 全局 AICPU ReduceScatter 算法表。
 * 以 HcclAicpuReduceScatterAlgoType 枚举值为数组下标，元素为对应的 HcclAlgorithm 实例
 * （hcclCmdType=HCCL_CMD_REDUCE_SCATTER, engineType=AICPU，topoMatch 为对应 executor 类模板参数的
 * TopoMatch 子类实例指针）。数组定义顺序必须与 HcclAicpuReduceScatterAlgoType 枚举顺序保持一致。
 *
 * TopoMatch 子类与算法对应关系（源自 src/ops/reduce_scatter/executor/ 注册宏）：
 *   TopoMatch1D         - InsReduceScatterMesh1D, InsReduceScatterNHR
 */
const HcclAlgorithm
    g_aicpuReduceScatterAlgoMap[static_cast<size_t>(HcclAicpuReduceScatterAlgoType::AICPU_REDUCESCATTER_ALGO_TYPE_COUNT)]
    = {
        MakeAicpuReduceScatterAlgo(std::make_shared<TopoMatch1D>(), MakeAicpuReduceScatterNhrAlgoExecDesc()),
        MakeAicpuReduceScatterAlgo(std::make_shared<TopoMatch1D>(), MakeAicpuReduceScatterMesh1DAlgoExecDesc()),
};

} // namespace ops_hccl
