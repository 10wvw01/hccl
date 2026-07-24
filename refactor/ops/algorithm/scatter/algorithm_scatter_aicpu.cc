/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "hccl_algorithm.h"
#include "scatter_template_desc.h"
#include "topo_match_1d.h"
#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "topo_match_pcie_mix.h"
#include "topo_match_concurrent.h"

namespace ops_hccl {

/**
 * 构造 AICPU Scatter 算法描述对象。
 * 统一填充 hcclCmdType=HCCL_CMD_SCATTER、engineType=AICPU。
 * algoExecDesc 由调用方传入；不传则默认构造为空。
 */
static HcclAlgorithm MakeAicpuScatterAlgo(
    std::shared_ptr<TopoMatchBase> topoMatch, AlgoExecDesc algoExecDesc = AlgoExecDesc{})
{
    HcclAlgorithm algo;
    algo.hcclCmdType = HcclCMDType::HCCL_CMD_SCATTER;
    algo.engineType = HcclAlgEngineType::AICPU;
    algo.topoMatch = std::move(topoMatch);
    algo.algoExecDesc = algoExecDesc;
    return algo;
}

/**
 * 构造 AICPU Scatter NHR 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXEC_V2(..., InsScatterNHR,
 *   InsV2ScatterSoleExecutor, TopoMatch1D, InsTempScatterNHR)。
 * execPolicy=SEQUENCE，children 仅一层 TemplateExecDesc（subCommIndex=0），dataSplitRatio=1。
 */
static AlgoExecDesc MakeAicpuScatterNhrAlgoExecDesc()
{
    TemplateDesc templateDesc = g_scatterTemplateDescMap[static_cast<size_t>(
        HcclScatterTemplateDescType::SCATTER_TEMPLATE_NHR_SINGLE_JETTY)];
    TemplateExecDesc templateExecDesc{templateDesc, SUB_COMM_INDEX_0};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {templateExecDesc};
    algoExecDesc.dataSplitRatio = {1};
    return algoExecDesc;
}

/**
 * 构造 AICPU Scatter Mesh1D 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXEC_V2(..., InsScatterMesh1D,
 *   InsV2ScatterSoleExecutor, TopoMatch1D, InsTempScatterMesh1D)。
 * execPolicy=SEQUENCE，children 仅一个 TemplateExecDesc 为 Mesh1D（FULLMESH），dataSplitRatio=1。
 */
static AlgoExecDesc MakeAicpuScatterMesh1DAlgoExecDesc()
{
    TemplateDesc templateDesc = g_scatterTemplateDescMap[static_cast<size_t>(
        HcclScatterTemplateDescType::SCATTER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateExecDesc templateExecDesc{templateDesc, SUB_COMM_INDEX_0};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {templateExecDesc};
    algoExecDesc.dataSplitRatio = {1};
    return algoExecDesc;
}

/**
 * 构造 AICPU Scatter ParallelMesh1DNHR 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsScatterParallelMesh1DNHR,
 *   InsV2ScatterParallelExecutor, TopoMatchMultilevel, InsTempScatterMesh1D, InsTempScatterNHR)。
 * 外层 execPolicy=SEQUENCE，children 为两个并行的 AlgoExecDesc 子树（串行组合）：
 *   - 子树0：execPolicy=PARALLEL，children=[FULLMESH→INTRA, NHR→INTER]，dataSplitRatio=1:1；
 *   - 子树1：execPolicy=PARALLEL，children=[NHR→INTER, FULLMESH→INTRA]（位置交换），dataSplitRatio=1:1；
 * 外层 dataSplitRatio=1:1。
 * 两阶段串行结构：Stage1 Mesh(data0)+NHR(data1) 并行，Stage2 NHR(data0)+Mesh(data1) 并行，
 * 阶段间由 UpdateDataSplitSequence 将上一阶段 output 作为下一阶段 input。
 */
static AlgoExecDesc MakeAicpuScatterParallelMesh1DNhrAlgoExecDesc()
{
    TemplateDesc fullmeshTemplateDesc = g_scatterTemplateDescMap[static_cast<size_t>(
        HcclScatterTemplateDescType::SCATTER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateDesc nhrTemplateDesc = g_scatterTemplateDescMap[static_cast<size_t>(
        HcclScatterTemplateDescType::SCATTER_TEMPLATE_NHR_SINGLE_JETTY)];

    // 第一个并行子树：fullmesh→INTRA，nhr→INTER
    auto parallelDesc0 = std::make_shared<AlgoExecDesc>();
    parallelDesc0->execPolicy = HcclAlgExecPolicy::PARALLEL;
    parallelDesc0->children = {
        TemplateExecDesc{fullmeshTemplateDesc, SUB_COMM_INDEX_0},
        TemplateExecDesc{nhrTemplateDesc, SUB_COMM_INDEX_1}};
    parallelDesc0->dataSplitRatio = {1, 1}; // 1:1

    // 第二个并行子树：nhr→INTER，fullmesh→INTRA（位置交换）
    auto parallelDesc1 = std::make_shared<AlgoExecDesc>();
    parallelDesc1->execPolicy = HcclAlgExecPolicy::PARALLEL;
    parallelDesc1->children = {
        TemplateExecDesc{nhrTemplateDesc, SUB_COMM_INDEX_1},
        TemplateExecDesc{fullmeshTemplateDesc, SUB_COMM_INDEX_0}};
    parallelDesc1->dataSplitRatio = {1, 1}; // 1:1

    // 两个并行子树串行组合
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {parallelDesc0, parallelDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 构造 AICPU Scatter ConcurrentMesh1DNHR 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsScatterConcurrentMesh1DNHR,
 *   InsV2ScatterConcurrentExecutor, TopoMatchConcurrent, InsTempScatterMesh1D, InsTempScatterNHR)。
 * execPolicy=PARALLEL，children 两个 TemplateExecDesc 分别为 Mesh1D（FULLMESH）和 NHR，dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuScatterConcurrentMesh1DNhrAlgoExecDesc()
{
    TemplateDesc fullmeshTemplateDesc = g_scatterTemplateDescMap[static_cast<size_t>(
        HcclScatterTemplateDescType::SCATTER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateDesc nhrTemplateDesc = g_scatterTemplateDescMap[static_cast<size_t>(
        HcclScatterTemplateDescType::SCATTER_TEMPLATE_NHR_SINGLE_JETTY)];
    TemplateExecDesc templateExecDesc0{fullmeshTemplateDesc, SUB_COMM_INDEX_0};
    TemplateExecDesc templateExecDesc1{nhrTemplateDesc, SUB_COMM_INDEX_1};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::PARALLEL;
    algoExecDesc.children = {templateExecDesc0, templateExecDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 全局 AICPU Scatter 算法表。
 * 以 HcclAicpuScatterAlgoType 枚举值为数组下标，元素为对应的 HcclAlgorithm 实例
 * （hcclCmdType=HCCL_CMD_SCATTER, engineType=AICPU，topoMatch 为对应 executor 类模板参数的
 * TopoMatch 子类实例指针）。数组定义顺序必须与 HcclAicpuScatterAlgoType 枚举顺序保持一致。
 *
 * TopoMatch 子类与算法对应关系（源自 src/ops/scatter/selector/scatter_auto_selector.cc 中
 * SelectAicpuAlgo 的算法名映射）：
 *   TopoMatch1D         - InsScatterNHR, InsScatterMesh1D
 *   TopoMatchMultilevel - InsScatterParallelMesh1DNHR, InsScatterSequenceNhrMesh1D
 *   TopoMatchUBX        - InsScatterParallelMesh1DNHRUBX
 *   TopoMatchPcieMix    - InsScatterParallelMesh1DNHRPcie
 *   TopoMatchConcurrent - InsScatterConcurrentMesh1DNHR
 */
const HcclAlgorithm
    g_aicpuScatterAlgoMap[static_cast<size_t>(HcclAicpuScatterAlgoType::AICPU_SCATTER_ALGO_TYPE_COUNT)]
    = {
        MakeAicpuScatterAlgo(std::make_shared<TopoMatch1D>(), MakeAicpuScatterNhrAlgoExecDesc()),
        MakeAicpuScatterAlgo(std::make_shared<TopoMatch1D>(), MakeAicpuScatterMesh1DAlgoExecDesc()),
        MakeAicpuScatterAlgo(
            std::make_shared<TopoMatchMultilevel>(), MakeAicpuScatterParallelMesh1DNhrAlgoExecDesc()),
        MakeAicpuScatterAlgo(
            std::make_shared<TopoMatchUBX>(), MakeAicpuScatterParallelMesh1DNhrAlgoExecDesc()),
        MakeAicpuScatterAlgo(
            std::make_shared<TopoMatchPcieMix>(), MakeAicpuScatterParallelMesh1DNhrAlgoExecDesc()),
        MakeAicpuScatterAlgo(
            std::make_shared<TopoMatchConcurrent>(), MakeAicpuScatterConcurrentMesh1DNhrAlgoExecDesc()),
};

} // namespace ops_hccl
