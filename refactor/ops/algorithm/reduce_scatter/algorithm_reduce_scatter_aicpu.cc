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
#include "topo_match_ubx.h"
#include "topo_match_pcie_mix.h"
#include "topo_match_squeeze_2d.h"
#include "topo_match_concurrent.h"

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
    TemplateExecDesc templateExecDesc{templateDesc, SUB_COMM_INDEX_0};
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
    TemplateExecDesc templateExecDesc{templateDesc, SUB_COMM_INDEX_0};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {templateExecDesc};
    algoExecDesc.dataSplitRatio = {1};
    return algoExecDesc;
}

/**
 * 构造 AICPU ReduceScatter ParallelMesh1DNHR 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsReduceScatterParallelMesh1DNHR,
 *   InsReduceScatterParallelExecutor, TopoMatchMultilevel, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR)。
 * 外层 execPolicy=SEQUENCE，children 为两个并行的 AlgoExecDesc 子树（串行组合）：
 *   - 子树0：execPolicy=PARALLEL，children=[FULLMESH→INTRA, NHR→INTER]，dataSplitRatio=1:1；
 *   - 子树1：execPolicy=PARALLEL，children=[NHR→INTER, FULLMESH→INTRA]（位置交换），dataSplitRatio=1:1；
 * 外层 dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuReduceScatterParallelMesh1DNhrAlgoExecDesc()
{
    TemplateDesc fullmeshTemplateDesc = g_reduceScatterTemplateDescMap[static_cast<size_t>(
        HcclReduceScatterTemplateDescType::REDUCESCATTER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateDesc nhrTemplateDesc = g_reduceScatterTemplateDescMap[static_cast<size_t>(
        HcclReduceScatterTemplateDescType::REDUCESCATTER_TEMPLATE_NHR_SINGLE_JETTY)];

    // 第一个并行子树：fullmesh→INTRA，nhr→INTER
    auto parallelDesc0 = std::make_shared<AlgoExecDesc>();
    parallelDesc0->execPolicy = HcclAlgExecPolicy::PARALLEL;
    parallelDesc0->children = {
        TemplateExecDesc{fullmeshTemplateDesc, SUB_COMM_INDEX_0},
        TemplateExecDesc{nhrTemplateDesc, SUB_COMM_INDEX_1}};
    parallelDesc0->dataSplitRatio = {1, 1}; // 1:1

    // 第二个并行子树：nhr→INTER，fullmesh→INTRA
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
 * 构造 AICPU ReduceScatter ParallelNHRNHRUboe 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsReduceScatterParallelNHRNHRUboe,
 *   InsReduceScatterParallelExecutor, TopoMatchSqueeze2D, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR)。
 * 拓扑与 ParallelMesh1DNHR 一致，复用相同的并行子树结构。
 */
static AlgoExecDesc MakeAicpuReduceScatterParallelNhrNhrUboeAlgoExecDesc()
{
    return MakeAicpuReduceScatterParallelMesh1DNhrAlgoExecDesc();
}

/**
 * 构造 AICPU ReduceScatter SequenceMesh1DNhr 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsReduceScatterSequenceMesh1DNhr,
 *   InsV2ReduceScatterSequenceExecutorAicpu, TopoMatchMultilevel,
 *   InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterNHR)。
 * execPolicy=SEQUENCE，children 两个 TemplateExecDesc 分别为 Mesh1D（FULLMESH）和 NHR，dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuReduceScatterSequenceNhrMesh1DAlgoExecDesc()
{
    TemplateDesc fullmeshTemplateDesc = g_reduceScatterTemplateDescMap[static_cast<size_t>(
        HcclReduceScatterTemplateDescType::REDUCESCATTER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateDesc nhrTemplateDesc = g_reduceScatterTemplateDescMap[static_cast<size_t>(
        HcclReduceScatterTemplateDescType::REDUCESCATTER_TEMPLATE_NHR_SINGLE_JETTY)];
    TemplateExecDesc templateExecDesc0{fullmeshTemplateDesc, SUB_COMM_INDEX_0};
    TemplateExecDesc templateExecDesc1{nhrTemplateDesc, SUB_COMM_INDEX_1};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {templateExecDesc0, templateExecDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 构造 AICPU ReduceScatter ConcurrentMeshNHR 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsReduceScatterConcurrentMeshNHR,
 *   InsReduceScatterConcurrentExecutor, TopoMatchUBX, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR)。
 * execPolicy=PARALLEL，children 两个 TemplateExecDesc 分别为 Mesh1D（FULLMESH）和 NHR，dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuReduceScatterConcurrentMeshNhrAlgoExecDesc()
{
    TemplateDesc fullmeshTemplateDesc = g_reduceScatterTemplateDescMap[static_cast<size_t>(
        HcclReduceScatterTemplateDescType::REDUCESCATTER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateDesc nhrTemplateDesc = g_reduceScatterTemplateDescMap[static_cast<size_t>(
        HcclReduceScatterTemplateDescType::REDUCESCATTER_TEMPLATE_NHR_SINGLE_JETTY)];
    TemplateExecDesc templateExecDesc0{fullmeshTemplateDesc, SUB_COMM_INDEX_0};
    TemplateExecDesc templateExecDesc1{nhrTemplateDesc, SUB_COMM_INDEX_1};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::PARALLEL;
    algoExecDesc.children = {templateExecDesc0, templateExecDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
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
 *   TopoMatchMultilevel - InsReduceScatterParallelMesh1DNHR, InsReduceScatterSequenceMesh1DNhr
 *   TopoMatchUBX        - InsReduceScatterParallelMesh1DNHRUBX, InsReduceScatterConcurrentMeshNHR
 *   TopoMatchPcieMix    - InsReduceScatterParallelMesh1DNHRPcie
 *   TopoMatchSqueeze2D  - InsReduceScatterParallelNHRNHRUboe
 */
const HcclAlgorithm
    g_aicpuReduceScatterAlgoMap[static_cast<size_t>(HcclAicpuReduceScatterAlgoType::AICPU_REDUCESCATTER_ALGO_TYPE_COUNT)]
    = {
        MakeAicpuReduceScatterAlgo(std::make_shared<TopoMatch1D>(), MakeAicpuReduceScatterNhrAlgoExecDesc()),
        MakeAicpuReduceScatterAlgo(std::make_shared<TopoMatch1D>(), MakeAicpuReduceScatterMesh1DAlgoExecDesc()),
        MakeAicpuReduceScatterAlgo(
            std::make_shared<TopoMatchMultilevel>(), MakeAicpuReduceScatterParallelMesh1DNhrAlgoExecDesc()),
        MakeAicpuReduceScatterAlgo(
            std::make_shared<TopoMatchUBX>(), MakeAicpuReduceScatterParallelMesh1DNhrAlgoExecDesc()),
        MakeAicpuReduceScatterAlgo(
            std::make_shared<TopoMatchPcieMix>(), MakeAicpuReduceScatterParallelMesh1DNhrAlgoExecDesc()),
        MakeAicpuReduceScatterAlgo(
            std::make_shared<TopoMatchSqueeze2D>(), MakeAicpuReduceScatterParallelNhrNhrUboeAlgoExecDesc()),
        MakeAicpuReduceScatterAlgo(
            std::make_shared<TopoMatchMultilevel>(), MakeAicpuReduceScatterSequenceNhrMesh1DAlgoExecDesc()),
        MakeAicpuReduceScatterAlgo(
            std::make_shared<TopoMatchConcurrent>(), MakeAicpuReduceScatterConcurrentMeshNhrAlgoExecDesc()),
};

} // namespace ops_hccl
