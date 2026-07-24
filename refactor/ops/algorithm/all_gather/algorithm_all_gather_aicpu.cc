/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_algorithm.h"
#include "all_gather_template_desc.h"
#include "topo_match_1d.h"
#include "topo_match_3_level.h"
#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "topo_match_pcie_mix.h"
#include "topo_match_squeeze_2d.h"
#include "topo_match_concurrent.h"

namespace ops_hccl {

/**
 * 构造 AICPU AllGather 算法描述对象。
 * 统一填充 hcclCmdType=HCCL_CMD_ALLGATHER、engineType=AICPU。
 * algoExecDesc 由调用方传入；不传则默认构造为空。
 */
static HcclAlgorithm MakeAicpuAllGatherAlgo(
    std::shared_ptr<TopoMatchBase> topoMatch, AlgoExecDesc algoExecDesc = AlgoExecDesc{})
{
    HcclAlgorithm algo;
    algo.hcclCmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    algo.engineType = HcclAlgEngineType::AICPU;
    algo.topoMatch = std::move(topoMatch);
    algo.algoExecDesc = algoExecDesc;
    return algo;
}

/**
 * 构造 AICPU AllGather NHR 算法的 AlgoExecDesc。
 * execPolicy=SEQUENCE，children 仅一层 TemplateExecDesc（subCommIndex=0，对应网络层级 level 0），
 * dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherNhrAlgoExecDesc()
{
    TemplateDesc templateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_NHR_SINGLE_JETTY)];
    TemplateExecDesc templateExecDesc{templateDesc, SUB_COMM_INDEX_0};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {templateExecDesc};
    algoExecDesc.dataSplitRatio = {1};
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather ParallelMesh1DNHRUboe 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsAllGatherParallelMesh1DNHRUboe,
 *   InsV2AllGatherParallelExecutor, TopoMatchSqueeze2D, InsTempAllGatherNHR, InsTempAllGatherNHR)。
 * 外层 execPolicy=SEQUENCE，children 为两个并行的 AlgoExecDesc 子树（串行组合）：
 *   - 子树0：execPolicy=PARALLEL，children=[NHR→INTRA, NHR→INTER]，dataSplitRatio=1:1；
 *   - 子树1：execPolicy=PARALLEL，children=[NHR→INTER, NHR→INTRA]（位置交换），dataSplitRatio=1:1；
 * 外层 dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherParallelMesh1DNhrUboeAlgoExecDesc()
{
    TemplateDesc nhrTemplateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_NHR_SINGLE_JETTY)];

    // 第一个并行子树：nhr→INTRA，nhr→INTER
    auto parallelDesc0 = std::make_shared<AlgoExecDesc>();
    parallelDesc0->execPolicy = HcclAlgExecPolicy::PARALLEL;
    parallelDesc0->children = {
        TemplateExecDesc{nhrTemplateDesc, SUB_COMM_INDEX_0},
        TemplateExecDesc{nhrTemplateDesc, SUB_COMM_INDEX_1}};
    parallelDesc0->dataSplitRatio = {1, 1}; // 1:1

    // 第二个并行子树：nhr→INTER，nhr→INTRA（位置交换）
    auto parallelDesc1 = std::make_shared<AlgoExecDesc>();
    parallelDesc1->execPolicy = HcclAlgExecPolicy::PARALLEL;
    parallelDesc1->children = {
        TemplateExecDesc{nhrTemplateDesc, SUB_COMM_INDEX_1},
        TemplateExecDesc{nhrTemplateDesc, SUB_COMM_INDEX_0}};
    parallelDesc1->dataSplitRatio = {1, 1}; // 1:1

    // 两个并行子树串行组合
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {parallelDesc0, parallelDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather SequenceNHRMesh1D 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsAllGatherSequenceNHRMesh1D,
 *   InsV2AllGatherSequenceExecutorAicpu, TopoMatchMultilevel,
 *   InsTempAllGatherMesh1D1DZAxisDetour, InsTempAllGatherNHR)。
 * execPolicy=SEQUENCE（对应 InsV2AllGatherSequenceExecutorAicpu），children 两个 TemplateExecDesc
 * 分别为 Mesh1D1DZAxisDetour（FULLMESH）和 NHR，dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherSequenceNhrMesh1DAlgoExecDesc()
{
    TemplateDesc fullmeshTemplateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateDesc nhrTemplateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_NHR_SINGLE_JETTY)];

        // 第一个并行子树：nhr→INTRA，nhr→INTER 替换原来的 Mesh1D1DZAxisDetour
    auto parallelDesc0 = std::make_shared<AlgoExecDesc>();
    parallelDesc0->execPolicy = HcclAlgExecPolicy::PARALLEL;
    parallelDesc0->children = {
        TemplateExecDesc{fullmeshTemplateDesc, SUB_COMM_INDEX_0},
        TemplateExecDesc{nhrTemplateDesc, SUB_COMM_INDEX_1}};
    parallelDesc0->dataSplitRatio = {1, 1}; // 1:1

    TemplateExecDesc templateExecDesc1{nhrTemplateDesc, SUB_COMM_INDEX_2};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {parallelDesc0, templateExecDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather ParallelMesh1DNHR 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsAllGatherParallelMesh1DNHR,
 *   InsV2AllGatherParallelExecutor, TopoMatchMultilevel, InsTempAllGatherMesh1D, InsTempAllGatherNHR)。
 * 外层 execPolicy=SEQUENCE，children 为两个并行的 AlgoExecDesc 子树（串行组合）：
 *   - 子树0：execPolicy=PARALLEL，children=[FULLMESH→INTRA, NHR→INTER]，dataSplitRatio=1:1；
 *   - 子树1：execPolicy=PARALLEL，children=[FULLMESH→INTER, NHR→INTRA]（位置交换），dataSplitRatio=1:1；
 * 外层 dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherParallelMesh1DNhrAlgoExecDesc()
{
    TemplateDesc fullmeshTemplateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateDesc nhrTemplateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_NHR_SINGLE_JETTY)];
    // 第一个并行子树：fullmesh→INTRA，nhr→INTER
    auto parallelDesc0 = std::make_shared<AlgoExecDesc>();
    parallelDesc0->execPolicy = HcclAlgExecPolicy::PARALLEL;
    parallelDesc0->children = {
        TemplateExecDesc{fullmeshTemplateDesc, SUB_COMM_INDEX_0},
        TemplateExecDesc{nhrTemplateDesc, SUB_COMM_INDEX_1}};
    parallelDesc0->dataSplitRatio = {1, 1}; // 1:1

    // 第二个并行子树：fullmesh→INTER，nhr→INTRA
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
 * 构造 AICPU AllGather Mesh1D 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXEC_V2(..., InsAllGatherMesh1D,
 *   InsV2AllGatherSoleExecutor, TopoMatch1D, InsTempAllGatherMesh1D)。
 * execPolicy=SEQUENCE（对应 InsV2AllGatherSoleExecutor），children 仅一个 TemplateExecDesc
 * 为 Mesh1D（FULLMESH），dataSplitRatio=1。
 */
static AlgoExecDesc MakeAicpuAllGatherMesh1DAlgoExecDesc()
{
    TemplateDesc templateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateExecDesc templateExecDesc{templateDesc, SUB_COMM_INDEX_0};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {templateExecDesc};
    algoExecDesc.dataSplitRatio = {1};
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather ConcurrentMesh1DNHR 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsAllGatherConcurrentMesh1DNHR,
 *   InsV2AllGatherConcurrentExecutor, TopoMatchUBX, InsTempAllGatherMesh1D, InsTempAllGatherNHR)。
 * execPolicy=PARALLEL（对应 InsV2AllGatherConcurrentExecutor），children 两个 TemplateExecDesc
 * 分别为 Mesh1D（FULLMESH）和 NHR，dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherConcurrentMesh1DNhrAlgoExecDesc()
{
    TemplateDesc fullmeshTemplateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateDesc nhrTemplateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_NHR_SINGLE_JETTY)];
    TemplateExecDesc templateExecDesc0{fullmeshTemplateDesc, SUB_COMM_INDEX_0};
    TemplateExecDesc templateExecDesc1{nhrTemplateDesc, SUB_COMM_INDEX_1};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::PARALLEL;
    algoExecDesc.children = {templateExecDesc0, templateExecDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather SequenceMesh1DNHRNHRMesh1DOcs 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsAllGatherSequenceMesh1DNHRNHRMesh1DOcs,
 *   InsV2AllGatherSequenceExecutorAicpu, TopoMatchMultilevel, InsTempAllGatherMesh1D1DZAxisDetour, InsTempAllGatherNHR)。
 * execPolicy=SEQUENCE（对应 InsV2AllGatherSequenceExecutorAicpu），children 两个 TemplateExecDesc
 * 分别为 Mesh1D1DZAxisDetour（FULLMESH）和 NHR，dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherSequenceMesh1DNHRNHRMesh1DOcsAlgoExecDesc()
{
    TemplateDesc fullmeshTemplateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateDesc nhrTemplateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_NHR_SINGLE_JETTY)];
    TemplateExecDesc templateExecDesc0{fullmeshTemplateDesc, SUB_COMM_INDEX_3};
    TemplateExecDesc templateExecDesc1{nhrTemplateDesc, SUB_COMM_INDEX_2};
    TemplateExecDesc templateExecDesc2{nhrTemplateDesc, SUB_COMM_INDEX_1};
    TemplateExecDesc templateExecDesc3{fullmeshTemplateDesc, SUB_COMM_INDEX_0};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {templateExecDesc0, templateExecDesc1, templateExecDesc2, templateExecDesc3};
    algoExecDesc.dataSplitRatio = {1, 1, 1, 1}; // 1:1:1:1
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather OmniPipeUboe 算法的 AlgoExecDesc。
 * 对应注册宏 InsV2AllGatherOmniPipeUboe，使用 TopoMatch3Level。
 * 编排结构（四层嵌套），其中 sequence0 在 parallel2 与 parallel3 中复用同一个 shared_ptr：
 *   final(SEQUENCE)
 *   ├── parallel2(PARALLEL) = [sequence0, NHR→SUB_COMM_INDEX_2]
 *   └── parallel3(PARALLEL) = [NHR→SUB_COMM_INDEX_2, sequence0]（位置交换）
 *   其中 sequence0(SEQUENCE) = [parallel0, parallel1]
 *     - parallel0(PARALLEL) = [FULLMESH→INTRA, NHR→INTER]
 *     - parallel1(PARALLEL) = [NHR→INTER, FULLMESH→INTRA]（位置交换）
 * 各层 dataSplitRatio 均为 1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherOmniPipeAlgoExecDesc()
{
    TemplateDesc fullmeshTemplateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
    TemplateDesc nhrTemplateDesc = g_allGatherTemplateDescMap[static_cast<size_t>(
        HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_NHR_SINGLE_JETTY)];

    // omin第一个节点表示慢轴，第二个节点表示快轴，只能有2个节点
    auto parallelDesc0 = std::make_shared<AlgoExecDesc>();
    parallelDesc0->execPolicy = HcclAlgExecPolicy::OMNIPIPE;
    parallelDesc0->children = {
        TemplateExecDesc{nhrTemplateDesc, SUB_COMM_INDEX_1}, TemplateExecDesc{fullmeshTemplateDesc, SUB_COMM_INDEX_0}};
    parallelDesc0->dataSplitRatio = {1, 1}; // 1:1

    // omin第一个节点表示慢轴，第二个节点表示快轴，支持嵌套，但是只能2个节点
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::OMNIPIPE;
    algoExecDesc.children = {parallelDesc0, TemplateExecDesc{nhrTemplateDesc, SUB_COMM_INDEX_2},};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 全局 AICPU AllGather 算法表。
 * 以 HcclAicpuAllGatherAlgoType 枚举值为数组下标，元素为对应的 HcclAlgorithm 实例
 * （hcclCmdType=HCCL_CMD_ALLGATHER, engineType=AICPU，topoMatch 为对应 executor 类模板参数的
 * TopoMatch 子类实例指针）。数组定义顺序必须与 HcclAicpuAllGatherAlgoType 枚举顺序保持一致。
 *
 * TopoMatch 子类与算法对应关系（源自 src/ops/all_gather/executor/ 注册宏）：
 *   TopoMatch1D         - InsAllGatherMesh1D, InsAllGatherMesh1D1DZAxisDetour, InsAllGatherNHR
 *   TopoMatch3Level     - InsV2AllGatherOmniPipeUboe
 *   TopoMatchMultilevel - InsAllGatherParallelMesh1DNHR, InsAllGatherSequenceNHRMesh1D
 *   TopoMatchUBX        - InsAllGatherConcurrentMesh1DNHR, 
 *   TopoMatchPcieMix    - InsAllGatherParallelMesh1DNHRPcie, InsV2AllGatherOmniPipePcie
 *   TopoMatchSqueeze2D  - InsAllGatherParallelMesh1DNHRUboe
 */
const HcclAlgorithm
    g_aicpuAllGatherAlgoMap[static_cast<size_t>(HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_ALGO_TYPE_COUNT)]
    = {
        MakeAicpuAllGatherAlgo(
            std::make_shared<TopoMatch3Level>(), MakeAicpuAllGatherOmniPipeAlgoExecDesc()),
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatch1D>(), MakeAicpuAllGatherNhrAlgoExecDesc()),
        MakeAicpuAllGatherAlgo(
            std::make_shared<TopoMatchSqueeze2D>(), MakeAicpuAllGatherParallelMesh1DNhrUboeAlgoExecDesc()),
        MakeAicpuAllGatherAlgo(
            std::make_shared<TopoMatchConcurrent>(), MakeAicpuAllGatherSequenceNhrMesh1DAlgoExecDesc()),
        MakeAicpuAllGatherAlgo(
            std::make_shared<TopoMatchMultilevel>(), MakeAicpuAllGatherParallelMesh1DNhrAlgoExecDesc()),
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatch1D>(), MakeAicpuAllGatherConcurrentMesh1DNhrAlgoExecDesc()),
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatch1D>(), MakeAicpuAllGatherMesh1DAlgoExecDesc()),
        MakeAicpuAllGatherAlgo(
            std::make_shared<TopoMatchPcieMix>(), MakeAicpuAllGatherParallelMesh1DNhrAlgoExecDesc()),
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatchPcieMix>(), MakeAicpuAllGatherOmniPipeAlgoExecDesc()),    // 待区分DPU
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatchConcurrent>(), MakeAicpuAllGatherConcurrentMesh1DNhrAlgoExecDesc()),
        MakeAicpuAllGatherAlgo(
            std::make_shared<TopoMatchMultilevel>(), MakeAicpuAllGatherSequenceMesh1DNHRNHRMesh1DOcsAlgoExecDesc()),
};

} // namespace ops_hccl
