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
#include "topo_match_1d.h"
#include "topo_match_3_level.h"
#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "topo_match_pcie_mix.h"
#include "topo_match_squeeze_2d.h"

namespace ops_hccl {

/**
 * 构造 AICPU AllGather 算法描述对象。
 * 统一填充 hcclCmdType=HCCL_CMD_ALLGATHER、engineType=AICPU。
 * algoExecDesc 由调用方传入；不传则默认构造为空。
 */
static HcclAlgorithm MakeAicpuAllGatherAlgo(std::shared_ptr<TopoMatchBase> topoMatch,
    AlgoExecDesc algoExecDesc = AlgoExecDesc{})
{
    HcclAlgorithm algo;
    algo.hcclCmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    algo.engineType = HcclAlgEngineType::AICPU;
    algo.topoMatch = std::move(topoMatch);
    algo.algoExecDesc = std::move(algoExecDesc);
    return algo;
}

/**
 * 构造 AICPU AllGather NHR 算法的 AlgoExecDesc。
 * execPolicy=SOLE，children 仅一层 TemplateExecDesc（subCommIndex=0，对应网络层级 level 0），
 * dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherNhrAlgoExecDesc()
{
    TemplateDesc templateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateExecDesc templateExecDesc{templateDesc, 0}; // subCommIndex=0，对应网络层级 level 0
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SOLE;
    algoExecDesc.children = {templateExecDesc};
    algoExecDesc.dataSplitRatio = {1};
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather ParallelMesh1DNHRUboe 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsAllGatherParallelMesh1DNHRUboe,
 *   InsV2AllGatherParallelExecutor, TopoMatchSqueeze2D, InsTempAllGatherNHR, InsTempAllGatherNHR)。
 * execPolicy=PARALLEL（对应 InsV2AllGatherParallelExecutor），children 两个 TemplateExecDesc
 * 均为 NHR（subCommIndex=0），dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherParallelMesh1DNhrUboeAlgoExecDesc()
{
    TemplateDesc templateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateExecDesc templateExecDesc0{templateDesc, 0};
    TemplateExecDesc templateExecDesc1{templateDesc, 1};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::PARALLEL;
    algoExecDesc.children = {templateExecDesc0, templateExecDesc1};
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
    TemplateDesc mesh1D1DZAxisDetourTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateDesc nhrTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateExecDesc templateExecDesc0{mesh1D1DZAxisDetourTemplateDesc, 0};
    TemplateExecDesc templateExecDesc1{nhrTemplateDesc, 1};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    algoExecDesc.children = {templateExecDesc0, templateExecDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather ParallelMesh1DNHR 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsAllGatherParallelMesh1DNHR,
 *   InsV2AllGatherParallelExecutor, TopoMatchMultilevel, InsTempAllGatherMesh1D, InsTempAllGatherNHR)。
 * execPolicy=PARALLEL（对应 InsV2AllGatherParallelExecutor），children 两个 TemplateExecDesc
 * 分别为 Mesh1D（FULLMESH）和 NHR，dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherParallelMesh1DNhrAlgoExecDesc()
{
    TemplateDesc mesh1DTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateDesc nhrTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateExecDesc templateExecDesc0{mesh1DTemplateDesc, 0};
    TemplateExecDesc templateExecDesc1{nhrTemplateDesc, 1};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::PARALLEL;
    algoExecDesc.children = {templateExecDesc0, templateExecDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather Mesh1D1DZAxisDetour 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXEC_V2(..., InsAllGatherMesh1D1DZAxisDetour,
 *   InsV2AllGatherSoleExecutor, TopoMatch1D, InsTempAllGatherMesh1D1DZAxisDetour)。
 * execPolicy=SOLE（对应 InsV2AllGatherSoleExecutor），children 仅一个 TemplateExecDesc
 * 为 Mesh1D1DZAxisDetour（FULLMESH），dataSplitRatio=1。
 */
static AlgoExecDesc MakeAicpuAllGatherMesh1D1DZAxisDetourAlgoExecDesc()
{
    TemplateDesc mesh1D1DZAxisDetourTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateExecDesc templateExecDesc{mesh1D1DZAxisDetourTemplateDesc, 0};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SOLE;
    algoExecDesc.children = {templateExecDesc};
    algoExecDesc.dataSplitRatio = {1};
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather Mesh1D 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXEC_V2(..., InsAllGatherMesh1D,
 *   InsV2AllGatherSoleExecutor, TopoMatch1D, InsTempAllGatherMesh1D)。
 * execPolicy=SOLE（对应 InsV2AllGatherSoleExecutor），children 仅一个 TemplateExecDesc
 * 为 Mesh1D（FULLMESH），dataSplitRatio=1。
 */
static AlgoExecDesc MakeAicpuAllGatherMesh1DAlgoExecDesc()
{
    TemplateDesc mesh1DTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateExecDesc templateExecDesc{mesh1DTemplateDesc, 0};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::SOLE;
    algoExecDesc.children = {templateExecDesc};
    algoExecDesc.dataSplitRatio = {1};
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather ParallelMesh1DNHRPcie 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsAllGatherParallelMesh1DNHRPcie,
 *   InsV2AllGatherParallelExecutor, TopoMatchPcieMix, InsTempAllGatherMesh1D, InsTempAllGatherNHR)。
 * execPolicy=PARALLEL（对应 InsV2AllGatherParallelExecutor），children 两个 TemplateExecDesc
 * 分别为 Mesh1D（FULLMESH）和 NHR，dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherParallelMesh1DNhrPcieAlgoExecDesc()
{
    TemplateDesc mesh1DTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateDesc nhrTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateExecDesc templateExecDesc0{mesh1DTemplateDesc, 0};
    TemplateExecDesc templateExecDesc1{nhrTemplateDesc, 1};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::PARALLEL;
    algoExecDesc.children = {templateExecDesc0, templateExecDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather ConcurrentMesh1DNHR 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsAllGatherConcurrentMesh1DNHR,
 *   InsV2AllGatherConcurrentExecutor, TopoMatchUBX, InsTempAllGatherMesh1D, InsTempAllGatherNHR)。
 * execPolicy=CONCURRENT（对应 InsV2AllGatherConcurrentExecutor），children 两个 TemplateExecDesc
 * 分别为 Mesh1D（FULLMESH）和 NHR，dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherConcurrentMesh1DNhrAlgoExecDesc()
{
    TemplateDesc mesh1DTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateDesc nhrTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY};
    TemplateExecDesc templateExecDesc0{mesh1DTemplateDesc, 0};
    TemplateExecDesc templateExecDesc1{nhrTemplateDesc, 1};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::CONCURRENT;
    algoExecDesc.children = {templateExecDesc0, templateExecDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 构造 AICPU AllGather ParallelMesh1DNHRMultiJetty 算法的 AlgoExecDesc。
 * 对应注册宏 REGISTER_EXECUTOR_BY_TWO_TEMPS(..., InsAllGatherParallelMesh1DNHRMultiJetty,
 *   InsV2AllGatherParallelExecutor, TopoMatchUBX, InsTempAllGatherMesh1D, InsTempAllGatherNHR)。
 * execPolicy=PARALLEL（对应 InsV2AllGatherParallelExecutor），children 两个 TemplateExecDesc
 * 分别为 Mesh1D（FULLMESH）和 NHR，jettyMode=MULTIPLE_JETTY，dataSplitRatio=1:1。
 */
static AlgoExecDesc MakeAicpuAllGatherParallelMesh1DNhrMultiJettyAlgoExecDesc()
{
    TemplateDesc mesh1DTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::MULTIPLE_JETTY};
    TemplateDesc nhrTemplateDesc{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR,
        HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::MULTIPLE_JETTY};
    TemplateExecDesc templateExecDesc0{mesh1DTemplateDesc, 0};
    TemplateExecDesc templateExecDesc1{nhrTemplateDesc, 1};
    AlgoExecDesc algoExecDesc;
    algoExecDesc.execPolicy = HcclAlgExecPolicy::PARALLEL;
    algoExecDesc.children = {templateExecDesc0, templateExecDesc1};
    algoExecDesc.dataSplitRatio = {1, 1}; // 1:1
    return algoExecDesc;
}

/**
 * 全局 AICPU AllGather 算法表。
 * key: HcclAicpuAllGatherAlgoType 枚举值
 * value: 对应的 HcclAlgorithm 实例（hcclCmdType=HCCL_CMD_ALLGATHER, engineType=AICPU,
 *        topoMatch 为对应 executor 类模板参数的 TopoMatch 子类实例指针）
 *
 * TopoMatch 子类与算法对应关系（源自 src/ops/all_gather/executor/ 注册宏）：
 *   TopoMatch1D         - InsAllGatherMesh1D, InsAllGatherMesh1D1DZAxisDetour, InsAllGatherNHR
 *   TopoMatch3Level     - InsV2AllGatherOmniPipeUboe
 *   TopoMatchMultilevel - InsAllGatherParallelMesh1DNHR, InsAllGatherSequenceNHRMesh1D
 *   TopoMatchUBX        - InsAllGatherConcurrentMesh1DNHR, InsAllGatherParallelMesh1DNHRMultiJetty
 *   TopoMatchPcieMix    - InsAllGatherParallelMesh1DNHRPcie, InsV2AllGatherOmniPipePcie
 *   TopoMatchSqueeze2D  - InsAllGatherParallelMesh1DNHRUboe
 */
const std::map<HcclAicpuAllGatherAlgoType, HcclAlgorithm> g_aicpuAllGatherAlgoMap = {
    {HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_OMNIPIPE_UBOE,
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatch3Level>())},
    {HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_NHR,
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatch1D>(), MakeAicpuAllGatherNhrAlgoExecDesc())},
    {HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_UBOE,
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatchSqueeze2D>(),
            MakeAicpuAllGatherParallelMesh1DNhrUboeAlgoExecDesc())},
    {HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_SEQUENCE_NHR_MESH1D,
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatchMultilevel>(),
            MakeAicpuAllGatherSequenceNhrMesh1DAlgoExecDesc())},
    {HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR,
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatchMultilevel>(),
            MakeAicpuAllGatherParallelMesh1DNhrAlgoExecDesc())},
    {HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_MESH1D1D_ZAXIS_DETOUR,
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatch1D>(),
            MakeAicpuAllGatherMesh1D1DZAxisDetourAlgoExecDesc())},
    {HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_MESH1D,
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatch1D>(),
            MakeAicpuAllGatherMesh1DAlgoExecDesc())},
    {HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_PCIE,
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatchPcieMix>(),
            MakeAicpuAllGatherParallelMesh1DNhrPcieAlgoExecDesc())},
    {HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_OMNIPIPE_PCIE,
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatchPcieMix>())},
    {HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_CONCURRENT_MESH1D_NHR,
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatchUBX>(),
            MakeAicpuAllGatherConcurrentMesh1DNhrAlgoExecDesc())},
    {HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_MULTIJETTY,
        MakeAicpuAllGatherAlgo(std::make_shared<TopoMatchUBX>(),
            MakeAicpuAllGatherParallelMesh1DNhrMultiJettyAlgoExecDesc())},
};




} // namespace ops_hccl
