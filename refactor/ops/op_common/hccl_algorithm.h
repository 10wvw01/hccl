/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @defgroup hccl_algorithm HcclAlg
 * @ingroup Hccl
 * @brief This file defines the HCCL algorithm description structure within multiple orthogonal dimentions.
 * This structure is represented with a class, which is instantiated by the upstream algorithm selector.
 * This algorithm class provides public APIs to construct corresponding executor to actually run the algorithm
 * upon the specified collective communication domain.
 *
 * Note, the algorhtm is organized in hierarchical structure. A executor works in higher level to
 * orchestrate multiple standalone workflows each for a specific logic sub-domain. We define the workflow
 * with template, which is described with TemplateDesc. As to how the workflows are figured out upon a
 * collective communication domain, it's the responsibility of algorithm selector, which will fill them
 * into the vector<TemplateDesc>. As to how these workflows will be orcheatrated, it can be uniquely determined
 * by executor's type.
 *
 * A logic sub-domain determines the scope a template works on. More specific, a sub-domain defines the scope
 * in which a groups of ranks to exchange data with specified manner. Multiple scopes may overlap in physical.
 * And. the upper layer executor takes the responsibility to schedule and synchronize its multiple templates.
 * The executor's scheduling and synchronizing procedure is defined in its orchestrate method. The ExecutorType
 * determines the different orchestrating methods.
 *
 * For a sub-domain a specific template works on, TopoMatchBase class provides the method to match/filter the
 * raw topo information into a specific topo view which the template cares and works on.
 */

#ifndef HCCL_ALG_H
#define HCCL_ALG_H

#include <memory>
#include <variant>
#include <vector>

#include <hccl/hccl_types.h>
#include "topo_match_base.h"
#include "alg_type.h"
#include "alg_param.h"

namespace ops_hccl {

enum class HcclAlgEngineType {
    AICPU,
    CCU_MS,
    CCU_SCHED,
    AIV,
};

enum class HcclAlgExecPolicy {
    SOLE,
    SEQUENCE,
    PARALLEL,
    CONCURRENT,
};

enum class HcclAlgShotMode {
    ONE_SHOT,
    TWO_SHOT,
};

enum class HcclAlgJettyMode {
    SINGLE_JETTY,
    MULTIPLE_JETTY,
};

// AICPU 模式 AllGather 算法枚举，对应 all_gather_auto_selector.cc 中 SelectAicpuAlgo 的 11 种算法
enum class HcclAicpuAllGatherAlgoType {
    AICPU_ALLGATHER_OMNIPIPE_UBOE,                  // InsV2AllGatherOmniPipeUboe
    AICPU_ALLGATHER_NHR,                            // InsAllGatherNHR
    AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_UBOE,       // InsAllGatherParallelMesh1DNHRUboe
    AICPU_ALLGATHER_SEQUENCE_NHR_MESH1D,            // InsAllGatherSequenceNHRMesh1D
    AICPU_ALLGATHER_PARALLEL_MESH1D_NHR,            // InsAllGatherParallelMesh1DNHR
    AICPU_ALLGATHER_MESH1D1D_ZAXIS_DETOUR,          // InsAllGatherMesh1D1DZAxisDetour
    AICPU_ALLGATHER_MESH1D,                         // InsAllGatherMesh1D
    AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_PCIE,       // InsAllGatherParallelMesh1DNHRPcie
    AICPU_ALLGATHER_OMNIPIPE_PCIE,                  // InsV2AllGatherOmniPipePcie
    AICPU_ALLGATHER_CONCURRENT_MESH1D_NHR,          // InsAllGatherConcurrentMesh1DNHR
    AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_MULTIJETTY, // InsAllGatherParallelMesh1DNHRMultiJetty
    AICPU_ALLGATHER_ALGO_TYPE_COUNT,                // 算法类型总数，用于数组下标上限
};

struct TemplateDesc {
    HcclCMDType hcclCmdType;
    HcclAlgoType algType;
    HcclAlgShotMode shotMode;
    HcclAlgJettyMode jettyMode;
};

// 子通信域索引：Intra=0（组内/网络层级 level 0），Inter=1（组间/网络层级 level 1）
enum SubCommIndexType : int {
    SUB_COMM_INDEX_INTRA = 0,
    SUB_COMM_INDEX_INTER = 1,
};

struct TemplateExecDesc {
    TemplateDesc templateDesc;
    int subCommIndex;
};

// SEQUENCE:   [[MESH], [NHR]]
// {
//     executeType: SEQ,
//     children = [MESH, NHR]
// }
//
// CONCURRENT: [[MESH, NHR]]
// {
//     executeType: PARALLEL,
//     children = [MESH, NHR]
// }
//
// PARALLEL: [[MESH, NHR, [NHR, MESH]]
// ROOT: {
//     executeType: SEQ,
//     templates = [NODE1, NODE2],
// }
//
// NODE1: {
//     executeType: PARALLEL,
//     templates = [MESH, NHR]
// }
//
// NODE2: {
//     executeType: PARALLEL,
//     templates = [NHR, MESH]
// }
struct AlgoExecDesc;
using VariantType = std::variant<TemplateExecDesc, std::shared_ptr<AlgoExecDesc>>;
struct AlgoExecDesc {
    HcclAlgExecPolicy execPolicy; // 描述children的并行策略：串行/并行
    std::vector<VariantType> children;
    std::vector<u32> dataSplitRatio; // 并行数据切分比例，元素个数必须和children个数一致,例如1:1:1
};

class OpsExecutor;
class BaseLauncher;
/**
 * 算法描述对象，由上游 Selector 选定后注入。
 * 提供 GetEngine/GetExecutor 能力，供 HcclExecOp 入口获取引擎与执行器。
 * 成员对执行器（OpsExecutor）公开，执行器在编排时直接读取算法描述信息。
 */
class HcclAlgorithm {
public:
    HcclAlgorithm() = default;
    ~HcclAlgorithm() = default;

    /**
     * 根据算法的引擎类型构造对应的 Launcher。
     * 输入参数：
     *   - comm: 通信域句柄，CCU Launcher 的 CreateRes 需要 comm 调用 HcclGetHcclBuffer/
     *     HcclChannelAcquire/HcclCommQueryCcuIns；AiCpuLauncher 不使用（保持默认构造）
     * 返回值：
     *   - AICPU: AiCpuLauncher
     *   - CCU_MS: CcuMsLauncher（传入 comm）
     *   - CCU_SCHED: CcuScheLauncher（传入 comm）
     * 返回 unique_ptr<BaseLauncher>，所有权移交调用方。
     */
    std::unique_ptr<BaseLauncher> GetEngine(HcclComm comm);

    /**
     * 根据算子类型与执行策略构造对应的 Executor。
     * 输入参数：
     *   - param: 算子参数，决定具体子类执行器（AllGather/ReduceScatter/... + Parallel/Sole）
     * 返回 unique_ptr<OpsExecutor>，所有权移交调用方。
     */
    std::unique_ptr<OpsExecutor> GetExecutor(OpParam &param);

    /**
     * 打印算法基本信息（引擎类型、cmd 类型、执行树等），用于调试。
     */
    void Dump();

    /**
     * 序列化算法描述，用于多机间算法选择一致性校验（Todo）。
     */
    void Serialize();

    // 算法描述成员，执行器在 CalcAlgHierarchyInfo/CalcRes/Orchestrate 中直接读取
    HcclCMDType hcclCmdType;
    HcclAlgEngineType engineType;
    std::shared_ptr<TopoMatchBase> topoMatch;
    AlgoExecDesc algoExecDesc;
};

// 全局 AICPU AllGather 算法表（定义在 algorithm/all_gather/algorithm_all_gather_aicpu.cc），以
// HcclAicpuAllGatherAlgoType 枚举值为数组下标。
extern const HcclAlgorithm
    g_aicpuAllGatherAlgoMap[static_cast<size_t>(HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_ALGO_TYPE_COUNT)];

} // namespace ops_hccl

#endif
