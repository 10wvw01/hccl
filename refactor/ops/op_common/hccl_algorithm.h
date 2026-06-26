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

#include <hccl/hccl_types.h>
#include "topo_match_base.h"
#include "alg_type.h"

namespace ops_hccl {

enum class HcclAlgEngineType {
    AICPU,
    CCU_MS,
    CCU_SCHED,
};

enum class HcclAlgExecutorType {
    PARALLEL,
    CONCURRENT,
    OMINIPIPE,
};

enum class HcclAlgShotMode {
    ONE_SHOT,
    TWO_SHOT,
};

enum class HcclAlgJettyMode {
    SINGLE_JETTY,
    MULTIPLE_JETTY,
};

struct TemplateDesc {
    HcclCMDType hcclCmdType;
    HcclAlgoType algType;
    HcclAlgShotMode shotMode;
    HcclAlgJettyMode jettyMode;
};

class HcclAlgorithm {
public:
    HcclCMDType hcclCmdType;
    HcclAlgEngineType engineType;
    HcclAlgExecutorType executorType;
    std::vector<TemplateDesc> templates;
    TopoMatchBase topoView;

    HcclResult GetExecutor();
    HcclResult GetTemplate();

private:
    std::shared_ptr<BaseExecutor> executor;
};

} // namespace ops_hccl

#endif


    // vector<vector<TemplateDesc>> templateDescs;   // 第一层表示stage，第二层表示数据part, 
    //                                               //  parallel: [[stage0_part0, stage0_part1],[stage1_part0, stage1_part1]] 
    //                                               //  concurrent: [[stage0_part0, stage0_part1]]
    //                                               //  omnipipe: [[stage0_part], [stage1_part0, stage1_part1]]
    //                                               //  sequece: [[stage0_part], [stage1_part], [stage2_part]]
    //                                               //  partConcurrent: [[stage0_part0, stage0_part1], [stage1_part0]]

    //                             Orch {
    //                                 thread (stage0_part0, stage0_part1)
    //                                 thread (stage1_part0)
    //                                 sync
    //                             }

    // vector<vector<int>> templateTopoIndex;        // 当前：topoIndex0表示板内的通信子域，1表示框内的通信子域，2表示超节点间的通信子域
                                                
    // 3: 
    // 串行Executor: 2 -> 1 -> 0  templateDescs: [[NHR], [NHR], [MESH]]    templateTopoIndex :[[2],[1],[0]]
    // OmniPipe 2 -> 1,0  templateDescs: [[NHR], [NHR, MESH]]    templateTopoIndex:[[2], [1, 0]]


    // [[NHR], [NHR]]


    // REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherParallelMesh1DNHR,
//                                InsV2AllGatherParallelExecutor, TopoMatchMultilevel, InsTempAllGatherMesh1D,
//                                InsTempAllGatherNHR);


// HcclAlgorithm { hcclCmdType = HCCL_CMD_ALLGATHER, engineType = AICPU, executorType = PARALLEL, templates = 
//     [[{hcclCmdType = HCCL_CMD_ALLGATHER, alg = mesh, ShotType = DEFAULT, JetttyType = DEFAULT}, 
//             {hcclCmdType = HCCL_CMD_ALLGATHER, alg = nhr, ShotType = DEFAULT, JetttyType = DEFAULT}],
//         [{hcclCmdType = HCCL_CMD_ALLGATHER, alg = nhr, ShotType = DEFAULT, JetttyType = DEFAULT}, 
//         {hcclCmdType = HCCL_CMD_ALLGATHER, alg = mesh, ShotType = DEFAULT, JetttyType = DEFAULT}]]
// }

// templateTopoIndex[[0, 1], [1, 0]]  // topoIndex0表示板内的通信子域，1表示框内的通信子域，2表示超节点间的通信子域
// template[[0, 1, 2]]  [1][3]




//REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLREDUCE, InsAllReduceMesh1DTwoShotMeshChunk, InsV2AllReduceSoleExecutor, 
//    TopoMatch1D, InsTempAllReduceMesh1DTwoShotMeshChunk);
// HcclAlgorithm {op = AllReduce, engine = aicpu, executor = Sole, templates = [[
//     {op = AllReduce, alg = mesh, CustomFeatures= {ShotType: TwoShotMeshChunk}},
// ],

// ]
// }