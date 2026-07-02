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

enum class HcclAlgExecPolicy {
    PARALLEL,
    CONCURRENT,
    OMINIPIPE,
    SOLE,
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

struct TemplateExecDesc {
    TemplateDesc templateDesc;
    int subCommIndex;
}

using VariantType = std::variant<TemplateExecDesc, std::shared_ptr<AlgoExecDesc>>;

enum ExecPolicy {
    PARALLEL,
    SEQUENCE,
}

// SEQUENCE:   [[MESH], [NHR]]
// {
//     executeType: SEQ,
//     children = [MESH, NHR]
// }

// CONCURRENT: [[MESH, NHR]]
// {
//     executeType: PARALLEL,
//     children = [MESH, NHR]
// }

// PARALLEL: [[MESH, NHR, [NHR, MESH]]
// ROOT: {
//     executeType: SEQ,
//     templates = [NODE1, NODE2],
// }

// NODE1: {
//     executeType: PARALLEL,
//     templates = [MESH, NHR]
// }

// NODE2: {
//     executeType: PARALLEL,
//     templates = [NHR, MESH]
// }


struct AlgoExecDesc {
    HcclAlgExecPolicy execPolicy;    // 描述children的并行策略：串行/并行
    std::vector<VariantType> children;
};

// /* This HcclExecDomainInstance should be moved to execturor_base.h */
// struct HcclExecDomainInstance {
//     std::shared_ptr<RankGroup> rankGroup;   /**< The rank group in this logic domain, which should be filterd
//                                                  by algorithm's topoMatcher. */
//     std::shared_ptr<DataGroup> dataGroup;   /**< The data group to be processed in this logic domain. */
//     /* Other sub-instances to be added. */
// }

// struct HcclAlgExecDomain {
//     TemplateDesc& templateDesc;             /**< The templated method to be executed on this logic domain. */
//     std::shared_ptr<HcclExecDomainInstance> execDomainInstance; /**< Executing domain instance, which should be
//                                                  instantiated by executor. */
// }

class HcclAlgorithm {
public:
    BaseExecutor& GetExecutor();

private:
    HcclCMDType hcclCmdType;
    HcclAlgEngineType engineType;
    TopoMatchBase topoMatcher;
    AlgoExecDesc algoExecDesc;

    // std::vector<TemplateDesc> templateDescs;  /**< The TemplateDescs to be used by the executor in this algorithm. */
    // std::vector<std::vector<std::shared_ptr<HcclAlgExecDomain>>> execDomains; /**< The splited logic executing domains
    //                                     in this algorithm applied to the collective communication doamin.
    //                                     Currently, executing domains and their executing sequence are represent with
    //                                     two-level vector. Executing domains in the same lower vector will be triggered 
    //                                     in parellel (same stage). While vectors (domain groups) in the top vector will
    //                                     be triggered in positive order. For examples:
    //                                     1. [[stage0_domain1, stage0_domain2],[stage1_domain1, stage1_domain2]], in this
    //                                     example, stage0_domain1 and stage0_domain2 will be triggered in parellel in the
    //                                     first stage, stage1_domain1 and stage1_domain2 will be triggered in parellel in
    //                                     the second stage. As to how the raw data will be sliced into different domains,
    //                                     and which ranks each domain contains, it's up to the executor to calculate by
    //                                     topoMatcher and corresponding data slicer. (Parellel)
    //                                     2. [[stage0_domain1, stage0_domain2]], in this example, two domains will be
    //                                     triggered in parellel within the only same stage. (Concurrent)
    //                                     3. [[stage0_domain1], [stage1_domain2, stage1_domain2]], in this example, 
    //                                     stage0 contains only one domain, stage2 contains two domains which will be
    //                                     triggered in parellel. (Ominipipe)
    //                                     4. [[stage0_domain1], [stage1_domain2], [stage2_domain3]], in this example,
    //                                     there are three stages each contains only one executing domain. (Sequence)
    //                                     5. [[stage0_doamin1, stage0_domain2], [stage1_domain3]], in this example, stage0
    //                                     contains two domains which will be triggered in parallel, stage1 contains only
    //                                     one domain. (PartConcurrent)
    //                                     */
    // std::shared_ptr<BaseExecutor> executor;
};

} // namespace ops_hccl

#endif