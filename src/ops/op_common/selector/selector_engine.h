/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_SELECTOR_ENGINE
#define HCCLV2_COLL_ALG_SELECTOR_ENGINE

#include <string>
#include <vector>

#include "alg_param.h"
#include "cost_model.h"
#include "cost_table.h"
#include "log.h"
#include "hccl_res.h"

namespace ops_hccl {

class SelectorEngine {
public:
    static SelectorEngine *Global();

    HcclResult Run(HcclComm comm, OpParam &param, TopoInfoWithNetLayerDetails *topoInfo,
                   std::string &algName);

    // 根据 algName 前缀推断所属引擎(OpExecuteConfig)
    static OpExecuteConfig GetEngineByAlgName(const std::string &algName);

    // 根据 topoInfo->hostDpuOnly + opExecuteConfig 返回可选择的引擎列表,按优先级高到低排序
    static std::vector<OpExecuteConfig> GetEnginePriority(TopoInfoWithNetLayerDetails *topoInfo,
                                                          OpExecuteConfig opExecuteConfig);

private:
    SelectorEngine() = default;

    HcclResult GetOrInitCostModel(HcclComm comm, const std::vector<OpExecuteConfig> &priorityOrder, CostModel *&cm);

    HcclResult GenAndProcessCostTable(OpParam &param, TopoInfoWithNetLayerDetails *topoInfo,
                                      CostTable &ct, CostModel &cm);

    HcclResult SelectMinCost(const CostTable &ct, const std::vector<OpExecuteConfig> &priorityOrder,
                             OpParam &param, std::string &algName);
};

} // namespace ops_hccl

#endif
