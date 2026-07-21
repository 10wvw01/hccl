/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cost_table.h"

#include <new>

namespace ops_hccl {

HcclResult CostTableManager::FilterCMByConfig(CostModel &cm, CostTable &ct,
                                              const TopoInfoWithNetLayerDetails *topoInfo,
                                              const OpParam &opParam)
{
    (void)cm;
    (void)ct;
    (void)topoInfo;
    (void)opParam;
    HCCL_DEBUG("[FilterCMByConfig] filter cost model by config.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CostTableManager::CostTableGen(CostModel &cm, CostTable &ct,
                                          const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam)
{
    HCCL_DEBUG("[CostTableGen] generate cost table, algCount=%d.", cm.count);
    HcclResult ret = FilterCMByConfig(cm, ct, topoInfo, opParam);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CostTableGen] FilterCMByConfig failed, ret=%d.", static_cast<int>(ret));
        return ret;
    }
    if (cm.count <= 0) {
        ct.costs = nullptr;
        ct.count = 0;
        return HcclResult::HCCL_SUCCESS;
    }

    ct.costs = new (std::nothrow) AlgoCost[cm.count];
    if (ct.costs == nullptr) {
        HCCL_ERROR("[CostTableGen] alloc AlgoCost failed, count=%d.", cm.count);
        return HcclResult::HCCL_E_PARA;
    }

    for (int i = 0; i < cm.count; ++i) {
        ct.costs[i].algName = cm.costAlgoParams[i].algName;
        ct.costs[i].cost = 0.0f;
    }
    ct.count = cm.count;
    return HcclResult::HCCL_SUCCESS;
}

CostTableManager *CostTableManager::Global()
{
    static CostTableManager *globalCostTableManager = new CostTableManager;
    return globalCostTableManager;
}

CostTableManager::~CostTableManager() {}

HcclResult CostTableManager::Load()
{
    const std::lock_guard<std::mutex> lock(mu_);
    HCCL_DEBUG("[CostTableManager] load cost table, count=%d.", costTable_.count);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CostTableManager::Query(const std::string &algName, u64 dataSize, double &cost) const
{
    const std::lock_guard<std::mutex> lock(mu_);
    for (int i = 0; i < costTable_.count; ++i) {
        if (algName == costTable_.costs[i].algName) {
            cost = costTable_.costs[i].cost;
            return HcclResult::HCCL_SUCCESS;
        }
    }
    HCCL_WARNING("[CostTableManager] no entry matched algName=%s dataSize=%llu.", algName.c_str(), dataSize);
    return HcclResult::HCCL_E_PARA;
}

} // namespace ops_hccl
