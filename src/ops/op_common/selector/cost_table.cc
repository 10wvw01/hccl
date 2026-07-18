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

namespace ops_hccl {

CostTable *CostTable::Global()
{
    static CostTable *globalCostTable = new CostTable;
    return globalCostTable;
}

HcclResult CostTable::Load()
{
    const std::lock_guard<std::mutex> lock(mu_);
    HCCL_DEBUG("[CostTable] load cost table, current entry count=%zu.", entries_.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CostTable::Query(const std::string &algName, u64 dataSize, double &cost) const
{
    const std::lock_guard<std::mutex> lock(mu_);
    for (const auto &entry : entries_) {
        if (entry.algName == algName && dataSize >= entry.minDataSize && dataSize <= entry.maxDataSize) {
            cost = entry.cost;
            return HcclResult::HCCL_SUCCESS;
        }
    }
    HCCL_WARNING("[CostTable] no entry matched algName=%s dataSize=%llu.", algName.c_str(), dataSize);
    return HcclResult::HCCL_E_PARA;
}

} // namespace ops_hccl
