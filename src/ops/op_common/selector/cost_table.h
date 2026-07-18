/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_SELECTOR_COST_TABLE
#define HCCLV2_COLL_ALG_SELECTOR_COST_TABLE

#include <mutex>
#include <string>

#include "alg_param.h"
#include "log.h"

namespace ops_hccl {

typedef struct {
    const char *algName;
    float cost;
} AlgoCost;

typedef struct {
    AlgoCost *costs;
    int count;
} CostTable;

class CostTableManager {
public:
    static CostTableManager *Global();

    ~CostTableManager();

    HcclResult Load();
    HcclResult Query(const std::string &algName, u64 dataSize, double &cost) const;

private:
    CostTableManager() = default;

    CostTable      costTable_{nullptr, 0};
    mutable std::mutex mu_;
};

} // namespace ops_hccl

#endif
