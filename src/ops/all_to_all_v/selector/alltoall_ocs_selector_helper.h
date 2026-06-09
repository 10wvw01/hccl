/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_OCS_SELECTOR_HELPER_H
#define ALLTOALL_OCS_SELECTOR_HELPER_H

#include "alg_param.h"

namespace ops_hccl {

// 与 ins_temp_all_to_all_v_mesh_1D.h 中 ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE 保持同名同值
constexpr u32 ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE = 16;

u32 GetOcsGroupNum(const TopoInfoWithNetLayerDetails* topoInfo);

// 是否选用 OCS 算法：HCCL_IS_USE_OCS 开关 + groupNum/并发准入（使用 topoInfo->userRankSize）。
bool IsUseOcsAlgorithm(const TopoInfoWithNetLayerDetails* topoInfo);

} // namespace ops_hccl

#endif // ALLTOALL_OCS_SELECTOR_HELPER_H
