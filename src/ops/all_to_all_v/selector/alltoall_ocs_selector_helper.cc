/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoall_ocs_selector_helper.h"

#include <algorithm>
#include <cstdlib>

#include "log.h"

namespace ops_hccl {

static u32 CalcOcsAdjustedConcurrent(u32 rankSize, u32 groupNum)
{
    if (groupNum == 0) {
        return 0;
    }
    return (ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE / groupNum) * groupNum;
}

u32 GetOcsGroupNum(const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (topoInfo != nullptr) {
        return topoInfo->ocsGroupNum == 0 ? 1 : topoInfo->ocsGroupNum;
    }
    return 1;
}

bool IsUseOcsAlgorithm(const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (topoInfo == nullptr) {
        return false;
    }

    u32 rankSize = topoInfo->userRankSize;
    u32 groupNum = GetOcsGroupNum(topoInfo);
    if (groupNum <= 1) {
        HCCL_INFO("[AlltoAllOcsHelper][IsUseOcsAlgorithm] groupNum[%u] <= 1, skip OCS", groupNum);
        return false;
    }
    // 通信域rank数小于最大并发数回退算法
    if (rankSize < ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE) {
        HCCL_INFO("[AlltoAllOcsHelper][IsUseOcsAlgorithm] rankSize[%u] < concurrent size %u, skip OCS", rankSize, ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE);
        return false;
    }
    if (rankSize == 0 || rankSize % groupNum != 0) {
        HCCL_INFO("[AlltoAllOcsHelper][IsUseOcsAlgorithm] rankSize[%u] %% groupNum[%u] != 0, skip OCS",
            rankSize, groupNum);
        return false;
    }
    // group间不均衡，回退算法
    u32 adjustedConcurrent = CalcOcsAdjustedConcurrent(rankSize, groupNum);
    if (adjustedConcurrent < groupNum) {
        HCCL_INFO("[AlltoAllOcsHelper][IsUseOcsAlgorithm] concurrent not enough, rankSize[%u] groupNum[%u] "
            "adjustedConcurrent[%u], skip OCS",       
            rankSize, groupNum, adjustedConcurrent);
        return false;
    }
    return true;
}

} // namespace ops_hccl