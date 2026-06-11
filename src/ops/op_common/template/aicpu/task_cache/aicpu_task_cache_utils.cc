/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_task_cache_utils.h"
namespace ops_hccl {

bool AicpuTaskCacheUtils::IsNonVariableOpType(HcclCMDType opType)
{
    // 注意: 当前hccl不支持HcclGather
    if (opType == HcclCMDType::HCCL_CMD_BROADCAST || opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
        opType == HcclCMDType::HCCL_CMD_REDUCE || opType == HcclCMDType::HCCL_CMD_ALLGATHER ||
        opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
        opType == HcclCMDType::HCCL_CMD_SCATTER) {
        return true;
    }
    return false;
}

} // namespace ops_hccl