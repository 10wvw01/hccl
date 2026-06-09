/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_task_cache_policy.h"
#include "alg_env_config.h"
#include "log.h"

namespace ops_hccl {

bool AicpuOpCachePolicy::IsOpTaskCacheEnable(const OpParam& param)
{
    if (!GetExternalInputHcclAicpuCacheEnable()) {
        return false;
    }

    const std::string& algName = param.algName;
    // 目前V类算子、batch类型算子、以及send/recv不考虑动态缓存 (使用白名单而非黑名单管理, 避免非预期算子进入cache机制)
    // 注意: 如果想要通过比较缓存刷新后的SQE与正常算子展开的SQE来debug, 可以将想要比较的算子从以下的cache白名单中移除,重新打包运行
    const HcclCMDType opType = param.opType;
    if (opType == HcclCMDType::HCCL_CMD_BROADCAST || opType == HcclCMDType::HCCL_CMD_REDUCE
        || opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
        || opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_SCATTER
        || opType == HcclCMDType::HCCL_CMD_ALLREDUCE) { // 非V类算子
        if (algName == "RunAlltoAllVStaged" || algName == "RunAlltoAllVFullMesh") {
            HCCL_INFO(
                "[AicpuOpCachePolicy][%s] algName[%s] is not supported for unfolding cache", __func__, algName.c_str());
            return false;
        }
        HCCL_INFO(
            "[AicpuOpCachePolicy][IsOpTaskCacheEnable] opType[%d] is supported for operator unfolding cache", opType);
        return true;
    }

    return false;
}

} // namespace ops_hccl
