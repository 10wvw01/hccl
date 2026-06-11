/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AICPU_TASK_CACHE_POLICY_H
#define HCCL_AICPU_TASK_CACHE_POLICY_H

#include "alg_param.h"

namespace ops_hccl {

class AicpuTaskCachePolicy {
public:
    static bool IsAicpuTaskCacheEnable(const OpParam& param, const TopoInfoWithNetLayerDetails& topoInfo, const AlgResourceCtxSerializable& resCtxHost);

private:
    static bool IsTopoSupported(const AlgResourceCtxSerializable& resCtxHost);
    static HcclResult IsInplace(const OpParam& param, bool& isInplace, const TopoInfoWithNetLayerDetails& topoInfo);
    static HcclResult ParseOpParamForCache(const OpParam& param, HcclDataType& sendType, HcclDataType& recvType,
        uint64_t& inputSize, uint64_t& outputSize, const TopoInfoWithNetLayerDetails& topoInfo);
};

} // namespace ops_hccl
#endif
