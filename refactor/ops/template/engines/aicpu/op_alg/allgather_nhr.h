/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_ALLGATHER_MESH_TEMPLATE_H
#define OPS_HCCL_ALLGATHER_MESH_TEMPLATE_H

#include "alg_primitive_types.h"

namespace ops_hccl {

/**
 * AllGather NHR算法模板
 * 执行流程：
 *   1. LocalDataCopy: input -> output/ccl buffer（本地数据预处理）
 *   2. RunAllGatherNhr:执行 SendRecv 完成数据收集
 *   3. PostLocalCopy: ccl buffer -> output（后处理，若需要）
 */
class AllGatherNhrTemplate : public BaseTemplate {
public:
    /**
     * 执行 AllGather Nhr kernel。
     * 输入参数：
     *   - params: 模板算法参数，包含 buffer 信息、slice 大小、repeat 次数等
     *   - templateResource: 通信资源
     * 返回值：
     *   - HCCL_SUCCESS: 执行成功
     *   - HCCL_E_INTERNAL: 执行失败
     */
    HcclResult KernelRun(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource) override;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_ALLGATHER_NHR_TEMPLATE_H
