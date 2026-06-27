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

#include "aicpu_base_template.h"
#include "alg_primitive_types.h"

namespace ops_hccl {

/**
 * AllGather Mesh 1D 算法模板
 * 职责：在 Mesh 1D 拓扑下实现 AllGather 集合通信操作。
 * 算法原理：
 *   - 每个 rank 持有自己的一片数据，最终所有 rank 获得全部数据；
 *   - Mesh 拓扑下，每个 rank 与相邻 rank 逐一通信，共 rankSize-1 步；
 *   - 每步将本 rank 当前持有的数据发送给下一个 rank，同时接收上一个 rank 的数据。
 * 执行流程：
 *   1. LocalDataCopy: input -> output/ccl buffer（本地数据预处理）
 *   2. RunAllGatherMesh: 逐 rank 执行 SendRecv 完成数据收集
 *   3. PostLocalCopy: ccl buffer -> output（后处理，若需要）
 */
class AllGatherMeshTemplate : public BaseTemplate {
public:
    /**
     * 执行 AllGather Mesh kernel。
     * 输入参数：
     *   - params: 模板算法参数，包含 buffer 信息、slice 大小、repeat 次数等
     *   - templateResource: 通信资源
     * 返回值：
     *   - HCCL_SUCCESS: 执行成功
     *   - HCCL_E_INTERNAL: 执行失败
     */
    HcclResult KernelRun(const TemplateAlgParams &params, TemplateResource &templateResource) override;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_ALLGATHER_MESH_TEMPLATE_H
