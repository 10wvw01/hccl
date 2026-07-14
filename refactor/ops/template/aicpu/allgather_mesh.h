/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_ALLGATHER_MESH_TEMPLATE_H
#define OPS_HCCL_ALLGATHER_MESH_TEMPLATE_H

#include "aicpu_base_template.h"

namespace ops_hccl {

/**
 * AllGather Mesh 1D 算法模板
 * 职责：在 Mesh 1D 拓扑下实现 AllGather 集合通信操作。
 * 算法原理：
 *   - 每个 rank 持有自己的一片数据，最终所有 rank 获得全部数据；
 *   - Mesh 拓扑下，每个 rank 与其他所有 rank 各通信一次，共 rankSize-1 步；
 *   - 每步将本 rank 当前持有的数据发送给对端，同时接收对端的数据。
 * 执行流程（由 AicpuBaseTemplate::KernelRun 编排）：
 *   1. PreCopy: input -> output / ccl buffer 本地数据预处理（基类默认实现）
 *   2. RunAlgorithm: 调 RunMeshAllGather 构造每对 rank 的 SendRecvInfo，
 *      按 PCIe/非 PCIe 选择 Read/Write 模式逐个执行 SendRecv
 *   3. PostCopy: ccl buffer -> output 后处理（基类默认实现，若需要）
 */
class AllGatherMeshTemplate : public AicpuBaseTemplate {
public:
    AllGatherMeshTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc)
        : AicpuBaseTemplate(myRank, std::move(ranks), std::move(templateDesc)) {}
    ~AllGatherMeshTemplate() = default;

protected:
    /** 通信编排：调 RunMeshAllGather 生成 SendRecvInfo 列表，由基类统一执行 SendRecv。 */
    HcclResult RunAlgorithm(TemplateResource &templateResource, std::vector<TxRxSlicesList> &txRxSlicesLists,
                            std::vector<u32> &ranksForOutputData) override;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_ALLGATHER_MESH_TEMPLATE_H
