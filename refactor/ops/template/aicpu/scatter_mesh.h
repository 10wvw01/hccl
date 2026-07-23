/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SCATTER_MESH_TEMPLATE_H
#define OPS_HCCL_SCATTER_MESH_TEMPLATE_H

#include "aicpu_base_template.h"

namespace ops_hccl {

/**
 * Scatter Mesh 1D 算法模板
 * 职责：在 Mesh 1D 拓扑下实现 Scatter 集合通信操作。
 * 算法原理：
 *   - root rank 持有全部 rankSize 个分片，其他 rank 无输入数据；
 *   - Mesh 拓扑下，root 与其他所有 rank 各通信一次，共 rankSize-1 步；
 *   - root 将每个 rank 对应的分片发送给该 rank，非 root rank 只接收属于自己的分片。
 * 执行流程（由 AicpuBaseTemplate::KernelRun 编排）：
 *   1. PreCopy（基类）: 仅 root rank 执行（将 input 中所有 rank 的分片拷到 ccl buffer 槽位），
 *      非 root rank 无 input，基类按 hcclCmdType==SCATTER && myRank!=root 跳过。
 *      Parallel 下第二个子算法的 inputBufferType=HCCL_BUFFER，基类也会跳过。
 *   2. RunAlgorithm: 调 RunMeshScatter 构造 tx/rx 描述符（root 只 tx，非 root 只 rx），
 *      由基类 SendAll 统一执行（reduceOp=RESERVED，纯搬运）
 *   3. PostCopy（基类）：将 ccl buffer[myRank 槽] 拷回 output
 */
class ScatterMeshTemplate : public AicpuBaseTemplate {
public:
    ScatterMeshTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc)
        : AicpuBaseTemplate(myRank, std::move(ranks), std::move(templateDesc)) {}
    ~ScatterMeshTemplate() = default;

protected:
    /** 通信编排：调 RunMeshScatter 生成 tx/rx 描述符，由基类统一执行 SendRecv。 */
    HcclResult RunAlgorithm(TemplateResource &templateResource, std::vector<TxRxSlicesList> &txRxSlicesLists,
                            std::vector<u32> &ranksForOutputData) override;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_SCATTER_MESH_TEMPLATE_H
