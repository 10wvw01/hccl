/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_REDUCESCATTER_MESH_TEMPLATE_H
#define OPS_HCCL_REDUCESCATTER_MESH_TEMPLATE_H

#include "aicpu_base_template.h"

namespace ops_hccl {

/**
 * ReduceScatter Mesh 1D 算法模板
 * 职责：在 Mesh 1D 拓扑下实现 ReduceScatter 集合通信操作。
 * 算法原理：
 *   - 每个 rank 将自己的数据按 rankSize 切分为 rankSize 个分片；
 *   - 每个 rank 保留自己的一个分片，并接收其他 rank 对应分片，进行归约；
 *   - Mesh 拓扑下，每个 rank 与其他所有 rank 各通信一次，共 rankSize-1 步；
 *   - 每步将本 rank 对应分片发送给对端，同时接收对端的对应分片并归约。
 * 执行流程（由 AicpuBaseTemplate::KernelRun 编排）：
 *   1. PreCopy: 只 LocalCopy 本卡 myRank 数据到 ccl buffer[myRank 槽]
 *   2. RunAlgorithm: 调 RunMeshReduceScatter 生成纯搬运 tx/rx，tx 源=input，
 *      把本卡 input 中对端 rank 的分片 Write 到对端 ccl buffer[myRank 槽]；
 *      rx 写入本卡 ccl buffer[connectedRank 槽]；reduceOp=RESERVED（不 inline reduce）
 *   3. SendAll: 通信完成后，LocalReduce 本卡 ccl buffer 各 rank 槽位到 ccl[myRank 槽]
 *   4. PostCopy（基类）：LocalCopy ccl buffer[myRank 槽]到 output
 */
class ReduceScatterMeshTemplate : public AicpuBaseTemplate {
public:
    ReduceScatterMeshTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc)
        : AicpuBaseTemplate(myRank, std::move(ranks), std::move(templateDesc)) {}
    ~ReduceScatterMeshTemplate() = default;

protected:
    /** 通信编排：调 RunMeshReduceScatter 生成 SendRecvInfo 列表，由基类统一执行 SendRecv。 */
    HcclResult RunAlgorithm(TemplateResource &templateResource, std::vector<TxRxSlicesList> &txRxSlicesLists,
                            std::vector<u32> &ranksForOutputData) override;

    /** PreCopy: 只 LocalCopy 本卡 myRank 数据到 ccl buffer[myRank 槽]。 */
    HcclResult PreCopy(const std::vector<ThreadHandle> &threads) override;

    /** SendAll: 通信纯搬运（reduceOp=RESERVED），完成后 LocalReduce 各 rank 槽到 ccl[myRank 槽]。 */
    HcclResult SendAll(BaseEngine &engine, const std::vector<TxRxSlicesList> &txRxSlicesLists,
                       TemplateResource &templateResource, const std::vector<ThreadHandle> &threads) override;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_REDUCESCATTER_MESH_TEMPLATE_H
