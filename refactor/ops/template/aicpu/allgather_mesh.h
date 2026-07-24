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
 *   1. PreCopy: input -> ccl buffer[myRank]（基类默认实现，供对端 Read 读取）
 *   2. RunAlgorithm: 调 RunMeshAllGather 构造每对 rank 的 SendRecvInfo，
 *      outputBufferType==OUTPUT 时 rx 直接落 output（不经本地 ccl 中转）
 *   3. SendAll: outputBufferType==OUTPUT 时走 READ 方向（本端主动从对端 ccl 读到本地 output）；
 *      否则走基类 WRITE 方向
 *   4. PostCopy: outputBufferType==OUTPUT 时只补搬 myRank 的 ccl->output（peer 已直接落 output）；
 *      outputBufferType==HCCL_BUFFER 时由基类将全部 ccl buffer 搬回 output
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

    /**
     * outputBufferType==OUTPUT 时 override SendAll：让 engine 走 READ 方向
     * （本端主动从对端 ccl buffer 读到本地 output，对端无法直接写本端 output）。
     * 否则调用基类默认实现（WRITE 方向）。
     */
    HcclResult SendAll(BaseEngine &engine, const std::vector<TxRxSlicesList> &txRxSlicesLists,
                       TemplateResource &templateResource, const std::vector<ThreadHandle> &threads) override;

    /**
     * outputBufferType==OUTPUT 时 override PostCopy：仅补搬 myRank 的 ccl->output
     * （peer 数据已在 SendAll 直接写入 output，不能复用基类以免 ccl[peer] 垃圾覆盖 output[peer]）。
     * 否则调用基类默认实现（ccl buffer -> output）。
     */
    HcclResult PostCopy(const std::vector<ThreadHandle> &threads) override;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_ALLGATHER_MESH_TEMPLATE_H
