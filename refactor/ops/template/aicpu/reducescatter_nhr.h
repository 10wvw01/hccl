/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_REDUCESCATTER_NHR_TEMPLATE_H
#define OPS_HCCL_REDUCESCATTER_NHR_TEMPLATE_H

#include "aicpu_base_template.h"

namespace ops_hccl {

/**
 * ReduceScatter NHR 算法模板
 * 职责：在 NHR（Non-Halving Recursive）拓扑下实现 ReduceScatter 集合通信操作。
 * 算法原理：
 *   - NHR 基于 halving doubling 思路，共 log2(rankSize) 步；
 *   - 第 step 步与距离 delta=2^(nSteps-1-step) 的对端双向收发；
 *   - 每步传输 nSlices 个分片，分片索引按 deltaSliceIndex=2^(nSteps-step) 递减；
 *   - 每步收到的分片与本地对应分片进行归约。
 * 执行流程（由 AicpuBaseTemplate::KernelRun 编排）：
 *   1. PreCopy: input -> ccl buffer 本地数据预处理（基类默认实现）
 *   2. RunAlgorithm: 调 RunNhrReduceScatter 构造 SendRecvInfo 列表，
 *      从中提取 sendRecvSlices_ 填充 txRxSlicesLists，由基类统一执行 SendRecv
 *   3. PostCopy: ccl buffer -> output 后处理（基类默认实现，若需要）
 */
class ReduceScatterNhrTemplate : public AicpuBaseTemplate {
public:
    ReduceScatterNhrTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc)
        : AicpuBaseTemplate(myRank, std::move(ranks), std::move(templateDesc)) {}
    ~ReduceScatterNhrTemplate() = default;

protected:
    /** 通信编排：调用 RunNhrReduceScatter 构造 SendRecvInfo 列表，提取 sendRecvSlices_ 填充 txRxSlicesLists，
     *  由基类统一执行 SendRecv。 */
    HcclResult RunAlgorithm(TemplateResource &templateResource, std::vector<TxRxSlicesList> &txRxSlicesLists,
                            std::vector<u32> &ranksForOutputData) override;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_REDUCESCATTER_NHR_TEMPLATE_H
