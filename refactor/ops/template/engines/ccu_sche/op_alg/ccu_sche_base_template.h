/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_SCHE_BASE_TEMPLATE_H
#define OPS_HCCL_CCU_SCHE_BASE_TEMPLATE_H

#include <memory>
#include <vector>

#include "base_template.h"
#include "alg_param.h"
#include "template_utils.h"
#include "die_strategy.h"
#include "ccu_kernel_arg_base.h"

namespace ops_hccl {

/**
 * CCU_SCHE (调度模式 / 非 mem2mem) 引擎模式 Host 侧 BaseTemplate。
 *
 * 职责：
 *   - 归一化 CCU_SCHE 模式的 KernelRun / FastLaunch 主流程；
 *   - 通过 IDieStrategy 处理 1Die/2Die 架构差异（仅 peer 集分片，每 die 独立 kernel）；
 *   - 不持有 IMultiJettyStrategy（SCHE 模式不支持 MultiJetty）；
 *   - 子类化点：CalcRes、GetKernelEntry、BuildKernelArg、BuildTaskArgs、SaveSubmitInfo。
 *
 * 与 CCU_MS 差异：
 *   - taskArgs 8 字段（inputAddr/outputAddr/token/offset + goSize[4]，无 scratch/die0Size/repeatStride）；
 *   - 数据经 CcuBuffer 中转，使用 GroupBroadcast/GroupReduce/GroupCopy LoopGroup 原语；
 *   - 不支持 MultiJetty；
 *   - LoopGroup 自动流水（loopCount=CCU_MS_DEFAULT_LOOP_COUNT=128, memSlice=4096）。
 *
 * 参考实现：
 *   - src/ops/all_gather/template/ccu/ccu_temp_all_gather_mesh_1D.cc（8 字段 taskArgs、LoopGroup 流水）
 */
class CcuScheBaseTemplate : public BaseTemplate {
public:
    CcuScheBaseTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc,
                        std::unique_ptr<IDieStrategy> dieStrategy);
    ~CcuScheBaseTemplate() override = default;

    /** 资源计算入口。子类实现 channel 决策并调用本类 BuildKernelInfo 构造 CcuKernelInfo 列表。 */
    HcclResult CalcRes(HcclComm comm, AlgResourceRequest &res) override = 0;

    /**
     * CCU_SCHE kernel 下发主流程：
     *   1. DecideDie 获取 die 划分（仅 peer 集分片）；
     *   2. 构造 LoopGroupConfig + CalGoSize；
     *   3. BuildTaskArgs 构造 8 字段 taskArgs；
     *   4. 多 die 时 PreSyncInterThreads；
     *   5. 遍历 die 调用 HcommCcuKernelLaunch；
     *   6. 多 die 时 PostSyncInterThreads；
     *   7. SaveSubmitInfo 填充 CcuKernelSubmitInfo。
     */
    HcclResult KernelRun(const TemplateDataParams &params, TemplateResource &templateResource,
                         std::vector<u32> &ranksForOutputData) override;

    /** 快速下发：从缓存 cachedArgs 补地址偏移后直接 HcommCcuKernelLaunch（无 scratch）。 */
    HcclResult FastLaunch(const OpParam &param, const TemplateFastLaunchCtx &ctx) override;

protected:
    /** 子类实现：返回 kernel 函数指针与名字。 */
    virtual HcclResult GetKernelEntry(void *&kernelFunc, const char *&kernelName) = 0;

    /** 子类实现：构造算子专属的 CcuKernelArgBaseSche 派生实例。 */
    virtual HcclResult BuildKernelArg(std::shared_ptr<CcuKernelArgBaseSche> &arg) = 0;

    /**
     * 子类实现：构造 taskArgs（SCHE 8 字段：inputAddr/outputAddr/token/offset + goSize[4]）。
     * goSize 由 CalGoSize(sliceSize, config) 计算，基类 KernelRun 负责构造 config 并传入。
     */
    virtual HcclResult BuildTaskArgs(const TemplateDataParams &params, const DieDecision &dieDecision,
                                     const std::vector<uint64_t> &goSize,
                                     std::vector<uint64_t> &taskArgs, uint64_t &argSize) = 0;

    /** 子类实现：从 taskArgs 填充 CcuKernelSubmitInfo.cachedArgs（追加 inBuffBaseOff/outBuffBaseOff）。 */
    virtual HcclResult SaveSubmitInfo(const std::vector<uint64_t> &taskArgs,
                                      TemplateResource &templateResource) = 0;

    /**
     * 子类实现：返回 FastLaunch 所需的 taskArgs 字段数（不含末尾 2 个 offset 字段）。
     * 约定 cachedArgs 布局：
     *   [0]           = inputAddr
     *   [1]           = outputAddr
     *   [argSize]     = inBuffBaseOff
     *   [argSize+1]   = outBuffBaseOff
     * SCHE 无 scratchAddr，故末尾仅 2 个 offset 字段。
     */
    virtual uint64_t GetTaskArgSize() const = 0;

    /**
     * 公共工具：构造 CcuKernelInfo 列表并回填到 res.ccuKernelInfos。
     * 流程与 CcuMsBaseTemplate::BuildKernelInfo 一致，区别：
     *   - 使用 CcuKernelArgBaseSche（含 ifHandleSelfRank 字段）；
     *   - slaveThreadNum/notifyNumOnMainThread 按 dieNum 填充（2 die 时 slaveThreadNum=1）。
     */
    HcclResult BuildKernelInfo(HcclComm comm, AlgResourceRequest &res, const DieDecision &dieDecision);

    /** 调用 dieStrategy_->Decide，根据 channelDescs 与 sliceSize 决策 die 划分。 */
    HcclResult DecideDie(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs,
                         u64 sliceSize, u32 dataTypeSize, DieDecision &decision);

    std::unique_ptr<IDieStrategy> dieStrategy_;

    OpParam opParam_{};
    BuffInfo buffInfo_{};
    u32 mySubCommRank_{INVALID_VALUE_RANKID};
    u32 templateRankSize_{0};
    HcclDataType dataType_{HcclDataType::HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceOp_{HcclReduceOp::HCCL_REDUCE_RESERVED};
    std::vector<std::vector<u32>> subCommRanks_;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_CCU_SCHE_BASE_TEMPLATE_H
