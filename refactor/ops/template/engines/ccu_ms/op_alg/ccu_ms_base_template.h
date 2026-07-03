/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_MS_BASE_TEMPLATE_H
#define OPS_HCCL_CCU_MS_BASE_TEMPLATE_H

#include <memory>
#include <vector>

#include "base_template.h"
#include "alg_param.h"
#include "template_utils.h"
#include "die_strategy.h"
#include "jetty_strategy.h"
#include "ccu_kernel_arg_base.h"

namespace ops_hccl {

/**
 * CCU_MS (mem2mem) 引擎模式 Host 侧 BaseTemplate。
 *
 * 职责：
 *   - 归一化 CCU_MS 模式的 KernelRun / FastLaunch 主流程；
 *   - 通过 IDieStrategy 处理 1Die/2Die 架构差异（peer 集分片或数据半分）；
 *   - 通过 IMultiJettyStrategy 处理 MultiJetty 特性（per-jetty event bitmask 切分）；
 *   - 子类化点：CalcRes（channel 决策）、GetKernelEntry、BuildKernelArg、BuildTaskArgs、SaveSubmitInfo。
 *
 * 与 CCU_SCHE 差异：
 *   - taskArgs 15+ 字段（含 scratchAddr/token/die0Size/die1Size/repeatStride 等）；
 *   - 数据搬运走 ccu::Write/ccu::Read 直接搬远端，不经 CcuBuffer 中转；
 *   - 支持 MultiJetty 切分；
 *   - LoopGroup 仅用于本地 GroupCopy（loopCount=CCU_M2M_LOCAL_COPY_LOOP_COUNT=16, memSlice=32KB）。
 *
 * 参考实现：
 *   - src/ops/all_reduce/template/ccu/ccu_temp_all_reduce_mesh_1D_mem2mem.cc（单 die）
 *   - src/ops/all_gather/template/ccu/ccu_temp_all_gather_nhr_1D_mem2mem.cc（2 die 数据半分）
 *   - src/ops/all_gather/template/ccu/ccu_temp_all_gather_2dies_mesh_1D.cc（2 die peer 集）
 */
class CcuMsBaseTemplate : public BaseTemplate {
public:
    CcuMsBaseTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc,
                      std::unique_ptr<IDieStrategy> dieStrategy,
                      std::unique_ptr<IMultiJettyStrategy> jettyStrategy);
    ~CcuMsBaseTemplate() override = default;

    /**
     * 资源计算入口。子类实现 channel 决策（CalcChannelRequestXxx）并调用本类 BuildKernelInfo
     * 构造 CcuKernelInfo 列表回填到 res.ccuKernelInfos。
     */
    HcclResult CalcRes(HcclComm comm, AlgResourceRequest &res) override = 0;

    /**
     * CCU_MS kernel 下发主流程：
     *   1. DecideDie + DecideJetty 获取决策；
     *   2. BuildTaskArgs 构造 taskArgs；
     *   3. 多 die 时 PreSyncInterThreads；
     *   4. 遍历 die 调用 HcommCcuKernelLaunch；
     *   5. 多 die 时 PostSyncInterThreads；
     *   6. SaveSubmitInfo 填充 CcuKernelSubmitInfo 供 FastLaunch 复用。
     */
    HcclResult KernelRun(const TemplateDataParams &params, TemplateResource &templateResource,
                         std::vector<u32> &ranksForOutputData) override;

    /**
     * 快速下发：从缓存 cachedArgs 补地址偏移后直接 HcommCcuKernelLaunch。
     */
    HcclResult FastLaunch(const OpParam &param, const TemplateFastLaunchCtx &ctx) override;

protected:
    /** 子类实现：返回 kernel 函数指针与名字，用于 kernelInfo 注册。 */
    virtual HcclResult GetKernelEntry(void *&kernelFunc, const char *&kernelName) = 0;

    /** 子类实现：构造算子专属的 CcuKernelArgBaseMs 派生实例（含 rankSize/rankId/opParam/subCommRanks）。 */
    virtual HcclResult BuildKernelArg(std::shared_ptr<CcuKernelArgBaseMs> &arg) = 0;

    /**
     * 子类实现：构造 taskArgs（CCU_MS 15+ 字段，含 inputAddr/outputAddr/token/scratchAddr/
     * die0Size/die1Size/repeatNum/inputSliceStride/outputSliceStride/inputRepeatStride/
     * outputRepeatStride/isInputOutputEqual/die0LastSize/die1LastSize + 可选 jetty 字段）。
     */
    virtual HcclResult BuildTaskArgs(const TemplateDataParams &params, const DieDecision &dieDecision,
                                     const JettyDecision &jettyDecision, std::vector<uint64_t> &taskArgs,
                                     uint64_t &argSize) = 0;

    /** 子类实现：从 taskArgs 填充 CcuKernelSubmitInfo.cachedArgs（追加 inBuffBaseOff/outBuffBaseOff/hcclBuffBaseOff）。 */
    virtual HcclResult SaveSubmitInfo(const std::vector<uint64_t> &taskArgs,
                                      TemplateResource &templateResource) = 0;

    /**
     * 子类实现：返回 FastLaunch 所需的 taskArgs 字段数（不含末尾 3 个 offset 字段）。
     * 约定 cachedArgs 布局：
     *   [0]           = inputAddr
     *   [1]           = outputAddr
     *   [3]           = scratchAddr（CCU_MS 专属，若不用则置 0）
     *   [argSize]     = inBuffBaseOff
     *   [argSize+1]   = outBuffBaseOff
     *   [argSize+2]   = hcclBuffBaseOff
     */
    virtual uint64_t GetTaskArgSize() const = 0;

    /**
     * 公共工具：构造 CcuKernelInfo 列表并回填到 res.ccuKernelInfos。
     * 子类 CalcRes 完成 channel 决策后调用本方法。
     * 流程：
     *   1. GetKernelEntry 获取 kernel 函数指针与名字；
     *   2. 按 dieDecision.dieNum 遍历，每 die 构造一个 CcuKernelInfo：
     *      a. strcpy_s kernelFuncName；
     *      b. kernelInfo.kernelFunc = reinterpret_cast<void*>(kernelFunc)；
     *      c. BuildKernelArg 构造 arg，setKernelArg；
     *      d. kernelInfo.channels = dieDecision.channelsPerDie[dieIdx]；
     *   3. res.ccuKernelNum.push_back(dieDecision.dieNum)；
     *   4. notifyNumOnMainThread/slaveThreadNum 按 dieNum 填充。
     */
    HcclResult BuildKernelInfo(HcclComm comm, AlgResourceRequest &res, const DieDecision &dieDecision);

    /** 调用 dieStrategy_->Decide，根据 channelDescs 与 sliceSize 决策 die 划分。 */
    HcclResult DecideDie(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs,
                         u64 sliceSize, u32 dataTypeSize, DieDecision &decision);

    /** 调用 jettyStrategy_->Decide，根据 sliceSize 决策 jetty 切分。 */
    HcclResult DecideJetty(u64 sliceSize, u32 dataTypeSize, JettyDecision &decision);

    std::unique_ptr<IDieStrategy> dieStrategy_;
    std::unique_ptr<IMultiJettyStrategy> jettyStrategy_;

    OpParam opParam_{};
    BuffInfo buffInfo_{};
    u32 mySubCommRank_{INVALID_VALUE_RANKID};
    u32 templateRankSize_{0};
    HcclDataType dataType_{HcclDataType::HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceOp_{HcclReduceOp::HCCL_REDUCE_RESERVED};
    std::vector<std::vector<u32>> subCommRanks_;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_CCU_MS_BASE_TEMPLATE_H
