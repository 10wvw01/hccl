/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_MS_LAUNCHER_H
#define OPS_HCCL_CCU_MS_LAUNCHER_H

#include "base_launcher.h"
#include "alg_param.h"
#include "ccu_ms_kernel_launch.h"

namespace ops_hccl {

/**
 * CCU_MS (mem2mem) 引擎 Launcher。
 *
 * 职责：
 *   - CreateRes：创建 CCU_MS 运行时资源（cclMem/thread/channel/ccuKernel）；
 *   - LaunchKernel：下发 CCU kernel 并驱动算法编排。
 *
 * 参考实现：
 *   - refactor/ops/template/engines/aicpu/launcher/aicpu_launcher.{h,cc}（Launcher 架构）
 *   - src/ops/op_common/op_common.cc:1526-1561（HcclAllocAlgResourceCcu 资源申请流程）
 *
 * 与 AICPU Launcher 差异：
 *   - 资源申请增加 CcuMsGetChannelForCcu + CcuMsGetCcuKernel（CCU 专属 channel/kernel 注册）；
 *   - LaunchKernel 调用 CcuMsLaunchKernel（batch mode + Orchestrate + profiling）。
 */
class CcuMsLauncher : public BaseLauncher {
public:
    explicit CcuMsLauncher(HcclComm comm) : comm_(comm) {}
    ~CcuMsLauncher() override = default;

    /**
     * 创建 CCU_MS 运行时资源。
     * 流程（迁移自 op_common.cc:1526-1561 HcclAllocAlgResourceCcu）：
     *   1. HcclGetHcclBuffer 获取 CCL buffer → resCtx_.cclMem；
     *   2. 回填 notifyNumOnMainThread/slaveThreadNum/notifyNumPerThread；
     *   3. HcclGetThread 创建主/从线程；
     *   4. CcuMsGetChannelForCcu 以 kernel 为粒度申请 channel；
     *   5. CcuMsGetCcuKernel 注册 CCU kernel 句柄。
     */
    HcclResult CreateRes(AlgResourceRequest &res) override;

    /**
     * 下发 CCU_MS kernel 到设备侧执行。
     * 流程（参考 aicpu_launcher.cc:87-102 + kernel_launch.cc:46-177）：
     *   1. 调用 CcuMsLaunchKernel 完成 batch mode + Orchestrate + profiling。
     */
    HcclResult LaunchKernel(const OpParam &param, OpsExecutor &executor) override;

    // BaseLauncher 的纯虚 Send()：refactor 早期 stub，尚未实现具体传输逻辑。
    HcclResult Send(const TransferContext &ctx) override { (void)ctx; return HCCL_SUCCESS; }

private:
    HcclComm comm_;
    AlgResourceCtxSerializable resCtx_;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_CCU_MS_LAUNCHER_H
