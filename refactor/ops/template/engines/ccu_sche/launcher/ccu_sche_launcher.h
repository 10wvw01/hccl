/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_SCHE_LAUNCHER_H
#define OPS_HCCL_CCU_SCHE_LAUNCHER_H

#include "base_launcher.h"
#include "alg_param.h"
#include "ccu_sche_kernel_launch.h"

namespace ops_hccl {

/**
 * CCU_SCHE (调度模式 / 非 mem2mem) 引擎 Launcher。
 *
 * 职责：
 *   - CreateRes：创建 CCU_SCHE 运行时资源（cclMem/thread/channel/ccuKernel）；
 *   - LaunchKernel：下发 CCU kernel 并驱动算法编排。
 *
 * 参考实现：
 *   - refactor/ops/template/engines/ccu_ms/launcher/ccu_ms_launcher.{h,cc}（对称架构）
 *   - src/ops/op_common/op_common.cc:1526-1561（HcclAllocAlgResourceCcu 资源申请流程）
 *
 * 与 CCU_MS Launcher 差异：
 *   - engine 类型为 CCU_SCHED（HcclAlgEngineType），底层 CommEngine 仍为 COMM_ENGINE_CCU；
 *   - baseTemplate 不持有 IMultiJettyStrategy（SCHE 模式不支持 MultiJetty）；
 *   - kernel_launch 流程与 CCU_MS 一致（batch mode + Orchestrate + profiling）。
 */
class CcuScheLauncher : public BaseLauncher {
public:
    explicit CcuScheLauncher(HcclComm comm) : comm_(comm) {}
    ~CcuScheLauncher() override = default;

    /**
     * 创建 CCU_SCHE 运行时资源。
     * 流程（迁移自 op_common.cc:1526-1561 HcclAllocAlgResourceCcu）：
     *   1. HcclGetHcclBuffer 获取 CCL buffer → resCtx_.cclMem；
     *   2. 回填 notifyNumOnMainThread/slaveThreadNum/notifyNumPerThread；
     *   3. HcclGetThread 创建主/从线程（占位，待接口扩展）；
     *   4. CcuScheGetChannelForCcu 以 kernel 为粒度申请 channel；
     *   5. CcuScheGetCcuKernel 注册 CCU kernel 句柄。
     */
    HcclResult CreateRes(AlgResourceRequest &res) override;

    /**
     * 下发 CCU_SCHE kernel 到设备侧执行。
     * 流程：调用 CcuScheLaunchKernel 完成 batch mode + Orchestrate + profiling。
     */
    HcclResult LaunchKernel(const OpParam &param, OpsExecutor &executor) override;

private:
    HcclComm comm_;
    AlgResourceCtxSerializable resCtx_;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_CCU_SCHE_LAUNCHER_H
