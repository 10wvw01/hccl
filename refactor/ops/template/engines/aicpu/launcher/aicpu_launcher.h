/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_AICPU_LAUNCHER_H
#define OPS_HCCL_AICPU_LAUNCHER_H

#include "base_launcher.h"
#include "kernel_launch.h"

namespace ops_hccl {

/**
 * AICPU 引擎 Launcher
 * 职责：实现 AICPU 引擎的资源创建与 kernel 下发能力。
 * 资源创建：通过 AicpuResourceCreator 创建 channel、notify、thread；
 * Kernel 下发：通过 HcclLaunchAicpuKernel 入口完成环境准备、算法编排与 profiling 上报。
 */
class AiCpuLauncher : public BaseLauncher {
public:
    AiCpuLauncher() = default;
    ~AiCpuLauncher() override = default;

    /**
     * 创建 AICPU 引擎所需的运行时资源。
     * 工作流程：
     *   1. 遍历资源请求列表；
     *   2. 为每个层级创建 channel（通信通道）、notify（同步通知）、thread（执行线程）；
     *   3. 将创建的资源句柄回填到 res 中。
     * 输入参数：
     *   - res: 资源请求列表，每项包含 slaveThreadNum、notifyNumPerThread、channels 等
     * 返回值：
     *   - HCCL_SUCCESS: 资源创建成功
     *   - HCCL_E_INTERNAL: 资源创建失败（如 channel 创建失败）
     */
    HcclResult CreateRes(std::vector<AlgResourceRequest> &res) override;

    /**
     * 下发 AICPU kernel 到设备侧执行。
     * 工作流程：
     *   1. 加载 AICPU kernel 二进制（LoadAICPUKernel）；
     *   2. 准备执行环境：获取主 thread、设置 batch mode、注册 DFX 信息；
     *   3. 主 thread 等待 Host stream 的 notify 通知；
     *   4. 根据 opType 获取对应的 executor，调用 executor.Orchestrate 驱动算法编排；
     *   5. 上报 profiling 信息并通知 Host stream 完成。
     * 输入参数：
     *   - param: 算子参数，包含 commName、tag、opType、数据描述、resCtx 等
     *   - executor: 执行器引用，提供 Orchestrate 接口驱动模板编排
     * 返回值：
     *   - HCCL_SUCCESS: kernel 下发并执行成功
     *   - HCCL_E_PARA: 参数非法（如 param 为空）
     *   - HCCL_E_INTERNAL: 下发或执行失败
     */
    HcclResult LaunchKernel(const OpParam &param, BaseExecutor &executor) override;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_AICPU_LAUNCHER_H
