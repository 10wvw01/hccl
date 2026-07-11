/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_MS_KERNEL_LAUNCH_H
#define OPS_HCCL_CCU_MS_KERNEL_LAUNCH_H

#include "alg_param.h"

namespace ops_hccl {

class OpsExecutor;

/**
 * CCU_MS kernel 下发入口。
 *
 * 工作流程（参考 aicpu/kernel_launch.cc:46-177 HcclLaunchAicpuKernel）：
 *   1. HcommAcquireComm 获取通信域句柄；
 *   2. 设置 batch mode，注册 DFX op 信息与 profiling；
 *   3. 主 thread 等待 Host stream 的 notify 通知（HcommThreadNotifyWaitOnThread）；
 *   4. 调用 executor.Orchestrate 驱动算法编排（baseTemplate.KernelRun）；
 *   5. 上报 profiling，通知 Host stream 完成（HcommThreadNotifyRecordOnThread）；
 *   6. 结束 batch mode，释放通信域句柄（HcommReleaseComm）。
 *
 * 输入参数：
 *   - param: 算子参数
 *   - executor: 执行器引用，提供 Orchestrate 接口
 *   - resCtx: 已创建的资源上下文（CreateRes 回填）
 * 返回值：
 *   - HCCL_SUCCESS: kernel 下发并执行成功
 *   - HCCL_E_INTERNAL: 下发或执行失败
 */
HcclResult CcuMsLaunchKernel(const OpParam &param, OpsExecutor &executor,
                             AlgResourceCtxSerializable &resCtx);

}  // namespace ops_hccl

#endif  // OPS_HCCL_CCU_MS_KERNEL_LAUNCH_H
