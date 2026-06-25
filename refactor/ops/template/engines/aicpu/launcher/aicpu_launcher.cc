/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_launcher.h"
#include "load_kernel.h"
#include "hccl_log.h"

namespace ops_hccl {

/**
 * 创建 AICPU 运行时资源。
 * 工作流程：
 *   1. 遍历 res 列表中的每一项资源请求 AlgResourceRequest；
 *   2. 为每个层级创建 channel：
 *      - 遍历 resourceRequest.channels 中的 HcclChannelDesc；
 *      - 调用 HcommChannelCreate 创建实际通信通道并回填句柄；
 *   3. 根据 slaveThreadNum 创建从线程：
 *      - 调用 HcommThreadCreate 创建主线程与从线程；
 *   4. 根据 notifyNumPerThread 与 notifyNumOnMainThread 创建 notify：
 *      - 调用 HcommNotifyCreate 创建同步通知资源；
 *   5. 将创建的资源句柄回填到 res 中供 LaunchKernel 使用。
 * 错误处理：
 *   - channel 创建失败：记录错误日志并返回 HCCL_E_INTERNAL；
 *   - thread 创建失败：记录错误日志并返回 HCCL_E_INTERNAL；
 *   - notify 创建失败：记录错误日志并返回 HCCL_E_INTERNAL。
 * 不同引擎资源创建差异说明：
 *   - AICPU: 创建 channel、notify、thread
 *   - AIV:   创建 channel、共享内存（symmetric memory）
 *   - CCU:   创建 cclMem、notify、thread、channel
 */
HcclResult AiCpuLauncher::CreateRes(std::vector<AlgResourceRequest> &res)
{

}

/**
 * 下发 AICPU kernel 到设备侧执行。
 * 工作流程：
 *   1. 加载 AICPU kernel 二进制文件（LoadAICPUKernel），确保 kernel 已加载到设备；
 *   2. 调用 HcclLaunchAicpuKernel 完成以下子步骤：
 *      a. 获取通信域句柄（HcommAcquireComm）；
 *      b. 反序列化资源上下文（AlgResourceCtxSerializable），还原 channel/notify/thread；
 *      c. 根据 opType 还原变长数据（如 AllGatherV 的 counts/displs）；
 *      d. 设置 batch mode，注册 DFX op 信息与 profiling；
 *      e. 主 thread 等待 Host stream 的 notify 通知；
 *      f. 根据 opType 与 algName 获取 executor，调用 executor.Orchestrate 驱动算法编排；
 *      g. 上报 profiling，通知 Host stream 完成，结束 batch mode。
 *   3. 释放通信域句柄（HcommReleaseComm）。
 * 错误处理：
 *   - param 为空返回错误；
 *   - 通信域获取失败、资源反序列化失败、executor 获取失败、Orchestrate 失败均记录日志并返回错误码。
 */
HcclResult AiCpuLauncher::LaunchKernel(const OpParam &param, BaseExecutor &executor)
{
    HCCL_INFO("[AiCpuLauncher][LaunchKernel] start, commName[%s], tag[%s], algTag[%s]",
              param.commName, param.tag, param.algTag);

    // 步骤1：加载 AICPU kernel 二进制
    CHK_RET(LoadAICPUKernel());

    // 步骤2：通过 HcclLaunchAicpuKernel 入口完成环境准备、算法编排与 profiling 上报
    // 该入口内部完成：通信域获取、资源反序列化、变长数据还原、batch mode 设置、
    //                DFX 注册、主 thread notify 等待、executor.Orchestrate 调用、
    //                profiling 上报、notify 通知 Host stream、batch mode 结束
    CHK_RET(HcclLaunchAicpuKernel(&param, executor));

    HCCL_INFO("[AiCpuLauncher][LaunchKernel] end, tag[%s], algTag[%s], commName[%s]",
              param.tag, param.algTag, param.commName);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
