/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_BASE_LAUNCHER_H
#define OPS_HCCL_BASE_LAUNCHER_H

#include <vector>
#include "hccl/base.h"
#include "alg_primitive_types.h"

namespace ops_hccl {

class BaseExecutor;

/**
 * 引擎 Launcher 基类
 * 职责：抽象不同设备引擎（AICPU/AIV/CCU）的 Kernel 下发与资源创建能力。
 * 不同引擎子类实现各自的资源创建方式与 kernel 下发流程：
 *   - AICPU: 创建 channel、notify、thread，下发 AICPU kernel
 *   - AIV:   创建 channel、共享内存，下发 AIV kernel
 *   - CCU:   创建 cclMem、notify、thread、channel，下发 CCU kernel
 */
class BaseLauncher {
public:
    BaseLauncher() = default;
    virtual ~BaseLauncher() = default;

    /**
     * 创建引擎所需的运行时资源。
     * 工作流程：
     *   1. 遍历 res 列表中的每一项资源请求；
     *   2. 根据引擎类型创建对应的资源（channel/notify/thread/cclMem 等）；
     *   3. 将创建的资源句柄回填到 res 中供后续 LaunchKernel 使用。
     * 输入参数：
     *   - res: 资源请求列表，由 executor.CalcRes 生成，包含每层级的资源需求
     * 返回值：
     *   - HCCL_SUCCESS: 资源创建成功
     *   - HCCL_E_INTERNAL: 资源创建失败
     */
    virtual HcclResult CreateRes(std::vector<AlgResourceRequest> &res) = 0;

    /**
     * 下发 kernel 到设备侧执行。
     * 工作流程：
     *   1. 准备执行环境（加载 kernel 二进制、初始化 thread）；
     *   2. 调用 executor.Orchestrate 驱动算法模板编排；
     *   3. 等待设备侧执行完成并上报 profiling。
     * 输入参数：
     *   - param: 算子参数，包含 commName、tag、opType、数据描述等
     *   - executor: 执行器引用，提供 Orchestrate 接口
     * 返回值：
     *   - HCCL_SUCCESS: kernel 下发并执行成功
     *   - HCCL_E_INTERNAL: 下发或执行失败
     */
    virtual HcclResult LaunchKernel(const OpParam &param, BaseExecutor &executor) = 0;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_BASE_LAUNCHER_H
