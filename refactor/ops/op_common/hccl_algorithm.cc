/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_algorithm.h"

#include "log.h"
#include "base_launcher.h"
#include "engines/aicpu/launcher/aicpu_launcher.h"
#include "engines/ccu_ms/launcher/ccu_ms_launcher.h"
#include "engines/ccu_sche/launcher/ccu_sche_launcher.h"
#include "executor/ops_executor.h"

namespace ops_hccl {

/**
 * 根据算法的引擎类型（engineType）构造对应的 Launcher。
 *   - AICPU: 返回 AiCpuLauncher（传入 comm 供 CreateRes 使用）
 *   - CCU_MS: 返回 CcuMsLauncher（传入 comm 供 CreateRes 使用）
 *   - CCU_SCHED: 返回 CcuScheLauncher（传入 comm 供 CreateRes 使用）
 */
std::unique_ptr<BaseLauncher> HcclAlgorithm::GetEngine(HcclComm comm)
{
    switch (engineType) {
        case HcclAlgEngineType::AICPU:
            return std::make_unique<AiCpuLauncher>(comm);
        case HcclAlgEngineType::CCU_MS:
            return std::make_unique<CcuMsLauncher>(comm);
        case HcclAlgEngineType::CCU_SCHED:
            return std::make_unique<CcuScheLauncher>(comm);
        default:
            HCCL_ERROR("[HcclAlgorithm][GetEngine] invalid engineType[%d]", static_cast<int>(engineType));
            return nullptr;
    }
}

/**
 * 构造统一Executor。
 * 输入参数：
 *   - param: 算子参数，传递给执行器构造函数用于初始化 rank/data 等信息
 */
std::unique_ptr<OpsExecutor> HcclAlgorithm::GetExecutor(OpParam &param)
{
    return std::make_unique<OpsExecutor>(*this, param);
}

/**
 * 打印算法基本信息，用于调试。
 */
void HcclAlgorithm::Dump()
{
    HCCL_INFO("[HcclAlgorithm][Dump] engineType[%d], hcclCmdType[%d], execPolicy[%d], childrenNum[%zu]",
              static_cast<int>(engineType), static_cast<int>(hcclCmdType),
              static_cast<int>(algoExecDesc.execPolicy), algoExecDesc.children.size());
}

/**
 * 序列化算法描述，用于多机间算法选择一致性校验。
 * Todo: 待实现序列化协议。
 */
void HcclAlgorithm::Serialize()
{
    // Todo: 序列化 engineType/hcclCmdType/algoExecDesc 供多机一致性校验
}

} // namespace ops_hccl
