/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_common.h"

#include "ops_executor.h"
#include "base_launcher.h"
#include "log.h"

namespace ops_hccl {

/**
 * 算子执行入口。
 * 流程：
 *   1. 通过 alg 获取引擎（launcher）与执行器（executor）；
 *   2. executor 计算算法分级信息与资源需求；
 *   3. 引擎按自身方式创建运行时资源；
 *   4. 引擎下发 kernel，内部回调 executor.Orchestrate 完成算法编排。
 */
HcclResult HcclExecOp(HcclComm comm, OpParam &param,
                      std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
                      HcclAlgorithm &alg, const ResPackGraphMode &resPack)
{
    // Todo：Kernel缓存

    // 1. 获取引擎与执行器
    auto engine = alg.GetEngine(comm);
    auto executor = alg.GetExecutor(param);

    // 2. 计算算法分级信息（topoMatch → algHierarchyInfo_）
    CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo.get()));

    // 3. 计算资源需求：executor.CalcRes 递归遍历 algoExecDesc，
    //    汇聚各层级 channel/notify/thread 需求到单个 AlgResourceRequest
    AlgResourceRequest resReq;
    CHK_RET(executor->CalcRes(resReq));

    // 4. 引擎创建运行时资源：
    //    AICPU → channel、notify、thread
    //    AIV   → channel、共享内存
    //    CCU   → cclMem、notify、thread、channel
    CHK_RET(engine->CreateRes(resReq));

    // 5. 下发 kernel：引擎内部回调 executor.Orchestrate 完成算法编排
    CHK_RET(engine->LaunchKernel(param, *executor));

    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
