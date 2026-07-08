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

HcclResult Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    HcclAlgorithm &alg)
{
    //判断通信域状态
    HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;
    if (HcommIsSupportHcclCommGetStatus()) {
        CHK_RET(HcclCommGetStatus(param.commName, &commStatus));
        if (commStatus != HCCL_COMM_STATUS_READY) {
            HCCL_ERROR("commStatus is not ready!, commStatus = %d", static_cast<int>(commStatus));
            return HCCL_E_SUSPENDING;
        }
    }
    HCCL_INFO("Start to execute Selector.");
    param.hcclComm = comm;
    // 获取基础拓扑
    CHK_RET(HcclCalcTopoInfo(comm, param, topoInfo));

    // 检查非对称拓扑支持情况，非对称场景仅 AllGather/AllReduce/ReduceScatter 可用
    CHK_RET(CheckAsymmetricTopoSupport(param.opType, topoInfo.get()));

    // 算法选择，选择完后顺便param.algTag设置了，资源的保存是以算子+算法为单位
    std::shared_ptr<ExecuteSelector> collAlgSelector = std::make_shared<ExecuteSelector>(ExecuteSelector());
    // ExecuteSelector::Run 内部调用 AutoSelectorBase::Select 输出 HcclAlgorithm
    CHK_RET(collAlgSelector->Run(param, topoInfo.get(), alg));
    CHK_RET(SetCommEngine(param));
    // AIV_ONLY 模式下禁止回退到非 AIV 引擎，未选中 AIV 时直接返回不支持。
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY && param.engine != CommEngine::COMM_ENGINE_AIV) {
        HCCL_ERROR("[HcclExecOp] opType[%d] currently do not select aiv mode, aiv only not support.",
            static_cast<int>(param.opType));
        return HCCL_E_NOT_SUPPORT;
    }
    // 如果一开始读取到的Engine不是aicpu，经过算法选择后回退到aipcu，则需要重新LoadAICPUKernel
    if ((param.engine == CommEngine::COMM_ENGINE_AICPU_TS) || (param.engine == CommEngine::COMM_ENGINE_CPU)) {
        HCCL_DEBUG("[Selector] is aicpu mode");
        CHK_RET(LoadAICPUKernel()); // 该函数内部有防止重复加载的逻辑
    }
    // 如果一开始读取到的Engine不是aiv，经过算法选择后回退到aiv，则需要重新RegisterKernel
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        HCCL_DEBUG("[Selector] is aiv mode");
        CHK_RET(RegisterKernel()); // 该函数内部有防止重复加载的逻辑
    }
    // SetOpParamAlgTag 依赖 algName 字符串，待后续 algName 字符串来源明确后补充
    // CHK_RET(SetOpParamAlgTag(param, algName));
    // 设定执行超时时间
    CHK_RET(SetExecTimeout(param));
    // 获取多维度切分比例
    CHK_RET(SetMultipleDimensionSplitRatio(param));
    HCCL_INFO("Success to execute Selector.");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
