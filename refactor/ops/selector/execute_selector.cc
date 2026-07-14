/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "execute_selector.h"
#include "auto_selector_base.h"
#include "all_gather_auto_selector.h"
#include "selector_registry.h"
#include "op_common.h"
#include "load_kernel.h"

namespace ops_hccl {

ExecuteSelector::ExecuteSelector()
{
    // 静态变量初始化在 .so 中可能被链接器优化掉，此处用懒加载确保 selector 已注册
    static bool initOnce = []() {
        SelectorRegistry::Global()->RegisterByOpType(
            HcclCMDType::HCCL_CMD_ALLGATHER, 18, new AllGatherAutoSelector());
        return true;
    }();
    (void)initOnce;
}

HcclResult ExecuteSelector::Run(OpParam &opParam, TopoInfoWithNetLayerDetails* topoInfo,
                                HcclAlgorithm &alg) const
{
    HCCL_DEBUG("[Algo][Selector] Run.");
    std::map<u32, AutoSelectorBase *> selectors = SelectorRegistry::Global()->GetAllSelectors();

    if (opParam.isMc2) {
        auto iter = selectors.find(18);
        if (iter == selectors.end()) {
            HCCL_ERROR("[Algo][Selector] CCU selector is not registried.");
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        if(iter->second->Select(opParam, topoInfo, alg) == SelectorStatus::MATCH) {
            HCCL_INFO("[Algo][Selector] The ccu selector[priority of %u] is matched.",
                iter->first);
            return HcclResult::HCCL_SUCCESS;
        }
        HCCL_ERROR("[Algo][Selector] CCU selector can not match for optype[%d].", opParam.opType);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    selectors = SelectorRegistry::Global()->GetSelectorsByOpType(opParam.opType);
    HCCL_INFO("[Algo][Selector] The selector nums of optype[%d] is [%zu].", opParam.opType, selectors.size());
    for (auto iter : selectors) {
        HCCL_DEBUG("[Algo][Selector] The selector[priority of %llu] is running.", iter.first);
        if (iter.second->Select(opParam, topoInfo, alg) == SelectorStatus::MATCH) {
            HCCL_INFO("[Algo][Selector] The selector[priority of %llu] is matched.",
                      iter.first);
            return HcclResult::HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[Algo][Selector] No selector is matched.");
    return HcclResult::HCCL_E_NOT_SUPPORT;
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
    // CHK_RET(SetCommEngine(param));
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

} // namespace ops_hccl
