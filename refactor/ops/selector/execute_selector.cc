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
#include "alg_env_config.h"
#include "dlhcomm_function.h"

namespace ops_hccl {

// 仅在本 .cc 内使用的常量，与 src/ops/op_common/op_common.cc 中保持一致
static constexpr uint32_t opExpansionModeCcuSched = 5;
static constexpr uint32_t opExpansionModeCcuMs = 4;

ExecuteSelector::ExecuteSelector()
{
}

HcclResult ExecuteSelector::Run(OpParam &opParam, TopoInfoWithNetLayerDetails *topoInfo, HcclAlgorithm &alg) const
{
    HCCL_DEBUG("[Algo][Selector] Run.");
    std::map<u32, AutoSelectorBase *> selectors = SelectorRegistry::Global()->GetAllSelectors();

    if (opParam.isMc2) {
        auto iter = selectors.find(18);
        if (iter == selectors.end()) {
            HCCL_ERROR("[Algo][Selector] CCU selector is not registried.");
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        if (iter->second->Select(opParam, topoInfo, alg) == SelectorStatus::MATCH) {
            HCCL_INFO("[Algo][Selector] The ccu selector[priority of %u] is matched.", iter->first);
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
            HCCL_INFO("[Algo][Selector] The selector[priority of %llu] is matched.", iter.first);
            return HcclResult::HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[Algo][Selector] No selector is matched.");
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult HcclGetOpExpansionMode(HcclComm comm, OpParam &param)
{
    HcclOpExpansionMode finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    // 第一步：决定使用哪种模式
    HcclResult ret = DecideHcclOpExpansionMode(comm, finalMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("DecideHcclOpExpansionMode failed, ret: %d", ret);
        return ret;
    }
    param.commOpExpansionMode = finalMode;

    // 第二步：应用选择的模式到param
    ret = ApplyOpExpansionMode(param, finalMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("ApplyOpExpansionMode failed, ret: %d", ret);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult DecideHcclOpExpansionMode(HcclComm comm, HcclOpExpansionMode &finalMode)
{
    HcclOpExpansionMode configOpExpansionMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    bool useConfigOpExpansionMode = false;
    auto &hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
    if (hcommFunction.dlHcclConfigGetInfo) {
        uint32_t infoLen = sizeof(HcclOpExpansionMode);
        CHK_RET(hcommFunction.dlHcclConfigGetInfo(
            comm, HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, infoLen, &configOpExpansionMode));
        finalMode = configOpExpansionMode;
        useConfigOpExpansionMode = true;
    } else {
        HCCL_INFO("[DecideHcclOpExpansionMode] HcclConfigGetInfo is not supported, use environment mode.");
        finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs);
    }

    // A5仅通过HcclConfigGetInfo获取展开模式，其他型号保留环境变量方式
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
#ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950 || !useConfigOpExpansionMode) {
#else
    if (deviceType != DevType::DEV_TYPE_910_95 || !useConfigOpExpansionMode) {
#endif
        if (GetExternalInputHcclAicpuUnfold() == true) {
            finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AI_CPU;
        } else if (GetExternalInputHcclAivOnlyMode() == true) {
            finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY;
        } else if (GetExternalInputHcclAivMode() == true) {
            finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AIV;
        } else if (GetExternalInputHcclCcuMSMode()) {
            finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs);
        } else if (GetExternalInputHcclCcuSchedMode()) {
            finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuSched);
        }
        if (useConfigOpExpansionMode && configOpExpansionMode != finalMode) {
            HCCL_DEBUG("[DecideHcclOpExpansionMode] configOpExpansionMode: %d, environment mode: %d, conflict, use "
                       "environment mode.",
                configOpExpansionMode, finalMode);
        }
    }
    HCCL_INFO("[DecideHcclOpExpansionMode] finalMode: %d.", finalMode);

    return HCCL_SUCCESS;
}

HcclResult ApplyOpExpansionMode(OpParam &param, HcclOpExpansionMode finalMode)
{
    switch (finalMode) {
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AI_CPU:
            param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
            param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
            CHK_RET(LoadAICPUKernel());
            HCCL_DEBUG("[ApplyOpExpansionMode] AICPU mode selected.");
            break;
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AIV:
            param.opExecuteConfig = OpExecuteConfig::AIV;
            param.engine = CommEngine::COMM_ENGINE_AIV;
            CHK_RET(RegisterKernel());
            HCCL_DEBUG("[ApplyOpExpansionMode] AIV mode selected.");
            break;
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY:
            param.opExecuteConfig = OpExecuteConfig::AIV_ONLY;
            param.engine = CommEngine::COMM_ENGINE_AIV;
            CHK_RET(RegisterKernel());
            HCCL_DEBUG("[ApplyOpExpansionMode] AIV_ONLY mode selected.");
            break;
        case static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs):
            param.opExecuteConfig = OpExecuteConfig::CCU_MS;
            param.engine = CommEngine::COMM_ENGINE_CCU;
            HCCL_DEBUG("[ApplyOpExpansionMode] CCU_MS mode selected.");
            break;
        case static_cast<HcclOpExpansionMode>(opExpansionModeCcuSched):
            param.opExecuteConfig = OpExecuteConfig::CCU_SCHED;
            param.engine = CommEngine::COMM_ENGINE_CCU;
            HCCL_DEBUG("[ApplyOpExpansionMode] CCU_SCHED mode selected.");
            break;
        default:
            // 回退到aicpu
            HCCL_WARNING("[ApplyOpExpansionMode] Invalid HcclOpExpansionMode: %d, fallback to AICPU_TS.", finalMode);
            param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
            param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
            CHK_RET(LoadAICPUKernel());
            break;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Selector(
    HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo, HcclAlgorithm &alg)
{
    // 判断通信域状态
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
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY
        && param.engine != CommEngine::COMM_ENGINE_AIV) {
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
