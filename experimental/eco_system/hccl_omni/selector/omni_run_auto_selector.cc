/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "omni_run_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {

bool OmniRunAutoSelector::IsOmniRunOp(const OpParam &opParam) const
{
    // Tag is "OMNIRUN_<commName>" set by HcclOmniRun
    return strstr(opParam.tag, "OMNIRUN_") != nullptr;
}

SelectorStatus OmniRunAutoSelector::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                    const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    if (!IsOmniRunOp(opParam)) {
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[OmniRunAutoSelector][%s] OMNIRUN tag detected, select OmniRunCcu for ccu_schedule", __func__);
    selectAlgName = "OmniRunCcu";
    return SelectorStatus::MATCH;
}

SelectorStatus OmniRunAutoSelector::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                    const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    if (!IsOmniRunOp(opParam)) {
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[OmniRunAutoSelector][%s] OMNIRUN tag detected, select OmniRunCcu for ccu_ms", __func__);
    selectAlgName = "OmniRunCcu";
    return SelectorStatus::MATCH;
}

SelectorStatus OmniRunAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                      const OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    if (!IsOmniRunOp(opParam)) {
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[OmniRunAutoSelector][%s] OMNIRUN tag detected, select OmniRunAicpu for aicpu", __func__);
    selectAlgName = "OmniRunAicpu";
    return SelectorStatus::MATCH;
}

// Priority 17 < 18 (alltoallv_auto_selector), so this selector is tried first.
// Only matches when tag starts with "OMNIRUN_", otherwise falls through to the default selector.
REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLTOALLV, 17, OmniRunAutoSelector);

} // namespace ops_hccl
