/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_template_desc.h"

namespace ops_hccl {

/**
 * 全局 Scatter TemplateDesc 表。
 * 将 gen_algorithm.cc（重构后为 algorithm_scatter_aicpu.cc）中重复构造的 TemplateDesc 去重，
 * 统一在此处构造，通过 HcclScatterTemplateDescType 枚举下标访问。
 * 数组定义顺序必须与 HcclScatterTemplateDescType 枚举顺序保持一致。
 *
 * 去重依据：所有 TemplateDesc 的 hcclCmdType 恒为 HCCL_CMD_SCATTER，
 * shotMode 恒为 ONE_SHOT，仅 algType 与 jettyMode 存在差异，共 2 种组合：
 *   - NHR + SINGLE_JETTY        （InsScatterNHR / Parallel 子项 NHR 复用）
 *   - FULLMESH + SINGLE_JETTY   （InsScatterMesh1D / Parallel 子项 Mesh1D 复用）
 * Parallel 复合算法不引入新的 TemplateDesc 组合，通过上述两项拼装得到：
 *   - Parallel（Mesh1D+NHR）: 子树 [FULLMESH→INTRA, NHR→INTER]，外层 PARALLEL/SEQUENCE。
 */
const TemplateDesc
    g_scatterTemplateDescMap[static_cast<size_t>(HcclScatterTemplateDescType::SCATTER_TEMPLATE_DESC_COUNT)]
    = {
        // SCATTER_TEMPLATE_NHR_SINGLE_JETTY
        TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER, HcclAlgoType::HCCL_ALGO_TYPE_NHR,
            HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY},
        // SCATTER_TEMPLATE_FULLMESH_SINGLE_JETTY
        TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
            HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY},
};

} // namespace ops_hccl
