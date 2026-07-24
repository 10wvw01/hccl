/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ALGORITHM_SCATTER_TEMPLATE_DESC_H
#define HCCL_ALGORITHM_SCATTER_TEMPLATE_DESC_H

#include "hccl_algorithm.h"

namespace ops_hccl {

/**
 * AICPU Scatter 算法中 TemplateDesc 的去重枚举。
 * gen_algorithm.cc（重构后为 algorithm_scatter_aicpu.cc）中所有 TemplateDesc 构造
 * 均可归并为以下 2 种组合（hcclCmdType 恒为 HCCL_CMD_SCATTER，shotMode 恒为 ONE_SHOT）：
 *   - NHR + SINGLE_JETTY
 *   - FULLMESH + SINGLE_JETTY
 * Parallel 复合算法不引入新的 TemplateDesc 组合，通过上述两项拼装得到
 * （与 AllGather/ReduceScatter 的并行复用基础 TemplateDesc 的方式一致）。
 * 枚举顺序与 g_scatterTemplateDescMap 数组定义顺序严格一致。
 */
enum class HcclScatterTemplateDescType {
    SCATTER_TEMPLATE_NHR_SINGLE_JETTY,        // NHR, ONE_SHOT, SINGLE_JETTY
    SCATTER_TEMPLATE_FULLMESH_SINGLE_JETTY,   // FULLMESH, ONE_SHOT, SINGLE_JETTY
    SCATTER_TEMPLATE_DESC_COUNT,              // 数组下标上限
};

/**
 * 全局 Scatter TemplateDesc 表（定义在 scatter_template_desc.cc）。
 * 以 HcclScatterTemplateDescType 枚举值为数组下标访问。
 * AlgoExecDesc 构造时通过枚举访问本表获取已构造好的 TemplateDesc，避免重复构造。
 */
extern const TemplateDesc g_scatterTemplateDescMap[static_cast<size_t>(
    HcclScatterTemplateDescType::SCATTER_TEMPLATE_DESC_COUNT)];

} // namespace ops_hccl

#endif // HCCL_ALGORITHM_SCATTER_TEMPLATE_DESC_H
