/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ALGORITHM_REDUCE_SCATTER_TEMPLATE_DESC_H
#define HCCL_ALGORITHM_REDUCE_SCATTER_TEMPLATE_DESC_H

#include "hccl_algorithm.h"

namespace ops_hccl {

/**
 * AICPU ReduceScatter 算法中 TemplateDesc 的去重枚举。
 * gen_algorithm.cc（重构后为 algorithm_reduce_scatter_aicpu.cc）中所有 TemplateDesc 构造
 * 均可归并为以下 2 种组合（hcclCmdType 恒为 HCCL_CMD_REDUCE_SCATTER，shotMode 恒为 ONE_SHOT）：
 *   - NHR + SINGLE_JETTY
 *   - FULLMESH + SINGLE_JETTY
 * 枚举顺序与 g_reduceScatterTemplateDescMap 数组定义顺序严格一致。
 */
enum class HcclReduceScatterTemplateDescType {
    REDUCESCATTER_TEMPLATE_NHR_SINGLE_JETTY,       // NHR, ONE_SHOT, SINGLE_JETTY
    REDUCESCATTER_TEMPLATE_FULLMESH_SINGLE_JETTY,   // FULLMESH, ONE_SHOT, SINGLE_JETTY
    REDUCESCATTER_TEMPLATE_DESC_COUNT,              // 数组下标上限
};

/**
 * 全局 ReduceScatter TemplateDesc 表（定义在 reduce_scatter_template_desc.cc）。
 * 以 HcclReduceScatterTemplateDescType 枚举值为数组下标访问。
 * AlgoExecDesc 构造时通过枚举访问本表获取已构造好的 TemplateDesc，避免重复构造。
 */
extern const TemplateDesc g_reduceScatterTemplateDescMap[static_cast<size_t>(
    HcclReduceScatterTemplateDescType::REDUCESCATTER_TEMPLATE_DESC_COUNT)];

} // namespace ops_hccl

#endif // HCCL_ALGORITHM_REDUCE_SCATTER_TEMPLATE_DESC_H
