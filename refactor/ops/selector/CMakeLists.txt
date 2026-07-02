# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/selector/CMakeLists.txt
# 算法选择器：按引擎（AICPU/AIV/CCU/HOSTCPU）选 + 按算子选
# 由 refactor/CMakeLists.txt 通过 add_subdirectory(ops/selector) 拉进来
#
# 当前状态：
#   ✅ engine_selector.cc      - 较完整（228+123 行），ConditionMask 引擎选择逻辑真实
#   ⚠️ base_selector.cc       - Selector() 缺返回值
#   ⚠️ all_gather_selector.cc  - 多次直接 return SelectorStatus::MATCH，缺条件分支
#   ❌ alg_selector.h          - 只有 31 行声明，无 .cc
#
# 当前只开 engine_selector.cc，其他等补完。

target_sources(hccl PRIVATE
    engine_selector.cc
    # base_selector.cc
    # all_gather_selector.cc
)