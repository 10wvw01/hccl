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
# 由 refactor/ops/CMakeLists.txt 通过 add_subdirectory(ops/selector) 拉进来
#
# 当前状态：
#   ✅ auto_selector_base.cc       - 选择器基类
#   ✅ selector_registry.cc        - 选择器注册表
#   ✅ all_gather_auto_selector.cc - AllGather 选择器
#   ✅ execute_selector.cc         - 执行入口 Selector()

target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../op_common
)

target_sources(hccl PRIVATE
    auto_selector_base.cc
    selector_registry.cc
    all_gather_auto_selector.cc
    execute_selector.cc
)