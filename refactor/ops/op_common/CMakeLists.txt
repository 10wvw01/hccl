# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/op_common/CMakeLists.txt
# 公共算法定义：HcclAlgorithm、HcclExecOp 等
# 由 refactor/CMakeLists.txt 通过 add_subdirectory(ops/op_common) 拉进来
#
# 当前状态（全部注释掉，补完后逐个打开）：
#   ❌ op_common.cc          - 13 行伪代码（缺分号、类型错乱）
#   ❌ hccl_algorithm.cc     - switch 语法非法、无分号
#
# 注：hccl_algorithm.h 是真实的类声明（142 行），但当前会引用 topo_match_base.h，
#     该头文件在 refactor/ 下没有，需要从 src/ops/op_common/topo/ 借（或新建）。

target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../template/engines
    ${CMAKE_CURRENT_LIST_DIR}/../template
    ${CMAKE_CURRENT_LIST_DIR}/../engines
    ${CMAKE_CURRENT_LIST_DIR}/../engines/aicpu
    ${CMAKE_CURRENT_LIST_DIR}/../engines/ccu_ms
    ${CMAKE_CURRENT_LIST_DIR}/../engines/ccu_sche
    ${CMAKE_CURRENT_LIST_DIR}/../engines/aiv
    ${CMAKE_CURRENT_LIST_DIR}/../executor
    ${CMAKE_CURRENT_LIST_DIR}/..
)
add_subdirectory(topo)
target_sources(hccl PRIVATE
    op_common.cc
    hccl_algorithm.cc
)