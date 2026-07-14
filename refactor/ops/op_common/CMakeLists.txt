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

target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../template
    ${CMAKE_CURRENT_LIST_DIR}/../engine
    ${CMAKE_CURRENT_LIST_DIR}/../engine/aicpu
    ${CMAKE_CURRENT_LIST_DIR}/../executor
    ${CMAKE_CURRENT_LIST_DIR}/..
)
add_subdirectory(topo)
target_sources(hccl PRIVATE
    op_common.cc
    hccl_algorithm.cc
    exec_timeout_manager.cc
)