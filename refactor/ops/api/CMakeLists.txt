# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/api/CMakeLists.txt
# 算子 C API 入口（AllGather 等）
# 由 refactor/CMakeLists.txt 通过 add_subdirectory(ops/api) 拉进来
#
# 当前状态：
#   ✅ all_gather_op.cc  - 完整真实代码（238 行）

target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${CMAKE_CURRENT_LIST_DIR}/../op_common
    ${CMAKE_CURRENT_LIST_DIR}/../op_common/inc    
    ${CMAKE_CURRENT_LIST_DIR}/../executor
)

target_sources(hccl PRIVATE
    all_gather_op.cc   # TODO: HcclAlgorithm/SingleRankProc/HcclExecOp 接口对齐
)