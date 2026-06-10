# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_subdirectory(executor)
add_subdirectory(selector)
add_subdirectory(template)
add_subdirectory(op_graph)

set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_op.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)
# 注: AICPU 注册层代码生成(HCCL_AICPU_CODEGEN)的 CMake 接入放在 src/CMakeLists.txt,
# 与 hccl 目标同目录, 以满足 generated 源文件 GENERATED 属性的作用域要求(CMake<3.20)。
# spec 输入: ${CMAKE_CURRENT_SOURCE_DIR}/aicpu.spec.yaml