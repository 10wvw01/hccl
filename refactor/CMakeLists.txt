# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/CMakeLists.txt
# 这是 refactor 路径的"第 3 层编排器"。
#
# 调用关系：
#   顶层 CMakeLists.txt                  → 定义 REFACTOR_OPS 开关
#   src/CMakeLists.txt                   → include(hccl.cmake)（创建 hccl target）
#                                          include(refactor/CMakeLists.txt) ← 本文件
#   refactor/CMakeLists.txt（本文件）    → 按各模块 option 决定是否进入子目录
#   refactor/ops/<模块>/CMakeLists.txt   → target_sources(hccl PRIVATE ...)
#
# 注意：本文件只在 src/CMakeLists.txt 已经 include(hccl.cmake) 之后才会被 include，
# 所以下面的 add_subdirectory() 进入的子目录里，可以用 target_sources(hccl ...)。

# 各模块独立开关：默认 OFF，由对应负责人写完后自行打开
option(REFACTOR_API       "Build refactor/ops/api (AllGather 入口)"     OFF)
option(REFACTOR_EXECUTOR  "Build refactor/ops/executor (Base/Sole/Parallel)" OFF)
option(REFACTOR_OP_COMMON "Build refactor/ops/op_common (HcclAlgorithm)"  OFF)
option(REFACTOR_SELECTOR  "Build refactor/ops/selector (EngineSelector)"  OFF)
option(REFACTOR_TEMPLATE  "Build refactor/ops/template (primitives + engines)" OFF)
option(REFACTOR_UTILS     "Build refactor/ops/utils (PreCopy/PostCopy)"  OFF)

# c_adaptor/common 是"直接拷贝"的稳定代码，不需要 option，默认编入
add_subdirectory(ops/c_adaptor/common)

# 各模块：仅当 option=ON 且对应 CMakeLists.txt 存在时才进入
# EXISTS 检查兼容 main 分支（没有 refactor/ 目录时不会报错）
if(REFACTOR_API AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ops/api/CMakeLists.txt)
    add_subdirectory(ops/api)
endif()

if(REFACTOR_EXECUTOR AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ops/executor/CMakeLists.txt)
    add_subdirectory(ops/executor)
endif()

if(REFACTOR_OP_COMMON AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/CMakeLists.txt)
    add_subdirectory(ops/op_common)
endif()

if(REFACTOR_SELECTOR AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ops/selector/CMakeLists.txt)
    add_subdirectory(ops/selector)
endif()

if(REFACTOR_TEMPLATE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ops/template/CMakeLists.txt)
    add_subdirectory(ops/template)
endif()

if(REFACTOR_UTILS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ops/utils/CMakeLists.txt)
    add_subdirectory(ops/utils)
endif()