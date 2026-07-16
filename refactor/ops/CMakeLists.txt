# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops 模块分发层。
# 模块默认都关闭；需要把哪个重构模块接入主构建，就显式打开对应 REFACTOR_<MODULE> 开关。
# REFACTOR 总开关：ON 时所有子模块全部打开，后续新增模块只需加一行
option(REFACTOR_API       "Build refactor/ops/api"       ${REFACTOR})
option(REFACTOR_EXECUTOR  "Build refactor/ops/executor"  ${REFACTOR})
option(REFACTOR_OP_COMMON "Build refactor/ops/op_common" ${REFACTOR})
option(REFACTOR_SELECTOR  "Build refactor/ops/selector"  ${REFACTOR})
option(REFACTOR_TEMPLATE  "Build refactor/ops/template"  ${REFACTOR})
option(REFACTOR_UTILS     "Build refactor/ops/utils"     ${REFACTOR})
option(REFACTOR_ALGORITHM "Build refactor/ops/algorithm" ${REFACTOR})
option(REFACTOR_ENGINE   "Build refactor/ops/engine"    ${REFACTOR})

# c_adaptor/common 是主工程 common 代码的早期接入点。
# 当主工程已经创建 hccl_compat 时，这里不能重复 add_subdirectory，否则会重复定义 target。
if(NOT TARGET hccl_compat AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/c_adaptor/common/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/c_adaptor/common
                     ${CMAKE_CURRENT_BINARY_DIR}/c_adaptor/common)
endif()

if(REFACTOR_API AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/api/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/api
                     ${CMAKE_CURRENT_BINARY_DIR}/api)
endif()

if(REFACTOR_EXECUTOR AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/executor/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/executor
                     ${CMAKE_CURRENT_BINARY_DIR}/executor)
endif()

if(REFACTOR_OP_COMMON AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/op_common/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/op_common
                     ${CMAKE_CURRENT_BINARY_DIR}/op_common)
endif()

if(REFACTOR_SELECTOR AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/selector/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/selector
                     ${CMAKE_CURRENT_BINARY_DIR}/selector)
endif()

if(REFACTOR_TEMPLATE AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/template/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/template
                     ${CMAKE_CURRENT_BINARY_DIR}/template)
endif()

if(REFACTOR_UTILS AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/utils/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/utils
                     ${CMAKE_CURRENT_BINARY_DIR}/utils)
endif()

if(REFACTOR_ALGORITHM AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/algorithm/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/algorithm
                     ${CMAKE_CURRENT_BINARY_DIR}/algorithm)
endif()

if(REFACTOR_ENGINE AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/engine/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/engine
                     ${CMAKE_CURRENT_BINARY_DIR}/engine)
endif()
