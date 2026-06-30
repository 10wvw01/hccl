# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# template/debug 只注册显式打开的单文件验证目标。
# 每个目标自己声明依赖的最小源码集合，避免为了验证一个 planner 把整个 refactor/template 编进来。
option(REFACTOR_TEMPLATE_DEBUG_MESH_ALLGATHER_PLAN "Build mesh allgather plan check target" OFF)

if(REFACTOR_TEMPLATE_DEBUG_MESH_ALLGATHER_PLAN)
    add_executable(mesh_allgather_plan_check
        mesh_allgather_plan_check.cc
        ../primitives/mesh_allgather_planner.cc
        mesh_allgather_plan_debug_template_utils.cc
        mesh_allgather_plan_debug_log_stubs.cc
    )

    target_include_directories(mesh_allgather_plan_check PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/stubs
        ${CMAKE_CURRENT_LIST_DIR}/../../op_common
        ${INCLUDE_LIST}
        ${CMAKE_CURRENT_LIST_DIR}/../primitives
    )

    target_compile_definitions(mesh_allgather_plan_check PRIVATE
        HOST_COMPILE
        CANN_VERSION_NUM=${HCCL_CANN_VERSION_NUM}
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:OPEN_BUILD_PROJECT>
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )

    target_compile_options(mesh_allgather_plan_check PRIVATE
        -fno-common
        -fno-strict-aliasing
        -pipe
        -ffunction-sections
        -fdata-sections
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
    )

    target_link_options(mesh_allgather_plan_check PRIVATE
        -Wl,--gc-sections
    )

    # 旧目标名只作为兼容入口保留，后续脚本可以逐步切到 mesh_allgather_plan_check。
    add_custom_target(mesh_allgather_plan_dump DEPENDS mesh_allgather_plan_check)
endif()
