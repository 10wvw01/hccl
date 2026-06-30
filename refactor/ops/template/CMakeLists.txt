# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/template 主编译入口。
# 这里放真正要进入 hccl target 的源码；临时验证、oracle 对比和单文件 check 放在 debug/CMakeLists.txt。
target_sources(hccl PRIVATE
    primitives/mesh_allgather_planner.cc
    primitives/mesh_primitives.cc
    primitives/nhr_primitives.cc

    engines/aicpu/launcher/load_kernel.cc
    engines/aicpu/launcher/aicpu_launcher.cc
    engines/aicpu/launcher/kernel_launch.cc

    # engines/aicpu/op_alg/allgather_mesh.cc
    # engines/aicpu/op_alg/allgather_nhr.cc

    # engines/aiv/launcher/load_kernel.cc
    # engines/aiv/launcher/kernel_launch.cc
)

# template 层 debug/check 总开关。默认关闭。
# 具体编译哪个单文件验证目标，由 debug/CMakeLists.txt 中的 REFACTOR_TEMPLATE_DEBUG_* 控制。
option(REFACTOR_TEMPLATE_DEBUG "Enable refactor/ops/template debug/check targets" OFF)
option(REFACTOR_TEMPLATE_DEBUG_MESH_ALLGATHER_PLAN "Build mesh allgather plan check target" OFF)

if(REFACTOR_TEMPLATE_DEBUG_MESH_ALLGATHER_PLAN)
    set(REFACTOR_TEMPLATE_DEBUG ON CACHE BOOL "Enable refactor/ops/template debug/check targets" FORCE)
endif()

if(REFACTOR_DEBUG AND REFACTOR_TEMPLATE_DEBUG AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/debug/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/debug
                     ${CMAKE_CURRENT_BINARY_DIR}/debug)
endif()
