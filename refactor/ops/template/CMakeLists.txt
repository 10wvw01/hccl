# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/template/CMakeLists.txt
# 通信模板：primitives（Mesh/NHR 算法实现）+ engines（AICPU/AIV launcher + op_alg）
# 由 refactor/CMakeLists.txt 通过 add_subdirectory(ops/template) 拉进来
#
# 当前状态：
#   ✅ primitives/mesh_primitives.cc                  - RunMeshAllGather/ReduceScatter 等（329 行）
#   ✅ primitives/nhr_primitives.cc                   - RunNhrAllGather/ReduceScatter 等（263 行）
#   ✅ engines/aicpu/launcher/load_kernel.cc          - LoadAICPUKernel（57 行）
#   ✅ engines/aicpu/launcher/aicpu_launcher.cc      - LaunchKernel 真实，CreateRes 空
#   ✅ engines/aicpu/launcher/kernel_launch.cc        - HcclLaunchAicpuKernel 真实（400 行）
#   ❌ engines/aicpu/op_alg/allgather_mesh.cc         - 18 行伪代码（错别字 nhr_primitivces）
#   ❌ engines/aicpu/op_alg/allgather_nhr.cc          - 19 行伪代码（类名大小写错乱）
#   ❌ engines/aiv/launcher/load_kernel.cc            - 直接复制 aicpu 的，应重写
#   ❌ engines/aiv/launcher/kernel_launch.cc          - 直接复制 aicpu 的，应重写

target_sources(hccl PRIVATE
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

option(REFACTOR_TEMPLATE_PLAN_DEBUG "Build refactor template planner debug tools" OFF)

if(REFACTOR_TEMPLATE_PLAN_DEBUG)
    add_executable(mesh_allgather_plan_dump
        debug/mesh_allgather_plan_dump.cc
    )

    target_include_directories(mesh_allgather_plan_dump PRIVATE
        ${INCLUDE_LIST}
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives
    )

    target_compile_definitions(mesh_allgather_plan_dump PRIVATE
        HOST_COMPILE
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:OPEN_BUILD_PROJECT>
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )

    target_compile_options(mesh_allgather_plan_dump PRIVATE
        -fno-common
        -fno-strict-aliasing
        -pipe
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
    )

    # 这个工具只验证 BuildMeshAllGatherSlicePlan 的输出，不进入默认构建，也不触发真实 SendRecv。
    target_link_libraries(mesh_allgather_plan_dump PRIVATE hccl)
endif()
