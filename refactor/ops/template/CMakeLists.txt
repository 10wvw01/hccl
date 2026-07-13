# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Refactor template sources that should enter the hccl target.
# 注：refactor/ops 下的各 CMakeLists.txt 通过共享 hccl target 累加 include 路径与源文件。
# 这里补充 op_common.cc / hccl_algorithm.cc 编译所需的所有路径，
# 因为它们依赖 engines/base_launcher.h 及 engines/aicpu|ccu_ms|ccu_sche 子模块。
target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/op_alg
    ${CMAKE_CURRENT_LIST_DIR}/op_alg/aicpu
    ${CMAKE_CURRENT_LIST_DIR}/op_alg/ccu_common
    ${CMAKE_CURRENT_LIST_DIR}/op_alg/ccu_common/strategies
    ${CMAKE_CURRENT_LIST_DIR}/op_alg/ccu_ms
    ${CMAKE_CURRENT_LIST_DIR}/op_alg/ccu_sche
    ${CMAKE_CURRENT_LIST_DIR}/primitives
    ${CMAKE_CURRENT_LIST_DIR}/../op_common
    ${CMAKE_CURRENT_LIST_DIR}/../op_common/inc
    ${CMAKE_CURRENT_LIST_DIR}/../op_common/topo
    ${CMAKE_CURRENT_LIST_DIR}/../executor
    ${CMAKE_CURRENT_LIST_DIR}/../engines
    ${CMAKE_CURRENT_LIST_DIR}/../engines/aicpu
    ${CMAKE_CURRENT_LIST_DIR}/../engines/ccu_ms
    ${CMAKE_CURRENT_LIST_DIR}/../engines/ccu_sche
    ${CMAKE_CURRENT_LIST_DIR}/../engines/aiv
    ${CMAKE_CURRENT_LIST_DIR}/../c_adaptor/common
    ${CMAKE_CURRENT_LIST_DIR}/../c_adaptor/common/hcomm_dlsym
    ${HCCL_DIR}/src/ops/op_common
    ${HCCL_DIR}/src/ops/op_common/inc
    ${HCCL_DIR}/src/common
    ${HCCL_DIR}/src/common/hcomm_dlsym
)

target_sources(hccl PRIVATE
    primitives/mesh_primitives.cc
    primitives/nhr_primitives.cc

    # engines/aicpu/launcher/load_kernel.cc
    # engines/aicpu/launcher/aicpu_launcher.cc
    # engines/aicpu/launcher/kernel_launch.cc

    op_alg/aicpu/aicpu_base_template.cc
    op_alg/aicpu/allgather_mesh.cc
    op_alg/aicpu/allgather_nhr.cc

    # engines/aiv/launcher/load_kernel.cc
    # engines/aiv/launcher/kernel_launch.cc
)
