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
target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../op_common
    ${HCCL_DIR}/src/ops/op_common/template
    ${HCCL_DIR}/src/ops/op_common/template/wrapper
)

target_sources(hccl PRIVATE
    primitives/mesh_primitives.cc
    # primitives/nhr_primitives.cc

    # engines/aicpu/launcher/load_kernel.cc
    # engines/aicpu/launcher/aicpu_launcher.cc
    # engines/aicpu/launcher/kernel_launch.cc

    # engines/aicpu/op_alg/allgather_mesh.cc
    # engines/aicpu/op_alg/allgather_nhr.cc

    # engines/aiv/launcher/load_kernel.cc
    # engines/aiv/launcher/kernel_launch.cc
)
