# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not be in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# engine/aicpu：AICPU 引擎（AiCpuEngine / KernelLaunch / LoadKernel）

target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_sources(hccl PRIVATE
    aicpu_engine.cc
    kernel_launch.cc
    load_kernel.cc
    dpu/kernel_launch.cc
    dfx/task_exception_fun.cc
)
