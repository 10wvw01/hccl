# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(NOT ENABLE_EXPERIMENTAL)
    return()
endif()

if(TARGET hccl)
    eco_system_add_sources(
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/omni_parser.cc
    )
elseif(TARGET scatter_aicpu_kernel)
    # Device AICPU：无 hccl 目标，omni_parser 需直接编入 scatter_aicpu_kernel
    target_sources(scatter_aicpu_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/omni_parser.cc)
    target_include_directories(scatter_aicpu_kernel PRIVATE
        ${ECO_SYSTEM_ROOT}/op_common
        ${ECO_SYSTEM_ROOT}/op_common/inc
    )
    target_compile_definitions(scatter_aicpu_kernel PRIVATE ENABLE_EXPERIMENTAL)
    message(STATUS "eco_system: omni_parser.cc -> scatter_aicpu_kernel")
endif()
