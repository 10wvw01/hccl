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

# 暂无 AIV 实现，预留目录结构
set(src_list
    # ${CMAKE_CURRENT_SOURCE_DIR}/omni_temp_aiv.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/kernel/aiv_kernel_omni.cc
)

if(NOT src_list)
    message(STATUS "eco_system: aiv template has no sources, skip")
    return()
endif()

eco_system_add_sources(
    SOURCES ${src_list}
)
