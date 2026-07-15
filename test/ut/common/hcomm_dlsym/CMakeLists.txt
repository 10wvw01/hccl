# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_executable(hccl_ut_hcomm_version_cache main.cc)

target_include_directories(hccl_ut_hcomm_version_cache PRIVATE
    ${HCCL_DIR}/src/common/hcomm_dlsym
)

target_compile_options(hccl_ut_hcomm_version_cache PRIVATE
    -U_FORTIFY_SOURCE
    -g
    --coverage
    -fprofile-arcs
    -ftest-coverage
    -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)

target_link_libraries(hccl_ut_hcomm_version_cache PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -lgcov
    -lrt
    -lpthread
)

run_llt_test(
    TARGET hccl_ut_hcomm_version_cache
    TASK_NUM 1
)
