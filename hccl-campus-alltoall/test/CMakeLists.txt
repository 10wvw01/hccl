# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 查找 GTest 库
find_package(GTest REQUIRED)
include(GoogleTest)

# 启用 CTest 测试框架
enable_testing()

set(HCCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")

# 定义测试目标
add_executable(test_alltoall
    # 业务代码
    ${HCCL_DIR}/op_host/alltoall.cc
    ${HCCL_DIR}/op_host/alltoall_aicpu.cc
    ${HCCL_DIR}/op_host/alltoall_ccu.cc
    ${HCCL_DIR}/op_host/exec_ccu_op.cc
    ${HCCL_DIR}/op_host/launch_aicpu_kernel.cc
    ${HCCL_DIR}/op_kernel_aicpu/aicpu_kernel.cc
    ${HCCL_DIR}/op_kernel_aicpu/exec_op.cc
    ${HCCL_DIR}/op_kernel_ccu/ccu_kernel.cc
    # 测试代码
    ${HCCL_DIR}/test/ut_alltoall.cc
    ${HCCL_DIR}/test/stub/ut_stub_acl_rt.cc
    ${HCCL_DIR}/test/stub/ut_stub_hcomm.cc
)
target_include_directories(test_alltoall PRIVATE
    ${HCCL_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
)
target_compile_definitions(test_alltoall PRIVATE
    LOG_LEVEL=LOG_LEVEL_DEBUG
)
target_compile_options(test_alltoall PRIVATE
    -g -O0
    -std=c++17
    -fstack-protector-all
)
target_link_options(test_alltoall PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_link_libraries(test_alltoall PRIVATE
    GTest::GTest
    GTest::Main
)

# 安装
install(TARGETS test_alltoall RUNTIME DESTINATION bin)

# 运行测试程序
gtest_discover_tests(test_alltoall)
