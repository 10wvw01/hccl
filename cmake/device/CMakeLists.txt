# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

set(PRODUCT_SIDE "device")
set(HCCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../" CACHE PATH "Project top directory")
include(${HCCL_DIR}/cmake/fetch_cann_cmake.cmake)
project(hccl_device)

# 初始化 CANN 项目配置
init_cann_project(PREPEND_MODULE_PATH)

message(STATUS "BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")
message(STATUS "ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}")
message(STATUS "CANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}")

# cmake 配置
include(${HCCL_DIR}/cmake/config.cmake)
include(${HCCL_DIR}/cmake/func.cmake)

# 添加子目录
add_cann_subdirectories_relative(${HCCL_DIR}
    ${HCCL_DIR}/src
)

# 实验特性: 将 BIRS 等实验算子编入 device 侧 scatter_aicpu_kernel
if(ENABLE_EXPERIMENTAL)
    target_compile_definitions(scatter_aicpu_kernel PRIVATE ENABLE_EXPERIMENTAL)
    target_include_directories(scatter_aicpu_kernel PRIVATE
        ${HCCL_DIR}/experimental/
        ${HCCL_DIR}/experimental/ops/
        ${HCCL_DIR}/experimental/ops/op_common/
        ${HCCL_DIR}/experimental/ops/op_common/template
        ${HCCL_DIR}/experimental/ops/op_common/topo
        ${HCCL_DIR}/experimental/ops/reduce_scatter/
        ${HCCL_DIR}/experimental/ops/reduce_scatter/birs/
        ${HCCL_DIR}/experimental/ops/reduce_scatter/birs/template/)
    target_sources(scatter_aicpu_kernel PRIVATE
        ${HCCL_DIR}/experimental/ops/op_common/template/alg_template_base_experimental.cc
        ${HCCL_DIR}/experimental/ops/op_common/topo/topo_experimental.cc
        ${HCCL_DIR}/experimental/ops/reduce_scatter/birs/reduce_scatter_birs_executor.cc
        ${HCCL_DIR}/experimental/ops/reduce_scatter/birs/reduce_scatter_executor_base.cc
        ${HCCL_DIR}/experimental/ops/reduce_scatter/birs/template/reduce_scatter_birs.cc
        ${HCCL_DIR}/experimental/ops/reduce_scatter/birs/template/reduce_scatter_birs_inter.cc)
endif()

pack_targets_and_files(
    OUTPUT_TARGET "generate_device_aicpu_package"
    OUTPUT "aicpu_hccl.tar.gz"
    TARGETS scatter_aicpu_kernel hccl_kernel_compat
    MANIFEST "bin_hash.cfg"
    TAR_ROOT_DIR "aicpu_kernels_device"
)
sign_file(
    INPUT "aicpu_hccl.tar.gz"
    CONFIG "${HCCL_DIR}/scripts/sign/hccl_check_cfg.xml"
    RESULT_VAR "aicpu_hccl_sign_file"
    DEPENDS generate_device_aicpu_package
)
add_dependencies(sign_aicpu_hccl generate_device_aicpu_package)

# 安装 aicpu_hccl.tar.gz
install(FILES ${aicpu_hccl_sign_file}
    DESTINATION ${INSTALL_AICPU_KERNEL_JSON_DIR}/kernel 
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

# 设置子工程打包
set_cann_subprj_package()