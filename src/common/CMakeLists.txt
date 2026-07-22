# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter_acl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_env_config.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/param_check.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/inconsistent_check.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_type.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter_error_manager_pub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/config_log.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/log.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/sal.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_mc2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/compat.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dfx/hccl_dump_impl.cpp
)

# 在静态模式下添加 static_restore.cc
message(STATUS "STATIC_MODE = ${STATIC_MODE}")
if(STATIC_MODE)
    list(APPEND src_list ${CMAKE_CURRENT_SOURCE_DIR}/static_restore.cc)
    message(STATUS "Added static_restore.cc to src_list")
endif()

message(STATUS "src_list contains ${src_list}")

if(TARGET hccl)
    target_sources(hccl PRIVATE
        ${src_list}
    )
endif()

if(NOT HCCL_CANN_COMPAT_850)
    set(HCCL_DFX_QUEUE_READ_KERNEL ${CMAKE_CURRENT_SOURCE_DIR}/dfx/hccl_dfx_queue_read_kernel.h)
    set_source_files_properties(
        ${HCCL_DFX_QUEUE_READ_KERNEL} PROPERTIES LANGUAGE CXX
    )
    set(CCE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin)
    add_custom_target(hccl_dfx_queue_read_kernel_910_95 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_dfx_queue_read_kernel_910_95.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_dfx_queue_read_kernel_910_95.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${HCCL_DFX_QUEUE_READ_KERNEL}
            -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_dfx_queue_read_kernel_910_95_tmp.o
            --cce-aicore-arch=dav-c310-vec
            --cce-aicore-input-parameter-size=512 --cce-aicore-only
            -mllvm -cce-aicore-function-stack-size=0x8000
            -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -mllvm -cce-aicore-record-overflow=false
            -mllvm -cce-aicore-addr-transform
            -mllvm --cce-aicore-jump-expand=true
            -std=c++17 -fstack-protector-all
        COMMAND ${CCE_DIR}/ld.lld -m aicorelinux -Ttext=0
            ${CMAKE_CURRENT_BINARY_DIR}/hccl_dfx_queue_read_kernel_910_95_tmp.o
            -static -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_dfx_queue_read_kernel_910_95.o
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_dfx_queue_read_kernel_910_95.o
        DESTINATION ${INSTALL_LIBRARY_DIR}
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
endif()

add_subdirectory(hcomm_dlsym)
