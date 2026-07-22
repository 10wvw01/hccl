# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(STATIC_MODE)
    target_sources(hccl PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_rank_graph_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_res_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_primitives_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_dlsym.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_inner_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_host_profiling_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_host_comm_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_res_expt_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu_res_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_ccu_res_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu_launch_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu_primitives_impl_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_diag_dl.cc
    )
else()
    add_library(hccl_compat SHARED)

    if(BUILD_OPEN_PROJECT)
        target_compile_definitions(hccl_compat PRIVATE
            OPEN_BUILD_PROJECT
            $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
        )
    else()
        target_compile_definitions(hccl_compat PRIVATE
            $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
        )
    endif()

    hccl_apply_cann_compat(hccl_compat)

    target_include_directories(hccl_compat PRIVATE
        ${INCLUDE_LIST}
    )

    if(BUILD_OPEN_PROJECT)
        # Open-source build: SDK headers are not in the default include paths.
        # Locate them relative to ASCEND_CANN_PACKAGE_PATH and add manually.
        find_path(_HCCL_SDK_INC
            NAMES hccl/hccl_types.h
            HINTS ${ASCEND_CANN_PACKAGE_PATH}
            PATH_SUFFIXES x86_64-linux/include
            NO_DEFAULT_PATH
        )
        find_path(_HCCL_SDK_PKG
            NAMES hccl/base.h
            HINTS ${ASCEND_CANN_PACKAGE_PATH}
            PATH_SUFFIXES x86_64-linux/pkg_inc
            NO_DEFAULT_PATH
        )
        find_path(_HCCL_SDK_LOG
            NAMES dlog_pub.h
            HINTS ${ASCEND_CANN_PACKAGE_PATH}
            PATH_SUFFIXES pkg_inc/base
            NO_DEFAULT_PATH
        )
        if(_HCCL_SDK_INC)
            get_filename_component(_HCCL_SDK_ROOT ${_HCCL_SDK_INC} DIRECTORY)
            target_include_directories(hccl_compat PRIVATE
                ${_HCCL_SDK_INC}
                ${_HCCL_SDK_INC}/hccl
                ${_HCCL_SDK_INC}/hcomm
                ${_HCCL_SDK_INC}/hcomm/ccu
            )
            target_link_directories(hccl_compat PRIVATE ${_HCCL_SDK_ROOT}/lib64)
        endif()
        if(_HCCL_SDK_PKG)
            target_include_directories(hccl_compat PRIVATE
                ${_HCCL_SDK_PKG}
                ${_HCCL_SDK_PKG}/hccl
            )
        endif()
        if(_HCCL_SDK_LOG)
            target_include_directories(hccl_compat PRIVATE ${_HCCL_SDK_LOG})
        endif()
    endif()

    target_compile_options(hccl_compat PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-g>
        -std=c++14
        -fstack-protector-all
    )

    target_link_options(hccl_compat PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
    )

    if(BUILD_OPEN_PROJECT)
        target_link_libraries(hccl_compat PRIVATE
            $<BUILD_INTERFACE:runtime_headers>
            $<BUILD_INTERFACE:hcomm_headers>
            -Wl,--no-as-needed
            unified_dlog
            acl_rt
            -Wl,--no-as-needed
        )
    else()
        target_link_libraries(hccl_compat PRIVATE
            $<BUILD_INTERFACE:slog_headers>
            -Wl,--no-as-needed
            unified_dlog
            acl_rt
            -Wl,--no-as-needed
        )
    endif()

    install(TARGETS hccl_compat
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )

    target_sources(hccl_compat PRIVATE
        hccl_rank_graph_dl.cc
        hccl_res_dl.cc
        hcomm_primitives_dl.cc
        hcomm_dlsym.cc
        hccl_inner_dl.cc
        hcomm_host_profiling_dl.cc
        hccl_host_comm_dl.cc
        hccl_res_expt_dl.cc
        ccu_res_dl.cc
        hccl_ccu_res_dl.cc
        ccu_launch_dl.cc
        ccu_primitives_impl_dl.cc
        hcomm_diag_dl.cc
    )
endif()