# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# aicpu_engine Send 单元测试
# 仅在 REFACTOR_LAUNCHER=ON 时编译
# -----------------------------------------------------------------------------------------------------------

if(NOT REFACTOR_LAUNCHER)
    message(STATUS "[UT] refactor/engine: skipped (REFACTOR_LAUNCHER=OFF)")
    return()
endif()

# ASCEND_CANN_PACKAGE_PATH fallback: build.sh build_ut 传的是 ASCEND_INSTALL_PATH
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH OR ASCEND_CANN_PACKAGE_PATH STREQUAL "")
    if(DEFINED ASCEND_INSTALL_PATH)
        set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_INSTALL_PATH})
    elseif(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH})
    endif()
endif()

# 自动发现当前目录下所有 .cc 文件（排除 ut_stubs.cc）
file(GLOB UT_AICPU_TEST_SRC "${CMAKE_CURRENT_LIST_DIR}/*.cc")
list(REMOVE_ITEM UT_AICPU_TEST_SRC "${CMAKE_CURRENT_LIST_DIR}/ut_stubs.cc")

set(UT_AICPU_SRC
    ${UT_AICPU_TEST_SRC}
    ut_stubs.cc
    # 直接编译被测源文件
    ${CMAKE_SOURCE_DIR}/refactor/ops/engine/aicpu/aicpu_engine.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/engine/aicpu/dpu/kernel_launch.cc
)

add_executable(hccl_ut_refactor_engine_aicpu ${UT_AICPU_SRC})

target_include_directories(hccl_ut_refactor_engine_aicpu PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_SOURCE_DIR}/refactor/ops
    ${CMAKE_SOURCE_DIR}/refactor/ops/engine
    ${CMAKE_SOURCE_DIR}/refactor/ops/engine/aicpu
    ${CMAKE_SOURCE_DIR}/refactor/ops/executor
    ${CMAKE_SOURCE_DIR}/refactor/ops/algorithm
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/inc
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/topo
    ${CMAKE_SOURCE_DIR}/refactor/ops/template
    ${CMAKE_SOURCE_DIR}/refactor/ops/c_adaptor/common
    ${CMAKE_SOURCE_DIR}/refactor/ops/c_adaptor/common/hcomm_dlsym
    ${CMAKE_SOURCE_DIR}/refactor/ops/selector
    ${CMAKE_SOURCE_DIR}/refactor/ops/utils
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/ops/op_common
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/inc
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/topo
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/template
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/executor
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/executor/registry
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/selector
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/op_graph
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/common/hcomm_dlsym
    ${CMAKE_SOURCE_DIR}/src/common/hcomm_dlsym/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/include/base
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
    ${CANN_3RD_LIB_PATH}/gtest/include
)

target_compile_definitions(hccl_ut_refactor_engine_aicpu PRIVATE
    CANN_VERSION_NUM=90099801
    CFG_BUILD_DEBUG
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(hccl_ut_refactor_engine_aicpu PRIVATE
    -std=c++17
    -include ${CMAKE_SOURCE_DIR}/test/ut/refactor/template/ut_version_macro.h
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    -fPIC -pipe
)

target_link_directories(hccl_ut_refactor_engine_aicpu PRIVATE
    ${CANN_3RD_LIB_PATH}/gtest/lib64
)

target_link_libraries(hccl_ut_refactor_engine_aicpu PRIVATE
    gtest gtest_main pthread gcov rt
)
