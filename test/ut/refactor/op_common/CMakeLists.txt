# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# op_common unit test (exec_timeout_manager + topo).
# Only build when REFACTOR_OP_COMMON is enabled.
# -----------------------------------------------------------------------------------------------------------

if(NOT REFACTOR_OP_COMMON)
    message(STATUS "[UT] refactor/op_common: skipped (REFACTOR_OP_COMMON=OFF)")
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

file(GLOB UT_OP_COMMON_TEST_SRC "${CMAKE_CURRENT_LIST_DIR}/*.cc")
list(REMOVE_ITEM UT_OP_COMMON_TEST_SRC "${CMAKE_CURRENT_LIST_DIR}/ut_stubs.cc")

set(UT_OP_COMMON_SRC
    ${UT_OP_COMMON_TEST_SRC}
    ut_stubs.cc
    # 直接编译被测源文件
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/exec_timeout_manager.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/topo/topo.cc
)

add_executable(hccl_ut_refactor_op_common ${UT_OP_COMMON_SRC})

target_include_directories(hccl_ut_refactor_op_common PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_SOURCE_DIR}/refactor/ops
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/inc
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/topo
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/channel
    ${CMAKE_SOURCE_DIR}/refactor/ops/engine
    ${CMAKE_SOURCE_DIR}/refactor/ops/engine/aicpu
    ${CMAKE_SOURCE_DIR}/refactor/ops/executor
    ${CMAKE_SOURCE_DIR}/refactor/ops/algorithm
    ${CMAKE_SOURCE_DIR}/refactor/ops/template
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/aicpu
    ${CMAKE_SOURCE_DIR}/refactor/ops/c_adaptor/common
    ${CMAKE_SOURCE_DIR}/refactor/ops/c_adaptor/common/hcomm_dlsym
    ${CMAKE_SOURCE_DIR}/refactor/ops/selector
    ${CMAKE_SOURCE_DIR}/refactor/ops/utils
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/ops/op_common
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/inc
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/common/hcomm_dlsym
    ${CMAKE_SOURCE_DIR}/src/common/hcomm_dlsym/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/acl
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

target_compile_definitions(hccl_ut_refactor_op_common PRIVATE
    CANN_VERSION_NUM=90099801
    CFG_BUILD_DEBUG
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(hccl_ut_refactor_op_common PRIVATE
    -std=c++17
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    -fPIC -pipe
)

target_link_directories(hccl_ut_refactor_op_common PRIVATE
    ${CANN_3RD_LIB_PATH}/gtest/lib64
)

target_link_libraries(hccl_ut_refactor_op_common PRIVATE
    gtest gtest_main pthread gcov rt
)
