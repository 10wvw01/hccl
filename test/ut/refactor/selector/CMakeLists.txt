# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# execute_selector 单元测试
# 仅在 REFACTOR_SELECTOR=ON 时编译
# -----------------------------------------------------------------------------------------------------------

if(NOT REFACTOR_SELECTOR)
    message(STATUS "[UT] refactor/selector: skipped (REFACTOR_SELECTOR=OFF)")
    return()
endif()

# 自动发现当前目录下所有 .cc 文件（排除 ut_stubs.cc）
file(GLOB UT_SELECTOR_TEST_SRC "${CMAKE_CURRENT_LIST_DIR}/*.cc")
list(REMOVE_ITEM UT_SELECTOR_TEST_SRC "${CMAKE_CURRENT_LIST_DIR}/ut_stubs.cc")

set(UT_SELECTOR_SRC
    ${UT_SELECTOR_TEST_SRC}
    # CHK_* 和 HCCL_ERROR depend on log symbols in production. Keep a tiny UT stub here.
    ut_stubs.cc
    # 直接编译被测源文件
    ${CMAKE_SOURCE_DIR}/refactor/ops/selector/execute_selector.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/selector/selector_registry.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/selector/auto_selector_base.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/selector/all_gather_auto_selector.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/algorithm/all_gather/algorithm_all_gather_aicpu.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/algorithm/all_gather/all_gather_template_desc.cc
)

add_executable(hccl_ut_refactor_selector ${UT_SELECTOR_SRC})

target_include_directories(hccl_ut_refactor_selector PRIVATE
    # refactor 头文件
    ${CMAKE_SOURCE_DIR}/refactor/ops
    ${CMAKE_SOURCE_DIR}/refactor/ops/selector
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/inc
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/topo
    ${CMAKE_SOURCE_DIR}/refactor/ops/c_adaptor/common
    ${CMAKE_SOURCE_DIR}/refactor/ops/c_adaptor/common/hcomm_dlsym
    ${CMAKE_SOURCE_DIR}/refactor/ops/algorithm/all_gather
    ${CMAKE_SOURCE_DIR}/refactor/ops/template
    ${CMAKE_SOURCE_DIR}/refactor/ops/engines/aicpu
    ${CMAKE_SOURCE_DIR}/src/common/hcomm_dlsym
    ${CMAKE_SOURCE_DIR}/src/common/hcomm_dlsym/ccu
    # SDK 头文件
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

target_compile_definitions(hccl_ut_refactor_selector PRIVATE
    CANN_VERSION_NUM=90099801
    CFG_BUILD_DEBUG
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(hccl_ut_refactor_selector PRIVATE
    -std=c++17
    # Some public headers use CANN_VERSION before the normal production version header is visible in UT.
    -include ${CMAKE_CURRENT_LIST_DIR}/ut_version_macro.h
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    -fPIC -pipe
)

target_link_directories(hccl_ut_refactor_selector PRIVATE
    ${CANN_3RD_LIB_PATH}/gtest/lib64
)

target_link_libraries(hccl_ut_refactor_selector PRIVATE
    gtest gtest_main pthread gcov rt
)
