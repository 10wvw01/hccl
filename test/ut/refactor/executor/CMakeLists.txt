# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ops_executor 单元测试
# 仅在 REFACTOR_EXECUTOR=ON 时编译
# -----------------------------------------------------------------------------------------------------------

if(NOT REFACTOR_EXECUTOR)
    message(STATUS "[UT] refactor/executor: skipped (REFACTOR_EXECUTOR=OFF)")
    return()
endif()

set(UT_EXECUTOR_SRC
    test_ops_executor.cc
    ut_stubs.cc
    # 直接编译被测源文件，避免依赖 hccl 库的链接顺序问题
    ${CMAKE_SOURCE_DIR}/refactor/ops/executor/ops_executor.cc
)

add_executable(hccl_ut_refactor_executor ${UT_EXECUTOR_SRC})

# refactor 头文件路径
target_include_directories(hccl_ut_refactor_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/refactor/ops
    ${CMAKE_SOURCE_DIR}/refactor/ops/executor
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common
    ${CMAKE_SOURCE_DIR}/refactor/ops/template
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/ops/op_common
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/inc
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/topo
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/template
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/common/hcomm_dlsym
    ${CMAKE_SOURCE_DIR}/src/common/hcomm_dlsym/ccu
    ${CMAKE_SOURCE_DIR}/src/common/op_graph
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/executor
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/selector
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/executor/registry
    # SDK 头文件路径（与 src/CMakeLists.txt 中 INCLUDE_LIST 对齐）
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

target_compile_definitions(hccl_ut_refactor_executor PRIVATE
    CANN_VERSION_NUM=90099801
    CFG_BUILD_DEBUG
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(hccl_ut_refactor_executor PRIVATE
    -std=c++17
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    -fPIC -pipe
)

target_link_directories(hccl_ut_refactor_executor PRIVATE
    ${CANN_3RD_LIB_PATH}/gtest/lib64
)

target_link_libraries(hccl_ut_refactor_executor PRIVATE
    gtest gtest_main pthread gcov rt
)

# TODO: run_llt_test ASAN timeout 参数待适配
# run_llt_test(
#     TARGET  hccl_ut_refactor_executor
#     TASK_NUM 1
# )
