# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ops_executor 单元测试
# 仅在 REFACTOR_EXECUTOR=ON 时编译
# -----------------------------------------------------------------------------------------------------------

if(NOT REFACTOR_EXECUTOR)
    message(STATUS "[UT] refactor/executor: skipped (REFACTOR_EXECUTOR=OFF)")
    return()
endif()

# 自动发现当前目录下所有 .cc 文件（排除 ut_stubs.cc）
file(GLOB UT_EXECUTOR_TEST_SRC "${CMAKE_CURRENT_LIST_DIR}/*.cc")
list(REMOVE_ITEM UT_EXECUTOR_TEST_SRC "${CMAKE_CURRENT_LIST_DIR}/ut_stubs.cc")

set(UT_EXECUTOR_SRC
    ${UT_EXECUTOR_TEST_SRC}
    ut_stubs.cc
    # 直接编译被测源文件
    ${CMAKE_SOURCE_DIR}/refactor/ops/executor/ops_executor.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/channel/channel.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/channel/channel_request.cc
    # template 源文件 (template_factory 内联实例化 AllGatherMeshTemplate，需 vtable)
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/aicpu/aicpu_base_template.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/aicpu/allgather_mesh.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/aicpu/allgather_nhr.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/primitives/mesh_primitives.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/primitives/nhr_primitives.cc
)

add_executable(hccl_ut_refactor_executor ${UT_EXECUTOR_SRC})

target_include_directories(hccl_ut_refactor_executor PRIVATE
    # refactor 头文件
    ${CMAKE_SOURCE_DIR}/refactor/ops
    ${CMAKE_SOURCE_DIR}/refactor/ops/executor
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/inc
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/topo
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/channel
    ${CMAKE_SOURCE_DIR}/refactor/ops/template
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/aicpu
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/primitives
    ${CMAKE_SOURCE_DIR}/refactor/ops/engine
    ${CMAKE_SOURCE_DIR}/refactor/ops/utils
    ${CMAKE_SOURCE_DIR}/refactor/ops/selector
    ${CMAKE_SOURCE_DIR}/refactor/ops/c_adaptor/common
    ${CMAKE_SOURCE_DIR}/refactor/ops/c_adaptor/common/hcomm_dlsym
    # src 头文件（channel.cc 依赖 topo.h 等）
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/executor/channel
    ${CMAKE_SOURCE_DIR}/src/ops/op_common
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/inc
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/topo
    ${CMAKE_SOURCE_DIR}/src/ops/op_common/template
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
