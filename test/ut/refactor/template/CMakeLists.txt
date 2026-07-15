# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# mesh_primitives unit test.
# Only build when REFACTOR_TEMPLATE is enabled.
# -----------------------------------------------------------------------------------------------------------

if(NOT REFACTOR_TEMPLATE)
    message(STATUS "[UT] refactor/template/primitives: skipped (REFACTOR_TEMPLATE=OFF)")
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

file(GLOB UT_PRIMITIVES_TEST_SRC "${CMAKE_CURRENT_LIST_DIR}/*.cc")
list(REMOVE_ITEM UT_PRIMITIVES_TEST_SRC "${CMAKE_CURRENT_LIST_DIR}/ut_stubs.cc")

set(UT_PRIMITIVES_SRC
    ${UT_PRIMITIVES_TEST_SRC}
    # CHK_* and HCCL_ERROR depend on log symbols in production. Keep a tiny UT stub here.
    ut_stubs.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/primitives/mesh_primitives.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/primitives/nhr_primitives.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/aicpu/aicpu_base_template.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/aicpu/allgather_mesh.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/aicpu/allgather_nhr.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/utils/utils.cc
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/channel/channel.cc
)

add_executable(hccl_ut_refactor_template_primitives ${UT_PRIMITIVES_SRC})

target_include_directories(hccl_ut_refactor_template_primitives PRIVATE
    ${CMAKE_SOURCE_DIR}/refactor/ops
    ${CMAKE_SOURCE_DIR}/refactor/ops/c_adaptor/common
    ${CMAKE_SOURCE_DIR}/refactor/ops/c_adaptor/common/hcomm_dlsym
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/inc
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/topo
    ${CMAKE_SOURCE_DIR}/refactor/ops/op_common/channel
    ${CMAKE_SOURCE_DIR}/refactor/ops/template
    ${CMAKE_SOURCE_DIR}/refactor/ops/template/aicpu
    ${CMAKE_SOURCE_DIR}/refactor/ops/engine
    ${CMAKE_SOURCE_DIR}/refactor/ops/utils
    ${CMAKE_SOURCE_DIR}/refactor/ops/executor
    ${CMAKE_SOURCE_DIR}/refactor/ops/algorithm
    ${CMAKE_SOURCE_DIR}/refactor/ops/selector
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/ops/op_common
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

target_compile_definitions(hccl_ut_refactor_template_primitives PRIVATE
    CANN_VERSION_NUM=90099801
    CFG_BUILD_DEBUG
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(hccl_ut_refactor_template_primitives PRIVATE
    -std=c++17
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    -fPIC -pipe
)

target_link_directories(hccl_ut_refactor_template_primitives PRIVATE
    ${CANN_3RD_LIB_PATH}/gtest/lib64
)

target_link_libraries(hccl_ut_refactor_template_primitives PRIVATE
    gtest_main gtest pthread gcov rt
)
