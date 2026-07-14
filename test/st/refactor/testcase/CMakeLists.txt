# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# refactor ST 用例：只编译 refactor 已实现的算子（当前仅 all_gather）
# -----------------------------------------------------------------------------------------------------------

set(src_list
    main.cc
    all_gather_aicpu_testcase.cc
)

add_executable(hccl_checker_ops_stest ${src_list})

if (BUILD_OPEN_PROJECT)
    target_include_directories(hccl_checker_ops_stest PUBLIC
        # 第三方头文件
        ${THIRD_PARTH_PATH}/gtest_shared/include

        # HCCL 仓 refactor 头文件（替换原 src/ 路径）
        ${HCCL_DEV_BASE}/refactor/ops
        ${HCCL_DEV_BASE}/refactor/ops/op_common
        ${HCCL_DEV_BASE}/refactor/ops/op_common/inc
        ${HCCL_DEV_BASE}/refactor/ops/c_adaptor/common

        # src/common 仍提供 log.h / hcomm_dlsym（refactor 复用）
        ${HCCL_DEV_BASE}/src/common
        ${HCCL_DEV_BASE}/src/common/hcomm_dlsym
        ${HCCL_DEV_BASE}/src/common/hcomm_dlsym/ccu

        # Ascend 安装包头文件
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/pkg_inc
        ${ASCEND_HOME_PATH}/pkg_inc/base
        ${ASCEND_HOME_PATH}/hcomm/include/hccl
        ${ASCEND_HOME_PATH}/include/hccl
        ${ASCEND_HOME_PATH}/pkg_inc/hccl
        ${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu

        # ST 引擎头文件
        ${UTILS_DIR}/sim_world
        ${UTILS_DIR}/common/utils
        ${UTILS_DIR}/common/exception
        ${UTILS_DIR}/hccl_verifier
        ${UTILS_DIR}/hccl_verifier/semantics_check
        ${UTILS_DIR}/hccl_verifier/task_graph_generator
        ${UTILS_DIR}/hccl_proxy
        ${UTILS_DIR}/hccl_proxy/communicator
        ${UTILS_DIR}/hccl_proxy/topo_model
    )

    target_link_libraries(hccl_checker_ops_stest PRIVATE
        gtest
        pthread
        hccl-model-proxy
        hccl-verifier
        -Wl,--allow-shlib-undefined
        ${HCCL_DEV_BASE}/build/src/libhccl.so
        -ldl
    )

    target_compile_options(hccl_checker_ops_stest PRIVATE
        -std=c++14
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
    )
endif()
