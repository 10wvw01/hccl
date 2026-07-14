# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# refactor ST 引擎：sim_world/hccl_verifier/hccl_proxy/common/hccl_depends_stub/aicpu
# -----------------------------------------------------------------------------------------------------------

add_library(hccl-model-proxy SHARED "")

add_subdirectory(hccl_verifier)
add_subdirectory(sim_world)
add_subdirectory(hccl_proxy)
add_subdirectory(common)
add_subdirectory(hccl_depends_stub)
add_subdirectory(aicpu)

target_compile_options(hccl-model-proxy PRIVATE
    -std=c++14
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O0
    -g
    -D_GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(hccl-model-proxy PUBLIC
    # 链接 hccl_depends_stub
    hcomm
    unified_dlog
    acl_rt
    scatter_aicpu_kernel
)

if (BUILD_OPEN_PROJECT)
    target_link_libraries(hccl-model-proxy PUBLIC
        ${ASCEND_HOME_PATH}/lib64/libc_sec.so
    )

    target_include_directories(hccl-model-proxy PRIVATE
        ${UTILS_DIR}/sim_world
        ${UTILS_DIR}/common/exception
        ${UTILS_DIR}/common/utils
        ${UTILS_DIR}/hccl_verifier
        ${UTILS_DIR}/hccl_proxy
        ${UTILS_DIR}/hccl_proxy/communicator
        ${UTILS_DIR}/hccl_proxy/topo_model

        # Ascend 安装包头文件
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/pkg_inc
        ${ASCEND_HOME_PATH}/pkg_inc/base
        ${ASCEND_HOME_PATH}/pkg_inc/runtime
        ${ASCEND_HOME_PATH}/pkg_inc/profiling
        ${ASCEND_HOME_PATH}/hcomm/include/hccl
        ${ASCEND_HOME_PATH}/include/hccl
        ${ASCEND_HOME_PATH}/include/hcomm
        ${ASCEND_HOME_PATH}/include/hcomm/ccu
        ${ASCEND_HOME_PATH}/pkg_inc/hccl
        ${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu

        # refactor 头文件（替换原 src/ops/ 路径）
        ${HCCL_DEV_BASE}/refactor/ops
        ${HCCL_DEV_BASE}/refactor/ops/op_common
        ${HCCL_DEV_BASE}/refactor/ops/op_common/inc
        ${HCCL_DEV_BASE}/refactor/ops/c_adaptor/common

        # src/common 复用（log.h / hcomm_dlsym）
        ${HCCL_DEV_BASE}/src/common
        ${HCCL_DEV_BASE}/src/common/hcomm_dlsym
        ${HCCL_DEV_BASE}/src/common/hcomm_dlsym/ccu
    )
endif()
