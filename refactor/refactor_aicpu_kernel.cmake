# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# refactor 目录的 AICPU device 侧 kernel 库构建。
# 参考 src/scatter_aicpu_kernel.cmake 的编译方式，将 refactor/ops 下的 template/primitives
# 等 device 侧代码编为 libscatter_aicpu_kernel.so。

set(REFACTOR_OPS_DIR ${CMAKE_CURRENT_LIST_DIR}/ops)

add_library(scatter_aicpu_kernel SHARED
    # ── c_adaptor/common：device 侧共用的基础工具 ──
    ${REFACTOR_OPS_DIR}/c_adaptor/common/utils.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/config_log.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/sal.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/log.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/adapter_error_manager_pub.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/alg_env_config.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/device_compat.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/alg_type.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/param_check.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/hccl_mc2.cc

    # ── op_common：device 侧共用逻辑 ──
    ${REFACTOR_OPS_DIR}/op_common/hccl_algorithm.cc
    ${REFACTOR_OPS_DIR}/op_common/exec_timeout_manager.cc
    ${REFACTOR_OPS_DIR}/op_common/channel/channel.cc
    ${REFACTOR_OPS_DIR}/op_common/channel/channel_request.cc

    # ── template/aicpu：AICPU 算法模板实现 ──
    ${REFACTOR_OPS_DIR}/template/aicpu/aicpu_base_template.cc
    ${REFACTOR_OPS_DIR}/template/aicpu/allgather_mesh.cc
    ${REFACTOR_OPS_DIR}/template/aicpu/allgather_nhr.cc
    ${REFACTOR_OPS_DIR}/template/aicpu/reducescatter_mesh.cc
    ${REFACTOR_OPS_DIR}/template/aicpu/reducescatter_nhr.cc
    ${REFACTOR_OPS_DIR}/template/aicpu/scatter_mesh.cc
    ${REFACTOR_OPS_DIR}/template/aicpu/scatter_nhr.cc

    # ── template/primitives：通信原语实现 ──
    ${REFACTOR_OPS_DIR}/template/primitives/mesh_primitives.cc
    ${REFACTOR_OPS_DIR}/template/primitives/nhr_primitives.cc

    # ── engine/aicpu：AICPU 引擎 device 侧 kernel_launch ──
    ${REFACTOR_OPS_DIR}/engine/aicpu/data_transfer.cc
    ${REFACTOR_OPS_DIR}/engine/aicpu/kernel_launch.cc
    ${REFACTOR_OPS_DIR}/engine/aicpu/dpu/kernel_launch.cc

    # ── engine/aicpu：device 侧 DPU/DFX 实现 ──
    ${REFACTOR_OPS_DIR}/engine/aicpu/dfx/task_exception_fun.cc

    # ── executor：统一执行器 ──
    ${REFACTOR_OPS_DIR}/executor/ops_executor.cc

    # ── utils ──
    ${REFACTOR_OPS_DIR}/utils/utils.cc
)

# ── include 路径 ──
target_include_directories(scatter_aicpu_kernel PRIVATE
    ${REFACTOR_OPS_DIR}
    ${REFACTOR_OPS_DIR}/op_common
    ${REFACTOR_OPS_DIR}/op_common/inc
    ${REFACTOR_OPS_DIR}/op_common/topo
    ${REFACTOR_OPS_DIR}/op_common/channel
    ${INCLUDE_LIST}
    ${REFACTOR_OPS_DIR}/c_adaptor/common
    ${REFACTOR_OPS_DIR}/c_adaptor/common/hcomm_dlsym
    ${REFACTOR_OPS_DIR}/engine
    ${REFACTOR_OPS_DIR}/engine/aicpu
    ${REFACTOR_OPS_DIR}/engine/aicpu/dfx
    ${REFACTOR_OPS_DIR}/engine/aicpu/dpu
    ${REFACTOR_OPS_DIR}/template/aicpu
    ${REFACTOR_OPS_DIR}/template/primitives
    ${REFACTOR_OPS_DIR}/template
    ${REFACTOR_OPS_DIR}/executor
    ${REFACTOR_OPS_DIR}/utils
)

# ── 编译选项：与 scatter_aicpu_kernel 保持一致 ──
target_compile_options(scatter_aicpu_kernel PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
    -fstack-protector-all
    -Werror
)

target_link_options(scatter_aicpu_kernel PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)

target_compile_definitions(scatter_aicpu_kernel PRIVATE
    -DAICPU_COMPILE
)

hccl_apply_cann_compat(scatter_aicpu_kernel)

target_link_directories(scatter_aicpu_kernel PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/devlib/device
)

# ── 链接库：与 scatter_aicpu_kernel 一致 ──
if(NOT HCCL_CANN_COMPAT_850)
    target_link_libraries(scatter_aicpu_kernel PRIVATE
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:hcomm_headers>
        unified_dlog
        -Wl,--no-as-needed
        ccl_kernel
        hccl_kernel_compat
        -Wl,--no-as-needed
    )
else()
    target_link_libraries(scatter_aicpu_kernel PRIVATE
        -Wl,--no-as-needed
        hccl_kernel_compat
        -Wl,--no-as-needed
    )
endif()
add_dependencies(scatter_aicpu_kernel hccl_kernel_compat)

# ── 安装 ──
install(TARGETS scatter_aicpu_kernel
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)