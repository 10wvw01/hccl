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
# 等 device 侧代码编为 librefactor_aicpu_kernel.so。

set(REFACTOR_OPS_DIR ${CMAKE_CURRENT_LIST_DIR}/ops)

add_library(refactor_aicpu_kernel SHARED
    # ── c_adaptor/common：device 侧共用的基础工具 ──
    ${REFACTOR_OPS_DIR}/c_adaptor/common/utils.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/config_log.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/sal.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/log.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/adapter_error_manager_pub.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/alg_env_config.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/device_compat.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/adapter_acl.cc
    ${REFACTOR_OPS_DIR}/c_adaptor/common/alg_type.cc

    # ── op_common：device 侧共用逻辑 ──
    ${REFACTOR_OPS_DIR}/op_common/hccl_algorithm.cc
    ${REFACTOR_OPS_DIR}/op_common/op_common.cc
    ${REFACTOR_OPS_DIR}/op_common/exec_timeout_manager.cc
    ${REFACTOR_OPS_DIR}/op_common/channel/channel.cc
    ${REFACTOR_OPS_DIR}/op_common/channel/channel_request.cc

    # ── op_common/topo：拓扑匹配 ──
    ${REFACTOR_OPS_DIR}/op_common/topo/topo.cc
    ${REFACTOR_OPS_DIR}/op_common/topo/topo_host.cc
    ${REFACTOR_OPS_DIR}/op_common/topo/topo_match_base.cc
    ${REFACTOR_OPS_DIR}/op_common/topo/topo_match_1d.cc
    ${REFACTOR_OPS_DIR}/op_common/topo/topo_match_multilevel.cc
    ${REFACTOR_OPS_DIR}/op_common/topo/topo_match_ubx.cc
    ${REFACTOR_OPS_DIR}/op_common/topo/topo_match_ubx_1d.cc
    ${REFACTOR_OPS_DIR}/op_common/topo/topo_match_pcie_mix.cc
    ${REFACTOR_OPS_DIR}/op_common/topo/topo_match_3_level.cc
    ${REFACTOR_OPS_DIR}/op_common/topo/topo_match_squeeze_2d.cc

    # ── template/aicpu：AICPU 算法模板实现 ──
    ${REFACTOR_OPS_DIR}/template/aicpu/aicpu_base_template.cc
    ${REFACTOR_OPS_DIR}/template/aicpu/allgather_mesh.cc
    ${REFACTOR_OPS_DIR}/template/aicpu/allgather_nhr.cc
    ${REFACTOR_OPS_DIR}/template/aicpu/reducescatter_mesh.cc
    ${REFACTOR_OPS_DIR}/template/aicpu/reducescatter_nhr.cc

    # ── template/primitives：通信原语实现 ──
    ${REFACTOR_OPS_DIR}/template/primitives/mesh_primitives.cc
    ${REFACTOR_OPS_DIR}/template/primitives/nhr_primitives.cc

    # ── algorithm：算法描述 ──
    ${REFACTOR_OPS_DIR}/algorithm/all_gather/algorithm_all_gather_aicpu.cc
    ${REFACTOR_OPS_DIR}/algorithm/all_gather/all_gather_template_desc.cc
    ${REFACTOR_OPS_DIR}/algorithm/reduce_scatter/algorithm_reduce_scatter_aicpu.cc
    ${REFACTOR_OPS_DIR}/algorithm/reduce_scatter/reduce_scatter_template_desc.cc

    # ── engine/aicpu：AICPU 引擎 device 侧 kernel_launch ──
    ${REFACTOR_OPS_DIR}/engine/aicpu/kernel_launch.cc
    ${REFACTOR_OPS_DIR}/engine/aicpu/load_kernel.cc
    ${REFACTOR_OPS_DIR}/engine/aicpu/dpu/kernel_launch.cc
    ${REFACTOR_OPS_DIR}/engine/aicpu/dfx/task_exception_fun.cc

    # ── executor：统一执行器 ──
    ${REFACTOR_OPS_DIR}/executor/ops_executor.cc

    # ── selector：算法选择器 ──
    ${REFACTOR_OPS_DIR}/selector/execute_selector.cc
    ${REFACTOR_OPS_DIR}/selector/auto_selector_base.cc
    ${REFACTOR_OPS_DIR}/selector/all_gather_auto_selector.cc
    ${REFACTOR_OPS_DIR}/selector/reduce_scatter_auto_selector.cc
    ${REFACTOR_OPS_DIR}/selector/selector_registry.cc
    ${REFACTOR_OPS_DIR}/selector/dlhcomm_function.cc

    # ── utils ──
    ${REFACTOR_OPS_DIR}/utils/utils.cc
)

# ── include 路径 ──
target_include_directories(refactor_aicpu_kernel PRIVATE
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
    ${REFACTOR_OPS_DIR}/algorithm/all_gather
    ${REFACTOR_OPS_DIR}/algorithm/reduce_scatter
    ${REFACTOR_OPS_DIR}/executor
    ${REFACTOR_OPS_DIR}/selector
    ${REFACTOR_OPS_DIR}/utils
    ${REFACTOR_OPS_DIR}/api
)

# ── 编译选项：与 scatter_aicpu_kernel 保持一致 ──
target_compile_options(refactor_aicpu_kernel PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
    -fstack-protector-all
    -Werror
)

target_link_options(refactor_aicpu_kernel PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)

target_compile_definitions(refactor_aicpu_kernel PRIVATE
    -DAICPU_COMPILE
)

hccl_apply_cann_compat(refactor_aicpu_kernel)

target_link_directories(refactor_aicpu_kernel PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/devlib/device
)

# ── 链接库：与 scatter_aicpu_kernel 一致 ──
if(NOT HCCL_CANN_COMPAT_850)
    target_link_libraries(refactor_aicpu_kernel PRIVATE
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
    target_link_libraries(refactor_aicpu_kernel PRIVATE
        -Wl,--no-as-needed
        hccl_kernel_compat
        -Wl,--no-as-needed
    )
endif()
add_dependencies(refactor_aicpu_kernel hccl_kernel_compat)

# ── 安装 ──
install(TARGETS refactor_aicpu_kernel
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)
