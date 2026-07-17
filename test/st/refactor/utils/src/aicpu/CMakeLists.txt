# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# refactor ST 的 aicpu kernel 库：把 refactor 真实入口编进 scatter_aicpu_kernel.so
#
# 设计要点：
#   - aclrt_stub.cc 通过 extern "C" HcclLaunchAicpuKernel(OpParam*) 触发 kernel
#   - 直接编入 refactor 的 kernel_launch.cc（签名已与 algorithm 统一）
#   - 与 algorithm ST 对齐：hccl_res_dl.h 已同步 src/ 版本，task_exception_fun.cc 编入 .so
#   - 仅编译 1 个 dlsym 桩（hcomm_primitives_dl.cc），与 algorithm ST 对齐
# -----------------------------------------------------------------------------------------------------------

set(REFACTOR_DIR ${HCCL_DEV_BASE}/refactor)

set(INCLUDE_LIST
    # refactor 头文件
    ${REFACTOR_DIR}/ops
    ${REFACTOR_DIR}/ops/op_common
    ${REFACTOR_DIR}/ops/op_common/inc
    ${REFACTOR_DIR}/ops/op_common/topo
    ${REFACTOR_DIR}/ops/op_common/channel
    ${REFACTOR_DIR}/ops/executor
    ${REFACTOR_DIR}/ops/template
    ${REFACTOR_DIR}/ops/template/aicpu
    ${REFACTOR_DIR}/ops/template/primitives
    ${REFACTOR_DIR}/ops/utils
    ${REFACTOR_DIR}/ops/engine
    ${REFACTOR_DIR}/ops/engine/aicpu
    ${REFACTOR_DIR}/ops/c_adaptor/common
    ${REFACTOR_DIR}/ops/c_adaptor/common/hcomm_dlsym

    # src/common 复用（log.h / hcomm_dlsym）
    ${HCCL_DEV_BASE}/src/common
    ${HCCL_DEV_BASE}/src/common/hcomm_dlsym

    # Ascend 安装包头文件
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/acl
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_HOME_PATH}/include/hcomm
    ${ASCEND_HOME_PATH}/include/base
    ${ASCEND_HOME_PATH}/include/driver
    ${ASCEND_HOME_PATH}/include/external
    ${ASCEND_HOME_PATH}/include/platform
    ${ASCEND_HOME_PATH}/hcomm/include/hccl
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/hccl
    ${ASCEND_HOME_PATH}/pkg_inc/hcomm
    ${ASCEND_HOME_PATH}/pkg_inc/base
    ${ASCEND_HOME_PATH}/include/mmpa
)

add_library(scatter_aicpu_kernel SHARED
    # refactor aicpu kernel 入口（与 algorithm 统一签名的 extern "C" HcclLaunchAicpuKernel(OpParam*)）
    ${REFACTOR_DIR}/ops/engine/aicpu/kernel_launch.cc
    # DFX / Scatter 实现（与 algorithm ST 对齐，直接编入 refactor 版本）
    ${REFACTOR_DIR}/ops/engine/aicpu/dfx/task_exception_fun.cc

    # refactor executor（CalcRes / Orchestrate / OrchestrateLoop）
    ${REFACTOR_DIR}/ops/executor/ops_executor.cc

    # refactor op_common（HcclAlgorithm 序列化 / 反序列化 + exec_timeout_manager）
    ${REFACTOR_DIR}/ops/op_common/hccl_algorithm.cc
    ${REFACTOR_DIR}/ops/op_common/exec_timeout_manager.cc

    # refactor 拓扑匹配（供 template 使用）
    ${REFACTOR_DIR}/ops/op_common/topo/topo.cc
    ${REFACTOR_DIR}/ops/op_common/topo/topo_host.cc
    ${REFACTOR_DIR}/ops/op_common/topo/topo_match_base.cc
    ${REFACTOR_DIR}/ops/op_common/topo/topo_match_1d.cc

    # refactor utils（SendRecv* / PreCopy / PostCopy / rank 规划）
    ${REFACTOR_DIR}/ops/utils/utils.cc

    # refactor template（base + mesh）
    ${REFACTOR_DIR}/ops/template/aicpu/aicpu_base_template.cc
    ${REFACTOR_DIR}/ops/template/aicpu/allgather_mesh.cc
    ${REFACTOR_DIR}/ops/template/primitives/mesh_primitives.cc

    # HCOMM 弱符号 dlsym 实现（仅 1 个，与 algorithm ST 对齐）
    ${REFACTOR_DIR}/ops/c_adaptor/common/hcomm_dlsym/hcomm_primitives_dl.cc

    # HCOMM / HCCL 公共桩（log / error_manager / sal / config_log 等）
    ${REFACTOR_DIR}/ops/c_adaptor/common/log.cc
    ${REFACTOR_DIR}/ops/c_adaptor/common/adapter_error_manager_pub.cc
    ${REFACTOR_DIR}/ops/c_adaptor/common/config_log.cc
    ${REFACTOR_DIR}/ops/c_adaptor/common/sal.cc
    ${REFACTOR_DIR}/ops/c_adaptor/common/utils.cc
)

target_include_directories(scatter_aicpu_kernel PRIVATE
    ${INCLUDE_LIST}
)

target_compile_options(scatter_aicpu_kernel PRIVATE
    -g -O0 -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17
    -fno-common -fno-strict-aliasing -pipe
)

# -DAICPU_COMPILE 与 algorithm ST 的 kernel 库保持一致
target_compile_definitions(scatter_aicpu_kernel PRIVATE
    -DAICPU_COMPILE
)

target_link_options(scatter_aicpu_kernel PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)

target_link_libraries(scatter_aicpu_kernel PRIVATE
    -Wl,--no-as-needed
)
