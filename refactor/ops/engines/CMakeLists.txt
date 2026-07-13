# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/engines/CMakeLists.txt
# 引擎 Launcher 模块：AICPU / AIV / CCU_MS / CCU_SCHE
# 由 refactor/ops/CMakeLists.txt 通过 add_subdirectory(engines) 拉进来
#
# 当前状态：
#   ✅ aicpu/aicpu_launcher.cc   - AICPU 引擎 Launcher 实现
#   ✅ aicpu/kernel_launch.cc    - AICPU kernel 下发入口
#   ✅ aicpu/load_kernel.cc      - AICPU kernel 二进制加载
#   ⚠️ ccu_ms/ccu_ms_launcher.cc - 依赖 ccu_ms_kernel_launch.cc（未通过编译）
#   ⚠️ ccu_ms/ccu_ms_kernel_launch.cc - 使用未声明的 OpsExecutorParam、Orchestrate 签名不匹配
#   ⚠️ ccu_ms/ccu_ms_res.cc      - 引用 HcclChannelAcquire/HcclCommQueryCcuIns 等 CCU dlsym 符号
#   ⚠️ ccu_sche/ccu_sche_launcher.cc      - 同 ccu_ms
#   ⚠️ ccu_sche/ccu_sche_kernel_launch.cc - 同 ccu_ms
#   ⚠️ ccu_sche/ccu_sche_res.cc           - 同 ccu_ms
#   ⚠️ aiv/load_kernel.cc        - 与 aicpu/load_kernel.cc 完全相同，会导致 g_binKernelHandle 重复定义
#   ⚠️ aiv/kernel_launch.cc      - 旧版 extern "C" 实现，引用旧版 executor_base.h/coll_alg_exec_registry.h

target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/aicpu
    ${CMAKE_CURRENT_LIST_DIR}/ccu_ms
    ${CMAKE_CURRENT_LIST_DIR}/ccu_sche
    ${CMAKE_CURRENT_LIST_DIR}/aiv
    ${CMAKE_CURRENT_LIST_DIR}/../op_common
    ${CMAKE_CURRENT_LIST_DIR}/../op_common/inc
    ${CMAKE_CURRENT_LIST_DIR}/../executor
    ${CMAKE_CURRENT_LIST_DIR}/../template
    ${CMAKE_CURRENT_LIST_DIR}/../c_adaptor/common
    ${CMAKE_CURRENT_LIST_DIR}/../c_adaptor/common/hcomm_dlsym
    ${HCCL_DIR}/src/common
    ${HCCL_DIR}/src/common/hcomm_dlsym
    ${HCCL_DIR}/src/ops/op_common
    ${HCCL_DIR}/src/ops/op_common/inc
    ${HCCL_DIR}/src/ops/op_common/executor
    ${HCCL_DIR}/src/ops/op_common/executor/registry
    ${HCCL_DIR}/src/ops/op_common/template
    ${HCCL_DIR}/src/ops/op_common/template/wrapper
)

target_sources(hccl PRIVATE
    aicpu/aicpu_launcher.cc
    aicpu/kernel_launch.cc
    aicpu/load_kernel.cc

    aiv/kernel_launch.cc
    aiv/load_kernel.cc

    ccu_ms/ccu_ms_launcher.cc
    ccu_ms/ccu_ms_kernel_launch.cc
    ccu_ms/ccu_ms_res.cc

    ccu_sche/ccu_sche_launcher.cc
    ccu_sche/ccu_sche_kernel_launch.cc
    ccu_sche/ccu_sche_res.cc
)
