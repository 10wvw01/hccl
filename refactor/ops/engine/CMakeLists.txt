# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not be in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/engine 模块。
# 引擎 Engine 层：base_engine + aicpu/aiv/ccu_ms/ccu_sche 子引擎。
# 子目录各自维护 CMakeLists.txt，通过 add_subdirectory 累加 include 路径与源文件到共享 hccl target。

target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
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

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/aicpu
                 ${CMAKE_CURRENT_BINARY_DIR}/aicpu)
