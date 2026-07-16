# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/utils/CMakeLists.txt
# 工具函数：数据搬运（SendRecv*/PreCopy/PostCopy）+ rank 规划
# 由 refactor/CMakeLists.txt 通过 add_subdirectory(ops/utils) 拉进来

target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../template
    ${CMAKE_CURRENT_LIST_DIR}/../template/primitives
    ${CMAKE_CURRENT_LIST_DIR}/../op_common
    ${CMAKE_CURRENT_LIST_DIR}/../op_common/inc
    ${CMAKE_CURRENT_LIST_DIR}/../c_adaptor/common
    ${CMAKE_CURRENT_LIST_DIR}/../c_adaptor/common/hcomm_dlsym
    ${HCCL_DIR}/src/ops/op_common
    ${HCCL_DIR}/src/ops/op_common/inc
    ${HCCL_DIR}/src/common
    ${HCCL_DIR}/src/common/hcomm_dlsym
)

target_sources(hccl PRIVATE
    utils.cc
)