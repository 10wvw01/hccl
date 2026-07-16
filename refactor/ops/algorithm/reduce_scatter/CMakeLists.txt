# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/algorithm/reduce_scatter/CMakeLists.txt
# ReduceScatter 算法描述构造：TemplateDesc 去重表 + AICPU ReduceScatter 算法表
# 由 refactor/ops/algorithm/CMakeLists.txt 通过 add_subdirectory(reduce_scatter) 拉进来

target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../../op_common
)

target_sources(hccl PRIVATE
    reduce_scatter_template_desc.cc
    algorithm_reduce_scatter_aicpu.cc
)
