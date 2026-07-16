# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/algorithm/CMakeLists.txt
# 算法描述构造模块分发层：按算子类型分发

option(REFACTOR_ALGORITHM "Build refactor/ops/algorithm" OFF)

if(REFACTOR_ALGORITHM AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/all_gather/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/all_gather
                     ${CMAKE_CURRENT_BINARY_DIR}/all_gather)
endif()
