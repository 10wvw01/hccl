# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor 总入口。
# src/CMakeLists.txt 只有在 REFACTOR_OPS=ON 时才 include 本文件，所以这里默认不影响主工程。
# 本层只负责把 refactor/ops 接入 CMake build tree，具体源码由下一层按模块开关控制。

if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/ops/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/ops
                     ${CMAKE_CURRENT_BINARY_DIR}/refactor/ops)
endif()
