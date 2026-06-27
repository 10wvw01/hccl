# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/executor/CMakeLists.txt
# Executor 基类与子类实现
# 由 refactor/CMakeLists.txt 通过 add_subdirectory(ops/executor) 拉进来
#
# 当前状态（全部注释掉，蔡鑫补完后逐个打开）：
#   ⚠️ base_executor.cc                - 主体代码（145+110 行），但 ctor 引用了未声明的 baseOpParam
#   ⚠️ sole/sole_executor.cc           - h 是 7 行空类，cc 引用了未声明的 algTemplate/SaveCtx/dataTypeSize_
#   ❌ parallel/parallel_executor.cc    - 含 if(){} 语法错误，无法编译
#   ❌ concurrent/base_concurrent_executor.cc  - 空文件
#   ❌ omnipipe/base_omnipipe_executor.cc      - 空文件

target_sources(hccl PRIVATE
    # base_executor.cc
    # sole/sole_executor.cc
    # parallel/parallel_executor.cc
    # concurrent/base_concurrent_executor.cc
    # omnipipe/base_omnipipe_executor.cc
)