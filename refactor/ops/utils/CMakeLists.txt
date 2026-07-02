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
#
# 当前状态（全部注释掉，补完后逐个打开）：
#   ❌ data_transfer.cc   - 16 行裸函数声明，没 include 任何东西
#   ❌ plan_ranks.cc      - 7 行伪代码（getNhrRanks/getMeshRanks）
#   ❌ data_copy.cc       - 11 行纯注释

target_sources(hccl PRIVATE
    # data_transfer.cc
    # plan_ranks.cc
    # data_copy.cc
)