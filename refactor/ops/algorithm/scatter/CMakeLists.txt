# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# -----------------------------------------------------------------------------------------------------------

# refactor/ops/algorithm/scatter/CMakeLists.txt
# Scatter 算法描述构造：TemplateDesc 去重表 + AICPU Scatter 算法表
# 由 refactor/ops/algorithm/CMakeLists.txt 通过 add_subdirectory(scatter) 拉进来

target_include_directories(hccl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../../op_common
)

target_sources(hccl PRIVATE
    scatter_template_desc.cc
    algorithm_scatter_aicpu.cc
)
