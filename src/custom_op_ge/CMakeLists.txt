# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# HCCL custom op .so for GE (EagerExecuteOp 适配层)
# 通过 GE 路径 B (op_graph/lib/<os>/<arch>/*.so) 加载
add_library(custop_hccl SHARED hccl_custom_op.cc)
set_target_properties(custop_hccl PROPERTIES OUTPUT_NAME custop_hccl)

# TODO: 安装路径待正式评审，当前用独立 OPP 包根隔离 libopgraph_hccl.so
set(HCCL_GE_OP_INSTALL_DIR opp/built-in/hccl_ge/op_graph/lib/linux/${CMAKE_SYSTEM_PROCESSOR})

install(TARGETS custop_hccl
    LIBRARY DESTINATION ${HCCL_GE_OP_INSTALL_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)
