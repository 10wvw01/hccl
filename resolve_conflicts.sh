#!/bin/bash
set -e
cd /home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039

echo "=== Strategy A: take cann/master for 9 files ==="
git checkout --theirs \
  src/ops/all_gather/executor/ins_v2_all_gather_sequence_executor_3level.cc \
  src/ops/all_gather/executor/ins_v2_all_gather_sequence_executor_3level.h \
  src/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor_aicpu_3level.cc \
  src/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor_aicpu_3level.h \
  src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_3level.cc \
  src/ops/all_gather/selector/all_gather_auto_selector.cc \
  src/ops/all_reduce/selector/all_reduce_auto_selector.cc \
  src/ops/reduce_scatter/selector/reduce_scatter_auto_selector.cc \
  src/common/hcomm_dlsym/hccl_res_dl.h
git add \
  src/ops/all_gather/executor/ins_v2_all_gather_sequence_executor_3level.cc \
  src/ops/all_gather/executor/ins_v2_all_gather_sequence_executor_3level.h \
  src/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor_aicpu_3level.cc \
  src/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor_aicpu_3level.h \
  src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_3level.cc \
  src/ops/all_gather/selector/all_gather_auto_selector.cc \
  src/ops/all_reduce/selector/all_reduce_auto_selector.cc \
  src/ops/reduce_scatter/selector/reduce_scatter_auto_selector.cc \
  src/common/hcomm_dlsym/hccl_res_dl.h
echo "Done A"

echo "=== Strategy B: topo_host.cc take cann/master ==="
git checkout --theirs src/ops/op_common/topo/topo_host.cc
git add src/ops/op_common/topo/topo_host.cc
echo "Done B1"

echo "=== Strategy B: AG 3level testcase take cann/master ==="
git checkout --theirs test/st/algorithm/testcase/all_gather_3level_testcase.cc
git add test/st/algorithm/testcase/all_gather_3level_testcase.cc
echo "Done B2"

echo "=== Strategy C: broadcast CMakeLists — merge both entries ==="
cat > src/ops/broadcast/executor/CMakeLists.txt << 'CMAKE_EOF'
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/ins_v2_broadcast_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ins_v2_broadcast_parallel_executor.cc
)
if(NOT HCCL_CANN_COMPAT_850)
    list(APPEND src_list
        ${CMAKE_CURRENT_SOURCE_DIR}/ins_v2_broadcast_sequence_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ins_v2_broadcast_sequence_executor_3level.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ins_v2_broadcast_omnipipe_2d_executor.cc
    )
endif()

if(TARGET hccl)
    target_sources(hccl PRIVATE
        ${src_list}
    )
endif()
CMAKE_EOF
git add src/ops/broadcast/executor/CMakeLists.txt
echo "Done C"

echo "=== Strategy D: testcase CMakeLists — take cann/master + add our test files ==="
git checkout --theirs test/st/algorithm/testcase/CMakeLists.txt
git add test/st/algorithm/testcase/CMakeLists.txt
echo "Done D"

echo "=== Verify no remaining conflicts ==="
git diff --name-only --diff-filter=U
echo "=== Conflict resolution done ==="
