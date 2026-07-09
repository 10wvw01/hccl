#!/bin/bash
export ASCEND_HOME_PATH=/home/crx/toolkit/cann-9.1.0
source $ASCEND_HOME_PATH/set_env.sh 2>/dev/null
STBUILD=/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/test/st/algorithm/build
export LD_LIBRARY_PATH=$STBUILD/utils/src/hccl_depends_stub:$STBUILD/utils/src/aicpu:$STBUILD/utils/src/hccl_verifier:$STBUILD/utils/src/hccl_proxy:$LD_LIBRARY_PATH

for SUITE in "ST_BROADCAST_3LEVEL_TEST" "ST_BROADCAST_TEST" "ST_ALL_GATHER_3LEVEL_TEST" "ST_ALL_REDUCE_MULTILEVEL_TEST" "ST_ALL_REDUCE_TEST" "ST_REDUCE_SCATTER_3LEVEL_TEST" "ST_REDUCE_SCATTER_AICPU_TEST" "ST_REDUCE_TEST" "ST_SCATTER_TEST"; do
  $STBUILD/testcase/hccl_checker_ops_stest --gtest_filter="${SUITE}.*" --gtest_brief=1 > /tmp/reg_out.txt 2>&1
  RESULT=$(grep -E "PASSED|FAILED" /tmp/reg_out.txt | tail -1)
  echo "${SUITE}: ${RESULT}"
done
