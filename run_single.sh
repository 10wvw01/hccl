#!/bin/bash
export ASCEND_HOME_PATH=/home/crx/toolkit/cann-9.1.0
source $ASCEND_HOME_PATH/set_env.sh 2>/dev/null
STBUILD=/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/test/st/algorithm/build
export LD_LIBRARY_PATH=$STBUILD/utils/src/hccl_depends_stub:$STBUILD/utils/src/aicpu:$STBUILD/utils/src/hccl_verifier:$STBUILD/utils/src/hccl_proxy:$LD_LIBRARY_PATH
$STBUILD/testcase/hccl_checker_ops_stest --gtest_filter="ST_BROADCAST_TEST.st_broadcast_a5_aicpu_Mesh1DNHR_bigdata_test32" --gtest_brief=1 > /tmp/st_result.txt 2>&1
EXIT=$?
echo "Exit code: $EXIT"
echo "=== Test result ==="
grep -E "PASSED|FAILED|RUN|tests from" /tmp/st_result.txt
echo "=== Errors (if any) ==="
grep -iE "error|fail|conflict|missing|semantic" /tmp/st_result.txt | head -20
