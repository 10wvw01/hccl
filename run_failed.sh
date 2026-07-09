#!/bin/bash
export ASCEND_HOME_PATH=/home/crx/toolkit/cann-9.1.0
source $ASCEND_HOME_PATH/set_env.sh 2>/dev/null
STBUILD=/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/test/st/algorithm/build
export LD_LIBRARY_PATH=$STBUILD/utils/src/hccl_depends_stub:$STBUILD/utils/src/aicpu:$STBUILD/utils/src/hccl_verifier:$STBUILD/utils/src/hccl_proxy:$LD_LIBRARY_PATH
$STBUILD/testcase/hccl_checker_ops_stest --gtest_filter="ST_BROADCAST_3LEVEL_TEST.st_broadcast_3level_2x2x2_fp32_send201" --gtest_brief=1 2>&1 | grep -iE "error|fail|missing|conflict|total size|semantic" | head -10
