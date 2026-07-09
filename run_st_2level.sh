#!/bin/bash
export ASCEND_HOME_PATH=/home/crx/toolkit/cann-9.1.0
source $ASCEND_HOME_PATH/set_env.sh 2>/dev/null
cd /home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039

echo "=== Building HCCL ==="
bash build.sh -p /home/crx/toolkit/cann-9.1.0 > /tmp/build_2l.log 2>&1
echo "BUILD_EXIT=$?"
chmod u+w /home/crx/toolkit/cann-9.1.0/x86_64-linux/lib64/libhccl.so
cp build/src/libhccl.so /home/crx/toolkit/cann-9.1.0/x86_64-linux/lib64/libhccl.so

cd test/st/algorithm/build && rm -rf ./*
cmake .. -DBUILD_OPEN_PROJECT=ON > /tmp/st_cmake_2l.log 2>&1
make -j8 > /tmp/st_make_2l.log 2>&1
echo "ST_EXIT=$?"

STBUILD=/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/test/st/algorithm/build
export LD_LIBRARY_PATH=$STBUILD/utils/src/hccl_depends_stub:$STBUILD/utils/src/aicpu:$STBUILD/utils/src/hccl_verifier:$STBUILD/utils/src/hccl_proxy:$LD_LIBRARY_PATH

echo "=== Running 2-level tests ==="
$STBUILD/testcase/hccl_checker_ops_stest --gtest_filter="ST_BROADCAST_2LEVEL_TEST.*" --gtest_brief=1 2>&1 | tail -5
echo "=== Running 3-level tests ==="
$STBUILD/testcase/hccl_checker_ops_stest --gtest_filter="ST_BROADCAST_3LEVEL_TEST.*" --gtest_brief=1 2>&1 | tail -5
