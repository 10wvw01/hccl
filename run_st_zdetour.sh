#!/bin/bash
set -e
export ASCEND_HOME_PATH=/home/crx/toolkit/cann-9.1.0
source $ASCEND_HOME_PATH/set_env.sh 2>/dev/null
cd /home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039

echo "=== Building HCCL library ==="
bash build.sh -p /home/crx/toolkit/cann-9.1.0 2>&1 | tail -5

echo "=== Installing libhccl.so ==="
chmod u+w /home/crx/toolkit/cann-9.1.0/x86_64-linux/lib64/libhccl.so
cp build/src/libhccl.so /home/crx/toolkit/cann-9.1.0/x86_64-linux/lib64/libhccl.so

echo "=== Building ST tests ==="
cd test/st/algorithm
mkdir -p ./build && cd ./build && rm -rf ./*
cmake .. -DBUILD_OPEN_PROJECT=ON > /tmp/st_cmake.log 2>&1
make -j8 > /tmp/st_make.log 2>&1
echo "=== ST build done ==="

echo "=== Running broadcast 3level tests ==="
STBUILD=/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/test/st/algorithm/build
export LD_LIBRARY_PATH=$STBUILD/utils/src/hccl_depends_stub:$STBUILD/utils/src/aicpu:$STBUILD/utils/src/hccl_verifier:$STBUILD/utils/src/hccl_proxy:$LD_LIBRARY_PATH
$STBUILD/testcase/hccl_checker_ops_stest --gtest_filter="ST_BROADCAST_3LEVEL_TEST.*" --gtest_brief=1 2>&1
echo "=== EXIT=$? ==="
