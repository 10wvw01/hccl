#!/bin/bash
set -e
trap 'echo "❌ Error occurred in build.sh at line $LINENO"; exit 1' ERR

# refactor ST 构建入口
# 用法：bash build.sh [--gtest_filter=...]
# 前置：仓库根已执行 bash build.sh --refactor 编出 build/src/libhccl.so

SHELL_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}) && pwd)

cd $SHELL_DIR
mkdir -p ./build && cd ./build/ && rm -rf ../build/*

# 编译用例工程
cmake .. -DBUILD_OPEN_PROJECT=ON && make -j8

# 运行时 LD_LIBRARY_PATH：
#   - hccl_depends_stub: ST 自带 hcomm/runtime 桩
#   - build/src + refactor/.../hcomm_dlsym: refactor 版 libhccl.so + libhccl_compat.so
#   - CANN lib64: refactor libhccl.so 的依赖（libhcomm/libacl_rt/libc_sec/libunified_dlog）
LIBRARY_DIR="${SHELL_DIR}/build/utils/src/hccl_depends_stub:"
REFACTOR_LIB_DIR="${SHELL_DIR}/../../../build/src:"
REFACTOR_COMPAT_DIR="${SHELL_DIR}/../../../build/src/refactor/ops/c_adaptor/common/hcomm_dlsym:"
export LD_LIBRARY_PATH=${LIBRARY_DIR}${REFACTOR_LIB_DIR}${REFACTOR_COMPAT_DIR}${LD_LIBRARY_PATH}

# 默认只跑 AllGather（refactor 当前唯一实现的算子），可由 $1 覆盖
FILTER="${1:---gtest_filter=ST_ALL_GATHER_*.*}"

${SHELL_DIR}/build/testcase/hccl_checker_ops_stest ${FILTER}

exit 0
