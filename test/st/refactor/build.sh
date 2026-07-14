#!/bin/bash
set -e
trap 'echo "❌ Error occurred in build.sh at line $LINENO"; exit 1' ERR

# refactor ST 构建入口
# 用法：bash build.sh [--gtest_filter=...]
# 会自动先构建仓库根的 libhccl.so（refactor 版本）

SHELL_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}) && pwd)
REPO_ROOT=$(cd ${SHELL_DIR}/../../.. && pwd)

# 步骤1：构建 refactor 版 libhccl.so
echo "==> Building libhccl.so (refactor)..."
cd ${REPO_ROOT}
bash build.sh --refactor

# 步骤2：编译 ST 用例工程
echo "==> Building ST testcases..."
cd ${SHELL_DIR}
mkdir -p ./build && cd ./build/ && rm -rf ../build/*
cmake .. -DBUILD_OPEN_PROJECT=ON && make -j8

# 步骤3：运行测试
LIBRARY_DIR="${SHELL_DIR}/build/utils/src/hccl_depends_stub:"
REFACTOR_LIB_DIR="${REPO_ROOT}/build/src:"
REFACTOR_COMPAT_DIR="${REPO_ROOT}/build/src/refactor/ops/c_adaptor/common/hcomm_dlsym:"
export LD_LIBRARY_PATH=${LIBRARY_DIR}${REFACTOR_LIB_DIR}${REFACTOR_COMPAT_DIR}${LD_LIBRARY_PATH}

# 默认只跑 AllGather（refactor 当前唯一实现的算子），可由 $1 覆盖
FILTER="${1:---gtest_filter=ST_ALL_GATHER_*.*}"

${SHELL_DIR}/build/testcase/hccl_checker_ops_stest ${FILTER}

exit 0
