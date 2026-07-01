# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

#!/bin/bash
set -e
trap 'echo "Error occurred in build.sh at line $LINENO"; exit 1' ERR

# 获取shell脚本目录作为根目录
SHELL_DIR=$(cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

# 获取CPU核数用于并发编译和执行
CPU_NUM=$(cat /proc/cpuinfo | grep "^processor" | wc -l)

# 创建build编译目录
cd $SHELL_DIR
mkdir -p ./build && cd ./build/ && rm -rf ../build/*

# 配置cmake参数
CMAKE_ARGS="-DBUILD_OPEN_PROJECT=ON"
if [ "${ENABLE_GCOV}" == "on" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GCOV=ON"
fi
if [ -n "${ST_TASKS}" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DST_TASKS=${ST_TASKS}"
fi

echo "CMAKE_ARGS=${CMAKE_ARGS}"
cmake .. ${CMAKE_ARGS}

# 并行编译所有目标
echo "Building ST targets with ${CPU_NUM} parallel jobs..."
cmake --build . -j ${CPU_NUM}

# 设置运行时库搜索路径
LIBRARY_PATHS="${SHELL_DIR}/build/utils/src"
LIBRARY_PATHS="${LIBRARY_PATHS}:${SHELL_DIR}/build/utils/src/hccl_verifier"
LIBRARY_PATHS="${LIBRARY_PATHS}:${SHELL_DIR}/build/utils/src/hccl_depends_stub"
LIBRARY_PATHS="${LIBRARY_PATHS}:${SHELL_DIR}/build/utils/src/aicpu"
export LD_LIBRARY_PATH="${LIBRARY_PATHS}:${LD_LIBRARY_PATH}"

# 并行执行所有测试用例
echo "Running ST tests with ${CPU_NUM} parallel jobs..."
set +e
ctest -j ${CPU_NUM} --timeout 300 --output-on-failure 2>&1 | tee ctest_run.log
ctest_ret=${PIPESTATUS[0]}
set -e

if [ ${ctest_ret} -ne 0 ]; then
    echo "Some ST tests failed (exit code: ${ctest_ret})"
fi

exit ${ctest_ret}
