#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set -u

script_dir=$(cd "$(dirname "$0")" && pwd)
temp_dir=$(mktemp -d)

# 作用：清理回归测试创建的临时目录
# 参数：无
# 返回值：始终返回零
cleanup()
{
    rm -rf "$temp_dir"
}

# 作用：输出失败原因并终止回归测试
# 参数：第一个参数为失败原因
# 返回值：始终以非零状态终止
fail()
{
    echo "FAIL: $1" >&2
    exit 1
}

# 作用：检查指定场景遗留的 rank 进程是否已被回收
# 参数：第一个参数为保存 rank PID 的目录
# 返回值：全部进程已结束时返回零，否则返回非零
assert_no_rank_process()
{
    local pid_dir=$1
    local pid_file
    local pid

    for pid_file in "$pid_dir"/*.pid
    do
        [ -e "$pid_file" ] || continue
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
    done
    return 0
}

# 作用：运行单个启动器回归场景并校验退出状态、日志和进程回收
# 参数：第一个参数为场景名，第二个参数为期望退出状态
# 返回值：所有断言通过时返回零，否则终止测试
run_case()
{
    local scenario=$1
    local expected_status=$2
    local case_dir="$temp_dir/$scenario"
    local status
    local header_count

    mkdir -p "$case_dir/pids"
    cp "$script_dir/run_tensorflow.sh" "$case_dir/run_tensorflow.sh"
    : > "$case_dir/hccl_tensorflow_allreduce_test.py"

    (
        cd "$case_dir" || exit 1
        PATH="$temp_dir/bin:$PATH" TEST_SCENARIO="$scenario" TEST_PID_DIR="$case_dir/pids" \
            bash ./run_tensorflow.sh > launcher.log 2>&1
    )
    status=$?

    if [ "$status" -ne "$expected_status" ]; then
        fail "$scenario expected status $expected_status but got $status"
    fi
    [ -f "$case_dir/tf_train.log" ] || fail "$scenario did not produce tf_train.log"
    header_count=$(grep -c '^------------------ train_[0-7]\.log ------------------$' "$case_dir/tf_train.log")
    [ "$header_count" -eq 8 ] || fail "$scenario log summary is incomplete"
    if compgen -G "$case_dir/train_*.log" >/dev/null; then
        fail "$scenario left rank log files behind"
    fi
    assert_no_rank_process "$case_dir/pids" || fail "$scenario left rank processes running"
}

trap cleanup EXIT
mkdir -p "$temp_dir/bin"

cat > "$temp_dir/bin/sleep" <<'EOF'
#!/bin/bash
/bin/sleep 0.05
EOF

cat > "$temp_dir/bin/python3" <<'EOF'
#!/bin/bash
echo "$$" > "$TEST_PID_DIR/rank-$DEVICE_ID.pid"
case "$TEST_SCENARIO" in
    success)
        echo "device:$DEVICE_ID tensorflow hccl test success"
        exit 0
        ;;
    failure)
        if [ "$DEVICE_ID" -eq 3 ]; then
            echo "device:$DEVICE_ID tensorflow hccl test fail"
            exit 7
        fi
        echo "device:$DEVICE_ID tensorflow hccl test success"
        exit 0
        ;;
    timeout)
        trap '' TERM
        while true
        do
            /bin/sleep 1
        done
        ;;
esac
exit 2
EOF

chmod +x "$temp_dir/bin/sleep" "$temp_dir/bin/python3"

run_case success 0
run_case failure 7
run_case timeout 124

echo "PASS: run_tensorflow.sh regression scenarios"
