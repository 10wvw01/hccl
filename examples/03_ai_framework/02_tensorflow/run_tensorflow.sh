#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set -e
server_num=1
single_npus=8
train_file="hccl_tensorflow_allreduce_test.py"
wait_interval=10
max_wait_count=11
terminate_grace=1

export RANK_TABLE_FILE=ranktable.json

pids=()
ranks=()
rank_size=$((server_num * single_npus))
for i in $(seq 0 $((single_npus-1)))
do
		export DEVICE_ID=$i
		export RANK_ID=$i
		export RANK_SIZE=$rank_size
		echo "------------------ train_$i.log ------------------" > train_$i.log
		python3 "$train_file" >> train_$i.log 2>&1 &
		pids+=("$!")
		ranks+=("$i")
done

timed_out=0
for i in $(seq 1 $max_wait_count)
do
		echo "wait ${wait_interval}s, Running......"
		sleep "$wait_interval"
		running_count=0
		for pid in "${pids[@]}"
		do
				if kill -0 "$pid" 2>/dev/null; then
						running_count=$((running_count + 1))
				fi
		done
		if [ "$running_count" -eq 0 ]; then
				break
		fi
		if [ "$i" -eq "$max_wait_count" ]; then
				timed_out=1
		fi
done

if [ "$timed_out" -ne 0 ]; then
		echo "ERROR: TensorFlow ranks timed out, terminating background tasks"
		for pid in "${pids[@]}"
		do
				if kill -0 "$pid" 2>/dev/null; then
						kill "$pid" 2>/dev/null || true
				fi
		done
		sleep "$terminate_grace"
		for pid in "${pids[@]}"
		do
				if kill -0 "$pid" 2>/dev/null; then
						kill -KILL "$pid" 2>/dev/null || true
				fi
		done
fi

overall_status=0
for index in "${!pids[@]}"
do
		if wait "${pids[$index]}"; then
				continue
		else
				rank_status=$?
				echo "ERROR: TensorFlow rank ${ranks[$index]} exited with status $rank_status"
				if [ "$overall_status" -eq 0 ]; then
						overall_status=$rank_status
				fi
		fi
done

if [ "$timed_out" -ne 0 ]; then
		overall_status=124
fi

rm -f tf_train.log
for i in $(seq 0 $((single_npus-1))); do cat "train_$i.log" >> tf_train.log; done
cat tf_train.log
rm -f train_*.log
rm -f *result*.json

exit "$overall_status"
