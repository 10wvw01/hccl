#!/bin/bash

set -e

source /home/yhb/local/torch-venv/bin/activate

export HCCL_SOCKET_IFNAME=lo
export HCCL_IF_BASE_PORT=50033

source env.sh

if [[ $# -ne 1 ]] || [[ ! "$1" =~ ^[0-1]$ ]]; then
    echo "Usage: $0 {0|1}"
    echo "  0: alltoallv_aicpu_ts_test (torchrun)"
    echo "  1: test_jit_api.py — hccl_omni JIT API (torchrun)"
    exit 1
fi

case $1 in
    0)
        echo "alltoallv test not supported"
        ;;
    1)
        torchrun --nproc_per_node=2 test_jit_api.py
        ;;
esac
