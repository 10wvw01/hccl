#!/bin/bash
set -e
#source /home/h30040628/myenv_new/bin/activate
export HCCL_SOCKET_IFNAME=lo

export HCCL_IF_BASE_PORT=50033

export HCCL_OMNI_JIT_CACHE=0
source env.sh
torchrun --nproc_per_node=2 test_full_omni.py -k test_allgather -v
