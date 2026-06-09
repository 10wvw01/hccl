#!/bin/bash
source /home/crx/toolkit/cann-9.1.0/set_env.sh
bash build.sh --build-type=debug
yes y | ./build_out/cann-hccl_9.1.0_linux-x86_64.run --full --install-path=/home/crx/toolkit
bash build.sh --st
