#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
export USE_CCACHE=1
export PATH=/usr/local/ccache/bin:$PATH
export CCACHE_SECONDARY_STORAGE=redis://10.0.0.135:6379
export CCACHE_COMPILERCHECK=content
export CCACHE_SLOPPINESS=include_file_mtime,time_macros,include_file_ctime
export CCACHE_UMASK=002
export CMAKE_CXX_COMPILER_LAUNCHER=/usr/local/ccache/bin/ccache
# export CCACHE_DEBUG=1
# export CCACHE_DEBUG=true
sudo apt update
sudo apt install -y redis-tools
sudo apt install -y libhiredis-dev
sudo apt install libhiredis0.14
redis-cli -h 10.0.0.135 -p 6379 ping && echo "Redis connection OK" || echo "Redis connection FAILED"
/usr/local/ccache/bin/ccache -V
/usr/local/ccache/bin/ccache -z
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
sudo update-alternatives --set gcc /usr/bin/gcc-14
gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e
case "${ut_type}" in
    ut)
        bash build.sh --ut --cann_3rd_lib_path=/home/jenkins/opensource
        ret=$?
        coverage_save="false"
        ;;
    st)
        if [ "${TARGET_BRANCH}x" != "masterx" ]; then
            exit 0
        fi
        pip3 install Pyyaml
        wget -nv https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/cann-hccl_linux-x86_64_ubuntu24.run
        chmod u+x cann-hccl_linux-x86_64_ubuntu24.run
        sudo chmod 777 /home/jenkins/Ascend
        yes "y" | bash cann-hccl_linux-x86_64_ubuntu24.run --full --install-path=/home/jenkins/Ascend
        export ASCEND_HOME_PATH=/home/jenkins/Ascend/cann
        echo "=====1===="
        source /home/jenkins/Ascend/cann/bin/setenv.bash
        echo "=====2===="
        ls -l /home/jenkins/Ascend/cann-9.1.0/share/info/
        echo "=====3===="
        cat /home/jenkins/change_cann_pkgs.log
        echo "=====4===="
        git log -n 10
        echo "=====5===="
        bash build.sh --st > test_st.log
        ret=$?
        coverage_save="false"
        ;;
esac
ret=0

if [ $ret -ne 200 ] && [ $ret -ne 0 ]; then
    echo "run ut fail"
    exit 1
fi
if [ $ret -eq 0 ]; then
    if [ "$coverage_save" = "true" ];then
    echo "ut_process=coverage" >> $ATOMGIT_OUTPUT
    else
    echo "ut_process=ut_cov" >> $ATOMGIT_OUTPUT
    fi
fi
exit 0
