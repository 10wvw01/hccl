#!/bin/bash


cd ${WORKSPACE}
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if [[ "${task_name}" == *ubuntu24* ]]; then
    sudo update-alternatives --set gcc /usr/bin/gcc-14
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
fi
gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e

echo "exec cmd: [bash build.sh --pkg --examples --cann_3rd_lib_path="/home/jenkins/opensource"]"
bash build.sh --pkg --examples --cann_3rd_lib_path="/home/jenkins/opensource"

ret=$?

cd build
tar -zcf examples.tar.gz examples
cp -rf examples.tar.gz ${WORKSPACE}/build_out
cd -

exit $ret