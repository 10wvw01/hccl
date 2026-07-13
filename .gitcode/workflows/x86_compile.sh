#!/bin/bash

whoami
echo "whoami"
cd ${WORKSPACE}
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if [[ "${task_name}" == *ubuntu24* ]]; then
    sudo update-alternatives --set gcc /usr/bin/gcc-14
    sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
fi
gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e

echo "exec cmd: [bash build.sh --pkg --cann_3rd_lib_path="/home/jenkins/opensource" --sign-script scripts/sign/community_sign_build.py --full -p /home/jenkins/Ascend/cann]"
bash build.sh --pkg --cann_3rd_lib_path="/home/jenkins/opensource" --sign-script scripts/sign/community_sign_build.py --full -p /home/jenkins/Ascend/cann

ret=$?

exit $ret
