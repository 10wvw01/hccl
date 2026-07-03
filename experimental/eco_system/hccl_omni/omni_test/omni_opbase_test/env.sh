HOME_PATH="/home/yhb/local"
RUN_PATH="/home/yhb/omni_opbase_test"
MPI_PATH=/home/hjh/mpich
export MPI_HOME=/home/hjh/mpich

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME_PATH/cann/toolkit/tools/profiler/lib64:$HOME_PATH/cann/toolkit/tools/profiler/lib64:$HOME_PATH/cann/toolkit/tools/aml/lib64:$HOME_PATH/cann/x86_64-linux/lib64:$HOME_PATH/cann/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME_PATH/cann/x86_64-linux/devlib/linux/aarch64:$HOME_PATH/cann/lib64:$HOME_PATH/cann/x86_64-linux/lib64:$HOME_PATH/cann/x86_64-linux/ascendc/include/highlevel_api/lib/hccl/:$HOME_PATH/cann/x86_64-linux/include/experiment/hccl/external/hccl/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_PATH/lib/
export PATH=$PATH:$MPI_PATH/bin

export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH=$RUN_PATH/log
export ASCEND_DIR=$HOME_PATH/cann/x86_64-linux
source $HOME_PATH/cann/set_env.sh
#export HCCL_ALGO="alltoall=level:OMNI"
#export HCCL_ALL_TO_ALL_V_ALGO="OMNI"
export ASCEND_MODULE_LOG_LEVEL=HCCL=1
export HCCL_TEST_USE_DEVS="0,1"
