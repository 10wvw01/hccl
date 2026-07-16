cmake_minimum_required(VERSION 3.14)
project(ascend_topology_aware_allreduce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 指定AICPU编译选项（仿真环境适用）
add_compile_options(-D__AICPU__ -O2 -fPIC)

# 头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ASCEND_CANN_PATH}/include  # 需要设置环境变量或手动指定
)

# 链接库（HCCL、TS引擎、aclrt）
link_directories(${ASCEND_CANN_PATH}/lib)

# 生成静态库（算子最终会被链接到AICPU镜像中）
add_library(allreduce_operator STATIC
    src/all_reduce_impl.cpp
    src/common/topology_utils.cpp
    src/common/reduce_ops.cpp
    src/communicator/ts_wrapper.cpp
    src/algorithm/ring_algorithms.cpp
)

target_include_directories(allreduce_operator PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# 单元测试可执行文件（在Host侧运行）
add_executable(allreduce_test test/test_main.cpp)
target_link_libraries(allreduce_test 
    allreduce_operator
    hccl
    ts_engine
    aclrt
    pthread
)