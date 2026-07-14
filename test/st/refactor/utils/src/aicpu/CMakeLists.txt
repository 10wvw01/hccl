# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# refactor ST 的 aicpu kernel 库：最小桩
#
# refactor 的算子实现在 build/src/libhccl.so 里（由 bash build.sh --refactor 产出），
# 这里只提供空 .so 满足 hccl-model-proxy 的链接依赖，以及 src 签名的 HcclLaunchAicpuKernel 空实现
# （aclrt_stub.cc 引用，refactor 走同步路径实际不调它）。
# -----------------------------------------------------------------------------------------------------------

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/scatter_aicpu_kernel_stub.cc
    "#include \"hccl/hccl_types.h\"\n"
    "extern \"C\" unsigned int HcclLaunchAicpuKernel(void *param) { (void)param; return 0; }\n"
)

add_library(scatter_aicpu_kernel SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/scatter_aicpu_kernel_stub.cc
)

target_include_directories(scatter_aicpu_kernel PRIVATE
    ${ASCEND_HOME_PATH}/include
)

target_compile_options(scatter_aicpu_kernel PRIVATE
    -g -O0 -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
)
