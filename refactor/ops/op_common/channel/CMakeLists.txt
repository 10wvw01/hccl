# refactor/ops/op_common/channel/CMakeLists.txt
# channel 请求相关函数：CalcChannelRequestMesh1D、CalcChannelRequestNhr 等

target_include_directories(hccl BEFORE PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_sources(hccl PRIVATE
    channel.cc
    channel_request.cc
)
