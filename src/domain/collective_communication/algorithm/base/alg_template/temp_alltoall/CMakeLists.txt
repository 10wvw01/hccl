set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/allltoall_pipeline_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/allltoall_pipeline_mesh_pairwise_ping_pong.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/allltoall_pipeline_mesh_pairwise_ccl_enough.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)