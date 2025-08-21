set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_recursive_hd.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nhr_oneshot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nhr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nhr_v1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_ahc.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_ahc_broke.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_reduce_broadcast.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_mesh_opbase.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_mesh_oneshot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_doubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_doubling_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_local_reduce_bcast.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_local_reduce.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_chunk_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_opbase_pipeline.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_hd_optim.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_graph_pipeline.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)