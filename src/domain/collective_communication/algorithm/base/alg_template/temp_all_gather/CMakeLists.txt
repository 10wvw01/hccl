set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/aligned_all_gather_double_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_halving_doubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_mesh_atomic.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_mesh_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_recursive_hd.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_ring_concurrent_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_ring_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_slim_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_mesh_mix.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_graph_pipeline.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_nhr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_nhr_v1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_pipeline.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_unified_march.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_nb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_ahc.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_ahc_broke.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_hd_stage.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_sio_hccs.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)