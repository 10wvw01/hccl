set(HCCL_PACKET_FLAG 1)

if(BUILD_OPEN_PROJECT)
    message(STATUS "open TOP_DIR = ${TOP_DIR}")
    set(HCCL_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
    set(HCCL_OPEN_CODE_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
    set(HCCL_THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../../)
else()
    message(STATUS "close TOP_DIR = ${TOP_DIR}")
    set(HCCL_BASE_DIR ${TOP_DIR}/ace/comop/hccl/whole/hccl/)
    set(HCCL_OPEN_CODE_BASE_DIR ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/)
    set(HCCL_THIRD_PARTY_DIR ${TOP_DIR})
endif()

set(HCCL_NEXT_CODE_ROOT ${TOP_DIR}/ace/comop/hccl/whole/hccl_next)

message(STATUS "HCCL_OPEN_CODE_BASE_DIR = ${HCCL_OPEN_CODE_BASE_DIR}")
message(STATUS "HCCL_BASE_DIR = ${HCCL_BASE_DIR}")
add_library(hccl  SHARED)

set(hccl_include_list
    ${HCCL_BASE_DIR}/algorithm/base/inc
    ${HCCL_BASE_DIR}/algorithm/pub_inc
    ${HCCL_BASE_DIR}/pub_inc
    ${HCCL_BASE_DIR}/pub_inc/hccl
    ${HCCL_BASE_DIR}/algorithm/base/alg_aiv_template
    ${HCCL_BASE_DIR}/algorithm/base/alg_template
    ${HCCL_BASE_DIR}/algorithm/base/mc2_handler
    ${HCCL_BASE_DIR}/algorithm/base/alg_template/component
    ${HCCL_BASE_DIR}/algorithm/base/communicator
    ${HCCL_BASE_DIR}/algorithm/base/communicator/legacy
    ${HCCL_BASE_DIR}/algorithm/impl
    ${HCCL_BASE_DIR}/algorithm/impl/legacy
    ${HCCL_BASE_DIR}/algorithm/impl/inc
    ${HCCL_BASE_DIR}/algorithm/impl/task
    ${HCCL_BASE_DIR}/algorithm/impl/resource_manager
    ${HCCL_BASE_DIR}/algorithm/impl/coll_executor

    ${HCCL_BASE_DIR}/framework/cluster_maintenance/health/heartbeat/
    ${HCCL_BASE_DIR}/framework/cluster_maintenance/recovery/operator_retry/
    ${HCCL_BASE_DIR}/framework/cluster_maintenance/detect/detect_connect_anomalies/
    ${HCCL_BASE_DIR}/framework/common/src/
    ${HCCL_BASE_DIR}/framework/common/src/host
    ${HCCL_BASE_DIR}/framework/common/src/config
    ${HCCL_BASE_DIR}/framework/common/src/topo
    ${HCCL_BASE_DIR}/framework/common/src/task
    ${HCCL_BASE_DIR}/framework/common/src/exception
    ${HCCL_BASE_DIR}/framework/common/src/thread
    ${HCCL_BASE_DIR}/framework/common/src/h2d_dto
    ${HCCL_BASE_DIR}/framework/common/src/onesided_memory_management
    ${HCCL_BASE_DIR}/framework/common/src/mgr
    ${HCCL_BASE_DIR}/framework/communicator/impl/
    ${HCCL_BASE_DIR}/framework/communicator/impl/resource_manager
    ${HCCL_BASE_DIR}/framework/communicator/impl/one_sided_service
    ${HCCL_BASE_DIR}/framework/communicator/impl/zero_copy_memory_agent
    ${HCCL_BASE_DIR}/framework/hcom/
    ${HCCL_BASE_DIR}/framework/inc/
    ${HCCL_BASE_DIR}/framework/nslbdp/
    ${HCCL_BASE_DIR}/framework/op_base/src
    ${HCCL_BASE_DIR}/common/health/
    ${HCCL_BASE_DIR}/common/stream/
)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hccl PRIVATE
        OPEN_BUILD_PROJECT
        LOG_CPP)

    target_include_directories(hccl PRIVATE
        ${HCCL_BASE_DIR}/../../../inc
        ${HCCL_BASE_DIR}/../../../inc/hccl
        ${HCCL_BASE_DIR}/../../../hccl/
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util/error_manager/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog/toolchain/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/mmpa
        ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/highlevel_api/
        ${THIRD_PARTY_NLOHMANN_PATH}
        ${HCCL_BASE_DIR}/common/debug/profiling/inc
        ${HCCL_BASE_DIR}/common/debug/profiling/inc/host
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/debug/config
    )

    target_link_directories(hccl PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )
else()
    target_include_directories(hccl PRIVATE
        ${TOP_DIR}/ace/comop/inc/external/hccl/             # ${ASCEND_CANN_PACKAGE_PATH}/include
        ${TOP_DIR}/ace/comop/inc/hccl/                      # ${ASCEND_CANN_PACKAGE_PATH}/include
        ${TOP_DIR}/ace/comop/hccl/whole/hccl/algorithm/pub_inc
        ${TOP_DIR}/abl/msprof/inc                           # experiment/msprof/
        ${TOP_DIR}/ace/npuruntime/inc                       # experiment/runtime/
        ${TOP_DIR}/inc/driver                               # experiment/ascend_hal/driver/
        ${TOP_DIR}/inc/external                             # experiment
        ${TOP_DIR}/metadef/inc/common/util/error_manager    # experiment/metadef/inc/common/util/error_manager/
        ${TOP_DIR}/metadef/inc/external/
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/abl/slog/inc/toolchain                   # experiment/slog/
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/abl/mmpa/inc/mmpa
        ${TOP_DIR}/ace/comop/hccl/
        ${HCCL_BASE_DIR}/common/debug/profiling/inc
        ${HCCL_BASE_DIR}/common/debug/profiling/inc/host
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/debug/config
    )
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_compile_options(hccl PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O3
    -std=c++14
    -fstack-protector-all
)

target_link_options(hccl PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    -s
)

target_compile_definitions(hccl PRIVATE
    $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
)
target_include_directories(hccl PRIVATE
    ${hccl_include_list}
)

if(BUILD_OPEN_PROJECT)
    target_link_libraries(hccl
            -Wl,--no-as-needed
            c_sec
            ascendalog
            ascendcl
            hccl_alg
            hccl_plf
            -Wl,--as-needed
            -lrt
            -ldl
            -lpthread
    )
else()
    target_link_libraries(hccl
            $<BUILD_INTERFACE:kernel_tiling_headers>
            $<BUILD_INTERFACE:slog_headers>
            $<BUILD_INTERFACE:mmpa_headers>
            -Wl,--no-as-needed
            ascendcl
            c_sec
            slog
            mmpa
            hccl_alg
            hccl_plf
            -Wl,--as-needed
            -lrt
            -ldl
            -lpthread
    )
endif()

install(TARGETS hccl
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)

message(STATUS "framework KERNEL_MODE=${KERNEL_MODE}")
message(STATUS "framework BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")
message(STATUS "framework ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")

if(NOT BUILD_OPEN_PROJECT OR (BUILD_OPEN_PROJECT AND KERNEL_MODE))
    add_library(ccl_kernel  SHARED)

    set(CCL_KERNEL_SRC_LIST
        ${HCCL_BASE_DIR}/algorithm/impl/alg_env_config.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_alg_utils.cc
        ${HCCL_BASE_DIR}/algorithm/impl/hccl_alg_device.cc
        ${HCCL_BASE_DIR}/algorithm/impl/hccl_aiv.cc
        ${HCCL_BASE_DIR}/algorithm/impl/hccl_aiv_device.cc
        ${HCCL_BASE_DIR}/algorithm/impl/topo_matcher.cc
        ${HCCL_BASE_DIR}/algorithm/impl/resource_manager/ccl_buffer_manager.cc
        ${HCCL_BASE_DIR}/algorithm/impl/resource_manager/stream_active_manager.cc
        ${HCCL_BASE_DIR}/algorithm/impl/resource_manager/workspace_mem.cc
        ${HCCL_BASE_DIR}/algorithm/impl/resource_manager/workspace_resource.cc
        ${HCCL_BASE_DIR}/algorithm/impl/resource_manager/workspace_resource_impl.cc
        ${HCCL_BASE_DIR}/algorithm/impl/resource_manager/op_base_stream_manager.cc
        ${HCCL_BASE_DIR}/algorithm/impl/resource_manager/offload_stream_manager.cc
        ${HCCL_BASE_DIR}/algorithm/impl/resource_manager/hccl_socket_manager.cc
        ${HCCL_BASE_DIR}/algorithm/impl/resource_manager/queue_notify_manager.cc

        ${HCCL_BASE_DIR}/algorithm/base/communicator/legacy/comm_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/legacy/comm_ring.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/legacy/comm_p2p.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/legacy/comm_mesh.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/legacy/comm_halving_doubling.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/legacy/comm_star.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_hd_transport_req.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_mesh_transport_req.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_partial_mesh_transport_req.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_ring_transport_req.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_transport_req_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_p2p_transport_req.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_hccs_plus_sio_transport_req.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_ahc_broke_transport_req.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_ahc_transport_req.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_ahc_transport_req_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_nb_transport_req.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_nhr_transport_req.cc
        ${HCCL_BASE_DIR}/algorithm/base/communicator/calc_nhr_v1_transport_req.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/alg_template_register.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/alg_template_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/nonuniform_hierarchical_ring_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/nonuniform_hierarchical_ring_v1_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/nonuniform_bruck_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/asymmetric_hierarchical_concatenate_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/asymmetric_hierarchical_concatenate_alg_template_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/asymmetric_hierarchical_concatenate_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/recursive_halvingdoubling_base.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/component/reducer.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/component/sender.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/aligned_all_gather_double_ring.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_ring_concurrent_direct.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_halving_doubling.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_mesh.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_ring.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_mesh_direct.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_mesh_atomic.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_mesh_mix.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_recursive_hd.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_nhr.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_nhr_v1.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_pipeline.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_unified_march.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_nb.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_ahc.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_ahc_broke.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_hd_stage.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather/all_gather_sio_hccs.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_recursive_hd.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_ring.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_nhr_oneshot.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_nhr.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_nhr_v1.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_ahc.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_ahc_broke.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_nb.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_reduce_broadcast.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_mesh_opbase.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_mesh_oneshot.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_doubling.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_doubling_direct.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_local_reduce_bcast.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_local_reduce.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_chunk_mesh.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_opbase_pipeline.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_hd_optim.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce/all_reduce_graph_pipeline.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoall/allltoall_pipeline_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoall/allltoall_pipeline_mesh_pairwise_ping_pong.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoall/allltoall_pipeline_mesh_pairwise_ccl_enough.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoallv/alltoallv_pairwise.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoallv/alltoallv_staged_base.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoallv/alltoallv_staged_pairwise.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoallv/alltoallv_staged_mesh.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoallv/alltoallv_staged_calculator.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoallv/alltoallv_mesh_read_only.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoallv/alltoallv_for_310p.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoallv/alltoallv_direct_fullmesh.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_broadcast/bcast_halvingdoubling.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_broadcast/bcast_recursive_halvingdoubling.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_broadcast/broadcast_ring.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_broadcast/broadcast_star.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_broadcast/broadcast_nhr_oneshot.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_broadcast/broadcast_nhr.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_broadcast/broadcast_nhr_v1.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_broadcast/broadcast_nb.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_broadcast/broadcast_nb_binary.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_broadcast/broadcast_oneshot.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce/reduce_ring.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce/reduce_recursive_hd.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce/reduce_nhr_oneshot.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/aligned_reduce_scatter_double_ring_with_serial_local_copy.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/aligned_reduce_scatter_double_ring.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_ring_concurrent_direct.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_halving_doubling.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_recursive_hd.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_ring.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh_atomic.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh_mix.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh_mix_single_stream.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh_atomic_opbase.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_nhr.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_nhr_v1.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_nb.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_ahc.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_ahc_broke.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_hd_stage.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_local_reduce.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_pipeline.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_unified_march.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_hccs_sio.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_scatter/scatter_ring_concurrent_direct.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_scatter/scatter_double_ring_direct.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_scatter/scatter_mesh.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_scatter/scatter_ring.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_scatter/multi_root_scatter_ring.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_scatter/scatter_nhr.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_scatter/scatter_nb.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_scatter/scatter_ring_direct.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_send_recv/send_receive.cc

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_gather/gather_ring.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_gather/gather_mesh.cc
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_gather/gather_star.cc

        ${HCCL_BASE_DIR}/algorithm/base/mc2_handler/mc2_handler.cc

        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_executor_base.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_native_executor_base.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_comm_executor.cc

        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_single_rank_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_fast_double_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_aligned_all_reduce_double_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_comm_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_double_ring_concurrent_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_mid_count_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_oneshot_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_pipeline_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_graph_pipeline_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_small_count_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_reduce_plus_bcast_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mix_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_zerocopy_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_order_preserved_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/310P/coll_all_reduce_for_310p_ring_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/310P/coll_all_reduce_for_310p_doubling_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/310P/coll_all_reduce_for_310p_doubling_direct_executor.cc

        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_comm_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ring_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_aligned_all_gather_double_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_opbase_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_opbase_pipeline_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_single_rank_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_double_ring_concurrent_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_small_count_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_semi_ring_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mix_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_sio_hccs_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ring_zerocopy_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ring_zerocopy_exchange_executor.cc

        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather_v/coll_aligned_all_gather_v_double_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_semi_ring_executor.cc

        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_ring_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_comm_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_smallcount_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mix_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_ring_zerocopy_executor.cc

        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce/coll_reduce_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce/coll_reduce_mesh_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce/coll_reduce_ring_plus_hd_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce/coll_reduce_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce/coll_reduce_comm_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce/coll_reduce_single_rank_executor.cc

        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_comm_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_deter_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_fast_double_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_aligned_reduce_scatter_double_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_dma_elimination.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_opbase_pipeline_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_single_rank_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_double_ring_concurrent_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_semi_ring_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mix_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_zerocopy_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_zerocopy_exchange_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_order_preserved_executor.cc

        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_aiv_big_count_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_aiv_smallcount_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_opbase_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_aligned_reduce_scatter_v_double_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_fast_double_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_semi_ring_executor.cc

        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_scatter/coll_scatter_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_scatter/coll_scatter_ring_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_scatter/coll_scatter_ring_for_910_93_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_scatter/coll_scatter_comm_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_scatter/coll_scatter_mesh_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_scatter/coll_scatter_single_rank_executor.cc

        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_retry_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_send_receive/coll_send_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_send_receive/coll_receive_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_send_receive/coll_batch_write_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_executor_aicpu.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_single_rank_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_fullmesh_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_direct_fullmesh_executor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_2level_pipeline_excecutor.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_to_all/310P/coll_all_to_all_v_for_310p_executor.cc

        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/registry/coll_alg_exec_registry.cc
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/alg_profiling.cc

        ${HCCL_BASE_DIR}/algorithm/impl/task/threadManage.cc
        ${HCCL_BASE_DIR}/algorithm/impl/task/parallel_task_loader.cc
        ${HCCL_BASE_DIR}/algorithm/impl/task/task_loader.cc

        # common
        ${HCCL_BASE_DIR}/common/debug/profiling/task_profiling.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/task_exception_handler.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/profiler_base.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/plugin_runner.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/plugin_runner_device.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/command_handle.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/profiling_manager.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/task_overflow.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/profiler_manager_impl.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/profiler_manager.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/profiling_manager_pub.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/adapter_prof.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/dlprof_function.cc
        ${HCCL_BASE_DIR}/common/debug/profiling/dlrt_function.cc
        ${HCCL_BASE_DIR}/common/debug/config/config_log.cc
        ${HCCL_BASE_DIR}/common/health/calc_crc.cc
        ${HCCL_BASE_DIR}/common/health/rank_consistentcy_checker.cc
        ${HCCL_BASE_DIR}/common/stream/stream_utils_aicpu.cc

        #framework
        ${HCCL_BASE_DIR}/framework/communicator/impl/one_sided_service/i_hccl_one_sided_service.cc
    )
    target_sources(ccl_kernel PRIVATE
        ${CCL_KERNEL_SRC_LIST}
    )

    target_compile_options(ccl_kernel PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O3
        -std=c++14
    )

    target_compile_definitions(ccl_kernel PRIVATE
        HCCD
        CCL_KERNEL_AICPU
    )
    target_include_directories(ccl_kernel PRIVATE
        ${HCCL_BASE_DIR}/algorithm/base/inc
        ${HCCL_BASE_DIR}/algorithm/pub_inc
        ${HCCL_BASE_DIR}/algorithm/base/alg_aiv_template
        ${HCCL_BASE_DIR}/algorithm/base/alg_template
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/component
        ${HCCL_BASE_DIR}/algorithm/base/mc2_handler
        ${HCCL_BASE_DIR}/algorithm/base/communicator
        ${HCCL_BASE_DIR}/algorithm/impl
        ${HCCL_BASE_DIR}/algorithm/impl/inc
        ${HCCL_BASE_DIR}/algorithm/impl/legacy
        ${HCCL_BASE_DIR}/algorithm/impl/legacy/operator
        ${HCCL_BASE_DIR}/algorithm/base/communicator/legacy
        ${HCCL_BASE_DIR}/algorithm/impl/resource_manager
        ${HCCL_BASE_DIR}/algorithm/impl/task
        ${HCCL_BASE_DIR}/algorithm/impl/operator
        ${HCCL_BASE_DIR}/algorithm/impl/operator/registry
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/registry
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_send_receive
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_reduce/310P
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather/310P
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_gather_v
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter/310P
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter_v
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_reduce_scatter_v/310P
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_all_to_all
        ${HCCL_BASE_DIR}/algorithm/impl/coll_executor/coll_scatter
        ${HCCL_BASE_DIR}/common/health
        ${HCCL_BASE_DIR}/common/debug/profiling/inc
        ${HCCL_BASE_DIR}/common/debug/profiling/inc/aicpu
        ${HCCL_BASE_DIR}/common/debug/config/
        ${HCCL_BASE_DIR}/common/debug/
        ${HCCL_BASE_DIR}/common/stream
        ${HCCL_BASE_DIR}/framework/inc
        ${HCCL_BASE_DIR}/framework/cluster_maintenance/health/heartbeat/
        ${HCCL_BASE_DIR}/framework/cluster_maintenance/detect/detect_connect_anomalies/
        ${HCCL_BASE_DIR}/framework/common/src
        ${HCCL_BASE_DIR}/framework/common/src/aicpu
        ${HCCL_BASE_DIR}/framework/common/src/config
        ${HCCL_BASE_DIR}/framework/common/src/task
        ${HCCL_BASE_DIR}/framework/common/src/topo
        ${HCCL_BASE_DIR}/framework/hcom
        ${HCCL_BASE_DIR}/framework/communicator/impl/
        ${HCCL_BASE_DIR}/framework/communicator/impl/one_sided_service/
        ${HCCL_BASE_DIR}/framework/communicator/impl/zero_copy_memory_agent
        ${HCCL_BASE_DIR}/framework/communicator/impl/resource_manager
        ${HCCL_BASE_DIR}/framework/device/inc/
        ${HCCL_BASE_DIR}/framework/op_base/src/
        ${HCCL_BASE_DIR}/framework/device/
        ${HCCL_BASE_DIR}/framework/device/common
        ${HCCL_BASE_DIR}/framework/device/inc
        ${HCCL_BASE_DIR}/framework/cluster_maintenance/recovery/operator_retry
        ${HCCL_BASE_DIR}/framework/common/src/exception

        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_gather
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_all_reduce
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoall
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_alltoallv
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_broadcast
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_reduce_scatter
        ${HCCL_BASE_DIR}/algorithm/base/alg_template/temp_scatter
        ${HCCL_BASE_DIR}/algorithm/base/alg_template
        ${HCCL_BASE_DIR}/algorithm/pub_inc

    )

    add_subdirectory(device)
endif()

if(NOT BUILD_OPEN_PROJECT)
    target_include_directories(ccl_kernel PRIVATE
        ${HCCL_BASE_DIR}/pub_inc
        ${HCCL_BASE_DIR}/pub_inc/inner
        ${HCCL_BASE_DIR}/pub_inc/aicpu

        ${HCCL_BASE_DIR}/hccl/hccl_comm/wrapper/notify/

        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/inc/aicpu/
        ${TOP_DIR}/ace/comop/hccl/whole/orion
        ${HCCL_BASE_DIR}/platform/
        ${HCCL_BASE_DIR}/platform/inc/
        ${HCCL_BASE_DIR}/platform/inc/adapter
        ${HCCL_BASE_DIR}/platform/resource/transport/heterog/
        ${HCCL_BASE_DIR}/platform/resource/transport/host/
        ${HCCL_BASE_DIR}/platform/resource/transport/
        ${HCCL_BASE_DIR}/platform/resource/notify/
        ${HCCL_BASE_DIR}/platform/task
        ${HCCL_BASE_DIR}/platform/common
        ${HCCL_BASE_DIR}/platform/common/unique
        ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
        ${TOP_DIR}/asl/hss/inc/
        ${TOP_DIR}/asl/hss/inc/pub
        ${TOP_DIR}/asl/hss/common/hash_table

        ${HCCL_BASE_DIR}/inc
        ${HCCL_BASE_DIR}/hdcs
        ${HCCL_BASE_DIR}/hccl/hccl_comm/inc
        ${HCCL_BASE_DIR}/hcom/src
        ${HCCL_BASE_DIR}/hcom/src/gradient_segment
        ${HCCL_BASE_DIR}/hccl/op_base/src
        ${HCCL_BASE_DIR}/hccl_heterog/common
        ${HCCL_BASE_DIR}/hccl_heterog/heterog_cpu
        ${HCCL_BASE_DIR}/hccl_heterog/rpc
        ${HCCL_BASE_DIR}/hccl/hccl_comm/executor/component/inc
        ${HCCL_BASE_DIR}/hccl/hccl_comm/impl/
    )
    target_link_libraries(ccl_kernel PRIVATE
        $<BUILD_INTERFACE:intf_pub_cxx14>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:hccl_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:network_headers>
        $<BUILD_INTERFACE:atrace_headers>
        $<BUILD_INTERFACE:kernel_tiling_headers>
        -Wl,--no-as-needed
        c_sec
        slog
        ccl_kernel_plf
        aicpu_sharder
        mmpa
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
        ofed_headers
    )

    FILE(GLOB_RECURSE orion_include_list LIST_DIRECTORIES TRUE ${TOP_DIR}/ace/comop/hccl/orion/*)
    foreach (dir ${orion_include_list})
        IF ((IS_DIRECTORY ${dir} ) AND (${dir} MATCHES "hccl/orion"))
            target_include_directories(ccl_kernel PRIVATE ${dir})
        endif ()
    endforeach ()

    FILE(GLOB_RECURSE hccl_next_dir_list LIST_DIRECTORIES TRUE ${HCCL_NEXT_CODE_ROOT})
    foreach (dir ${hccl_next_dir_list})
        IF ((IS_DIRECTORY ${dir} ) AND (${dir} MATCHES "whole/hccl_next"))
            target_include_directories(ccl_kernel PRIVATE ${dir})
        endif ()
    endforeach ()

    install(TARGETS  ccl_kernel
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}  OPTIONAL
    )
endif()
add_subdirectory(common)
add_subdirectory(communicator)
add_subdirectory(hcom)
add_subdirectory(op_base)
add_subdirectory(cluster_maintenance)
add_subdirectory(nslbdp)

if(BUILD_OPEN_PROJECT AND KERNEL_MODE)
    target_include_directories(ccl_kernel PRIVATE
        ${hccl_include_list}
        ${HCCL_BASE_DIR}/../../../inc
        ${HCCL_BASE_DIR}/../../../inc/hccl
        ${HCCL_BASE_DIR}/../../../hccl/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/datagw/aicpu/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/datagw/aicpu/common/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/atrace/utrace/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl/aicpu
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/external/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal/driver/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog/toolchain/
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/mmpa
        ${HCCL_THIRD_PARTY_DIR}/ace/comop/inc/external
        ${HCCL_THIRD_PARTY_DIR}/ace/comop/inc/hccl
        ${HCCL_THIRD_PARTY_DIR}/ace/comop/inc
        ${HCCL_THIRD_PARTY_DIR}/abl/libc_sec/include
        ${HCCL_THIRD_PARTY_DIR}/abl/slog/inc/toolchain
        ${HCCL_THIRD_PARTY_DIR}/abl/mmpa/inc/mmpa
        ${HCCL_THIRD_PARTY_DIR}/abl/mmpa/inc/mmpa/sub_inc
        ${THIRD_PARTY_NLOHMANN_PATH}
        )
    if(BUILD_OPEN_PROJECT)
        target_compile_definitions(ccl_kernel PRIVATE
            OPEN_BUILD_PROJECT
            -D_GLIBCXX_USE_CXX11_ABI=1) # 链接ascendalog需要LOG_CPP，链接slog不需要
        message(STATUS "framework CMAKE_HOST_SYSTEM_PROCESSOR=${CMAKE_HOST_SYSTEM_PROCESSOR}")
        if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
            message(STATUS "Host is x86_64")
            set(ASCEND_CANN_PACKAGE_ENV_PATH ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux)
        elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
            message(STATUS "Host is ARM64")
            set(ASCEND_CANN_PACKAGE_ENV_PATH ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux)
        endif()
        message(STATUS "framework ASCEND_CANN_PACKAGE_ENV_PATH=${ASCEND_CANN_PACKAGE_ENV_PATH}")
        target_link_libraries(ccl_kernel PRIVATE
            -Wl,--no-as-needed
            ascend_hal
            c_sec
            slog
            aicpu_sharder
            mmpa
            -Wl,--as-needed
            -Wl,--whole-archive
            ${ASCEND_CANN_PACKAGE_ENV_PATH}/lib64/device/lib64/libccl_kernel_plf.a
            -Wl,--no-whole-archive
            -Wl,-Bsymbolic
            -lrt
            -ldl
            -lpthread
        )

        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libaicpu_custom.json
            COMMAND ${HI_PYTHON} ${HCCL_BASE_DIR}/../../../cmake/scripts/parser_ini.py ${CMAKE_CURRENT_SOURCE_DIR}/device/framework/aicpu_custom.ini ${CMAKE_CURRENT_BINARY_DIR}/libaicpu_custom.json
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        add_custom_target(aicpu_custom_json DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libaicpu_custom.json)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libaicpu_custom.json
            DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
        )

        target_link_directories(ccl_kernel PRIVATE
            ${ASCEND_CANN_PACKAGE_PATH}/lib64
        )

        set_target_properties(ccl_kernel PROPERTIES
            OUTPUT_NAME aicpu_custom)
    endif()
endif()