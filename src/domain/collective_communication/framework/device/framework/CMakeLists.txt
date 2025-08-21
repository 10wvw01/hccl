set(src_list
    aicpu_hccl_process.cc
    aicpu_rpc_server.cc
    aicpu_communicator.cc
    aicpu_prof.cc
    aicpu_rpc_serverv2.cc
    aicpu_hdc.cc
    zero_copy_address_mgr.cc
    aicpu_zero_copy_exchanger.cc
    aicpu_share_data_manager.cc
)

target_sources(ccl_kernel PRIVATE
    ${src_list}
)