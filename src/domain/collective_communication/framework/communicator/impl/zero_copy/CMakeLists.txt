set(src_list
    zero_copy_address_mgr.cc
    zero_copy_address_mgr_host.cc
    zero_copy_memory_agent.cc
)

if(NOT BUILD_OPEN_PROJECT OR (BUILD_OPEN_PROJECT AND KERNEL_MODE))
    set(src_list_device
        zero_copy_address_mgr_device.cc
    )

    target_sources(ccl_kernel PRIVATE
        ${src_list_device}
    )
endif()

target_sources(hccl PRIVATE
    ${src_list}
)