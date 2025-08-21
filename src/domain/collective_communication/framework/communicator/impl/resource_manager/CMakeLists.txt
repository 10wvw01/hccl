set(src_list
    transport_manager.cc
    preempt_port_manager.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)

if(NOT BUILD_OPEN_PROJECT OR (BUILD_OPEN_PROJECT AND KERNEL_MODE))
    target_sources(ccl_kernel PRIVATE
        ${src_list}
    )
endif()