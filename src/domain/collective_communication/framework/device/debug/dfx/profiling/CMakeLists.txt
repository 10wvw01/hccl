set(src_list
    profiling_manager_device.cc
    profiling_extend_info.cc
)

target_sources(ccl_kernel PRIVATE
    ${src_list}
)