set(src_list
        dfx_extend_info.cc
)

target_sources(ccl_kernel PRIVATE
    ${src_list}
)

add_subdirectory(trace)
add_subdirectory(profiling)