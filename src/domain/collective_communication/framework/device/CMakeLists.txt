if(NOT BUILD_OPEN_PROJECT)
    set(src_list
        hccl_aicpu_kernel.cc
    )
    target_sources(ccl_kernel PRIVATE
        ${src_list}
        mc2_aicpu_interface.cc
    )
endif()

if(BUILD_OPEN_PROJECT AND KERNEL_MODE)
    target_sources(ccl_kernel PRIVATE
        mc2_aicpu_interface.cc
    )
endif()
add_subdirectory(algorithm)
add_subdirectory(common)
add_subdirectory(framework)
add_subdirectory(utils)
add_subdirectory(debug)