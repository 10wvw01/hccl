set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/config.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/param_check.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/json_utils.cc
)

target_sources(hccl PRIVATE
    ${src_list}
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_hash_utils.cc
)


if(NOT BUILD_OPEN_PROJECT OR (BUILD_OPEN_PROJECT AND KERNEL_MODE))
    target_sources(ccl_kernel PRIVATE
        ${src_list}
    )
endif()

add_subdirectory(opexecounter)
add_subdirectory(topo)
add_subdirectory(task)
add_subdirectory(exception)
add_subdirectory(thread)
add_subdirectory(config)
add_subdirectory(onesided_memory_management)