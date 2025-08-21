set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/env_config.cc
)

target_sources(hccl PRIVATE
    ${src_list}
    ${CMAKE_CURRENT_SOURCE_DIR}/env_config_host.cc
)

if(NOT BUILD_OPEN_PROJECT OR (BUILD_OPEN_PROJECT AND KERNEL_MODE))
    target_sources(ccl_kernel PRIVATE
        ${src_list}
        ${CMAKE_CURRENT_SOURCE_DIR}/env_config_aicpu.cc
    )
endif()