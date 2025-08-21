target_sources(hccl_alg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/config_log.cc
)

target_sources(hccl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/config_log.cc
)

if(NOT BUILD_OPEN_PROJECT OR (BUILD_OPEN_PROJECT AND KERNEL_MODE))
    target_sources(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/config_log.cc
    )
endif()