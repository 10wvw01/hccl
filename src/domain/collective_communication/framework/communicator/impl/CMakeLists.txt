set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_communicator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_communicator_attrs.cc 
)

target_sources(hccl PRIVATE
    ${src_list}
    ${CMAKE_CURRENT_SOURCE_DIR}/aclgraph/zero_copy_acl_graph.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_communicator_host.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_communicator_attrs_host.cc
    task_abort_handler.cc
)

if(NOT BUILD_OPEN_PROJECT OR (BUILD_OPEN_PROJECT AND KERNEL_MODE))
    target_sources(ccl_kernel PRIVATE
        ${src_list}
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_communicator_device.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_communicator_attrs_device.cc
    )
endif()

add_subdirectory(resource_manager)
add_subdirectory(one_sided_service)
add_subdirectory(zero_copy)