set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/global_mem_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/global_mem_record.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)