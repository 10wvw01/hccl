# Auto-generated script for copying headers
file(GLOB H_FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/src/es_hccl_build/generated_code/*.h")
foreach(h_file ${H_FILES})
    get_filename_component(filename ${h_file} NAME)
    # 使用 configure_file 替代 COPY_FILE (兼容 CMake 3.16)
    configure_file(${h_file} "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/es_output/include/es_hccl/${filename}" COPYONLY)
    message(STATUS "Copied header: ${filename}")
endforeach()
