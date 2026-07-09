# Auto-generated script for copying whl files
file(GLOB WHL_FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/src/es_hccl_build/python_package/dist/*.whl")
foreach(whl_file ${WHL_FILES})
    get_filename_component(filename ${whl_file} NAME)
    # 使用 configure_file 替代 COPY_FILE (兼容 CMake 3.16)
    configure_file(${whl_file} "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/es_output/whl/${filename}" COPYONLY)
    message(STATUS "Copied wheel: ${filename}")
endforeach()
