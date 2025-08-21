set(src_list
    mc2_aicpu_utils.cc
    aicpu_hdc_utils.cc
)

target_sources(ccl_kernel PRIVATE
    ${src_list}
)