set(src_list
    mc2_trace_utils.cc
    executor_tracer.cc
    cann_error_reporter.cc
)

target_sources(ccl_kernel PRIVATE
    ${src_list}
)