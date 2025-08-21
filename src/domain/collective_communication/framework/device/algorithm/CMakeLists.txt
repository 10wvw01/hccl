set(src_list
    aicpu_hccl_allgather.cc
    aicpu_hccl_allreduce.cc
    aicpu_hccl_dispatcher.cc
    aicpu_hccl_dmy_cal_allreduce.cc
    aicpu_hccl_reduce_scatter.cc
    aicpu_hccl_alltoall.cc
)

target_sources(ccl_kernel PRIVATE
    ${src_list}
)