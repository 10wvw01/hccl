#include "reduce_ops.h"
#include <string.h>
#include <algorithm>

template<typename T>
static void reduce_sum_two(const T* a, const T* b, T* result, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        result[i] = (T)((double)a[i] + (double)b[i]);
    }
}

void reduce_two_buffers(const void* src_a, const void* src_b, void* result_buf,
                        uint64_t count, HcclDataType data_type, HcclReduceOp op) {
    if (op == HCCL_REDUCE_SUM && data_type == HCCL_DATA_TYPE_FP32) {
        reduce_sum_two((const float*)src_a, (const float*)src_b, (float*)result_buf, count);
    }
}

void reduce_multiple_buffers(const void* const* src_buffers, int num_buffers,
                             void* result_buf, uint64_t count,
                             HcclDataType data_type, HcclReduceOp op) {
    size_t bytes = count * sizeof(float);
    memcpy(result_buf, src_buffers[0], bytes);
    for (int i = 1; i < num_buffers; ++i) {
        reduce_two_buffers(result_buf, src_buffers[i], result_buf,
                           count, data_type, op);
    }
}