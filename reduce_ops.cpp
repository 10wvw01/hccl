#include "reduce_ops.h"
#include <string.h>
#include <float.h>
#include <algorithm>

// 辅助：将数据按类型转换为double进行累加，保证精度和固定顺序
template <typename T> static void reduce_sum_two(const T *a, const T *b, T *result, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i) {
        // 强制先加a，再加b，顺序绝对固定
        double tmp = (double)a[i] + (double)b[i];
        result[i] = (T)tmp;
    }
}

template <typename T> static void reduce_max_two(const T *a, const T *b, T *result, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i) {
        result[i] = std::max(a[i], b[i]);
    }
}

template <typename T> static void reduce_min_two(const T *a, const T *b, T *result, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i) {
        result[i] = std::min(a[i], b[i]);
    }
}

void reduce_two_buffers(
    const void *src_a, const void *src_b, void *result_buf, uint64_t count, Hcc1DataType data_type, Hcc1ReduceOp op)
{
    // 为了确保完全确定性，若src_a与result重叠，需用临时buffer
    // 此处假设已由调用方处理，直接计算

    if (op == HCCL_REDUCE_SUM) {
        switch (data_type) {
            case HCCL_DATA_TYPE_FP32:
                reduce_sum_two((const float *)src_a, (const float *)src_b, (float *)result_buf, count);
                break;
            // 添加其他类型...
            default:
                break;
        }
    } else if (op == HCCL_REDUCE_MAX) {
        // 类似实现...
    }
}

void reduce_multiple_buffers(const void *const *src_buffers, int num_buffers, void *result_buf, uint64_t count,
    Hcc1DataType data_type, Hcc1ReduceOp op)
{
    // 先将第一个buffer拷贝到结果
    size_t bytes = count * sizeof(float); // 简化，实际需根据type
    memcpy(result_buf, src_buffers[0], bytes);
    // 依次与后续buffer归约，顺序固定为0,1,2...
    for (int i = 1; i < num_buffers; ++i) {
        reduce_two_buffers(result_buf, src_buffers[i], result_buf, count, data_type, op);
    }
}