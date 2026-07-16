#ifndef REDUCE_OPS_H
#define REDUCE_OPS_H

#include <stdint.h>
#include "all_reduce.h"

// 对两个buffer执行确定性归约，结果存入result_buf。
// 固定顺序：先累加src_a，再累加src_b（保证浮点数次序固定）
void reduce_two_buffers(
    const void *src_a, const void *src_b, void *result_buf, uint64_t count, Hcc1DataType data_type, Hcc1ReduceOp op);

// 对单个buffer内的数据执行跨通道归约（用于Ring阶段将多卡数据压缩）
// 这里为了确定性，按传入顺序逐个累加
void reduce_multiple_buffers(const void *const *src_buffers, int num_buffers, void *result_buf, uint64_t count,
    Hcc1DataType data_type, Hcc1ReduceOp op);

#endif // REDUCE_OPS_H