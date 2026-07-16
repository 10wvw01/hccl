#ifndef RING_ALGORITHMS_H
#define RING_ALGORITHMS_H

#include <stdint.h>
#include "all_reduce.h"

// 机内8卡 Ring AllReduce（归约并分发中间结果）
// 最终所有卡都获得本Server的局部归约值
void intra_server_ring_reduce(const void *send_buf, void *recv_buf, uint64_t count, Hcc1DataType data_type,
    Hcc1ReduceOp op, const TopologyInfo *topo, aclrtStream stream);

// 机内广播：网关将全局结果发给同Server其他7卡
void intra_server_broadcast(const void *src_buf, void *dst_buf, uint64_t count, Hcc1DataType data_type,
    const TopologyInfo *topo, aclrtStream stream);

#endif // RING_ALGORITHMS_H