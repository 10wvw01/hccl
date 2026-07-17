#ifndef RING_ALGORITHMS_H
#define RING_ALGORITHMS_H

#include <stdint.h>
#include "../../include/all_reduce.h"
#include "../common/topology_utils.h"

void intra_server_ring_reduce(
    const void* send_buf,
    void* recv_buf,
    uint64_t count,
    HcclDataType data_type,
    HcclReduceOp op,
    const TopologyInfo* topo,
    aclrtStream stream
);

void intra_server_broadcast(
    const void* src_buf,
    void* dst_buf,
    uint64_t count,
    HcclDataType data_type,
    const TopologyInfo* topo,
    aclrtStream stream
);

#endif // RING_ALGORITHMS_H