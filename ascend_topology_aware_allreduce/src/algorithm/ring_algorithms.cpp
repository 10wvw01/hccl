#include "ring_algorithms.h"
#include "../common/topology_utils.h"
#include "../communicator/ts_wrapper.h"
#include "../common/reduce_ops.h"
#include <stdlib.h>
#include <string.h>

void intra_server_ring_reduce(const void* send_buf, void* recv_buf,
                              uint64_t count, HcclDataType data_type,
                              HcclReduceOp op, const TopologyInfo* topo,
                              aclrtStream stream) {
    int local = topo->local_rank;
    int server = topo->server_id;
    int world_base = server * NPU_PER_SERVER;
    size_t elem_size = 4;
    size_t chunk_bytes = count * elem_size;

    void* tmp_buf = malloc(chunk_bytes);
    memcpy(recv_buf, send_buf, chunk_bytes);

    for (int step = 0; step < NPU_PER_SERVER - 1; ++step) {
        int send_rank = world_base + ((local + 1) % NPU_PER_SERVER);
        int recv_rank = world_base + ((local - 1 + NPU_PER_SERVER) % NPU_PER_SERVER);

        ts_send(recv_buf, chunk_bytes, send_rank, stream);
        ts_recv(tmp_buf, chunk_bytes, recv_rank, stream);
        ts_synchronize_stream(stream);

        reduce_two_buffers(recv_buf, tmp_buf, recv_buf, count, data_type, op);
    }

    free(tmp_buf);
}

void intra_server_broadcast(const void* src_buf, void* dst_buf,
                            uint64_t count, HcclDataType data_type,
                            const TopologyInfo* topo, aclrtStream stream) {
    int local = topo->local_rank;
    int server = topo->server_id;
    int gateway_rank = topo->gateway_rank;
    size_t bytes = count * 4;

    if (local == 0) {
        for (int dst_local = 1; dst_local < NPU_PER_SERVER; ++dst_local) {
            int dst_rank = server * NPU_PER_SERVER + dst_local;
            ts_send(src_buf, bytes, dst_rank, stream);
        }
        memcpy(dst_buf, src_buf, bytes);
    } else {
        ts_recv(dst_buf, bytes, gateway_rank, stream);
    }
    ts_synchronize_stream(stream);
}