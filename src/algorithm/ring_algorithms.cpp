#include "ring_algorithms.h"
#include "common/topology_utils.h"
#include "communicator/ts_wrapper.h"
#include "common/reduce_ops.h"
#include <stdlib.h>
#include <string.h>

void intra_server_ring_reduce(const void *send_buf, void *recv_buf, uint64_t count, Hcc1DataType data_type,
    Hcc1ReduceOp op, const TopologyInfo *topo, aclrtStream stream)
{
    int local = topo->local_rank;
    int server = topo->server_id;
    int world_base = server * NPU_PER_SERVER;

    size_t elem_size = 4; // 简化按float
    size_t chunk_bytes = count * elem_size;

    // 临时buffer用于接收邻卡数据
    void *tmp_buf = malloc(chunk_bytes);
    // 将send_buf拷贝到recv_buf作为初始值
    memcpy(recv_buf, send_buf, chunk_bytes);

    // 环形归约：每个卡向右发送，向左接收
    for (int step = 0; step < NPU_PER_SERVER - 1; ++step) {
        int send_rank = world_base + ((local + 1) % NPU_PER_SERVER);
        int recv_rank = world_base + ((local - 1 + NPU_PER_SERVER) % NPU_PER_SERVER);

        // 异步发送自己当前的recv_buf
        ts_send(recv_buf, chunk_bytes, send_rank, stream);
        // 接收左邻数据到tmp_buf
        ts_recv(tmp_buf, chunk_bytes, recv_rank, stream);
        ts_synchronize_stream(stream);

        // 归约：recv_buf = recv_buf + tmp_buf (固定顺序)
        reduce_two_buffers(recv_buf, tmp_buf, recv_buf, count, data_type, op);
    }

    free(tmp_buf);
}

void intra_server_broadcast(const void *src_buf, void *dst_buf, uint64_t count, Hcc1DataType data_type,
    const TopologyInfo *topo, aclrtStream stream)
{
    int local = topo->local_rank;
    int server = topo->server_id;
    int gateway_rank = topo->gateway_rank;
    size_t bytes = count * 4; // 简化float

    if (local == 0) {
        // 网关：发送给所有其他local_rank (1~7)
        for (int dst_local = 1; dst_local < NPU_PER_SERVER; ++dst_local) {
            int dst_rank = server * NPU_PER_SERVER + dst_local;
            ts_send(src_buf, bytes, dst_rank, stream);
        }
        // 自己拷贝
        memcpy(dst_buf, src_buf, bytes);
    } else {
        // 非网关：从网关接收
        ts_recv(dst_buf, bytes, gateway_rank, stream);
    }
    ts_synchronize_stream(stream);
}