#include "C:\Users\zhuwenqi\hccl_workspace\hccl\ascend_topology_aware_allreduce\include\all_reduce.h"
#include "common/topology_utils.h"
#include "algorithm/ring_algorithms.h"
#include "communicator/ts_wrapper.h"
#include "common/reduce_ops.h"
#include <stdlib.h>
#include <string.h>

HcclResult HcclAllReduce(void* sendBuf, void* recvBuf, uint64_t count,
                          HcclDataType dataType, HcclReduceOp op,
                          HcclComm comm, aclrtStream stream) {
    int rank = 0;

    TopologyInfo topo;
    get_topology_info(comm, rank, &topo);

    size_t elem_size = 4;
    size_t total_bytes = count * elem_size;

    void* intra_buf = malloc(total_bytes);
    void* global_buf = NULL;
    if (is_gateway(&topo)) {
        global_buf = malloc(total_bytes);
    }

    intra_server_ring_reduce(sendBuf, intra_buf, count, dataType, op, &topo, stream);
    ts_synchronize_stream(stream);

    if (is_gateway(&topo)) {
        int peer = topo.peer_gateway;
        if (topo.server_id == 0) {
            ts_send(intra_buf, total_bytes, peer, stream);
            ts_recv(global_buf, total_bytes, peer, stream);
            ts_synchronize_stream(stream);
            reduce_two_buffers(intra_buf, global_buf, global_buf, count, dataType, op);
        } else {
            ts_recv(global_buf, total_bytes, peer, stream);
            ts_send(intra_buf, total_bytes, peer, stream);
            ts_synchronize_stream(stream);
            reduce_two_buffers(global_buf, intra_buf, global_buf, count, dataType, op);
        }
    }

    intra_server_broadcast(
        is_gateway(&topo) ? global_buf : intra_buf,
        recvBuf,
        count,
        dataType,
        &topo,
        stream
    );

    free(intra_buf);
    if (global_buf) free(global_buf);

    return HCCL_SUCCESS;
}