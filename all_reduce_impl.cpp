#include "all_reduce.h"
#include "common/topology_utils.h"
#include "algorithm/ring_algorithms.h"
#include "communicator/ts_wrapper.h"
#include "common/reduce_ops.h"
#include <stdlib.h>
#include <string.h>

Hcc1Result Hcc1AllReduce(void *sendBuf, void *recvBuf, uint64_t count, Hcc1DataType dataType, Hcc1ReduceOp op,
    Hcc1Comm comm, aclrtStream stream)
{
    // 1. 获取当前rank（实际应调用HcclGetRank）
    int rank = 0; // 仿真占位，实际从comm提取
    // 假设通过某种方式获得rank，例如环境变量或HCCL内部接口
    // 此处为演示，固定写0，实际需替换为真实获取逻辑

    TopologyInfo topo;
    get_topology_info(comm, rank, &topo);

    size_t elem_size = 4; // 简化，可根据dataType查表
    size_t total_bytes = count * elem_size;

    // 2. 分配机内临时buffer
    void *intra_buf = malloc(total_bytes);
    void *global_buf = NULL;
    if (is_gateway(&topo)) {
        global_buf = malloc(total_bytes);
    }

    // ====== 阶段1：机内Ring归约 ======
    // 输入sendBuf，输出intra_buf（本Server 8卡之和）
    intra_server_ring_reduce(sendBuf, intra_buf, count, dataType, op, &topo, stream);
    ts_synchronize_stream(stream);

    // ====== 阶段2：跨机单网关归约 ======
    if (is_gateway(&topo)) {
        int peer = topo.peer_gateway;
        if (topo.server_id == 0) {
            // Server0：先发后收
            ts_send(intra_buf, total_bytes, peer, stream);
            ts_recv(global_buf, total_bytes, peer, stream);
            ts_synchronize_stream(stream);
            // 归约：本地(先) + 对端(后)
            reduce_two_buffers(intra_buf, global_buf, global_buf, count, dataType, op);
        } else { // Server1
            // Server1：先收后发（避免死锁）
            ts_recv(global_buf, total_bytes, peer, stream);
            ts_send(intra_buf, total_bytes, peer, stream);
            ts_synchronize_stream(stream);
            // 归约：对端(先) + 本地(后) -> 写入global_buf
            // 注意此时global_buf存放对端数据，intra_buf存放本地
            // 统一为 对端 + 本端
            reduce_two_buffers(global_buf, intra_buf, global_buf, count, dataType, op);
        }
    }

    // ====== 阶段3：机内广播 ======
    // 网关将global_buf广播给同Server其他卡，最终写入recvBuf
    const void *src_for_bcast = is_gateway(&topo) ? global_buf : NULL;
    // 非网关占位，实际函数内部会根据local_rank判断接收
    intra_server_broadcast(is_gateway(&topo) ? global_buf : intra_buf, // 非网关此参数无用
        recvBuf, count, dataType, &topo, stream);

    // 清理
    free(intra_buf);
    if (global_buf)
        free(global_buf);

    return HCCL_SUCCESS;
}
