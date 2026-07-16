#include "topology_utils.h"
#include <stdlib.h>

// 假设底层有获取rank的API，这里仿真直接模拟
void get_topology_info(Hcc1Comm comm, int rank, TopologyInfo *info)
{
    // 实际环境中应调用: HcclGetRank(comm, &rank); HcclGetRankSize(comm, &size);
    // 此处根据题目固定为16卡
    info->global_rank = rank;
    info->global_size = TOTAL_RANKS;
    info->server_id = rank / NPU_PER_SERVER;
    info->local_rank = rank % NPU_PER_SERVER;
    info->gateway_rank = info->server_id * NPU_PER_SERVER;            // local_rank 0
    info->peer_gateway = (info->server_id == 0) ? NPU_PER_SERVER : 0; // 8 或 0
}