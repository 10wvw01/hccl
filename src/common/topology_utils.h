#ifndef TOPOLOGY_UTILS_H
#define TOPOLOGY_UTILS_H

#include <stdint.h>

// 硬件固定配置：2 Server，每Server 8 NPU
#define SERVER_NUM 2
#define NPU_PER_SERVER 8
#define TOTAL_RANKS (SERVER_NUM * NPU_PER_SERVER) // 16

typedef struct TopologyInfo {
    int global_rank;  // 0~15
    int global_size;  // 16
    int server_id;    // 0 或 1
    int local_rank;   // 0~7
    int gateway_rank; // 本Server网关rank（local_rank==0的全局rank）
    int peer_gateway; // 对端Server网关rank
} TopologyInfo;

// 从comm和rank中提取拓扑信息（仿真环境直接计算）
void get_topology_info(Hcc1Comm comm, int rank, TopologyInfo *info);

// 检查是否为网关卡
static inline int is_gateway(const TopologyInfo *info)
{
    return info->local_rank == 0;
}

#endif // TOPOLOGY_UTILS_H