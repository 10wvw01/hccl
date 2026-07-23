/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include <thread>
#include "alg_env_config.h"

using namespace HcclSim;
using namespace ops_hccl;

constexpr uint32_t DATATYPE_SIZE_TABLE_SCATTER_ST[HCCL_DATA_TYPE_RESERVED]
    = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t), 2, sizeof(float), sizeof(int64_t), sizeof(uint64_t),
        sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t), 8, 2, 16, 2, 1, 1, 1, 1};

class ST_SCATTER_AICPU_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_SIM_FORCE_MESH1D_CLOS_TOPO");
    }

    static void SetUpTestCase()
    {
    }
    static void TearDownTestCase()
    {
    }
};

static u32 AnalyseRankSize(const TopoMeta &topoInfo)
{
    u32 rankSize = 0;
    for (const auto &superPod : topoInfo) {
        for (const auto &podIdx : superPod) {
            rankSize += podIdx.size();
        }
    }
    return rankSize;
}

void RunScatterAicpuA5(const TopoMeta &topoInfo, const u64 &recvCount, const HcclDataType &dataType,
    const u32 &root)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoInfo, DevType::DEV_TYPE_950);

    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_SCATTER_ST[dataType];
    auto rankSize = AnalyseRankSize(topoInfo);
    // 算子执行参数设置
    // 多线程运行SCATTER算子
    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            // 1.SetDevice
            aclrtSetDevice(rankId);

            // 2.创建流
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            // 3.初始化通信域
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void *sendBuf = nullptr;
            void *recvBuf = nullptr;
            // Scatter: root 持有全部 rank 的数据，其他 rank sendBuf 可为空（这里统一分配）
            u64 sendBufSize = recvCount * dataTypeSize * rankSize; // root 输入包含所有 rank 的数据
            u64 recvBufSize = recvCount * dataTypeSize;            // 输出为本 rank 的数据
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            HCCL_INFO("[ST_SCATTER_AICPU_TEST]Run HcclScatter, root=%u", root);

            // 4.算子下发
            CHK_RET(HcclScatter(sendBuf, recvBuf, recvCount, dataType, root, comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    // 等待多线程执行完成
    for (auto &thread : threads) {
        thread.join();
    }

    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckScatter(taskQueues, rankSize, dataType, recvCount, root);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

// ═══════════════════════════════════════════════════════════════════
// Mesh 1D 拓扑用例
// ═══════════════════════════════════════════════════════════════════

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_mesh_1d_2rank_int64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto recvCount = 100;                                // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64; // 数据类型
    auto root = 0u;                                      // 根节点
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_mesh_1d_2rank_int8_500M_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto recvCount = 250 * 1024 * 1024;                 // 单卡数据量250M(int8:1B), 总数据250M*2=500M
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto root = 0u;                                      // 根节点
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_mesh_1d_8rank_fp64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto recvCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP64; // 数据类型
    auto root = 0u;                                      // 根节点
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_mesh_1d_4rank_fp32_16G_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}}; // 三维数组指定超节点-Server-Device信息
    // 算子执行参数设置
    auto recvCount = 1024 * 1024 * 1024;               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    auto root = 0u;                                      // 根节点
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

// ═══════════════════════════════════════════════════════════════════
// NHR 拓扑用例
// ═══════════════════════════════════════════════════════════════════

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_NHR_2rank_int64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {1}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto recvCount = 100;                                // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64; // 数据类型
    auto root = 0u;                                      // 根节点
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_NHR_2rank_int8_500M_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {1}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto recvCount = 250 * 1024 * 1024;                 // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto root = 0u;                                      // 根节点
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_NHR_8rank_fp64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto recvCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP64; // 数据类型
    auto root = 0u;                                      // 根节点
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_NHR_4rank_fp32_16G_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {1}, {2}, {3}}}; // 三维数组指定超节点-Server-Device信息
    // 算子执行参数设置
    auto recvCount = 1024 * 1024 * 1024;               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    auto root = 0u;                                      // 根节点
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

// 非 0 root 用例：校验 NHR/Mesh Scatter 在 root != 0 时仍正确分发
TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_NHR_4rank_non_zero_root_int32_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {1}, {2}, {3}}}; // 三维数组指定超节点-Server-Device信息
    // 算子执行参数设置
    auto recvCount = 100;                                // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto root = 2u;                                      // 非 0 root
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_mesh_1d_4rank_non_zero_root_int32_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}}; // 三维数组指定超节点-Server-Device信息
    // 算子执行参数设置
    auto recvCount = 100;                                // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto root = 3u;                                      // 非 0 root
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

// ═══════════════════════════════════════════════════════════════════
// Parallel 拓扑用例（多 server，触发 ParallelMesh1DNHR 算法）
// 拓扑 {{{0,1},{0,1}}} 表示 2 server × 2 卡 = 4 rank，layer0 Mesh1D + layer1 NHR
// ═══════════════════════════════════════════════════════════════════

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_parallel_2x2rank_int8_32M_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    // 算子执行参数设置
    // recvCount=8M(int8:1B) -> 单卡8MB > 4MB阈值触发Parallel; 总数据8M*4=32MB < 4GB -> 走Parallel而非Sequence
    auto recvCount = 8 * 1024 * 1024;                  // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto root = 0u;                                      // 根节点
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_parallel_2x4rank_int64_4G_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}}; // 三维数组指定超节点-Server-Device信息
    // 算子执行参数设置
    auto recvCount = 64 * 1024 * 1024;                  // 单卡数据量512M,总数据512*8=4G
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64; // 数据类型
    auto root = 0u;                                      // 根节点
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

// Parallel 非 0 root：校验 Parallel 模式下 root != 0 的分发语义
TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_parallel_2x2rank_int32_non_zero_root_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    // 算子执行参数设置
    // recvCount=8M(int32:4B) -> 单卡32MB > 4MB阈值触发Parallel; 总数据32M*4=128MB < 4GB -> 走Parallel而非Sequence
    auto recvCount = 8 * 1024 * 1024;                  // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto root = 2u;                                      // 非 0 root（在第二 server）
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);
}

// ═══════════════════════════════════════════════════════════════════
// Concurrent 拓扑用例（Mesh1D-CLOS，layer0 Mesh + layer1 NHR 共享同一组卡）
// 需通过 HCCL_SIM_FORCE_MESH1D_CLOS_TOPO 强制 CLOS 拓扑识别
// ═══════════════════════════════════════════════════════════════════

TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_concurrent_mesh1dclos_fp16_512mb_data_test)
{
    setenv("HCCL_SIM_FORCE_MESH1D_CLOS_TOPO", "1", 1);

    // 仿真模型初始化：1 server 4 卡，layer0 Mesh + layer1 NHR 共享同一组卡
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    // 算子执行参数设置
    auto recvCount = 256 * 1024 * 1024;            // 单卡数据量（FP16: 2字节 -> 总 512MB > 512KB 阈值）
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 数据类型
    auto root = 0u;                                      // 根节点
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);

    unsetenv("HCCL_SIM_FORCE_MESH1D_CLOS_TOPO");
}

// Concurrent 非 0 root：覆盖 Mesh1D-CLOS 拓扑下 root != 0 的并发分发
TEST_F(ST_SCATTER_AICPU_TEST, st_scatter_a5_aicpu_concurrent_mesh1dclos_int32_non_zero_root_test)
{
    setenv("HCCL_SIM_FORCE_MESH1D_CLOS_TOPO", "1", 1);

    // 仿真模型初始化：1 server 4 卡，layer0 Mesh + layer1 NHR 共享同一组卡
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    // 算子执行参数设置
    auto recvCount = 100;                                // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto root = 2u;                                      // 非 0 root
    RunScatterAicpuA5(topoMeta, recvCount, dataType, root);

    unsetenv("HCCL_SIM_FORCE_MESH1D_CLOS_TOPO");
}
