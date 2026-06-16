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

constexpr uint32_t DATATYPE_SIZE_TABLE_ALL_GATHER_ST[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t),
    sizeof(int32_t), 2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t),
    sizeof(uint32_t), 8, 2, 16, 2, 1, 1, 1, 1};

// 统计 topoMeta 中参与集合通信的卡数(超节点-Server-Device 三层求和)
u32 AnalyseRankSize(const TopoMeta &topo)
{
    u32 n = 0;
    for (auto &&pod : topo) {
        for (auto &&srv : pod) {
            n += srv.size();
        }
    }
    return n;
}

class ST_ALL_GATHER_AICPU_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

void RunAllGatherAicpuA5(const TopoMeta &topo, const u64 &sendCount, const HcclDataType &dataType)
{
    // 段1:仿真模型初始化(机型 A5 / DEV_TYPE_950)
    SimWorld::Global()->Init(topo, DevType::DEV_TYPE_950);

    // 段2:设置展开模式为 AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    const u32 dtSize = DATATYPE_SIZE_TABLE_ALL_GATHER_ST[dataType];
    const u32 rankSize = AnalyseRankSize(topo);

    // 段3:多线程运行 AllGather 算子,每线程模拟一个 rank
    std::vector<std::thread> threads;
    for (u32 rankId = 0; rankId < rankSize; ++rankId) {
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
            u64 sendBufSize = sendCount * dtSize;             // 输入 1 份
            u64 recvBufSize = sendCount * dtSize * rankSize;  // 输出 N 份
            // 打桩实现,仿真运行需标记内存是 INPUT 和 OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            // 4.算子下发
            CHK_RET(HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    // 等待多线程执行完成
    for (auto &thread : threads) {
        thread.join();
    }

    // 段4:结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllGather(taskQueues, rankSize, dataType, sendCount);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 段5:资源清理
    SimWorld::Global()->Deinit();
}

// ---- 单层 MESH_1D(level0 单 server)→ InsAllGatherMesh1D ----
TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_2rank_int64_small_data_test)
{
    TopoMeta topoMeta {{{0, 1}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_3rank_int64_small_data_test)
{
    TopoMeta topoMeta {{{0, 1, 2}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_4rank_int32_small_data_test)
{
    TopoMeta topoMeta {{{0, 1, 2, 3}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_6rank_int16_small_data_test)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 5}}};  // 非连续 PhyId
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_8rank_int8_small_data_test)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

// ---- 多层每 server 1 卡 → InsAllGatherNHR ----
TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_2rank_fp64_small_data_test)
{
    TopoMeta topoMeta {{{0}, {0}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP64;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_3rank_fp32_small_data_test)
{
    TopoMeta topoMeta {{{0}, {0}, {0}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_4rank_fp16_small_data_test)
{
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_8rank_bfp16_small_data_test)
{
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_6rank_fp8e5m2_small_data_test)
{
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}, {0}, {0}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_6rank_fp8e8m0_big_data_test)
{
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}, {0}, {0}}};
    u64 sendCount = 1024ULL * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

// ---- 多层多卡(MESH_1D 内嵌)→ 小数据 Parallel / 大数据 Sequence ----
TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x2rank_small_data_test)
{
    TopoMeta topoMeta {{{0, 1}, {0, 1}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x3rank_small_data_test)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x3rank_big_data_test)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}}};
    u64 sendCount = 300ULL * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT16;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_3x3rank_big_data_test)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    u64 sendCount = 300ULL * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}
