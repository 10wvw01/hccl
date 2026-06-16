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

// 各数据类型对应的单元字节数(下标即 HcclDataType 枚举值)
constexpr uint32_t DATATYPE_SIZE_TABLE_REDUCE_SCATTER[HCCL_DATA_TYPE_RESERVED] = {
    sizeof(int8_t),   // HCCL_DATA_TYPE_INT8
    sizeof(int16_t),  // HCCL_DATA_TYPE_INT16
    sizeof(int32_t),  // HCCL_DATA_TYPE_INT32
    2,                // HCCL_DATA_TYPE_FP16
    sizeof(float),    // HCCL_DATA_TYPE_FP32
    sizeof(int64_t),  // HCCL_DATA_TYPE_INT64
    sizeof(uint64_t), // HCCL_DATA_TYPE_UINT64
    sizeof(uint8_t),  // HCCL_DATA_TYPE_UINT8
    sizeof(uint16_t), // HCCL_DATA_TYPE_UINT16
    sizeof(uint32_t), // HCCL_DATA_TYPE_UINT32
    8,                // HCCL_DATA_TYPE_FP64
    2,                // HCCL_DATA_TYPE_BFP16
    16,               // HCCL_DATA_TYPE_INT128
    2, 1, 1, 1, 1
};

// 统计拓扑中参与通信的 rank 总数(遍历所有 pod / server / device)
static u32 CalsRankSize(const TopoMeta& topo)
{
    u32 n = 0;
    for (auto& pod : topo) {
        for (auto& srv : pod) {
            n += srv.size();
        }
    }
    return n;
}

// 公共执行函数(五段式):初始化 -> 设置 AICPU 展开模式 -> 多线程下发 -> 成图校验 -> 清理
static void RunReduceScatterAicpuA5(const TopoMeta& topo, const u64& recvCount, const HcclDataType& dataType,
    const HcclReduceOp& reduceOp)
{
    // 段1:仿真模型初始化,机型 A5(DEV_TYPE_950)
    SimWorld::Global()->Init(topo, DevType::DEV_TYPE_950);

    // 段2:设置展开模式为 AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    const u32 dtSize = DATATYPE_SIZE_TABLE_REDUCE_SCATTER[dataType];
    const u32 rankSize = CalsRankSize(topo);

    // 段3:多线程模拟各 rank 下发 ReduceScatter 算子
    std::vector<std::thread> threads;
    for (u32 rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));
            void *sendBuf = nullptr;
            void *recvBuf = nullptr;
            u64 sendBufSize = recvCount * dtSize * rankSize;  // 输入 N 份(与 AllGather 相反)
            u64 recvBufSize = recvCount * dtSize;             // 输出 1 份
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    // 段4:结果成图 + 语义校验
    auto q = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckReduceScatter(q, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 段5:资源清理
    SimWorld::Global()->Deinit();
}

class ST_REDUCE_SCATTER_AICPU_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

// ===================== mesh:单层 MESH 算法族 =====================
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_3rank_int8_sum_test)
{
    TopoMeta topoMeta{{{0, 1, 2}}};
    auto recvCount = 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_3rank_int16_min_test)
{
    TopoMeta topoMeta{{{0, 1, 2}}};
    auto recvCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_8rank_fp32_max_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};
    auto recvCount = 1 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_8rank_fp16_sum_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};
    auto recvCount = 100 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_4rank_int32_min_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto recvCount = 1 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

// ===================== nhr:多层每层 1 卡 算法族 =====================
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_nhr_2rank_fp32_sum_test)
{
    TopoMeta topoMeta{{{0}, {0}}};
    auto recvCount = 1 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_nhr_2rank_int8_max_test)
{
    TopoMeta topoMeta{{{0}, {0}}};
    auto recvCount = 512;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_nhr_8rank_fp32_min_test)
{
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};
    auto recvCount = 1 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_nhr_8rank_bfp16_sum_test)
{
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};
    auto recvCount = 100 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_nhr_4rank_int16_max_test)
{
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}};
    auto recvCount = 1 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

// ===================== meshnhr:多层多卡 算法族 =====================
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x2rank_fp32_sum_test)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    auto recvCount = 1 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x2rank_int32_max_test)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    auto recvCount = 2048;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_3x2rank_fp32_min_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};
    auto recvCount = 300 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_3x2rank_fp16_sum_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};
    auto recvCount = 1 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_3x3rank_fp32_max_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    auto recvCount = 1 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_3x3rank_int8_sum_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    auto recvCount = 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_3x3rank_bfp16_min_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    auto recvCount = 1 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_4x4rank_fp32_sum_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};
    auto recvCount = 1 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_4x3rank_int16_max_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    auto recvCount = 1 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

// ===================== meshchunk:单层大数据分块 算法族 =====================
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_4rank_fp32_sum_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto recvCount = 400 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_4rank_fp16_min_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto recvCount = 400 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_8rank_int32_max_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};
    auto recvCount = 200 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_4rank_bfp16_sum_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto recvCount = 300 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}
