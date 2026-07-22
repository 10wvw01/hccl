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
#include "alg_env_config.h"
#include <thread>
#include <vector>

using namespace HcclSim;
using namespace ops_hccl;

constexpr uint32_t DATATYPE_SIZE_TABLE_REDUCE_2LEVEL[HCCL_DATA_TYPE_RESERVED] = {
    sizeof(int8_t), sizeof(int16_t), sizeof(int32_t), 2, sizeof(float), sizeof(int64_t), sizeof(uint64_t),
    sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t), 8, 2, 16, 2, 1, 1, 1, 1};

class ST_REDUCE_2LEVEL_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }

    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_INDEPENDENT_OP");
    }

    static void SetUpTestCase()
    {}

    static void TearDownTestCase()
    {}
};

void RunReduce2LevelA5(const TopoMeta &topoMeta, const u64 &recvCount, const HcclDataType &dataType,
    const HcclReduceOp &reduceOp, const uint32_t root)
{
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    auto rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_REDUCE_2LEVEL[dataType];
    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);

            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void *sendBuf = nullptr;
            void *recvBuf = nullptr;
            u64 sendBufSize = recvCount * dataTypeSize * rankSize;
            u64 recvBufSize = recvCount * dataTypeSize;
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            CHK_RET(HcclReduce(sendBuf, recvBuf, recvCount, dataType, reduceOp, root, comm, stream));

            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckReduce(taskQueues, rankSize, dataType, recvCount, reduceOp, root);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x2x8_int32_sum_root0_basic)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x4x8_fp32_sum_root_mid_tail)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 8);
    auto recvCount = 200 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 17;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x3x4_int32_max_root_last)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 3, 4);
    auto recvCount = 513;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    uint32_t root = 11;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x2x4_bfp16_min_root_nonzero)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 4);
    auto recvCount = 300;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    uint32_t root = 5;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x2x8_fp32_sum_recv1_root_last)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    auto recvCount = 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 15;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x4x2_int8_sum_recv8_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 2);
    auto recvCount = 8 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 6;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x3x5_fp32_min_non_power_topo_recv4_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 3, 5);
    auto recvCount = 4 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    uint32_t root = 7;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x2x2_fp32_sum_multi_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 2);
    auto recvCount = 1 * 1024 * 1024 + 73;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 3;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x8x8_int16_max_root_last)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 8, 8);
    auto recvCount = 64 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    uint32_t root = 63;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x2x8_bfp16_sum_recv100k_plus_8)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    auto recvCount = 100 * 1024 + 8;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x5x3_int8_sum_odd_topo_root_mid)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 5, 3);
    auto recvCount = 262;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 8;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x1x8_int32_sum_single_server)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 3;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x4x1_fp32_sum_l0_degenerate)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 1);
    auto recvCount = 128 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

TEST_F(ST_REDUCE_2LEVEL_TEST, st_reduce_2level_1x1x1_fp32_sum_single_rank)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 1);
    auto recvCount = 17;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce2LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}
