/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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
#include "check_utils.h"
#include "hccl_verifier.h"
#include <thread>
#include "alg_env_config.h"

using namespace HcclSim;
using namespace ops_hccl;

constexpr uint32_t DATATYPE_SIZE_TABLE_REDUCE_SCATTER[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

// 4层(net_layer 0/1/2/3, SuperNode 级 OCS) ReduceScatter ST。
// 通过 HCCL_SIM_ENABLE_LEVEL3=1 让 ST 模拟器追加第4层(net_layer_3)建模，
// 由 selector 的 level3Ocs 标志位选中 InsReduceScatterSequenceMesh1DNHRNHRMesh1D。
class ST_REDUCE_SCATTER_4LEVEL_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        // 显式启用 ST 模拟器的第4层(net_layer_3)建模
        setenv("HCCL_SIM_ENABLE_LEVEL3", "1", 1);
    }
    void TearDown() override
    {
        unsetenv("HCCL_SIM_ENABLE_LEVEL3");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_OP_EXPANSION_MODE");

    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

void RunReduceScatter4LevelA5(const TopoMeta &topoMeta, const u64 &recvCount, const HcclDataType &dataType,
    const HcclReduceOp &reduceOp)
{
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);

    auto rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_REDUCE_SCATTER[dataType];
    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);

            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void *recvBuf = nullptr;
            void *sendBuf = nullptr;
            u64 sendBufSize = recvCount * dataTypeSize * rankSize;
            u64 recvBufSize = recvCount * dataTypeSize;

            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));

            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));

            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    SimWorld::Global()->Deinit();
}

// P0: #1 - 4-level basic correctness on multi-pod topology (8x8x2 => 4层)
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_8x8x2_fp32_sum_basic)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 8, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P0: #2 - 4-level correctness with 3 pods (inter3 repeatNum>1 path)
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_4x4x3_fp32_sum_repeatnum_gt1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 4, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P1: #3 - different scale, int32 max op
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_4x4x2_int32_max_different_scale)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto recvCount = 500;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P1: #4 - small-scale large-data multi-loop segmentation
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_4x2x2_fp32_sum_multi_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto recvCount = 500 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P1: #5 - boundary rank, recvCount=200+1
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_8x2x2_fp32_sum_small_cluster_recv200_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 200 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P2: #6 - higher repeatNum (repeatNum=L3=4)
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_4x2x4_int32_sum_repeatnum4)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 2, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P2: #7 - BFP16 data type
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_4x2x2_bfp16_max_dtype)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    auto recvCount = 300;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P2: #8 - min op on 4-level topology
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_4x4x3_fp32_min_level3_3cluster)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 4, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P2: #9 - asymmetric dimensions across 4 levels
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_4x3x2_int32_sum_asymmetric_all)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// --- Degenerate Level (dimension=1) edge cases ---

// L1=1: single server per pod, degenerate L1, min op
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_8x1x3_fp32_min_l1_degenerate)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 1, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// L0=1: degenerate L0 + recvCount=8+1
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_1x2x3_int8_sum_l0_degenerate_recv8_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 1);
    auto recvCount = 8 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// --- recvCount aligned boundary + 1 cases ---

// recvCount=4+1=5: stride slicing remainder
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_4x3x2_fp32_min_recv4_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto recvCount = 4 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// recvCount=64K+1: loop slicing remainder
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_4x4x2_int16_max_recv64k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto recvCount = 64 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// L0=1 + recvCount=128K+1: degenerate L0 + large data with remainder
TEST_F(ST_REDUCE_SCATTER_4LEVEL_TEST, st_reduce_scatter_4level_1x4x2_fp32_sum_l0_degenerate_recv128k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    auto recvCount = 128 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter4LevelA5(topoMeta, recvCount, dataType, reduceOp);
}
