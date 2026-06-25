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

constexpr uint32_t DATATYPE_SIZE_TABLE_ALL_GATHER_4LEVEL[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t),
    sizeof(int32_t), 2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t),
    sizeof(uint32_t), 8, 2, 16, 2, 1, 1, 1, 1};

// 4层(net_layer 0/1/2/3, SuperNode 级 OCS) AllGather ST。
// 通过 HCCL_SIM_ENABLE_LEVEL3=1 让 ST 模拟器追加第4层(net_layer_3)建模，
// 由 selector 的 level3Ocs 标志位选中 InsAllGatherSequenceNHRNHRMesh1DOcs。
class ST_ALL_GATHER_4LEVEL_TEST : public ::testing::Test {
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

void RunAllGather4LevelA5(const TopoMeta &topoMeta, const u64 &sendCount, const HcclDataType &dataType)
{
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);

    auto rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_ALL_GATHER_4LEVEL[dataType];
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
            u64 sendBufSize = sendCount * dataTypeSize;
            u64 recvBufSize = sendCount * dataTypeSize * rankSize;
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            CHK_RET(HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream));

            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllGather(taskQueues, rankSize, dataType, sendCount);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    SimWorld::Global()->Deinit();
}

// P0: #1 - 4-level basic correctness on multi-pod topology (8x8x2 => 4层)
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_8x8x2_fp32_basic)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 8, 8);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// P0: #2 - 4-level correctness with 3 pods (inter3 repeatNum>1 path)
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_4x4x3_fp32_repeatnum_gt1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 4, 4);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// P1: #3 - different scale, int32
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_4x4x2_int32_different_scale)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto sendCount = 500;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// P1: #4 - small-scale large-data multi-loop segmentation
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_4x2x2_fp32_multi_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto sendCount = 500 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// P1: #5 - boundary rank, sendCount=200+1
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_8x2x2_fp32_small_cluster_send200_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto sendCount = 200 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// P2: #6 - higher repeatNum (repeatNum=L3=4)
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_4x2x4_int32_repeatnum4)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 2, 4);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// P2: #7 - BFP16 data type
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_4x2x2_bfp16_dtype)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    auto sendCount = 300;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// P2: #8 - 4-level topology on 3 clusters
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_4x4x3_fp32_level3_3cluster)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 4, 4);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// P2: #9 - asymmetric dimensions across 4 levels
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_4x3x2_int32_asymmetric_all)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// --- Degenerate Level (dimension=1) edge cases ---

// L1=1: single server per pod, degenerate L1
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_8x1x3_fp32_l1_degenerate)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 1, 8);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// L0=1: degenerate L0 + sendCount=8+1
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_1x2x3_int8_l0_degenerate_send8_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 1);
    auto sendCount = 8 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// --- sendCount aligned boundary + 1 cases ---

// sendCount=4+1=5: stride slicing remainder
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_4x3x2_fp32_send4_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto sendCount = 4 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// sendCount=64K+1: loop slicing remainder
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_4x4x2_int16_send64k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto sendCount = 64 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}

// L0=1 + sendCount=128K+1: degenerate L0 + large data with remainder
TEST_F(ST_ALL_GATHER_4LEVEL_TEST, st_allgather_4level_1x4x2_fp32_l0_degenerate_send128k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    auto sendCount = 128 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather4LevelA5(topoMeta, sendCount, dataType);
}
