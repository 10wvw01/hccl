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

constexpr uint32_t DATATYPE_SIZE_TABLE_AG_ZDETOUR[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_ALL_GATHER_ZDETOUR_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }

    void TearDown() override
    {
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
    }

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

u32 AnalyseAllGatherZDetourRankSize(const TopoMeta &topoInfo)
{
    u32 rankSize = 0;
    for (const auto &superPod : topoInfo) {
        for (const auto &server : superPod) {
            rankSize += server.size();
        }
    }
    return rankSize;
}

void RunAllGatherZDetourAicpuA5(const TopoMeta &topoInfo, const u64 &sendCount, const HcclDataType &dataType)
{
    SimWorld::Global()->Init(topoInfo, DevType::DEV_TYPE_950);

    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_AG_ZDETOUR[dataType];
    auto rankSize = AnalyseAllGatherZDetourRankSize(topoInfo);
    std::vector<std::thread> threads;
    for (auto rankIdx = 0; rankIdx < rankSize; ++rankIdx) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankIdx);

            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankIdx, &comm));

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

    for (auto &thread : threads) {
        thread.join();
    }

    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllGather(taskQueues, rankSize, dataType, sendCount);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    SimWorld::Global()->Deinit();
}

TEST_F(ST_ALL_GATHER_ZDETOUR_TEST, st_all_gather_a5_aicpu_mesh_1d_8rank_fp32_basic_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherZDetourAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_ZDETOUR_TEST, st_all_gather_a5_aicpu_mesh_1d_2rank_int64_small_data_test)
{
    TopoMeta topoMeta{{{0, 1}}};
    auto sendCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64;
    RunAllGatherZDetourAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_ZDETOUR_TEST, st_all_gather_a5_aicpu_mesh_1d_4rank_int32_recv1_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto sendCount = 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGatherZDetourAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_ZDETOUR_TEST, st_all_gather_a5_aicpu_mesh_1d_8rank_bfp16_tail_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};
    auto sendCount = 200 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGatherZDetourAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_ZDETOUR_TEST, st_all_gather_a5_aicpu_mesh_1d_8rank_int8_multi_loop_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};
    auto sendCount = 1 * 1024 * 1024 + 73;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherZDetourAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_ZDETOUR_TEST, st_all_gather_a5_aicpu_zdetour_mesh1d_2x2rank_fp32_big_data_test)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    auto sendCount = 270 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherZDetourAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_ZDETOUR_TEST, st_all_gather_a5_aicpu_zdetour_mesh1d_2x3rank_int16_big_data_tail_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};
    auto sendCount = 360 * 1024 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    RunAllGatherZDetourAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_ZDETOUR_TEST, st_all_gather_a5_aicpu_zdetour_mesh1d_2x2rank_bfp16_big_data_tail_test)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    auto sendCount = 540 * 1024 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGatherZDetourAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_ZDETOUR_TEST, st_all_gather_a5_aicpu_zdetour_mesh1d_2x4rank_uint16_big_data_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}};
    auto sendCount = 270 * 1024 * 1024 + 3;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT16;
    RunAllGatherZDetourAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_ZDETOUR_TEST, st_all_gather_a5_aicpu_zdetour_mesh1d_3x3rank_uint8_big_data_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    auto sendCount = 512 * 1024 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;
    RunAllGatherZDetourAicpuA5(topoMeta, sendCount, dataType);
}
