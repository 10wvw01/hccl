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

constexpr uint32_t DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_BROADCAST_3LEVEL_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}

    void RunBroadcast3LevelTest(TopoMeta topoMeta, u32 rankSize, uint64_t count,
        HcclDataType dataType, u32 root, u32 dataTypeSize)
    {
        SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);

        std::vector<std::thread> threads;
        for (auto rankId = 0u; rankId < rankSize; ++rankId) {
            threads.emplace_back([=]() {
                aclrtSetDevice(rankId);

                aclrtStream stream = nullptr;
                aclrtCreateStream(&stream);

                HcclComm comm = nullptr;
                CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

                void *buf = nullptr;
                u64 bufSize = count * dataTypeSize;
                aclrtMalloc(&buf, bufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));

                CHK_RET(HcclBroadcast(buf, count, dataType, root, comm, stream));

                CHK_RET(HcclCommDestroy(comm));
                return HCCL_SUCCESS;
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
        HcclResult res = CheckBroadcast(taskQueues, rankSize, dataType, count, root);
        EXPECT_TRUE(res == HCCL_SUCCESS);

        SimWorld::Global()->Deinit();
    }
};

TEST_F(ST_BROADCAST_3LEVEL_TEST, st_broadcast_3level_2x2x2_int8_send1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto rankSize = 2 * 2 * 2;
    auto count = 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    auto root = 0;
    auto dataTypeSize = DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[dataType];
    RunBroadcast3LevelTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(ST_BROADCAST_3LEVEL_TEST, st_broadcast_3level_2x2x2_fp32_root0)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto rankSize = 2 * 2 * 2;
    auto count = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto root = 0;
    auto dataTypeSize = DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[dataType];
    RunBroadcast3LevelTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(ST_BROADCAST_3LEVEL_TEST, st_broadcast_3level_4x2x2_int32_root0)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 2, 2);
    auto rankSize = 4 * 2 * 2;
    auto count = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto root = 0;
    auto dataTypeSize = DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[dataType];
    RunBroadcast3LevelTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(ST_BROADCAST_3LEVEL_TEST, st_broadcast_3level_2x4x4_fp16_root0)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto rankSize = 2 * 4 * 4;
    auto count = 300;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    auto root = 0;
    auto dataTypeSize = DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[dataType];
    RunBroadcast3LevelTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(ST_BROADCAST_3LEVEL_TEST, st_broadcast_3level_2x2x2_int8_root3)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto rankSize = 2 * 2 * 2;
    auto count = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    auto root = 3;
    auto dataTypeSize = DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[dataType];
    RunBroadcast3LevelTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(ST_BROADCAST_3LEVEL_TEST, st_broadcast_3level_2x2x2_bfp16_root4)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto rankSize = 2 * 2 * 2;
    auto count = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto root = 4;
    auto dataTypeSize = DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[dataType];
    RunBroadcast3LevelTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(ST_BROADCAST_3LEVEL_TEST, st_broadcast_3level_2x2x2_int64_root7)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto rankSize = 2 * 2 * 2;
    auto count = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64;
    auto root = 7;
    auto dataTypeSize = DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[dataType];
    RunBroadcast3LevelTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(ST_BROADCAST_3LEVEL_TEST, st_broadcast_3level_2x1x4_fp32_skip_l1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    auto rankSize = 2 * 1 * 4;
    auto count = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto root = 0;
    auto dataTypeSize = DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[dataType];
    RunBroadcast3LevelTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(ST_BROADCAST_3LEVEL_TEST, st_broadcast_3level_2x4x2_fp32_multi_server)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 2);
    auto rankSize = 2 * 4 * 2;
    auto count = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto root = 0;
    auto dataTypeSize = DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[dataType];
    RunBroadcast3LevelTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(ST_BROADCAST_3LEVEL_TEST, st_broadcast_3level_2x2x2_fp32_send501m_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto rankSize = 2 * 2 * 2;
    auto count = 501 * 1024 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto root = 0;
    auto dataTypeSize = DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[dataType];
    RunBroadcast3LevelTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(ST_BROADCAST_3LEVEL_TEST, st_broadcast_3level_2x2x2_fp32_send201)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto rankSize = 2 * 2 * 2;
    auto count = 201;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto root = 0;
    auto dataTypeSize = DATATYPE_SIZE_TABLE_BROADCAST_3LEVEL[dataType];
    RunBroadcast3LevelTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}
