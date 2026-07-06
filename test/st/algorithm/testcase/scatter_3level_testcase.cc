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

constexpr uint32_t DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_SCATTER_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

uint64_t CountElements(const TopoMeta &topoMeta) {
    uint64_t total = 0;
    for (const auto& vec2d : topoMeta) {
        for (const auto& vec1d : vec2d) {
            // 直接加上最内层 vector 的大小
            total += vec1d.size();
        }
    }
    return total;
}

void RunScatterTest(int root, TopoMeta &topoMeta, int dataCount, HcclDataType dataType) 
{
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
    
    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto rankSize = CountElements(topoMeta);  // 参与集合通信的卡数(同topoMeta卡数一致)
    auto recvCount = dataCount;  // 接收数据量
    // auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型

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
            u64 sendDataSize = recvCount * DATATYPE_SIZE_TABLE[dataType] * rankSize;
            u64 recvDataSize = recvCount * DATATYPE_SIZE_TABLE[dataType];
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendDataSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvDataSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            // 4.算子下发
            CHK_RET(HcclScatter(sendBuf, recvBuf, recvCount, dataType, root, comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    // 等待多线程执行完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckScatter(taskQueues, rankSize, dataType, recvCount, root);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit(); 
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh_1d_success_2x8x8_root0_fp32_small_data)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 8, 8);
    RunScatterTest(0, topoMeta, 200, HcclDataType::HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_3x4x4_fp32_sum_repeatnum_gt1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 4, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_2level_backward_compat_meshnhr)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_2level_backward_compat_meshnhr_2)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 4);
    auto recvCount = 400 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_2x4x4_int32_max_different_scale)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto recvCount = 500;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_2x2x2_fp32_sum_multi_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto recvCount = 500 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_2x2x8_fp32_sum_small_cluster_recv200_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 200 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_4x2x4_int32_sum_repeatnum4)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 2, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_2x2x4_bfp16_max_dtype)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    auto recvCount = 300;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_3x2x8_fp32_sum_level2_3cluster)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_2x3x4_int32_sum_asymmetric_all)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

// --- Degenerate Level (dimension=1) edge cases ---

TEST_F(ST_SCATTER_TEST, st_scatter_3level_3x1x8_fp32_min_l1_degenerate)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 1, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_2x1x4_int8_sum_l1_degenerate_recv8_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    auto recvCount = 8 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_4x1x1_fp32_sum_double_degenerate_recv16_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 1, 1);
    auto recvCount = 16 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_2x3x4_fp32_min_recv4_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto recvCount = 4 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_2x4x4_int16_max_recv64k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto recvCount = 64 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_TEST, st_scatter_3level_2x4x1_fp32_sum_l0_degenerate_recv128k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    auto recvCount = 128 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    
    RunScatterTest(0, topoMeta, recvCount, dataType);
}