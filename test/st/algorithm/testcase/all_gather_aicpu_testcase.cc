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

// 数据类型字节数表，索引同 HcclDataType 枚举顺序
constexpr uint32_t DATATYPE_SIZE_TABLE_AG_AICPU_ST[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t),
    sizeof(int32_t), 2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t),
    sizeof(uint32_t), 8, 2, 16, 2, 1, 1, 1, 1};

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

// AICPU 展开模式下运行 HcclAllGather 并做成图校验，机型为 A5(DEV_TYPE_950)
void RunAllGatherAicpuA5(const TopoMeta &topoMeta, u64 sendCount, HcclDataType dataType)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    // 设置展开模式为 AI_CPU(独立算子展开，区别于 host-DPU 模式)
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    // 算子执行参数设置
    uint32_t rankSize = 0;  // 参与集合通信的卡数(同 topoMeta 卡数一致)
    for (auto &&superPod : topoMeta) {
        for (auto &&server : superPod) {
            rankSize += server.size();
        }
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_AG_AICPU_ST[dataType];

    // 多线程运行 AllGather 算子
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
            u64 sendBufSize = sendCount * dataTypeSize;             // 数据量转化为字节数
            u64 recvBufSize = sendCount * dataTypeSize * rankSize;  // AllGather 输出为 rankSize 倍
            // 打桩实现，仿真运行需标记内存是 INPUT 和 OUTPUT
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
    for (auto& thread : threads) {
        thread.join();
    }

    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllGather(taskQueues, rankSize, dataType, sendCount);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

// ---------------------------------------------------------------------------
// 拓扑维度覆盖：单卡、单机多卡、多机、多超节点
// ---------------------------------------------------------------------------

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_single_rank_fp32)
{
    TopoMeta topoMeta{{{0}}};  // 1 超节点 / 1 server / 1 device
    u64 sendCount = 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_single_server_2p_fp32)
{
    TopoMeta topoMeta{{{0, 1}}};  // 单机 2 卡
    u64 sendCount = 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_single_server_8p_fp16)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 单机 8 卡
    u64 sendCount = 4096;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_two_server_2x2_fp32)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};  // 2 server，每 server 2 卡
    u64 sendCount = 4096;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_three_server_3x2_fp16)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}}};  // 单超节点 3 server，每 server 2 卡
    u64 sendCount = 4096;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_multi_server_4x4_bfp16)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};  // 4 server x 4 卡
    u64 sendCount = 8192;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

// ---------------------------------------------------------------------------
// 数据类型覆盖(固定 2x2 拓扑 + 10M sendCount)
// ---------------------------------------------------------------------------

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_int8)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 10 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_int16)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 10 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_int32)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 10 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_int64)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 10 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT64;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_fp16)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 10 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_fp32)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 10 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_bfp16)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 10 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

// ---------------------------------------------------------------------------
// 数据量覆盖(固定 2x2 拓扑 + fp32)：单元素 / 小 / 中 / 大 / 非对齐
// ---------------------------------------------------------------------------

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_count_1)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_count_odd)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 13;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_count_1m)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 1 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_count_210m)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 210 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, aicpu_all_gather_count_unaligned)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 sendCount = 1 * 1024 * 1024 + 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}
