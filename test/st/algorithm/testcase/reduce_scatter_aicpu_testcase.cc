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

// HcclReduceScatter 算子 AICPU 展开模式系统测试(机型 A5 / DEV_TYPE_950)
// 覆盖维度:不同拓扑(单超节点/多超节点、不同 Server 数与 Device 数)、
//          不同数据类型、不同数据量、不同规约类型(reduceOp)。
class ST_REDUCE_SCATTER_AICPU_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

namespace {
// 按数据类型返回单元字节数
size_t DataUnitSizeOf(HcclDataType dataType)
{
    switch (dataType) {
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case HcclDataType::HCCL_DATA_TYPE_INT16:
        case HcclDataType::HCCL_DATA_TYPE_FP16:
        case HcclDataType::HCCL_DATA_TYPE_BFP16:
            return sizeof(int16_t);
        case HcclDataType::HCCL_DATA_TYPE_INT32:
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return sizeof(float);
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return sizeof(int64_t);
        default:
            return sizeof(float);
    }
}

// 公共执行体:在 AICPU 展开模式下,按给定拓扑/数据类型/数据量/规约类型运行 ReduceScatter 并做成图校验。
void RunReduceScatterAicpu(const TopoMeta &topoMeta, HcclDataType dataType, u64 recvCount,
    HcclReduceOp reduceOp)
{
    // 1.参与集合通信的卡数(同 topoMeta 卡数一致)
    auto rankSize = 0;
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    // 2.仿真模型初始化(机型 A5 / DEV_TYPE_950)
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    // 3.设置展开模式为 AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    size_t dataUnitSize = DataUnitSizeOf(dataType);

    // 4.多线程运行 ReduceScatter 算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现,仿真运行需标记内存是 INPUT 和 OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
            // 销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }

    // 5.结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 6.资源清理
    SimWorld::Global()->Deinit();
}
}  // namespace

// ============================ 拓扑维度 ============================
// 单超节点 单 Server 8 卡(纯卡间)
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, topo_single_server_8dev)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_FP32, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

// 单超节点 多 Server 各 2 卡
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, topo_single_supernode_4server_2dev)
{
    TopoMeta topoMeta {{{0, 1}, {0, 1}, {0, 1}, {0, 1}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_FP32, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

// 多超节点 各 1 Server 3 卡
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, topo_3supernode_3dev)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_FP32, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

// 多超节点 各 1 Server 4 卡
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, topo_4supernode_4dev)
{
    TopoMeta topoMeta {{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_FP32, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

// 多超节点 单卡(纯超节点间/Server 间)
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, topo_8supernode_1dev)
{
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_FP32, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

// ============================ 数据类型维度 ============================
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, dtype_int8)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_INT8, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, dtype_int16)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_INT16, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, dtype_int32)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_INT32, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, dtype_fp16)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_FP16, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, dtype_bfp16)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_BFP16, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

// ============================ 数据量维度 ============================
// 极小数据量
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, count_tiny_1)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_INT8, 1,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

// 小数据量
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, count_small_1k)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_FP32, 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

// 大数据量
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, count_large_100m)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_FP32, 100 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_SUM);
}

// ============================ 规约类型维度 ============================
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, reduceop_max)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_FP32, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, reduceop_min)
{
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunReduceScatterAicpu(topoMeta, HcclDataType::HCCL_DATA_TYPE_FP32, 1 * 1024 * 1024,
        HcclReduceOp::HCCL_REDUCE_MIN);
}

// 注:ReduceScatter 不支持 HCCL_REDUCE_PROD(算子下发即返回失败),故规约类型仅覆盖 SUM/MAX/MIN。
