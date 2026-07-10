/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "v_testcase_common.h"
#include "gtest/gtest.h"

#include "alg_env_config.h"
#include <algorithm>

class ST_ALL_GATHER_V_TEST : public ::testing::Test {

protected:
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_ST_FAIL_LOCAL_COPY_ON_THREAD");
    }
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    static void TearDownTestCase()
    {}
    static void SetUpTestCase()
    {}
};

static HcclResult AllGatherVDispatch(u32 rankId, u64 totalCount, VDataDesTag vDataDes,
    HcclComm comm, aclrtStream stream)
{
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    u64 sendBufSize = vDataDes.counts[rankId] * sizeof(vDataDes.dataType);
    u64 recvBufSize = totalCount * sizeof(vDataDes.dataType);
    aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
    aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
    return HcclAllGatherV(sendBuf, vDataDes.counts[rankId], recvBuf, vDataDes.counts.data(),
        vDataDes.displs.data(), vDataDes.dataType, comm, stream);
}

static void RunAllGatherVMultilevel(const TopoMeta &topoInfo, VDataDesTag vDataDes)
{
    RunVMultilevelTest(topoInfo, vDataDes, nullptr, AllGatherVDispatch, CheckAllGatherV);
}

// 运行 AllGatherV AICPU 并注入本地拷贝失败
// topoInfo 表示仿真拓扑
// vDataDes 表示 AllGatherV 的 counts、displs 和数据类型
// 返回值表示是否观测到非成功返回码
static bool RunAllGatherVLocalCopyFailure(const TopoMeta &topoInfo, VDataDesTag vDataDes)
{
    SimWorld::Global()->Init(topoInfo, DevType::DEV_TYPE_950);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_ST_FAIL_LOCAL_COPY_ON_THREAD", "1", 1);

    auto rankSize = AnalyseRankSize(topoInfo);
    std::vector<HcclResult> results(rankSize, HCCL_SUCCESS);
    std::vector<std::thread> threads;
    for (u32 rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=, &results]() {
            aclrtSetDevice(rankId);
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);
            HcclComm comm = nullptr;
            HcclResult ret = HcclCommInitClusterInfo("./ranktable.json", rankId, &comm);
            if (ret == HCCL_SUCCESS) {
                void *sendBuf = nullptr;
                void *recvBuf = nullptr;
                u64 sendBufSize = vDataDes.counts[rankId] * sizeof(int32_t);
                u64 recvBufSize = (vDataDes.displs.back() + vDataDes.counts.back()) * sizeof(int32_t);
                aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
                aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
                ret = HcclAllGatherV(sendBuf, vDataDes.counts[rankId], recvBuf, vDataDes.counts.data(),
                    vDataDes.displs.data(), vDataDes.dataType, comm, stream);
                HcclCommDestroy(comm);
            }
            results[rankId] = ret;
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    SimWorld::Global()->Deinit();
    unsetenv("HCCL_ST_FAIL_LOCAL_COPY_ON_THREAD");
    return std::any_of(results.begin(), results.end(), [](HcclResult ret) { return ret != HCCL_SUCCESS; });
}

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_aicpu_test)
{
    TopoMeta topoMeta{{{0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {89478485, 178956970};
    vDataDes.displs = {0, 89478485};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_mesh_1d_localcopy_fail_retcode_test)
{
    TopoMeta topoMeta{{{0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100};
    vDataDes.displs = {0, 100};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    EXPECT_TRUE(RunAllGatherVLocalCopyFailure(topoMeta, vDataDes));
}

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_multilevel_2pod_4rank_int32_equal_test)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100};
    vDataDes.displs = {0, 100, 200, 300};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_multilevel_2pod_6rank_fp16_equal_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {3, 4, 5}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {200, 200, 200, 200, 200, 200};
    vDataDes.displs = {0, 200, 400, 600, 800, 1000};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

// 3-pod topologies unsupported: simulator has only 2 net layers (no inter-superpod links at layer 2).
// Unequal AllGatherV unsupported: mesh1D slave streams end with WAIT not LOCAL_POST_TO (pre-existing bug).

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_3layer_2pod_2server_8rank_fp16_equal_test)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}, {{0, 1}, {0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {200, 200, 200, 200, 200, 200, 200, 200};
    vDataDes.displs = {0, 200, 400, 600, 800, 1000, 1200, 1400};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_3layer_2pod_1server_4rank_int32_equal_test)
{
    TopoMeta topoMeta{{{0, 1}}, {{0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100};
    vDataDes.displs = {0, 100, 200, 300};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_3layer_2pod_1server_8rank_fp16_equal_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}, {{0, 1, 2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100, 100, 100, 100, 100};
    vDataDes.displs = {0, 100, 200, 300, 400, 500, 600, 700};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

