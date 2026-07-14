/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Selector 函数测试：验证 AllGatherAutoSelector 的算法选择正确性。
 * 两个用例分别验证返回 SEQUENCE 与 PARALLEL 执行策略算法。
 */

#include "test_helpers.h"

namespace ops_hccl {

// 用例1：Selector 返回 SEQUENCE 算法 (AICPU_ALLGATHER_SEQUENCE_NHR_MESH1D)
// 条件：dataSize > 1MB 且 dataSize * userRankSize > 4GB
TEST_F(SelectorTest, SelectReturnsSequenceAlgorithm)
{
    // count=2M FP32 -> dataSize=8MB; userRankSize=1024 -> 8GB > 4GB
    auto param = MakeAicpuAllGatherParam(2 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
    auto topoInfo = MakeTwoLevelMesh1DTopo(1024);
    HcclAlgorithm alg;

    HcclResult ret = Selector(nullptr, *param, topoInfo, alg);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(alg.algoExecDesc.execPolicy, HcclAlgExecPolicy::SEQUENCE);
    EXPECT_EQ(alg.engineType, HcclAlgEngineType::AICPU);
    EXPECT_EQ(alg.hcclCmdType, HcclCMDType::HCCL_CMD_ALLGATHER);
}

// 用例2：Selector 返回 PARALLEL 算法 (AICPU_ALLGATHER_PARALLEL_MESH1D_NHR)
// 条件：dataSize > 1MB 且 dataSize * userRankSize <= 4GB
TEST_F(SelectorTest, SelectReturnsParallelAlgorithm)
{
    // count=2M FP32 -> dataSize=8MB; userRankSize=2 -> 16MB <= 4GB
    auto param = MakeAicpuAllGatherParam(2 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
    auto topoInfo = MakeTwoLevelMesh1DTopo(2);
    HcclAlgorithm alg;

    HcclResult ret = Selector(nullptr, *param, topoInfo, alg);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(alg.algoExecDesc.execPolicy, HcclAlgExecPolicy::PARALLEL);
    EXPECT_EQ(alg.engineType, HcclAlgEngineType::AICPU);
    EXPECT_EQ(alg.hcclCmdType, HcclCMDType::HCCL_CMD_ALLGATHER);
}

} // namespace ops_hccl
