/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * AllGatherMeshTemplate 单元测试。
 * 覆盖: RunAlgorithm (委托 RunMeshAllGather) / KernelRun 完整流程。
 */

#include "test_helpers.h"

namespace ops_hccl {
namespace testing {

// ═══════════════════════════════════════════════════════════════════
// 1. RunAlgorithm 分组
// ═══════════════════════════════════════════════════════════════════

// TC01 单 rank RunAlgorithm 不生成 txRxSlicesLists
TEST(AllGatherMeshRunAlgorithmTest, SingleRankNoTransfer)
{
    AllGatherMeshTemplate tmpl(0, {0}, TemplateDesc{HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.scratchStride = 16;
    tmpl.tempAlgParams_.dataStride = 16;
    tmpl.tempAlgParams_.ranksForInputData = {0};
    tmpl.templateRankSize_ = 1;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
}

// TC02 多 rank RunAlgorithm 生成 txRxSlicesLists
TEST(AllGatherMeshRunAlgorithmTest, MultiRankBuildsTxRxSlicesLists)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    AllGatherMeshTemplate tmpl(0, ranks, TemplateDesc{HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.scratchStride = 16;
    tmpl.tempAlgParams_.dataStride = 16;
    tmpl.tempAlgParams_.ranksForInputData = {0};
    tmpl.templateRankSize_ = 4;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 3 对端 rank → 3 个 txRxSlicesLists
    EXPECT_EQ(txRxSlicesLists.size(), 3u);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 1u);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 2u);
    EXPECT_EQ(txRxSlicesLists[2].dstRankId_, 3u);
}

// TC03 空 ranksForInputData 返回错误
TEST(AllGatherMeshRunAlgorithmTest, EmptyInputRanksReturnsError)
{
    AllGatherMeshTemplate tmpl(0, {0, 1}, TemplateDesc{HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.scratchStride = 16;
    tmpl.tempAlgParams_.dataStride = 16;
    // ranksForInputData 为空
    tmpl.templateRankSize_ = 2;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// ═══════════════════════════════════════════════════════════════════
// 2. KernelRun 完整流程分组
// ═══════════════════════════════════════════════════════════════════

// TC04 KernelRun 多 rank 调用 engine.Send
TEST_F(AicpuBaseTemplateTest, AllGatherMeshKernelRunMultiRankCallsSend)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    AllGatherMeshTemplate tmpl(0, ranks, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // RunMeshAllGather 为 3 个对端生成 txRxSlicesLists → engine.Send 调用 3 次
    EXPECT_EQ(engine_.GetSendCount(), 3u);
    // ranksForOutputData 应包含所有 rank
    EXPECT_EQ(ranksForOutputData.size(), 4u);
}

// TC05 KernelRun 单 rank 不调用 engine.Send
TEST_F(AicpuBaseTemplateTest, AllGatherMeshKernelRunSingleRankNoSend)
{
    AllGatherMeshTemplate tmpl(0, {0}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(engine_.GetSendCount(), 0u);
}

} // namespace testing
} // namespace ops_hccl
