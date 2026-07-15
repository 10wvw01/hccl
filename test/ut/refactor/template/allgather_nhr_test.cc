/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * AllGatherNhrTemplate 单元测试。
 * 覆盖: RunAlgorithm (委托 RunNhrAllGather) / KernelRun 完整流程。
 */

#include "test_helpers.h"

namespace ops_hccl {
namespace testing {

// ═══════════════════════════════════════════════════════════════════
// 1. RunAlgorithm 分组
// ═══════════════════════════════════════════════════════════════════

// TC01 单 rank RunAlgorithm 不生成 txRxSlicesLists
TEST(AllGatherNhrRunAlgorithmTest, SingleRankNoTransfer)
{
    AllGatherNhrTemplate tmpl(0, {0}, TemplateDesc{HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.stride = 16;
    tmpl.tempAlgParams_.ranksForInputData = {0};
    tmpl.templateRankSize_ = 1;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
}

// TC02 多 rank (4 rank, log2(4)=2 步) RunAlgorithm 生成 txRxSlicesLists
TEST(AllGatherNhrRunAlgorithmTest, FourRankTwoStepsBuildsTxRxSlicesLists)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    AllGatherNhrTemplate tmpl(0, ranks, TemplateDesc{HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.stride = 16;
    tmpl.tempAlgParams_.ranksForInputData = {0};
    tmpl.templateRankSize_ = 4;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 4 rank → 2 步 (ceil(log2(4))=2), 每步 1 个 TxRxSlicesList
    EXPECT_EQ(txRxSlicesLists.size(), 2u);
}

// TC03 空 ranksForInputData 返回错误
TEST(AllGatherNhrRunAlgorithmTest, EmptyInputRanksReturnsError)
{
    AllGatherNhrTemplate tmpl(0, {0, 1}, TemplateDesc{HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.stride = 16;
    // ranksForInputData 为空
    tmpl.templateRankSize_ = 2;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// TC04 8 rank NHR 生成 3 步
TEST(AllGatherNhrRunAlgorithmTest, EightRankThreeStepsBuildsTxRxSlicesLists)
{
    std::vector<u32> ranks = {0, 1, 2, 3, 4, 5, 6, 7};
    AllGatherNhrTemplate tmpl(0, ranks, TemplateDesc{HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.stride = 16;
    tmpl.tempAlgParams_.ranksForInputData = {0};
    tmpl.templateRankSize_ = 8;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 8 rank → 3 步 (ceil(log2(8))=3)
    EXPECT_EQ(txRxSlicesLists.size(), 3u);
}

// ═══════════════════════════════════════════════════════════════════
// 2. KernelRun 完整流程分组
// ═══════════════════════════════════════════════════════════════════

// TC05 KernelRun 多 rank 调用 engine.Send
TEST_F(AicpuBaseTemplateTest, AllGatherNhrKernelRunMultiRankCallsSend)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    AllGatherNhrTemplate tmpl(0, ranks, MakeNhrDesc());
    TemplateDataParams params = MakeTmplParams({0});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 4 rank → 2 步 → 2 次 Send
    EXPECT_EQ(engine_.GetSendCount(), 2u);
}

// TC06 KernelRun 单 rank 不调用 engine.Send
TEST_F(AicpuBaseTemplateTest, AllGatherNhrKernelRunSingleRankNoSend)
{
    AllGatherNhrTemplate tmpl(0, {0}, MakeNhrDesc());
    TemplateDataParams params = MakeTmplParams({0});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(engine_.GetSendCount(), 0u);
}

} // namespace testing
} // namespace ops_hccl
