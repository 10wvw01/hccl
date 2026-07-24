/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ScatterNhrTemplate 单元测试。
 * 覆盖: RunAlgorithm (委托 RunNhrScatter) / KernelRun 完整流程。
 */

#include "test_helpers.h"

namespace ops_hccl {
namespace testing {

// ═══════════════════════════════════════════════════════════════════
// 1. RunAlgorithm 分组
// ═══════════════════════════════════════════════════════════════════

// TC01 单 rank RunAlgorithm 不生成 txRxSlicesLists
TEST(ScatterNhrRunAlgorithmTest, SingleRankNoTransfer)
{
    ScatterNhrTemplate tmpl(0, {0}, TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.scratchStride = 16;
    tmpl.tempAlgParams_.dataStride = 16;
    tmpl.tempAlgParams_.root = 0;
    tmpl.tempAlgParams_.ranksForInputData = {0};
    tmpl.templateRankSize_ = 1;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
}

// TC02 4 rank (log2(4)=2 步) root rank 生成 txRxSlicesLists
TEST(ScatterNhrRunAlgorithmTest, FourRankRootBuildsTxRxSlicesLists)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    ScatterNhrTemplate tmpl(0, ranks, TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.scratchStride = 16;
    tmpl.tempAlgParams_.dataStride = 16;
    tmpl.tempAlgParams_.root = 0;
    // root 持有全部 rank 的输入数据
    tmpl.tempAlgParams_.ranksForInputData = {0, 1, 2, 3};
    tmpl.templateRankSize_ = 4;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 4 rank → 2 步 → 2 个 txRxSlicesLists
    EXPECT_EQ(txRxSlicesLists.size(), 2u);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({0}));
}

// TC03 8 rank NHR 生成 3 步
TEST(ScatterNhrRunAlgorithmTest, EightRankThreeStepsBuildsTxRxSlicesLists)
{
    std::vector<u32> ranks = {0, 1, 2, 3, 4, 5, 6, 7};
    ScatterNhrTemplate tmpl(0, ranks, TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.scratchStride = 16;
    tmpl.tempAlgParams_.dataStride = 16;
    tmpl.tempAlgParams_.root = 0;
    // root 持有全部 rank 的输入数据
    tmpl.tempAlgParams_.ranksForInputData = {0, 1, 2, 3, 4, 5, 6, 7};
    tmpl.templateRankSize_ = 8;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 8 rank → 3 步 (ceil(log2(8))=3)
    EXPECT_EQ(txRxSlicesLists.size(), 3u);
}

// TC04 空 ranksForInputData 返回错误
TEST(ScatterNhrRunAlgorithmTest, EmptyInputRanksReturnsError)
{
    ScatterNhrTemplate tmpl(0, {0, 1}, TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.scratchStride = 16;
    tmpl.tempAlgParams_.dataStride = 16;
    tmpl.tempAlgParams_.root = 0;
    // ranksForInputData 为空
    tmpl.templateRankSize_ = 2;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// ═══════════════════════════════════════════════════════════════════
// 2. PreCopy 分组（基类按 hcclCmdType==SCATTER && myRank!=root 跳过）
// ═══════════════════════════════════════════════════════════════════

// TC05 非 root rank（ranksForInputData 为空）基类 PreCopy 自动跳过
TEST(ScatterNhrPreCopyTest, NonRootRankSkipsPreCopy)
{
    ScatterNhrTemplate tmpl(2, {0, 1, 2, 3}, TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER,
        HcclAlgoType::HCCL_ALGO_TYPE_NHR, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.root = 0;
    // 非 root rank 的 ranksForInputData 为空
    tmpl.tempAlgParams_.ranksForInputData = {};

    std::vector<ThreadHandle> threads{static_cast<ThreadHandle>(0x10)};
    HcclResult ret = tmpl.PreCopy(threads);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ═══════════════════════════════════════════════════════════════════
// 3. KernelRun 完整流程分组
// ═══════════════════════════════════════════════════════════════════

// TC06 KernelRun root rank 调用 engine.Send 2 次 (4 rank → 2 步)
TEST_F(AicpuBaseTemplateTest, ScatterNhrKernelRunRootRankCallsSend)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    ScatterNhrTemplate tmpl(0, ranks, MakeScatterNhrDesc());
    TemplateDataParams params = MakeScatterTmplParams(0, ranks);
    TemplateResource res = MakeTmplResourceWithChannels(ranks, 0);
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 4 rank → 2 步 → 2 次 Send
    EXPECT_EQ(engine_.GetSendCount(), 2u);
    EXPECT_EQ(ranksForOutputData.size(), 1u);
    EXPECT_EQ(ranksForOutputData[0], 0u);
}

// TC07 KernelRun 非 root rank 调用 engine.Send
TEST_F(AicpuBaseTemplateTest, ScatterNhrKernelRunNonRootRankCallsSend)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    ScatterNhrTemplate tmpl(2, ranks, MakeScatterNhrDesc());
    TemplateDataParams params = MakeScatterTmplParams(0, ranks);
    // 非 root rank 不需要 PreCopy
    params.inputBufferType = BufferType::HCCL_BUFFER;
    TemplateResource res = MakeTmplResourceWithChannels(ranks, 2);
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 非 root 接收数据，Send 次数大于 0
    EXPECT_GT(engine_.GetSendCount(), 0u);
    EXPECT_EQ(ranksForOutputData.size(), 1u);
    EXPECT_EQ(ranksForOutputData[0], 2u);
}

// TC08 KernelRun 单 rank 不调用 engine.Send
TEST_F(AicpuBaseTemplateTest, ScatterNhrKernelRunSingleRankNoSend)
{
    ScatterNhrTemplate tmpl(0, {0}, MakeScatterNhrDesc());
    TemplateDataParams params = MakeScatterTmplParams(0, {0});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(engine_.GetSendCount(), 0u);
}

} // namespace testing
} // namespace ops_hccl
