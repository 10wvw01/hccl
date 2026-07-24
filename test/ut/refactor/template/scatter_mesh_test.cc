/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ScatterMeshTemplate 单元测试。
 * 覆盖: RunAlgorithm (委托 RunMeshScatter) / KernelRun 完整流程。
 */

#include "test_helpers.h"

namespace ops_hccl {
namespace testing {

// ═══════════════════════════════════════════════════════════════════
// 1. RunAlgorithm 分组
// ═══════════════════════════════════════════════════════════════════

// TC01 单 rank RunAlgorithm 不生成 txRxSlicesLists
TEST(ScatterMeshRunAlgorithmTest, SingleRankNoTransfer)
{
    ScatterMeshTemplate tmpl(0, {0}, TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
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

// TC02 root rank: 4 rank 生成 3 个 tx TxRxSlicesLists（发给其他 3 个 rank）
TEST(ScatterMeshRunAlgorithmTest, RootRankBuildsTxSlicesLists)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    ScatterMeshTemplate tmpl(0, ranks, TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
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
    // root 只发不收，3 个对端 → 3 个 tx TxRxSlicesLists
    EXPECT_EQ(txRxSlicesLists.size(), 3u);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 1u);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 2u);
    EXPECT_EQ(txRxSlicesLists[2].dstRankId_, 3u);
    // 输出归属恒为本 rank
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({0}));
}

// TC03 非 root rank: 只收不发，生成 1 个 rx TxRxSlicesList
TEST(ScatterMeshRunAlgorithmTest, NonRootRankBuildsRxSlicesList)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    ScatterMeshTemplate tmpl(2, ranks, TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
    tmpl.tempAlgParams_ = TemplateDataParams{};
    tmpl.tempAlgParams_.cclBufferPtr = reinterpret_cast<void *>(0x10000000);
    tmpl.tempAlgParams_.dataType = HCCL_DATA_TYPE_INT32;
    tmpl.tempAlgParams_.sliceCount = 4;
    tmpl.tempAlgParams_.scratchStride = 16;
    tmpl.tempAlgParams_.dataStride = 16;
    tmpl.tempAlgParams_.root = 0;
    // scatter 下 ranksForInputData 为全部 rank（由 executor 的 InitAlgoExecDataDesc 设置）
    tmpl.tempAlgParams_.ranksForInputData = {0, 1, 2, 3};
    tmpl.templateRankSize_ = 4;

    TemplateResource res;
    std::vector<TxRxSlicesList> txRxSlicesLists;
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.RunAlgorithm(res, txRxSlicesLists, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 非 root 只收不发，1 个 rx TxRxSlicesList
    EXPECT_EQ(txRxSlicesLists.size(), 1u);
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 0u);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 0u);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({2}));
}

// TC04 空 ranksForInputData 返回错误
TEST(ScatterMeshRunAlgorithmTest, EmptyInputRanksReturnsError)
{
    ScatterMeshTemplate tmpl(0, {0, 1}, TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
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
TEST(ScatterMeshPreCopyTest, NonRootRankSkipsPreCopy)
{
    ScatterMeshTemplate tmpl(2, {0, 1, 2, 3}, TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER,
        HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY});
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

// TC06 KernelRun root rank 调用 DataTransferSend 3 次
TEST_F(AicpuBaseTemplateTest, ScatterMeshKernelRunRootRankCallsSend)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    ScatterMeshTemplate tmpl(0, ranks, MakeScatterMeshDesc());
    TemplateDataParams params = MakeScatterTmplParams(0, ranks);
    TemplateResource res = MakeTmplResourceWithChannels(ranks, 0);
    std::vector<u32> ranksForOutputData;
    ClearTmplMock();

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // root 发给 3 个对端 → 3 次 Write（基类 SendAll 走 WRITE 方向）
    EXPECT_EQ(CountTmplCalls("Write"), 3u);
    // 输出归属恒为本 rank
    EXPECT_EQ(ranksForOutputData.size(), 1u);
    EXPECT_EQ(ranksForOutputData[0], 0u);
}

// TC07 KernelRun 非 root rank 触发接收侧数据传输
TEST_F(AicpuBaseTemplateTest, ScatterMeshKernelRunNonRootRankCallsSendOnce)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    ScatterMeshTemplate tmpl(2, ranks, MakeScatterMeshDesc());
    TemplateDataParams params = MakeScatterTmplParams(0, ranks);
    // 非 root rank 不需要 PreCopy，因此设置 inputBufferType=HCCL_BUFFER 跳过 PreCopy
    params.inputBufferType = BufferType::HCCL_BUFFER;
    TemplateResource res = MakeTmplResourceWithChannels(ranks, 2);
    std::vector<u32> ranksForOutputData;
    ClearTmplMock();

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 非 root 只接收：SendAll 走 WRITE 方向的 RecvWrite 分支（仅 NotifyRecord/NotifyWait 同步，
    // 不产生 HcommWriteOnThread）。断言 NotifyRecord 发生过即可证明进入了 SendAll。
    EXPECT_GT(CountTmplCalls("NotifyRecord"), 0u);
    EXPECT_EQ(ranksForOutputData.size(), 1u);
    EXPECT_EQ(ranksForOutputData[0], 2u);
}

// TC08 KernelRun 单 rank 不调用 DataTransferSend
TEST_F(AicpuBaseTemplateTest, ScatterMeshKernelRunSingleRankNoSend)
{
    ScatterMeshTemplate tmpl(0, {0}, MakeScatterMeshDesc());
    TemplateDataParams params = MakeScatterTmplParams(0, {0});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;
    ClearTmplMock();

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(CountTmplCalls("Write"), 0u);
}

} // namespace testing
} // namespace ops_hccl
