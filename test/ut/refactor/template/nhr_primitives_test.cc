/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * RunNhrAllGather public behavior test.
 */

#include "test_helpers.h"

namespace ops_hccl {
namespace testing {

class NhrAllGatherParamTest : public NhrAllGatherTest {};

TEST_F(NhrAllGatherParamTest, EmptyInputRanksReturnsError)
{
    std::vector<u32> ranks = {0, 1};
    TemplateDataParams params = MakeParams({});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_NE(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
}

TEST_F(NhrAllGatherParamTest, SingleRankReturnsWithoutTransfer)
{
    std::vector<u32> ranks = {0};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
    EXPECT_EQ(ranksForOutputData, params.ranksForInputData);
}

TEST_F(NhrAllGatherParamTest, MultiRankBuildsSlicesWithoutChannel)
{
    std::vector<u32> ranks = {0, 1};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 1U);
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 1U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 1U);
}

class NhrAllGatherTransferTest : public NhrAllGatherTest {};

TEST_F(NhrAllGatherTransferTest, BuildTransferForEachStep)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[1].srcRankId_, 3U);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 1U);
}

TEST_F(NhrAllGatherTransferTest, BuildStepSlicesByGlobalRankOffset)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    ASSERT_EQ(txRxSlicesLists[1].txSlicesList_.srcSlices_.size(), 2U);
    ASSERT_EQ(txRxSlicesLists[1].rxSlicesList_.dstSlices_.size(), 2U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).addr_, localCclMem_);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).offset_, 0U);
    EXPECT_EQ(TxDst(txRxSlicesLists[0]).offset_, 0U);
    EXPECT_EQ(RxSrc(txRxSlicesLists[0]).offset_, 32U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).offset_, 32U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[1], 0).offset_, 0U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[1], 1).offset_, 32U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1], 0).offset_, 48U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1], 1).offset_, 16U);
}

TEST_F(NhrAllGatherTransferTest, BuildMultipleInputRankSlices)
{
    std::vector<u32> ranks = {0, 4, 8, 12};
    TemplateDataParams params = MakeParams({0, 4});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    ASSERT_EQ(txRxSlicesLists[0].rxSlicesList_.dstSlices_.size(), 2U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0], 0).offset_, 128U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0], 1).offset_, 192U);
}

TEST_F(NhrAllGatherTransferTest, BuildTailRankSlice)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0});
    params.tailCount = 2;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    ASSERT_EQ(txRxSlicesLists[1].rxSlicesList_.dstSlices_.size(), 2U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1], 0).offset_, 48U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1], 0).size_, 8U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1], 0).count_, 2U);
}

TEST_F(NhrAllGatherTransferTest, UpdateOutputRanksForPostCopy)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ranksForOutputData, ranks);
}

class NhrScatterParamTest : public NhrAllGatherTest {};

TEST_F(NhrScatterParamTest, EmptyInputRanksReturnsError)
{
    std::vector<u32> ranks = {0, 1};
    TemplateDataParams params = MakeParams({});
    params.root = 0;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_NE(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
}

TEST_F(NhrScatterParamTest, SingleRankReturnsWithoutTransfer)
{
    std::vector<u32> ranks = {0};
    TemplateDataParams params = MakeParams({0});
    params.root = 0;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({0}));
}

class NhrScatterTransferTest : public NhrAllGatherTest {};

TEST_F(NhrScatterTransferTest, RootBuildsTreeTxSteps)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    params.root = 0;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({0}));
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[1].srcRankId_, 1U);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 1U);
    EXPECT_TRUE(txRxSlicesLists[0].rxSlicesList_.srcSlices_.empty());
    ASSERT_EQ(txRxSlicesLists[0].txSlicesList_.srcSlices_.size(), 2U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 0).offset_, 32U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 1).offset_, 48U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[1]).offset_, 16U);
}

TEST_F(NhrScatterTransferTest, IntermediateRankReceivesThenForwards)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    params.root = 0;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrScatter(params, ranks, 2, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({2}));
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 0U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 0U);
    EXPECT_TRUE(txRxSlicesLists[0].txSlicesList_.srcSlices_.empty());
    ASSERT_EQ(txRxSlicesLists[0].rxSlicesList_.dstSlices_.size(), 2U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0], 0).offset_, 32U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0], 1).offset_, 48U);
    EXPECT_EQ(txRxSlicesLists[1].srcRankId_, 3U);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 3U);
    EXPECT_TRUE(txRxSlicesLists[1].rxSlicesList_.dstSlices_.empty());
    EXPECT_EQ(TxSrc(txRxSlicesLists[1]).offset_, 48U);
}

TEST_F(NhrScatterTransferTest, LeafRankReceivesOnlyWhenItsStepArrives)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    params.root = 0;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrScatter(params, ranks, 3, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 1U);
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 2U);
    EXPECT_TRUE(txRxSlicesLists[0].txSlicesList_.srcSlices_.empty());
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).offset_, 48U);
}

TEST_F(NhrScatterTransferTest, BuildTailRankSlice)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    params.root = 0;
    params.tailCount = 2;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    ASSERT_EQ(txRxSlicesLists[0].txSlicesList_.srcSlices_.size(), 2U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 1).offset_, 48U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 1).size_, 8U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 1).count_, 2U);
}

TEST_F(NhrScatterTransferTest, NonZeroRootUsesRootRelativeTreeOrder)
{
    std::vector<u32> ranks = {4, 5, 6, 7};
    TemplateDataParams params = MakeParams({4, 5, 6, 7});
    params.root = 5;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrScatter(params, ranks, 5, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({5}));
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 7U);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 6U);
    ASSERT_EQ(txRxSlicesLists[0].txSlicesList_.srcSlices_.size(), 2U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 0).offset_, 112U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 1).offset_, 64U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[1]).offset_, 96U);
}

class NhrReduceScatterParamTest : public NhrAllGatherTest {};

// 校验输入数据归属为空时，构造通信描述符前返回错误。
TEST_F(NhrReduceScatterParamTest, EmptyInputRanksReturnsError)
{
    std::vector<u32> ranks = {0, 1};
    TemplateDataParams params = MakeParams({});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrReduceScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_NE(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
}

// 校验单 rank 不需要通信，输出归属仍为本 rank。
TEST_F(NhrReduceScatterParamTest, SingleRankReturnsWithoutTransfer)
{
    std::vector<u32> ranks = {0};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrReduceScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({0}));
}

class NhrReduceScatterTransferTest : public NhrAllGatherTest {};

// 校验 2 rank 只有 1 步：tx 发对端槽位，rx 落本 rank 槽位，输出归属恒为 myRank。
TEST_F(NhrReduceScatterTransferTest, TwoRankBuildsSingleStep)
{
    std::vector<u32> ranks = {0, 1};
    TemplateDataParams params = MakeParams({0, 1});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrReduceScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 1U);
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 1U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 1U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).addr_, localCclMem_);
    EXPECT_EQ(TxDst(txRxSlicesLists[0]).addr_, nullptr);
    EXPECT_EQ(RxSrc(txRxSlicesLists[0]).addr_, nullptr);
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).addr_, localCclMem_);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).offset_, 16U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).offset_, 0U);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({0}));
}

// 校验 4 rank 走 ceil(log2(4))=2 步，每步生成一个 TxRxSlicesList。
TEST_F(NhrReduceScatterTransferTest, FourRankBuildsTwoSteps)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrReduceScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    // step0: delta=1, 发给 rank3、收自 rank1；step1: delta=2, 与 rank2 互为对端
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 1U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 3U);
    EXPECT_EQ(txRxSlicesLists[1].srcRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 2U);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({0}));
}

// 校验槽位按全局 rankId 寻址：tx 发对端归约集合槽位，rx 落本端归约集合槽位。
TEST_F(NhrReduceScatterTransferTest, BuildStepSlicesByGlobalRankOffset)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrReduceScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    // step0: algRankStep=2, tx 槽位 [3, 1]，rx 槽位 [0, 2]
    ASSERT_EQ(txRxSlicesLists[0].txSlicesList_.srcSlices_.size(), 2U);
    ASSERT_EQ(txRxSlicesLists[0].rxSlicesList_.dstSlices_.size(), 2U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 0).offset_, 48U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 0).size_, 16U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 1).offset_, 16U);
    EXPECT_EQ(TxDst(txRxSlicesLists[0], 0).offset_, 48U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0], 0).offset_, 0U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0], 1).offset_, 32U);
    // step1: algRankStep=4, tx 槽位 [2]，rx 槽位 [0]
    ASSERT_EQ(txRxSlicesLists[1].txSlicesList_.srcSlices_.size(), 1U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[1], 0).offset_, 32U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1], 0).offset_, 0U);
}

// 校验非 0 rank 视角：归约集合与对端都按 myAlgRank 相对计算。
TEST_F(NhrReduceScatterTransferTest, NonZeroRankBuildsOwnReduceSet)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrReduceScatter(params, ranks, 2, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    // myAlgRank=2, step0: 发给 rank1、收自 rank3，tx 槽位 [1, 3]，rx 槽位 [2, 0]
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 3U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 1U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 0).offset_, 16U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 1).offset_, 48U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0], 0).offset_, 32U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0], 1).offset_, 0U);
    // step1: 与 rank0 互为对端，tx 槽位 [0]，rx 槽位 [2]
    EXPECT_EQ(txRxSlicesLists[1].srcRankId_, 0U);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 0U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[1], 0).offset_, 0U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1], 0).offset_, 32U);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({2}));
}

// 校验非 2 的幂 rankSize：nSteps=ceil(log2(3))=2，每步 nSlices=1。
TEST_F(NhrReduceScatterTransferTest, NonPowerOfTwoRankSizeDividesSlices)
{
    std::vector<u32> ranks = {0, 1, 2};
    TemplateDataParams params = MakeParams({0, 1, 2});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrReduceScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    // step0: delta=1, 发给 rank2、收自 rank1，tx 槽位 [2]，rx 槽位 [0]
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 1U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 2U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 0).offset_, 32U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0], 0).offset_, 0U);
    // step1: delta=2, 发给 rank1、收自 rank2，tx 槽位 [1]，rx 槽位 [0]
    EXPECT_EQ(txRxSlicesLists[1].srcRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 1U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[1], 0).offset_, 16U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1], 0).offset_, 0U);
}

// 校验最后一个 rank 的槽位使用 tailCount 生成尾块 slice。
TEST_F(NhrReduceScatterTransferTest, BuildTailRankSlice)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    params.tailCount = 2;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunNhrReduceScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    // step0 tx 槽位 [3, 1]：rank3 是尾块（tailCount=2 → 8B），rank1 是整块（sliceCount=4 → 16B）
    ASSERT_EQ(txRxSlicesLists[0].txSlicesList_.srcSlices_.size(), 2U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 0).offset_, 48U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 0).size_, 8U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 0).count_, 2U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 1).offset_, 16U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 1).size_, 16U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0], 1).count_, 4U);
    // 本端归约集合不含尾块槽位，rx 均为整块
    EXPECT_EQ(RxDst(txRxSlicesLists[0], 0).size_, 16U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0], 1).size_, 16U);
}

} // namespace testing
} // namespace ops_hccl
