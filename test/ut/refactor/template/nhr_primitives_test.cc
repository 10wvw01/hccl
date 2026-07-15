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

} // namespace testing
} // namespace ops_hccl
