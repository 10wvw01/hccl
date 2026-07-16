/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * RunMeshAllGather public behavior test.
 */

#include "test_helpers.h"

namespace ops_hccl {
namespace testing {

class MeshAllGatherParamTest : public MeshAllGatherTest {};

// 校验输入数据归属为空时，构造通信描述符前返回错误。
TEST_F(MeshAllGatherParamTest, EmptyInputRanksReturnsError)
{
    std::vector<u32> ranks = {0, 1};
    TemplateDataParams params = MakeParams({});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_NE(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
}

// 校验单 rank mesh 不需要通信，直接按成功返回。
TEST_F(MeshAllGatherParamTest, SingleRankReturnsWithoutTransfer)
{
    std::vector<u32> ranks = {0};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
    EXPECT_EQ(ranksForOutputData, params.ranksForInputData);
}

// 校验 channel 已从 primitive 移出，多 rank 只构造 tx/rx slice 描述符。
TEST_F(MeshAllGatherParamTest, MultiRankBuildsSlicesWithoutChannel)
{
    std::vector<u32> ranks = {0, 1};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 1U);
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 1U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 1U);
}

class MeshAllGatherTransferTest : public MeshAllGatherTest {};

// 校验每个对端 rank 都会按 ranks 顺序生成一个 TxRxSlicesList。
TEST_F(MeshAllGatherTransferTest, BuildTransferForEachPeer)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 3U);
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 1U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 1U);
    EXPECT_EQ(txRxSlicesLists[1].srcRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[2].srcRankId_, 3U);
    EXPECT_EQ(txRxSlicesLists[2].dstRankId_, 3U);
}

// 校验定长 slice 使用全局 rank 计算 CCL buffer 偏移。
TEST_F(MeshAllGatherTransferTest, BuildGlobalRankOffsetSlices)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 3U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).addr_, localCclMem_);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).offset_, 0U);
    EXPECT_EQ(TxDst(txRxSlicesLists[0]).offset_, 0U);
    EXPECT_EQ(RxSrc(txRxSlicesLists[0]).offset_, 16U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).offset_, 16U);
    EXPECT_EQ(RxSrc(txRxSlicesLists[1]).offset_, 32U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1]).offset_, 32U);
    EXPECT_EQ(RxSrc(txRxSlicesLists[2]).offset_, 48U);
    EXPECT_EQ(RxDst(txRxSlicesLists[2]).offset_, 48U);
}

// 校验多个本地输入 block 会展开成对端的多组接收 slice。
TEST_F(MeshAllGatherTransferTest, BuildMultipleInputRankSlices)
{
    std::vector<u32> ranks = {0, 4, 8, 12};
    TemplateDataParams params = MakeParams({0, 4});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 3U);
    ASSERT_EQ(txRxSlicesLists[1].rxSlicesList_.dstSlices_.size(), 2U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1], 0).offset_, 128U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1], 1).offset_, 192U);
}

// 校验最后一个 rank 使用 tailCount 生成尾块 slice。
TEST_F(MeshAllGatherTransferTest, BuildTailRankSlice)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0});
    params.tailCount = 2;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 3U);
    EXPECT_EQ(RxDst(txRxSlicesLists[2]).offset_, 48U);
    EXPECT_EQ(RxDst(txRxSlicesLists[2]).size_, 8U);
    EXPECT_EQ(RxDst(txRxSlicesLists[2]).count_, 2U);
}

// 校验输出数据归属表按全局 rank 排序后返回给 PostCopy。
TEST_F(MeshAllGatherTransferTest, UpdateOutputRanksForPostCopy)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshAllGather(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ranksForOutputData, ranks);
}

class MeshScatterParamTest : public MeshAllGatherTest {};

TEST_F(MeshScatterParamTest, EmptyInputRanksReturnsError)
{
    std::vector<u32> ranks = {0, 1};
    TemplateDataParams params = MakeParams({});
    params.root = 0;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_NE(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
}

TEST_F(MeshScatterParamTest, SingleRankReturnsWithoutTransfer)
{
    std::vector<u32> ranks = {0};
    TemplateDataParams params = MakeParams({0});
    params.root = 0;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({0}));
}

class MeshScatterTransferTest : public MeshAllGatherTest {};

TEST_F(MeshScatterTransferTest, RootBuildsTxOnlyForEachPeer)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    params.root = 0;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 3U);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({0}));
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 1U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 1U);
    EXPECT_EQ(txRxSlicesLists[1].srcRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[2].srcRankId_, 3U);
    EXPECT_EQ(txRxSlicesLists[2].dstRankId_, 3U);
    EXPECT_TRUE(txRxSlicesLists[0].rxSlicesList_.srcSlices_.empty());
    EXPECT_TRUE(txRxSlicesLists[0].rxSlicesList_.dstSlices_.empty());
}

TEST_F(MeshScatterTransferTest, RootBuildsGlobalRankOffsetTxSlices)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    params.root = 0;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 3U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).addr_, localCclMem_);
    EXPECT_EQ(TxDst(txRxSlicesLists[0]).addr_, nullptr);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).offset_, 16U);
    EXPECT_EQ(TxDst(txRxSlicesLists[0]).offset_, 16U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[1]).offset_, 32U);
    EXPECT_EQ(TxDst(txRxSlicesLists[1]).offset_, 32U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[2]).offset_, 48U);
    EXPECT_EQ(TxDst(txRxSlicesLists[2]).offset_, 48U);
}

TEST_F(MeshScatterTransferTest, NonRootBuildsRxOnlyFromRoot)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    params.root = 0;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshScatter(params, ranks, 2, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 1U);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({2}));
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 0U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 0U);
    EXPECT_TRUE(txRxSlicesLists[0].txSlicesList_.srcSlices_.empty());
    EXPECT_TRUE(txRxSlicesLists[0].txSlicesList_.dstSlices_.empty());
    EXPECT_EQ(RxSrc(txRxSlicesLists[0]).addr_, nullptr);
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).addr_, localCclMem_);
    EXPECT_EQ(RxSrc(txRxSlicesLists[0]).offset_, 32U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).offset_, 32U);
}

TEST_F(MeshScatterTransferTest, BuildTailRankSlice)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    params.root = 0;
    params.tailCount = 2;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 3U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[2]).offset_, 48U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[2]).size_, 8U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[2]).count_, 2U);
}

TEST_F(MeshScatterTransferTest, NonZeroRootSkipsSelfAndKeepsRankOrder)
{
    std::vector<u32> ranks = {4, 5, 6};
    TemplateDataParams params = MakeParams({4, 5, 6});
    params.root = 5;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshScatter(params, ranks, 5, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({5}));
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 4U);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 6U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).offset_, 64U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[1]).offset_, 96U);
}

class MeshReduceScatterParamTest : public MeshAllGatherTest {};

TEST_F(MeshReduceScatterParamTest, EmptyInputRanksReturnsError)
{
    std::vector<u32> ranks = {0, 1};
    TemplateDataParams params = MakeParams({});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshReduceScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_NE(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
}

TEST_F(MeshReduceScatterParamTest, SingleRankReturnsWithoutTransfer)
{
    std::vector<u32> ranks = {0};
    TemplateDataParams params = MakeParams({0});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshReduceScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(txRxSlicesLists.empty());
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({0}));
}

class MeshReduceScatterTransferTest : public MeshAllGatherTest {};

TEST_F(MeshReduceScatterTransferTest, BuildTransferForEachPeerInRingOrder)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshReduceScatter(params, ranks, 1, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 3U);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({1}));
    EXPECT_EQ(txRxSlicesLists[0].srcRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[0].dstRankId_, 2U);
    EXPECT_EQ(txRxSlicesLists[1].srcRankId_, 3U);
    EXPECT_EQ(txRxSlicesLists[1].dstRankId_, 3U);
    EXPECT_EQ(txRxSlicesLists[2].srcRankId_, 0U);
    EXPECT_EQ(txRxSlicesLists[2].dstRankId_, 0U);
}

TEST_F(MeshReduceScatterTransferTest, BuildTxPeerSliceAndRxLocalSlice)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshReduceScatter(params, ranks, 1, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 3U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).addr_, localCclMem_);
    EXPECT_EQ(TxDst(txRxSlicesLists[0]).addr_, nullptr);
    EXPECT_EQ(RxSrc(txRxSlicesLists[0]).addr_, nullptr);
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).addr_, localCclMem_);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).offset_, 32U);
    EXPECT_EQ(TxDst(txRxSlicesLists[0]).offset_, 16U);
    EXPECT_EQ(RxSrc(txRxSlicesLists[0]).offset_, 16U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).offset_, 32U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[1]).offset_, 48U);
    EXPECT_EQ(TxDst(txRxSlicesLists[1]).offset_, 16U);
    EXPECT_EQ(RxDst(txRxSlicesLists[1]).offset_, 48U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[2]).offset_, 0U);
    EXPECT_EQ(TxDst(txRxSlicesLists[2]).offset_, 16U);
    EXPECT_EQ(RxDst(txRxSlicesLists[2]).offset_, 0U);
}

TEST_F(MeshReduceScatterTransferTest, UsesAlgRankContributionSlotForNonZeroRanks)
{
    std::vector<u32> ranks = {4, 5, 6};
    TemplateDataParams params = MakeParams({4, 5, 6});
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshReduceScatter(params, ranks, 5, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 2U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[0]).offset_, 96U);
    EXPECT_EQ(TxDst(txRxSlicesLists[0]).offset_, 16U);
    EXPECT_EQ(RxSrc(txRxSlicesLists[0]).offset_, 80U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).offset_, 32U);
}

TEST_F(MeshReduceScatterTransferTest, BuildTailPeerTxSlice)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    params.tailCount = 2;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshReduceScatter(params, ranks, 0, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 3U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[2]).offset_, 48U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[2]).size_, 8U);
    EXPECT_EQ(TxSrc(txRxSlicesLists[2]).count_, 2U);
    EXPECT_EQ(RxDst(txRxSlicesLists[2]).offset_, 48U);
    EXPECT_EQ(RxDst(txRxSlicesLists[2]).size_, 16U);
    EXPECT_EQ(RxDst(txRxSlicesLists[2]).count_, 4U);
    EXPECT_EQ(TxDst(txRxSlicesLists[2]).offset_, 0U);
    EXPECT_EQ(TxDst(txRxSlicesLists[2]).size_, 8U);
    EXPECT_EQ(TxDst(txRxSlicesLists[2]).count_, 2U);
}

TEST_F(MeshReduceScatterTransferTest, LocalTailRankBuildsTailRxSlice)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0, 1, 2, 3});
    params.tailCount = 2;
    std::vector<u32> ranksForOutputData;
    std::vector<TxRxSlicesList> txRxSlicesLists;

    HcclResult ret = RunMeshReduceScatter(params, ranks, 3, ranksForOutputData, txRxSlicesLists);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(txRxSlicesLists.size(), 3U);
    EXPECT_EQ(ranksForOutputData, std::vector<u32>({3}));
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).offset_, 0U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).size_, 8U);
    EXPECT_EQ(RxDst(txRxSlicesLists[0]).count_, 2U);
}

} // namespace testing
} // namespace ops_hccl
