/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * RunMeshAllGather public behavior test.
 */

#include "test_helpers.h"

namespace ops_hccl {
namespace testing {

// ============================================================
// 1. Parameter check
// ============================================================

class MeshAllGatherParamTest : public MeshAllGatherTest {};

// 校验输入数据归属为空时，构造通信描述符前返回错误。
TEST_F(MeshAllGatherParamTest, EmptyInputRanksReturnsError)
{
    std::vector<u32> ranks = {0, 1};
    TemplateDataParams params = MakeParams({});
    TemplateResource resource = MakeResource(ranks, 0);
    std::vector<u32> ranksForOutputData;
    std::vector<SendRecvInfo> sendRecvInfos;

    HcclResult ret = RunMeshAllGather(params, resource, ranks, 0, ranksForOutputData, sendRecvInfos);

    EXPECT_NE(ret, HCCL_SUCCESS);
    EXPECT_TRUE(sendRecvInfos.empty());
}

// 校验单 rank mesh 不需要通信，直接按成功返回。
TEST_F(MeshAllGatherParamTest, SingleRankReturnsWithoutTransfer)
{
    std::vector<u32> ranks = {0};
    TemplateDataParams params = MakeParams({0});
    TemplateResource resource;
    std::vector<u32> ranksForOutputData;
    std::vector<SendRecvInfo> sendRecvInfos;

    HcclResult ret = RunMeshAllGather(params, resource, ranks, 0, ranksForOutputData, sendRecvInfos);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(sendRecvInfos.empty());
    EXPECT_EQ(ranksForOutputData, params.ranksForInputData);
}

// 校验每个非本端 rank 都必须提前规划 channel。
TEST_F(MeshAllGatherParamTest, MissingPeerChannelReturnsError)
{
    std::vector<u32> ranks = {0, 1};
    TemplateDataParams params = MakeParams({0});
    TemplateResource resource;
    std::vector<u32> ranksForOutputData;
    std::vector<SendRecvInfo> sendRecvInfos;

    HcclResult ret = RunMeshAllGather(params, resource, ranks, 0, ranksForOutputData, sendRecvInfos);

    EXPECT_NE(ret, HCCL_SUCCESS);
    EXPECT_TRUE(sendRecvInfos.empty());
}

// ============================================================
// 2. SendRecvInfo build
// ============================================================

class MeshAllGatherTransferTest : public MeshAllGatherTest {};

// 校验每个对端 rank 都会按 ranks 顺序生成一个 SendRecvInfo。
TEST_F(MeshAllGatherTransferTest, BuildTransferForEachPeer)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0});
    TemplateResource resource = MakeResource(ranks, 0);
    std::vector<u32> ranksForOutputData;
    std::vector<SendRecvInfo> sendRecvInfos;

    HcclResult ret = RunMeshAllGather(params, resource, ranks, 0, ranksForOutputData, sendRecvInfos);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(sendRecvInfos.size(), 3U);
    EXPECT_EQ(sendRecvInfos[0].sendRecvChannels_.txChannel_.remoteRank, 1U);
    EXPECT_EQ(sendRecvInfos[1].sendRecvChannels_.txChannel_.remoteRank, 2U);
    EXPECT_EQ(sendRecvInfos[2].sendRecvChannels_.txChannel_.remoteRank, 3U);
}

// 校验定长 slice 使用全局 rank 计算 CCL buffer 偏移。
TEST_F(MeshAllGatherTransferTest, BuildGlobalRankOffsetSlices)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0});
    TemplateResource resource = MakeResource(ranks, 0);
    std::vector<u32> ranksForOutputData;
    std::vector<SendRecvInfo> sendRecvInfos;

    HcclResult ret = RunMeshAllGather(params, resource, ranks, 0, ranksForOutputData, sendRecvInfos);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(sendRecvInfos.size(), 3U);
    EXPECT_EQ(TxSrc(sendRecvInfos[0]).addr_, localCclMem_);
    EXPECT_EQ(TxSrc(sendRecvInfos[0]).offset_, 0U);
    EXPECT_EQ(TxDst(sendRecvInfos[0]).offset_, 0U);
    EXPECT_EQ(RxSrc(sendRecvInfos[0]).offset_, 16U);
    EXPECT_EQ(RxDst(sendRecvInfos[0]).offset_, 16U);
    EXPECT_EQ(RxSrc(sendRecvInfos[1]).offset_, 32U);
    EXPECT_EQ(RxDst(sendRecvInfos[1]).offset_, 32U);
    EXPECT_EQ(RxSrc(sendRecvInfos[2]).offset_, 48U);
    EXPECT_EQ(RxDst(sendRecvInfos[2]).offset_, 48U);
}

// 校验多个本地输入 block 会展开成对端的多组接收 slice。
TEST_F(MeshAllGatherTransferTest, BuildMultipleInputRankSlices)
{
    std::vector<u32> ranks = {0, 4, 8, 12};
    TemplateDataParams params = MakeParams({0, 4});
    TemplateResource resource = MakeResource(ranks, 0);
    std::vector<u32> ranksForOutputData;
    std::vector<SendRecvInfo> sendRecvInfos;

    HcclResult ret = RunMeshAllGather(params, resource, ranks, 0, ranksForOutputData, sendRecvInfos);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(sendRecvInfos.size(), 3U);
    ASSERT_EQ(sendRecvInfos[1].sendRecvSlices_.rxSlicesList_.dstSlices_.size(), 2U);
    EXPECT_EQ(RxDst(sendRecvInfos[1], 0).offset_, 128U);
    EXPECT_EQ(RxDst(sendRecvInfos[1], 1).offset_, 192U);
}

// ============================================================
// 3. Output rank table
// ============================================================

// 预留用例：待 RunMeshAllGather 更新 PostCopy 数据归属表后启用。
TEST_F(MeshAllGatherTransferTest, DISABLED_UpdateOutputRanksForPostCopy)
{
    std::vector<u32> ranks = {0, 1, 2, 3};
    TemplateDataParams params = MakeParams({0});
    TemplateResource resource = MakeResource(ranks, 0);
    std::vector<u32> ranksForOutputData;
    std::vector<SendRecvInfo> sendRecvInfos;

    HcclResult ret = RunMeshAllGather(params, resource, ranks, 0, ranksForOutputData, sendRecvInfos);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ranksForOutputData, ranks);
}

} // namespace testing
} // namespace ops_hccl
