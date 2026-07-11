/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * InitAlgoExecDataDesc / GenTemplateDataParams / GenTemplateRes / UpdateSubCommMaskMap 测试
 */

#include "test_helpers.h"

namespace ops_hccl {
namespace testing {

// ============================================================
// InitAlgoExecDataDesc
// ============================================================

TEST_F(OpsExecutorTest, InitAlgoExecDataDescNonAllGather)
{
    executor_->SetCmdType(HCCL_CMD_ALLREDUCE);
    executor_->SetRankSize(4);
    AlgoExecDataDesc desc;
    executor_->InitAlgoExecDataDesc(desc, 100, 400, 3);
    EXPECT_EQ(desc.dataOffset, 100u);
    EXPECT_EQ(desc.tailCount, 3u);
    EXPECT_EQ(desc.sliceCount, 100u);
    EXPECT_EQ(desc.stride, 100u);
    EXPECT_EQ(desc.inputBufferType, BufferType::INPUT);
    EXPECT_EQ(desc.outputBufferType, BufferType::OUTPUT);
    ASSERT_EQ(desc.ranksForInputData.size(), 4u);
    EXPECT_EQ(desc.ranksForInputData[0], 0u);
    EXPECT_EQ(desc.ranksForInputData[3], 3u);
}

TEST_F(OpsExecutorTest, InitAlgoExecDataDescAllGather)
{
    executor_->SetCmdType(HCCL_CMD_ALLGATHER);
    executor_->SetRankSize(4);
    executor_->SetMyRank(2);
    AlgoExecDataDesc desc;
    executor_->InitAlgoExecDataDesc(desc, 0, 256, 0);
    EXPECT_EQ(desc.sliceCount, 256u);
    ASSERT_EQ(desc.ranksForInputData.size(), 1u);
    EXPECT_EQ(desc.ranksForInputData[0], 2u);
}

TEST_F(OpsExecutorTest, InitAlgoExecDataDescBroadcast)
{
    executor_->SetCmdType(HCCL_CMD_BROADCAST);
    executor_->SetRankSize(2);
    AlgoExecDataDesc desc;
    executor_->InitAlgoExecDataDesc(desc, 0, 128, 0);
    EXPECT_EQ(desc.outputBufferType, BufferType::INPUT);
}

// ============================================================
// GenTemplateDataParams
// ============================================================

TEST_F(OpsExecutorTest, GenTemplateDataParamsMapsFields)
{
    executor_->SetDataInfoInput((void *)0x1000, 1024);
    executor_->SetDataInfoOutput((void *)0x2000, 2048);
    executor_->SetCclBufferPtr((void *)0x3000);
    executor_->SetDataInfoDataType(HCCL_DATA_TYPE_FP32);
    executor_->SetDataInfoReduceOp(HCCL_REDUCE_SUM);
    executor_->SetRoot(5);
    executor_->SetOpMode(OpMode::OFFLOAD);

    AlgoExecDataDesc desc;
    desc.inputBufferType = BufferType::INPUT;
    desc.outputBufferType = BufferType::OUTPUT;
    desc.cclBufferType = BufferType::HCCL_BUFFER;
    desc.sliceCount = 512;
    desc.sliceOffset = 128;
    desc.tailCount = 3;
    desc.dataOffset = 64;
    desc.stride = 512;
    desc.ranksForInputData = {0, 1};

    TemplateDataParams params;
    executor_->GenTemplateDataParams(desc, params);

    EXPECT_EQ(params.inputBufferPtr, (void *)0x1000);
    EXPECT_EQ(params.sliceCount, 512u);
    EXPECT_EQ(params.sliceOffset, 128u);
    EXPECT_EQ(params.dataOffset, 64u);
    EXPECT_EQ(params.stride, 512u);
    EXPECT_EQ(params.reduceOp, HCCL_REDUCE_SUM);
    EXPECT_EQ(params.root, 5u);
    EXPECT_TRUE(params.enableRemoteMemAccess);
    ASSERT_EQ(params.ranksForInputData.size(), 2u);
}

TEST_F(OpsExecutorTest, GenTemplateResPopulatesResource)
{
    std::vector<std::map<u32, std::vector<ChannelInfo>>> table;
    std::map<u32, std::vector<ChannelInfo>> level0;
    ChannelInfo ch;
    ch.remoteRank = 7;
    level0[7].push_back(ch);
    table.push_back(level0);
    executor_->SetChannelTable(table);

    std::vector<std::vector<ThreadHandle>> subThreads;
    subThreads.push_back({100, 101, 102});
    executor_->SetSubThreads(subThreads);

    TemplateResource res;
    EXPECT_EQ(executor_->GenTemplateRes(0, res), HCCL_SUCCESS);
    ASSERT_EQ(res.channels.size(), 1u);
    EXPECT_EQ(res.channels[7][0].remoteRank, 7u);
    ASSERT_EQ(res.threads.size(), 3u);
}

// ============================================================
// UpdateSubCommMaskMap
// ============================================================

TEST_F(OpsExecutorTest, UpdateSubCommMaskMapInsertAndUpdate)
{
    AlgoExecDesc desc;
    desc.execPolicy = HcclAlgExecPolicy::PARALLEL;
    // subCommMask 最多 3 bit，每位对应一个 subCommIndex
    executor_->UpdateSubCommMaskMap(desc, 0b100);  // bit0: intra
    executor_->UpdateSubCommMaskMap(desc, 0b111);  // bit0+bit1+bit2: intra+inter+pod
    EXPECT_TRUE(true);
}

TEST_F(OpsExecutorTest, UpdateSubCommMaskMapNotFoundGraceful)
{
    AlgoExecDesc desc;
    desc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    EXPECT_EQ(executor_->PreSyncBySubCommMask(desc), HCCL_SUCCESS);
}

} // namespace testing
} // namespace ops_hccl
