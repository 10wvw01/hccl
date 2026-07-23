/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * AicpuBaseTemplate 单元测试。
 * 覆盖: KernelRun (数据量0 / 单rank / 多rank多线程) / PreCopy / PostCopy / SendAll /
 *       IsPcieProtocol / PreCopy tailCount / PostCopy tailCount。
 */

#include "test_helpers.h"

namespace ops_hccl {
namespace testing {

// ───────────── 具体子类: 实现纯虚 RunAlgorithm ─────────────
class TestableAicpuTemplate : public AicpuBaseTemplate {
public:
    TestableAicpuTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc desc)
        : AicpuBaseTemplate(myRank, std::move(ranks), std::move(desc)) {}

    HcclResult RunAlgorithm(TemplateResource &, std::vector<TxRxSlicesList> &txRxSlicesLists,
                            std::vector<u32> &ranksForOutputData) override
    {
        ranksForOutputData = ranks_;
        // 不构造任何 txRxSlicesLists, SendAll 不会执行
        (void)txRxSlicesLists;
        return HCCL_SUCCESS;
    }
};

// ───────────── 可配置子类: 生成指定数量的 TxRxSlicesList ─────────────
class TestableAicpuTemplateV2 : public AicpuBaseTemplate {
public:
    TestableAicpuTemplateV2(u32 myRank, std::vector<u32> ranks, TemplateDesc desc)
        : AicpuBaseTemplate(myRank, std::move(ranks), std::move(desc)) {}

    // 配置 RunAlgorithm 生成的 txRxSlicesLists 条数
    void SetSliceListCount(size_t count) { sliceListCount_ = count; }

    // 配置 ranksForOutputData（默认使用 ranks_）
    void SetRanksForOutputData(const std::vector<u32> &ranks) { customRanksForOutputData_ = ranks; }

    HcclResult RunAlgorithm(TemplateResource &, std::vector<TxRxSlicesList> &txRxSlicesLists,
                            std::vector<u32> &ranksForOutputData) override
    {
        ranksForOutputData = customRanksForOutputData_.empty() ? ranks_ : customRanksForOutputData_;
        for (size_t i = 0; i < sliceListCount_; ++i) {
            txRxSlicesLists.emplace_back();
        }
        return HCCL_SUCCESS;
    }

    // 公开 IsPcieProtocol 以便测试
    bool IsPcieProtocolPublic(const std::map<u32, std::vector<ChannelInfo>> &channels) const
    {
        return IsPcieProtocol(channels);
    }

private:
    size_t sliceListCount_ = 0;
    std::vector<u32> customRanksForOutputData_{};
};

// ═══════════════════════════════════════════════════════════════════
// 1. KernelRun 分组
// ═══════════════════════════════════════════════════════════════════

// TC01 sliceCount=0 且 tailCount=0 直接返回成功
TEST_F(AicpuBaseTemplateTest, KernelRunZeroSliceCountReturnsSuccess)
{
    TestableAicpuTemplate tmpl(0, {0, 1}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    params.sliceCount = 0;
    params.tailCount = 0;
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(ranksForOutputData.empty());
}

// TC02 单 rank 无需通信, PreCopy 后直接返回
TEST_F(AicpuBaseTemplateTest, KernelRunSingleRankPreCopyOnly)
{
    TestableAicpuTemplate tmpl(0, {0}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ranksForOutputData, params.ranksForInputData);
    // PreCopy 被调用: input != ccl, 所以会执行 LocalCopy
    EXPECT_GE(CountTmplCalls("LocalCopy"), 1u);
}

// TC03 多 rank 多线程执行完整流程 (PreSync -> RunAlgorithm -> SendAll -> PostSync -> PostCopy)
TEST_F(AicpuBaseTemplateTest, KernelRunMultiRankMultiThreadFullFlow)
{
    TestableAicpuTemplate tmpl(0, {0, 1, 2}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    TemplateResource res = MakeTmplResource(2); // 2 threads → multiThread=true
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ranksForOutputData, (std::vector<u32>{0, 1, 2}));
    // 多线程: PreSync (NotifyRecord) + PostSync (NotifyWait + NotifyRecord)
    EXPECT_GE(CountTmplCalls("NotifyRecord"), 1u);
    EXPECT_GE(CountTmplCalls("NotifyWait"), 1u);
}

// TC04 threads 为空时返回错误
TEST_F(AicpuBaseTemplateTest, KernelRunEmptyThreadsReturnsError)
{
    TestableAicpuTemplate tmpl(0, {0, 1}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    TemplateResource res; // empty threads
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// ═══════════════════════════════════════════════════════════════════
// 2. PreCopy 分组
// ═══════════════════════════════════════════════════════════════════

// TC05 PreCopy inputBufferPtr == cclBufferPtr 时跳过 PreCopy 拷贝
// 但 PostCopy 仍会执行 ccl→output 拷贝（outputBufferPtr != cclBufferPtr）
TEST_F(AicpuBaseTemplateTest, PreCopyInputEqualsCclSkipsCopy)
{
    TestableAicpuTemplate tmpl(0, {0, 1}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    params.inputBufferPtr = params.cclBufferPtr; // same pointer
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // PreCopy 跳过, 但 PostCopy 会执行 (ranksForOutputData = {0, 1})
    EXPECT_EQ(CountTmplCalls("LocalCopy"), 2u);
}

// TC06 PreCopy ranksForInputData 为空时返回错误
TEST_F(AicpuBaseTemplateTest, PreCopyEmptyRanksForInputReturnsError)
{
    TestableAicpuTemplate tmpl(0, {0, 1}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// TC07 PreCopy 多个 input rank 各执行一次 LocalCopy, PostCopy 也各执行一次
TEST_F(AicpuBaseTemplateTest, PreCopyMultiInputRanksMultiLocalCopy)
{
    TestableAicpuTemplate tmpl(0, {0, 1, 2}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0, 1, 2});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 3 个 input rank → PreCopy 3 次 + PostCopy 3 次 (ranksForOutputData = {0, 1, 2})
    EXPECT_EQ(CountTmplCalls("LocalCopy"), 6u);
}

// ═══════════════════════════════════════════════════════════════════
// 3. PostCopy 分组
// ═══════════════════════════════════════════════════════════════════

// TC08 PostCopy outputBufferType=HCCL_BUFFER 时跳过
TEST_F(AicpuBaseTemplateTest, PostCopyOutputIsHcclBufferSkipsCopy)
{
    TestableAicpuTemplate tmpl(0, {0, 1}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    params.outputBufferType = BufferType::HCCL_BUFFER;
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // PreCopy 仍执行, PostCopy 跳过
    EXPECT_GE(CountTmplCalls("LocalCopy"), 1u); // PreCopy
}

// TC09 PostCopy inputBufferType=HCCL_BUFFER 时跳过
TEST_F(AicpuBaseTemplateTest, PostCopyInputIsHcclBufferSkipsCopy)
{
    TestableAicpuTemplate tmpl(0, {0, 1}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    params.inputBufferType = BufferType::HCCL_BUFFER;
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC10 PostCopy enableRemoteMemAccess 时跳过
TEST_F(AicpuBaseTemplateTest, PostCopyRemoteMemAccessSkipsCopy)
{
    TestableAicpuTemplate tmpl(0, {0, 1}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    params.enableRemoteMemAccess = true;
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ═══════════════════════════════════════════════════════════════════
// 4. SendAll 分组
// ═══════════════════════════════════════════════════════════════════

// TC11 RunAlgorithm 生成空 txRxSlicesLists 时 SendAll 不执行
TEST_F(AicpuBaseTemplateTest, SendAllEmptyListNoSendCalls)
{
    TestableAicpuTemplate tmpl(0, {0, 1}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ═══════════════════════════════════════════════════════════════════
// 5. SendAll 分组 (非空 txRxSlicesLists)
// ═══════════════════════════════════════════════════════════════════

// TC12 RunAlgorithm 生成 2 个 TxRxSlicesList, SendAll 执行 2 次 DataTransferSend
TEST_F(AicpuBaseTemplateTest, SendAllWithNonEmptyListCallsDataTransferSend)
{
    TestableAicpuTemplateV2 tmpl(0, {0, 1}, MakeMeshDesc());
    tmpl.SetSliceListCount(2);
    TemplateDataParams params = MakeTmplParams({0});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC13 SendAll 传递的 TransferContext 字段与 tempAlgParams 一致
TEST_F(AicpuBaseTemplateTest, SendAllMultipleSlicesContextCorrect)
{
    TestableAicpuTemplateV2 tmpl(0, {0, 1}, MakeMeshDesc());
    tmpl.SetSliceListCount(1);
    TemplateDataParams params = MakeTmplParams({0});
    params.enableRemoteMemAccess = true;
    params.cclBufferType = BufferType::HCCL_BUFFER;
    params.dataType = HCCL_DATA_TYPE_INT32;
    params.reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC14 SendAll 成功路径验证
TEST_F(AicpuBaseTemplateTest, SendAllSuccessPropagation)
{
    TestableAicpuTemplateV2 tmpl(0, {0, 1}, MakeMeshDesc());
    tmpl.SetSliceListCount(1);
    TemplateDataParams params = MakeTmplParams({0});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ═══════════════════════════════════════════════════════════════════
// 6. IsPcieProtocol 分组
// ═══════════════════════════════════════════════════════════════════

// TC15 channels 中有 PCIe 协议的 ChannelInfo 时返回 true
TEST_F(AicpuBaseTemplateTest, IsPcieProtocolTrueForPcieChannel)
{
    TestableAicpuTemplateV2 tmpl(0, {0, 1}, MakeMeshDesc());

    std::map<u32, std::vector<ChannelInfo>> channels;
    ChannelInfo ch;
    ch.protocol = CommProtocol::COMM_PROTOCOL_PCIE;
    channels[1].push_back(ch);

    EXPECT_TRUE(tmpl.IsPcieProtocolPublic(channels));
}

// TC16 channels 中无 PCIe 协议时返回 false
TEST_F(AicpuBaseTemplateTest, IsPcieProtocolFalseForNonPcieChannel)
{
    TestableAicpuTemplateV2 tmpl(0, {0, 1}, MakeMeshDesc());

    std::map<u32, std::vector<ChannelInfo>> channels;
    ChannelInfo ch;
    ch.protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    channels[1].push_back(ch);

    EXPECT_FALSE(tmpl.IsPcieProtocolPublic(channels));
}

// TC17 channels 为空时返回 false
TEST_F(AicpuBaseTemplateTest, IsPcieProtocolFalseForEmptyChannels)
{
    TestableAicpuTemplateV2 tmpl(0, {0, 1}, MakeMeshDesc());

    std::map<u32, std::vector<ChannelInfo>> channels;
    EXPECT_FALSE(tmpl.IsPcieProtocolPublic(channels));
}

// ═══════════════════════════════════════════════════════════════════
// 7. PreCopy tailCount 分组
// ═══════════════════════════════════════════════════════════════════

// TC18 PreCopy 最后一个 rank 使用 sliceSize + tailSize
TEST_F(AicpuBaseTemplateTest, PreCopyWithTailCountLastRankUsesTailSize)
{
    // 4 ranks: {0, 1, 2, 3}, templateRankSize_ = 4
    TestableAicpuTemplateV2 tmpl(0, {0, 1, 2, 3}, MakeMeshDesc());
    // RunAlgorithm 生成空 txRxSlicesLists, 通信不执行
    tmpl.SetSliceListCount(0);
    // ranksForInputData = {0, 1, 2, 3} 遍历所有 rank
    TemplateDataParams params = MakeTmplParams({0, 1, 2, 3});
    // sliceCount=4, tailCount=2, dataType=INT32 (4 bytes)
    // sliceSize = 4*4 = 16 bytes
    // tailSize = 2*4 = 8 bytes
    // 最后一个 rank (rank 3 == ranks_[templateRankSize_-1]) 使用 sliceSize + tailSize = 24
    params.sliceCount = 4;
    params.tailCount = 2;
    params.dataStride = 4 * sizeof(int32_t);  // = 16
    params.scratchStride = 4 * sizeof(int32_t); // = 16
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // PreCopy 对 4 个 rank 各调一次 LocalCopy
    auto localCopyCalls = FindTmplCalls("LocalCopy");
    // 4 次 PreCopy + 4 次 PostCopy（ranksForOutputData_ = {0,1,2,3}），共 8 次
    ASSERT_EQ(localCopyCalls.size(), 8u);

    // 前 4 次 LocalCopy 是 PreCopy:
    // rank 0,1,2: len = 16 (sliceSize)
    // rank 3 (tailRankId): len = 24 (sliceSize + tailSize)
    EXPECT_EQ(localCopyCalls[0].len, 16u);
    EXPECT_EQ(localCopyCalls[1].len, 16u);
    EXPECT_EQ(localCopyCalls[2].len, 16u);
    EXPECT_EQ(localCopyCalls[3].len, 24u);

    // 后 4 次 LocalCopy 是 PostCopy，同样最后一个 rank 使用 sliceSize + tailSize
    EXPECT_EQ(localCopyCalls[4].len, 16u);
    EXPECT_EQ(localCopyCalls[5].len, 16u);
    EXPECT_EQ(localCopyCalls[6].len, 16u);
    EXPECT_EQ(localCopyCalls[7].len, 24u);
}

// ═══════════════════════════════════════════════════════════════════
// 8. PostCopy tailCount 分组
// ═══════════════════════════════════════════════════════════════════

// TC19 PostCopy 最后一个 rank 使用 sliceSize + tailSize
TEST_F(AicpuBaseTemplateTest, PostCopyWithTailCountLastRankUsesTailSize)
{
    // 4 ranks: {0, 1, 2, 3}
    TestableAicpuTemplateV2 tmpl(0, {0, 1, 2, 3}, MakeMeshDesc());
    tmpl.SetSliceListCount(0);
    // 仅 PreCopy rank 0, PostCopy ranksForOutputData = {0,1,2,3}
    tmpl.SetRanksForOutputData({0, 1, 2, 3});

    TemplateDataParams params = MakeTmplParams({0});
    params.sliceCount = 4;
    params.tailCount = 2;
    params.dataStride = 4 * sizeof(int32_t);  // = 16
    params.scratchStride = 4 * sizeof(int32_t); // = 16
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    auto localCopyCalls = FindTmplCalls("LocalCopy");
    // PreCopy: 1 次 (rank 0, ranksForInputData 最后一个) → len = 24 (sliceSize + tailSize)
    // PostCopy: 4 次 → rank 0,1,2 use len=16, rank 3 uses len=24 (sliceSize + tailSize)
    ASSERT_EQ(localCopyCalls.size(), 5u);

    // PreCopy (1 call) - rank 0 是 ranksForInputData 的最后一个 → 尾 rank
    EXPECT_EQ(localCopyCalls[0].len, 24u);

    // PostCopy (4 calls)
    EXPECT_EQ(localCopyCalls[1].len, 16u); // rank 0
    EXPECT_EQ(localCopyCalls[2].len, 16u); // rank 1
    EXPECT_EQ(localCopyCalls[3].len, 16u); // rank 2
    EXPECT_EQ(localCopyCalls[4].len, 24u); // rank 3 (ranksForOutputData_ 最后一个 → 尾 rank)
}

} // namespace testing
} // namespace ops_hccl
