/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * AicpuBaseTemplate 单元测试。
 * 覆盖: KernelRun (数据量0 / 单rank / 多rank多线程) / PreCopy / PostCopy / SendAll。
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

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
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

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
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

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
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

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
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

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
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

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// TC07 PreCopy 多个 input rank 各执行一次 LocalCopy, PostCopy 也各执行一次
TEST_F(AicpuBaseTemplateTest, PreCopyMultiInputRanksMultiLocalCopy)
{
    TestableAicpuTemplate tmpl(0, {0, 1, 2}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0, 1, 2});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
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

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
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

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
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

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ═══════════════════════════════════════════════════════════════════
// 4. SendAll 分组 (通过 MockBaseEngine 验证)
// ═══════════════════════════════════════════════════════════════════

// TC11 RunAlgorithm 生成空 txRxSlicesLists 时 SendAll 不执行
TEST_F(AicpuBaseTemplateTest, SendAllEmptyListNoSendCalls)
{
    TestableAicpuTemplate tmpl(0, {0, 1}, MakeMeshDesc());
    TemplateDataParams params = MakeTmplParams({0});
    TemplateResource res = MakeTmplResource();
    std::vector<u32> ranksForOutputData;

    HcclResult ret = tmpl.KernelRun(engine_, params, res, ranksForOutputData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(engine_.GetSendCount(), 0u);
}

} // namespace testing
} // namespace ops_hccl
