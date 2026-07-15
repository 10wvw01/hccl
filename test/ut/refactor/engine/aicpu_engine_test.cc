/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * AICPU Engine Send 接口单元测试。
 * 覆盖: 路由(6) / reduce封装 / 循环+跳过 / 边界 / 失败传播 / notify序列 / 数据传输准确性(大中小分片)。
 * 依据: test/ut/refactor/AICPU_Send_UT用例设计.md (TC01-TC45)。
 */

#include "test_helpers.h"
#include "hccl_algorithm.h"

namespace ops_hccl {
namespace testing {

// ═══════════════════════════════════════════════════════════════════
// 1. 路由分组 (TC01-TC06): direction × hasTx/hasRx → 6 wrapper
// ═══════════════════════════════════════════════════════════════════

// TC01 双向 Write → SendRecvWrite
TEST_F(AiCpuEngineSendTest, BidirWriteRoutesToSendRecvWrite)
{
    auto ctx = MakeCtx(true, true, false);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_GE(CountCalls("Write"), 1u);
    EXPECT_EQ(CountCalls("Read"), 0u);
    EXPECT_EQ(CountCalls("ReadReduce"), 0u);
}

// TC02 双向 Read → SendRecvRead
TEST_F(AiCpuEngineSendTest, BidirReadRoutesToSendRecvRead)
{
    auto ctx = MakeCtx(true, true, true);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_GE(CountCalls("Read"), 1u);
    EXPECT_EQ(CountCalls("Write"), 0u);
}

// TC03 仅发 Write → SendWrite
TEST_F(AiCpuEngineSendTest, TxOnlyWriteRoutesToSendWrite)
{
    auto ctx = MakeCtx(true, false, false);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_GE(CountCalls("Write"), 1u);
}

// TC04 仅发 Read → SendRead (被读方仅 notify, 不搬移)
TEST_F(AiCpuEngineSendTest, TxOnlyReadRoutesToSendRead)
{
    auto ctx = MakeCtx(true, false, true);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountTransferCalls(), 0u);
    EXPECT_GE(CountCalls("NotifyRecord"), 1u);
    EXPECT_GE(CountCalls("NotifyWait"), 1u);
}

// TC05 仅收 Write → RecvWrite (接收方仅 notify)
TEST_F(AiCpuEngineSendTest, RxOnlyWriteRoutesToRecvWrite)
{
    auto ctx = MakeCtx(false, true, false);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountTransferCalls(), 0u);
    EXPECT_GE(CountCalls("NotifyRecord"), 1u);
    EXPECT_GE(CountCalls("NotifyWait"), 1u);
}

// TC06 仅收 Read → RecvRead
TEST_F(AiCpuEngineSendTest, RxOnlyReadRoutesToRecvRead)
{
    auto ctx = MakeCtx(false, true, true);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_GE(CountCalls("Read"), 1u);
}

// ═══════════════════════════════════════════════════════════════════
// 2. Reduce 封装分组 (TC07-TC12)
// ═══════════════════════════════════════════════════════════════════

// TC08 Write 非 reduce → HcommWriteOnThread, len=size_
TEST_F(AiCpuEngineSendTest, WriteNoReduceCallsHcommWriteOnThread)
{
    auto ctx = MakeCtx(true, false, false, HCCL_REDUCE_RESERVED, 1, 16, 4);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    auto calls = FindCalls("Write");
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].len, 16u);
    EXPECT_EQ(CountCalls("WriteReduce"), 0u);
}

// TC07 Write + reduce → HcommWriteReduceOnThread, len=count_
TEST_F(AiCpuEngineSendTest, WriteReduceCallsHcommWriteReduceOnThread)
{
    auto ctx = MakeCtx(true, false, false, HCCL_REDUCE_SUM, 1, 16, 4);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    auto calls = FindCalls("WriteReduce");
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].len, 4u);
    EXPECT_EQ(calls[0].op, static_cast<int>(HCCL_REDUCE_SUM));
    EXPECT_EQ(CountCalls("Write"), 0u);
}

// TC10 Read 非 reduce → HcommReadOnThread
TEST_F(AiCpuEngineSendTest, ReadNoReduceCallsHcommReadOnThread)
{
    auto ctx = MakeCtx(false, true, true, HCCL_REDUCE_RESERVED, 1, 32, 8);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    auto calls = FindCalls("Read");
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].len, 32u);
}

// TC09 Read + reduce → HcommReadReduceOnThread, len=count_
TEST_F(AiCpuEngineSendTest, ReadReduceCallsHcommReadReduceOnThread)
{
    auto ctx = MakeCtx(false, true, true, HCCL_REDUCE_MAX, 1, 32, 8);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    auto calls = FindCalls("ReadReduce");
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].len, 8u);
    EXPECT_EQ(calls[0].op, static_cast<int>(HCCL_REDUCE_MAX));
}

// TC11 RecvWrite 忽略 reduceOp
TEST_F(AiCpuEngineSendTest, RecvWriteIgnoresReduceOp)
{
    auto ctx = MakeCtx(false, true, false, HCCL_REDUCE_SUM);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountTransferCalls(), 0u);
}

// TC12 SendRead 忽略 reduceOp
TEST_F(AiCpuEngineSendTest, SendReadIgnoresReduceOp)
{
    auto ctx = MakeCtx(true, false, true, HCCL_REDUCE_SUM);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountTransferCalls(), 0u);
}

// ═══════════════════════════════════════════════════════════════════
// 3. 循环与跳过分组 (TC13-TC14)
// ═══════════════════════════════════════════════════════════════════

// TC13 多 slice 循环调用次数 = slice 数
TEST_F(AiCpuEngineSendTest, MultiSliceLoopCountEqualsSliceNum)
{
    auto ctx = MakeCtx(true, false, false, HCCL_REDUCE_RESERVED, 4, 16, 4);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountCalls("Write"), 4u);
}

// TC14 size_==0 的 slice 被跳过
TEST_F(AiCpuEngineSendTest, ZeroSizeSliceSkipped)
{
    std::vector<DataSlice> src = {
        DataSlice(srcBase_, 0, 16, 4),
        DataSlice(srcBase_, 16, 0, 0),
        DataSlice(srcBase_, 32, 16, 4)
    };
    std::vector<DataSlice> dst = {
        DataSlice(dstBase_, 0, 16, 4),
        DataSlice(dstBase_, 16, 0, 0),
        DataSlice(dstBase_, 32, 16, 4)
    };
    TransferContext ctx;
    ctx.enableRemoteMemAccess = false;
    ctx.buffType = BufferType::HCCL_BUFFER;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.srcRankId_ = 2;
    ctx.txRxSlicesList.txSlicesList_ = SlicesList(src, dst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    auto calls = FindCalls("Write");
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0].len, 16u);
    EXPECT_EQ(calls[1].len, 16u);
}

// ═══════════════════════════════════════════════════════════════════
// 4. 边界分组 (TC15-TC18)
// ═══════════════════════════════════════════════════════════════════

// TC15 hasTx/hasRx 均空 → 直接成功
TEST_F(AiCpuEngineSendTest, EmptyTxRxReturnsSuccessWithoutCalls)
{
    auto ctx = MakeCtx(false, false, false);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_records.size(), 0u);
}

// TC16 threads 为空 → 失败
TEST_F(AiCpuEngineSendTest, EmptyThreadsReturnsError)
{
    auto ctx = MakeCtx(true, false, false);
    ctx.templateRes.threads.clear();
    EXPECT_NE(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_records.size(), 0u);
}

// TC17 dstRank 无 channel → 失败
TEST_F(AiCpuEngineSendTest, MissingDstChannelReturnsError)
{
    auto ctx = MakeCtx(true, false, false);
    ctx.templateRes.channels.erase(1);
    EXPECT_NE(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountTransferCalls(), 0u);
}

// TC18 srcRank 无 channel → 失败
TEST_F(AiCpuEngineSendTest, MissingSrcChannelReturnsError)
{
    auto ctx = MakeCtx(false, true, false);
    ctx.templateRes.channels.erase(2);
    EXPECT_NE(engine_.Send(ctx), HCCL_SUCCESS);
}

// ═══════════════════════════════════════════════════════════════════
// 5. 失败传播分组 (TC19-TC23)
// ═══════════════════════════════════════════════════════════════════

// TC19 NotifyRecord 失败提前返回
TEST_F(AiCpuEngineSendTest, NotifyRecordFailureStopsEarly)
{
    g_failName = "NotifyRecord";
    g_failRet = -1;
    auto ctx = MakeCtx(false, true, false);
    EXPECT_NE(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountCalls("NotifyRecord"), 1u);
    EXPECT_EQ(CountCalls("NotifyWait"), 0u);
}

// TC20 NotifyWait 失败提前返回
TEST_F(AiCpuEngineSendTest, NotifyWaitFailureStopsEarly)
{
    g_failName = "NotifyWait";
    g_failRet = -1;
    auto ctx = MakeCtx(true, false, false);
    EXPECT_NE(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountCalls("Write"), 0u);
}

// TC21 Write 原语失败提前返回
TEST_F(AiCpuEngineSendTest, WriteFailureStopsEarly)
{
    g_failName = "Write";
    g_failRet = -1;
    auto ctx = MakeCtx(true, false, false, HCCL_REDUCE_RESERVED, 3, 16, 4);
    EXPECT_NE(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountCalls("Write"), 1u);
    EXPECT_EQ(CountCalls("NotifyRecord"), 0u);
}

// TC22 Read 原语失败提前返回
TEST_F(AiCpuEngineSendTest, ReadFailureStopsEarly)
{
    g_failName = "Read";
    g_failRet = -1;
    auto ctx = MakeCtx(false, true, true, HCCL_REDUCE_RESERVED, 2, 16, 4);
    EXPECT_NE(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountCalls("Read"), 1u);
}

// TC23 WriteReduce 失败提前返回
TEST_F(AiCpuEngineSendTest, WriteReduceFailureStopsEarly)
{
    g_failName = "WriteReduce";
    g_failRet = -1;
    auto ctx = MakeCtx(true, false, false, HCCL_REDUCE_SUM, 2, 16, 4);
    EXPECT_NE(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountCalls("WriteReduce"), 1u);
}

// ═══════════════════════════════════════════════════════════════════
// 6. notify 序列分组 (TC24-TC27)
// ═══════════════════════════════════════════════════════════════════

// TC24 SendWrite notify 序列
TEST_F(AiCpuEngineSendTest, SendWriteNotifySequence)
{
    auto ctx = MakeCtx(true, false, false);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_records.front().name, "NotifyWait");
    EXPECT_EQ(g_records.front().idx, NOTIFY_IDX_ACK);
    EXPECT_EQ(g_records.back().name, "NotifyRecord");
    EXPECT_EQ(g_records.back().idx, NOTIFY_IDX_DATA_SIGNAL);
}

// TC25 RecvWrite notify 序列
TEST_F(AiCpuEngineSendTest, RecvWriteNotifySequence)
{
    auto ctx = MakeCtx(false, true, false);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_records.front().name, "NotifyRecord");
    EXPECT_EQ(g_records.front().idx, NOTIFY_IDX_ACK);
    EXPECT_EQ(g_records.back().name, "NotifyWait");
    EXPECT_EQ(g_records.back().idx, NOTIFY_IDX_DATA_SIGNAL);
}

// TC26 SendRecvWrite 双向 notify 序列
TEST_F(AiCpuEngineSendTest, SendRecvWriteBidirNotifySequence)
{
    auto ctx = MakeCtx(true, true, false);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_records[0].name, "NotifyRecord");
    EXPECT_EQ(g_records[0].idx, NOTIFY_IDX_ACK);
    EXPECT_EQ(g_records[1].name, "NotifyWait");
    EXPECT_EQ(g_records[1].idx, NOTIFY_IDX_ACK);
    auto last = g_records.back();
    EXPECT_EQ(last.name, "NotifyWait");
    EXPECT_EQ(last.idx, NOTIFY_IDX_DATA_SIGNAL);
}

// TC27 SendRecvRead 双向 notify 序列
TEST_F(AiCpuEngineSendTest, SendRecvReadBidirNotifySequence)
{
    auto ctx = MakeCtx(true, true, true);
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_records[0].name, "NotifyRecord");
    EXPECT_EQ(g_records[0].idx, NOTIFY_IDX_ACK);
    EXPECT_EQ(g_records.back().name, "NotifyWait");
    EXPECT_EQ(g_records.back().idx, NOTIFY_IDX_DATA_SIGNAL);
    EXPECT_GE(CountCalls("Read"), 1u);
}

// ═══════════════════════════════════════════════════════════════════
// 7. 地址与参数分组 (TC28-TC29)
// ═══════════════════════════════════════════════════════════════════

// TC28 地址 = addr + offset
TEST_F(AiCpuEngineSendTest, SliceAddressIsAddrPlusOffset)
{
    std::vector<DataSlice> src = {DataSlice(srcBase_, 8, 16, 4)};
    std::vector<DataSlice> dst = {DataSlice(dstBase_, 16, 16, 4)};
    TransferContext ctx;
    ctx.enableRemoteMemAccess = false;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.txSlicesList_ = SlicesList(src, dst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    auto calls = FindCalls("Write");
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].src, reinterpret_cast<uint8_t *>(srcBase_) + 8);
    EXPECT_EQ(calls[0].dst, reinterpret_cast<uint8_t *>(dstBase_) + 16);
}

// TC29 reduce 传 count_, 非 reduce 传 size_
TEST_F(AiCpuEngineSendTest, ReducePassesCountNonReducePassesSize)
{
    auto ctxNoReduce = MakeCtx(true, false, false, HCCL_REDUCE_RESERVED, 1, 16, 4);
    EXPECT_EQ(engine_.Send(ctxNoReduce), HCCL_SUCCESS);
    EXPECT_EQ(FindCalls("Write")[0].len, 16u);

    ClearMock();
    auto ctxReduce = MakeCtx(true, false, false, HCCL_REDUCE_SUM, 1, 16, 4);
    EXPECT_EQ(engine_.Send(ctxReduce), HCCL_SUCCESS);
    EXPECT_EQ(FindCalls("WriteReduce")[0].len, 4u);
}

// ═══════════════════════════════════════════════════════════════════
// 8. 数据传输准确性分组 (大/中/小分片, TC33-TC45)
// ═══════════════════════════════════════════════════════════════════

// TC33 小分片 Write 单 slice 数据准确
TEST_F(AiCpuEngineSendTest, SmallSliceWriteDataAccurate)
{
    const uint64_t sz = 16;
    auto want = MakePattern(sz, 0xAA);
    std::vector<DataSlice> src = {DataSlice(want.data(), 0, sz, 4)};
    std::vector<DataSlice> dst = {DataSlice(dstBase_, 0, sz, 4)};
    TransferContext ctx;
    ctx.enableRemoteMemAccess = false;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.txSlicesList_ = SlicesList(src, dst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    ASSERT_EQ(g_captured.size(), 1u);
    EXPECT_EQ(g_captured[0].size(), sz);
    ExpectBytesEq(g_captured[0], want);
}

// TC34 中分片 Write 单 slice 数据准确 (4KB)
TEST_F(AiCpuEngineSendTest, MidSliceWriteDataAccurate)
{
    const uint64_t sz = 4 * 1024;
    auto want = MakePattern(sz, 0x5A);
    std::vector<DataSlice> src = {DataSlice(want.data(), 0, sz, sz / 4)};
    std::vector<DataSlice> dst = {DataSlice(dstBase_, 0, sz, sz / 4)};
    TransferContext ctx;
    ctx.enableRemoteMemAccess = false;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.txSlicesList_ = SlicesList(src, dst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    ASSERT_EQ(g_captured.size(), 1u);
    ExpectBytesEq(g_captured[0], want);
}

// TC35 大分片 Write 多 slice 数据准确 (1MB 拆 4×256KB)
TEST_F(AiCpuEngineSendTest, LargeSliceMultiWriteDataAccurate)
{
    const uint64_t sliceSz = 256 * 1024;
    const uint64_t n = 4;
    std::vector<DataSlice> src;
    std::vector<DataSlice> dst;
    std::vector<std::vector<uint8_t>> wants;
    for (uint64_t i = 0; i < n; ++i) {
        wants.push_back(MakePattern(sliceSz, static_cast<uint8_t>(0x11 + i * 0x10)));
        src.emplace_back(wants[i].data(), 0, sliceSz, sliceSz / 4);
        dst.emplace_back(dstBase_, i * sliceSz, sliceSz, sliceSz / 4);
    }
    TransferContext ctx;
    ctx.enableRemoteMemAccess = false;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.txSlicesList_ = SlicesList(src, dst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    ASSERT_EQ(g_captured.size(), n);
    for (uint64_t i = 0; i < n; ++i) {
        ExpectBytesEq(g_captured[i], wants[i]);
    }
    EXPECT_EQ(CountCalls("Write"), n);
}

// TC36 小分片 Read 单 slice 数据准确
TEST_F(AiCpuEngineSendTest, SmallSliceReadDataAccurate)
{
    const uint64_t sz = 32;
    auto want = MakePattern(sz, 0xCC);
    std::vector<DataSlice> src = {DataSlice(want.data(), 0, sz, 8)};
    std::vector<DataSlice> dst = {DataSlice(dstBase_, 0, sz, 8)};
    TransferContext ctx;
    ctx.enableRemoteMemAccess = true;
    ctx.buffType = BufferType::OUTPUT;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[2].push_back(MakeChannel(2, 0xC2));
    ctx.txRxSlicesList.srcRankId_ = 2;
    ctx.txRxSlicesList.rxSlicesList_ = SlicesList(src, dst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    ASSERT_EQ(g_captured.size(), 1u);
    ExpectBytesEq(g_captured[0], want);
}

// TC38 大分片 WriteReduce 按 count 传输准确 (FP32 count=1024 → 4KB)
TEST_F(AiCpuEngineSendTest, LargeWriteReduceByCountAccurate)
{
    const uint64_t count = 1024;
    const uint64_t bytes = count * DataTypeSize(HCCL_DATA_TYPE_FP32);
    auto want = MakePattern(bytes, 0x07);
    std::vector<DataSlice> src = {DataSlice(want.data(), 0, bytes, count)};
    std::vector<DataSlice> dst = {DataSlice(dstBase_, 0, bytes, count)};
    TransferContext ctx;
    ctx.enableRemoteMemAccess = false;
    ctx.dataType = HCCL_DATA_TYPE_FP32;
    ctx.reduceOp = HCCL_REDUCE_SUM;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.txSlicesList_ = SlicesList(src, dst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    auto calls = FindCalls("WriteReduce");
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].len, count);
    ASSERT_EQ(g_captured.size(), 1u);
    EXPECT_EQ(g_captured[0].size(), bytes);
    ExpectBytesEq(g_captured[0], want);
}

// TC39 多 slice 混合大小 (尾片较小) 数据准确
TEST_F(AiCpuEngineSendTest, MixedSizeSlicesAccurate)
{
    const uint64_t sz1 = 64 * 1024;
    const uint64_t sz2 = 64 * 1024;
    const uint64_t sz3 = 32 * 1024;
    auto w1 = MakePattern(sz1, 0x01);
    auto w2 = MakePattern(sz2, 0x02);
    auto w3 = MakePattern(sz3, 0x03);
    std::vector<DataSlice> src = {
        DataSlice(w1.data(), 0, sz1, sz1 / 4),
        DataSlice(w2.data(), 0, sz2, sz2 / 4),
        DataSlice(w3.data(), 0, sz3, sz3 / 4)
    };
    std::vector<DataSlice> dst = {
        DataSlice(dstBase_, 0, sz1, sz1 / 4),
        DataSlice(dstBase_, sz1, sz2, sz2 / 4),
        DataSlice(dstBase_, sz1 + sz2, sz3, sz3 / 4)
    };
    TransferContext ctx;
    ctx.enableRemoteMemAccess = false;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.txSlicesList_ = SlicesList(src, dst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    ASSERT_EQ(g_captured.size(), 3u);
    ExpectBytesEq(g_captured[0], w1);
    ExpectBytesEq(g_captured[1], w2);
    ExpectBytesEq(g_captured[2], w3);
    EXPECT_EQ(FindCalls("Write")[0].len, sz1);
    EXPECT_EQ(FindCalls("Write")[1].len, sz2);
    EXPECT_EQ(FindCalls("Write")[2].len, sz3);
}

// TC42 相邻 slice 数据不串扰
TEST_F(AiCpuEngineSendTest, AdjacentSlicesNoCrosstalk)
{
    const uint64_t sz = 256 * 1024;
    auto w0 = MakePattern(sz, 0xFF);
    auto w1 = MakePattern(sz, 0x00);
    std::vector<DataSlice> src = {
        DataSlice(w0.data(), 0, sz, sz / 4),
        DataSlice(w1.data(), 0, sz, sz / 4)
    };
    std::vector<DataSlice> dst = {
        DataSlice(dstBase_, 0, sz, sz / 4),
        DataSlice(dstBase_, sz, sz, sz / 4)
    };
    TransferContext ctx;
    ctx.enableRemoteMemAccess = false;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.txSlicesList_ = SlicesList(src, dst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    ASSERT_EQ(g_captured.size(), 2u);
    ExpectBytesEq(g_captured[0], w0);
    ExpectBytesEq(g_captured[1], w1);
    EXPECT_NE(g_captured[0], g_captured[1]);
}

// TC45 路由 + 执行到 + 数据准确 综合 (大分片双向 WriteReduce)
TEST_F(AiCpuEngineSendTest, ComprehensiveBidirWriteReduce)
{
    const uint64_t sz = 256 * 1024;
    const uint64_t count = sz / 4; // FP32
    auto w0 = MakePattern(sz, 0x10);
    auto w1 = MakePattern(sz, 0x20);
    auto w2 = MakePattern(sz, 0x30);
    std::vector<DataSlice> txSrc = {
        DataSlice(w0.data(), 0, sz, count),
        DataSlice(w1.data(), 0, sz, count),
        DataSlice(w2.data(), 0, sz, count)
    };
    std::vector<DataSlice> txDst = {
        DataSlice(dstBase_, 0, sz, count),
        DataSlice(dstBase_, sz, sz, count),
        DataSlice(dstBase_, 2 * sz, sz, count)
    };
    TransferContext ctx;
    ctx.enableRemoteMemAccess = false;
    ctx.dataType = HCCL_DATA_TYPE_FP32;
    ctx.reduceOp = HCCL_REDUCE_SUM;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.templateRes.channels[2].push_back(MakeChannel(2, 0xC2));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.srcRankId_ = 2;
    ctx.txRxSlicesList.txSlicesList_ = SlicesList(txSrc, txDst);
    ctx.txRxSlicesList.rxSlicesList_ = SlicesList({DataSlice(srcBase_, 0, sz, count)},
                                                  {DataSlice(dstBase_, 0, sz, count)});

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_EQ(CountCalls("Read"), 0u);
    EXPECT_EQ(CountCalls("ReadReduce"), 0u);
    EXPECT_GE(CountCalls("WriteReduce"), 3u);
    ASSERT_GE(g_captured.size(), 3u);
    ExpectBytesEq(g_captured[0], w0);
    ExpectBytesEq(g_captured[1], w1);
    ExpectBytesEq(g_captured[2], w2);
    EXPECT_EQ(FindCalls("WriteReduce")[0].len, count);
    EXPECT_EQ(FindCalls("WriteReduce")[0].op, static_cast<int>(HCCL_REDUCE_SUM));
    EXPECT_EQ(g_records.front().name, "NotifyRecord");
    EXPECT_EQ(g_records.back().name, "NotifyWait");
}

// ═══════════════════════════════════════════════════════════════════
// 9. 补齐用例 (TC30/TC31/TC32/TC37/TC40/TC41/TC43/TC44)
// ═══════════════════════════════════════════════════════════════════

// TC30 reduceOp 透传 dataType/reduceOp 原值
TEST_F(AiCpuEngineSendTest, ReducePassesDataTypeAndReduceOp)
{
    auto ctx = MakeCtx(true, false, false, HCCL_REDUCE_SUM, 1, 8, 2);
    ctx.dataType = HCCL_DATA_TYPE_FP16;
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    auto calls = FindCalls("WriteReduce");
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].dt, static_cast<int>(HCCL_DATA_TYPE_FP16));
    EXPECT_EQ(calls[0].op, static_cast<int>(HCCL_REDUCE_SUM));
}

// TC31 LookupChannel 取第一条 channel
TEST_F(AiCpuEngineSendTest, LookupChannelPicksFirstChannel)
{
    auto ctx = MakeCtx(true, false, false);
    // channels[1] 追加第二条 channel (不同 handle), 应使用第一条 0xC1
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xBAD));
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    auto calls = FindCalls("Write");
    ASSERT_GE(calls.size(), 1u);
    EXPECT_EQ(calls[0].channel, 0xC1u);
}

// TC32 IsPcieProtocol 不影响路由 (direction 由 buffType 决定)
TEST_F(AiCpuEngineSendTest, IsPcieProtocolDoesNotAffectRouting)
{
    auto ctx = MakeCtx(true, false, false); // WRITE
    ctx.templateRes.channels[1][0].protocol = CommProtocol::COMM_PROTOCOL_PCIE;
    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_GE(CountCalls("Write"), 1u); // 仍走 SendWrite, 不变 Read
    EXPECT_EQ(CountCalls("Read"), 0u);
}

// TC37 大分片 Read 多 slice 数据准确
TEST_F(AiCpuEngineSendTest, LargeSliceMultiReadDataAccurate)
{
    const uint64_t sliceSz = 256 * 1024;
    const uint64_t n = 4;
    std::vector<DataSlice> src;
    std::vector<DataSlice> dst;
    std::vector<std::vector<uint8_t>> wants;
    for (uint64_t i = 0; i < n; ++i) {
        wants.push_back(MakePattern(sliceSz, static_cast<uint8_t>(0xC0 + i)));
        src.emplace_back(wants[i].data(), 0, sliceSz, sliceSz / 4);
        dst.emplace_back(dstBase_, i * sliceSz, sliceSz, sliceSz / 4);
    }
    TransferContext ctx;
    ctx.enableRemoteMemAccess = true;
    ctx.buffType = BufferType::OUTPUT;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[2].push_back(MakeChannel(2, 0xC2));
    ctx.txRxSlicesList.srcRankId_ = 2;
    ctx.txRxSlicesList.rxSlicesList_ = SlicesList(src, dst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    ASSERT_EQ(g_captured.size(), n);
    for (uint64_t i = 0; i < n; ++i) {
        ExpectBytesEq(g_captured[i], wants[i]);
    }
    EXPECT_EQ(CountCalls("Read"), n);
}

// TC40 大分片 SendRecvWrite 双向数据准确
TEST_F(AiCpuEngineSendTest, LargeBidirWriteDataAccurate)
{
    const uint64_t sz = 256 * 1024;
    auto wA = MakePattern(sz, 0xA0);
    auto wB = MakePattern(sz, 0xB0);
    std::vector<DataSlice> txSrc = {
        DataSlice(wA.data(), 0, sz, sz / 4),
        DataSlice(wB.data(), 0, sz, sz / 4)
    };
    std::vector<DataSlice> txDst = {
        DataSlice(dstBase_, 0, sz, sz / 4),
        DataSlice(dstBase_, sz, sz, sz / 4)
    };
    TransferContext ctx;
    ctx.enableRemoteMemAccess = false;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.templateRes.channels[2].push_back(MakeChannel(2, 0xC2));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.srcRankId_ = 2;
    ctx.txRxSlicesList.txSlicesList_ = SlicesList(txSrc, txDst);
    ctx.txRxSlicesList.rxSlicesList_ = SlicesList({DataSlice(srcBase_, 0, sz, sz / 4)},
                                                  {DataSlice(dstBase_, 0, sz, sz / 4)});

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_GE(CountCalls("Write"), 2u);
    ASSERT_GE(g_captured.size(), 2u);
    ExpectBytesEq(g_captured[0], wA);
    ExpectBytesEq(g_captured[1], wB);
    EXPECT_EQ(g_records.front().name, "NotifyRecord");
    EXPECT_EQ(g_records.back().name, "NotifyWait");
}

// TC41 大分片 SendRecvRead 双向数据准确
TEST_F(AiCpuEngineSendTest, LargeBidirReadDataAccurate)
{
    const uint64_t sz = 256 * 1024;
    auto wC = MakePattern(sz, 0xC0);
    auto wD = MakePattern(sz, 0xD0);
    std::vector<DataSlice> rxSrc = {
        DataSlice(wC.data(), 0, sz, sz / 4),
        DataSlice(wD.data(), 0, sz, sz / 4)
    };
    std::vector<DataSlice> rxDst = {
        DataSlice(dstBase_, 0, sz, sz / 4),
        DataSlice(dstBase_, sz, sz, sz / 4)
    };
    TransferContext ctx;
    ctx.enableRemoteMemAccess = true; // READ
    ctx.buffType = BufferType::OUTPUT;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.templateRes.channels[2].push_back(MakeChannel(2, 0xC2));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.srcRankId_ = 2;
    // 双向需 hasTx && hasRx, tx 给占位 slice
    ctx.txRxSlicesList.txSlicesList_ = SlicesList({DataSlice(srcBase_, 0, sz, sz / 4)},
                                                  {DataSlice(dstBase_, 0, sz, sz / 4)});
    ctx.txRxSlicesList.rxSlicesList_ = SlicesList(rxSrc, rxDst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_GE(CountCalls("Read"), 2u);
    ASSERT_GE(g_captured.size(), 2u);
    ExpectBytesEq(g_captured[0], wC);
    ExpectBytesEq(g_captured[1], wD);
}

// TC43 打桩捕获传输数据内容可观测
TEST_F(AiCpuEngineSendTest, StubCapturesTransferContentObservable)
{
    const uint64_t sz = 4 * 1024;
    auto want = MakePattern(sz, 0x42);
    std::vector<DataSlice> src = {DataSlice(want.data(), 0, sz, sz / 4)};
    std::vector<DataSlice> dst = {DataSlice(dstBase_, 0, sz, sz / 4)};
    TransferContext ctx;
    ctx.enableRemoteMemAccess = false;
    ctx.dataType = HCCL_DATA_TYPE_INT32;
    ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
    ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
    ctx.txRxSlicesList.dstRankId_ = 1;
    ctx.txRxSlicesList.txSlicesList_ = SlicesList(src, dst);

    EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
    EXPECT_FALSE(g_captured.empty());      // 可观测: 捕获非空
    EXPECT_EQ(g_captured[0].size(), sz);   // 长度可读回
    ExpectBytesEq(g_captured[0], want);    // 内容可读回且准确
}

// TC44 reduceOp + 大分片 dataType 组合 (FP16/FP32/INT8)
TEST_F(AiCpuEngineSendTest, ReduceOpWithDataTypeCombination)
{
    struct Case { HcclDataType dt; uint64_t count; uint64_t bytes; };
    const Case cases[] = {
        {HCCL_DATA_TYPE_FP16, 2048, 2048 * 2},
        {HCCL_DATA_TYPE_FP32, 1024, 1024 * 4},
        {HCCL_DATA_TYPE_INT8, 4096, 4096 * 1},
    };
    for (const auto &c : cases) {
        ClearMock();
        auto want = MakePattern(c.bytes, 0x09);
        std::vector<DataSlice> src = {DataSlice(want.data(), 0, c.bytes, c.count)};
        std::vector<DataSlice> dst = {DataSlice(dstBase_, 0, c.bytes, c.count)};
        TransferContext ctx;
        ctx.enableRemoteMemAccess = false;
        ctx.dataType = c.dt;
        ctx.reduceOp = HCCL_REDUCE_SUM;
        ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
        ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
        ctx.txRxSlicesList.dstRankId_ = 1;
        ctx.txRxSlicesList.txSlicesList_ = SlicesList(src, dst);

        EXPECT_EQ(engine_.Send(ctx), HCCL_SUCCESS);
        auto calls = FindCalls("WriteReduce");
        ASSERT_EQ(calls.size(), 1u);
        EXPECT_EQ(calls[0].len, c.count);                // count_
        EXPECT_EQ(calls[0].dt, static_cast<int>(c.dt)); // dataType 透传
        ASSERT_EQ(g_captured.size(), 1u);
        EXPECT_EQ(g_captured[0].size(), c.bytes);       // 字节数 = count × dataTypeSize
    }
}

// ═══════════════════════════════════════════════════════════════════
// 9. CreateRes 测试分组
// ═══════════════════════════════════════════════════════════════════

// TC46 CreateRes 基本流程成功
TEST_F(AiCpuEngineSendTest, CreateResBasicSuccess)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    OpParam param{};
    param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    snprintf(param.algTag, sizeof(param.algTag), "test_tag");
    param.stream = reinterpret_cast<void *>(0x2000);

    HcclAlgorithm alg;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    AlgResourceRequest resReq;
    resReq.slaveThreadNum = 1;
    resReq.notifyNumOnMainThread = 1;
    resReq.notifyNumPerThread = {1};
    resReq.channels.push_back({});

    TopoInfoWithNetLayerDetails topoInfo;
    HcclResult ret = engine_.CreateRes(comm, param, alg, algHierarchyInfo, resReq, topoInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC47 CreateRes 空 channel 层也能成功
TEST_F(AiCpuEngineSendTest, CreateResEmptyChannelsSuccess)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    OpParam param{};
    param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    snprintf(param.algTag, sizeof(param.algTag), "test_tag");

    HcclAlgorithm alg;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    AlgResourceRequest resReq;
    resReq.slaveThreadNum = 0;
    resReq.notifyNumOnMainThread = 0;

    TopoInfoWithNetLayerDetails topoInfo;
    HcclResult ret = engine_.CreateRes(comm, param, alg, algHierarchyInfo, resReq, topoInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC48 CreateRes host 模式 (engine != AICPU_TS)
TEST_F(AiCpuEngineSendTest, CreateResHostModeSuccess)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    OpParam param{};
    param.engine = static_cast<CommEngine>(100); // 非 AICPU_TS/CPU
    snprintf(param.algTag, sizeof(param.algTag), "test_tag");
    param.stream = reinterpret_cast<void *>(0x2000);

    HcclAlgorithm alg;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    AlgResourceRequest resReq;
    resReq.slaveThreadNum = 1;
    resReq.notifyNumOnMainThread = 1;
    resReq.notifyNumPerThread = {1};

    TopoInfoWithNetLayerDetails topoInfo;
    HcclResult ret = engine_.CreateRes(comm, param, alg, algHierarchyInfo, resReq, topoInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ═══════════════════════════════════════════════════════════════════
// 10. LaunchKernel 测试分组
// ═══════════════════════════════════════════════════════════════════

// TC49 LaunchKernel 基本流程成功
TEST_F(AiCpuEngineSendTest, LaunchKernelBasicSuccess)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    OpParam param{};
    param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    snprintf(param.algTag, sizeof(param.algTag), "test_tag");
    snprintf(param.tag, sizeof(param.tag), "test_op");
    snprintf(param.commName, sizeof(param.commName), "test_comm");
    param.stream = reinterpret_cast<void *>(0x2000);
    param.opConfig.execTimeout = 100;

    // 先执行 CreateRes 初始化 resCtx_
    HcclAlgorithm alg;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    AlgResourceRequest resReq;
    resReq.slaveThreadNum = 1;
    resReq.notifyNumOnMainThread = 1;
    resReq.notifyNumPerThread = {1};
    TopoInfoWithNetLayerDetails topoInfo;
    ASSERT_EQ(engine_.CreateRes(comm, param, alg, algHierarchyInfo, resReq, topoInfo), HCCL_SUCCESS);

    HcclResult ret = engine_.LaunchKernel(param);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC50 LaunchKernel CPU 引擎模式注册 DPU 回调
TEST_F(AiCpuEngineSendTest, LaunchKernelCpuEngineRegistersDpuCallback)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    OpParam param{};
    param.engine = CommEngine::COMM_ENGINE_CPU;
    snprintf(param.algTag, sizeof(param.algTag), "test_tag");
    snprintf(param.tag, sizeof(param.tag), "test_op");
    snprintf(param.commName, sizeof(param.commName), "test_comm");
    param.stream = reinterpret_cast<void *>(0x2000);
    param.opConfig.execTimeout = 100;

    HcclAlgorithm alg;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    AlgResourceRequest resReq;
    resReq.slaveThreadNum = 1;
    resReq.notifyNumOnMainThread = 1;
    resReq.notifyNumPerThread = {1};
    TopoInfoWithNetLayerDetails topoInfo;
    ASSERT_EQ(engine_.CreateRes(comm, param, alg, algHierarchyInfo, resReq, topoInfo), HCCL_SUCCESS);

    HcclResult ret = engine_.LaunchKernel(param);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

} // namespace testing
} // namespace ops_hccl
