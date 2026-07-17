/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * AICPU Engine Send UT helpers.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "base_engine.h"
#include "aicpu_engine.h"

namespace ops_hccl {
namespace testing {

// ───────────── stub 记录项 (与 ut_stubs.cc 共享) ─────────────
struct Hcall {
    std::string name;          // Write / Read / WriteReduce / ReadReduce / NotifyRecord / NotifyWait
    unsigned long thread = 0;
    unsigned long channel = 0;
    uint32_t idx = 0xFFFFFFFF; // notify idx
    void *dst = nullptr;
    const void *src = nullptr;
    uint64_t len = 0;          // 非 reduce=size_, reduce=count_
    int dt = 0;                // HcclDataType (reduce 时有效)
    int op = 0;                // HcclReduceOp (reduce 时有效)
};

extern std::vector<Hcall> g_records;
extern std::vector<std::vector<uint8_t>> g_captured;
extern bool g_failNext;
extern int32_t g_failRet;
extern std::string g_failName;

// ───────────── mock 控制 ─────────────
inline void ClearMock()
{
    g_records.clear();
    g_captured.clear();
    g_failNext = false;
    g_failRet = 0;
    g_failName.clear();
}

inline size_t CountCalls(const std::string &name)
{
    size_t n = 0;
    for (const auto &c : g_records) {
        if (c.name == name) { ++n; }
    }
    return n;
}

inline std::vector<Hcall> FindCalls(const std::string &name)
{
    std::vector<Hcall> out;
    for (const auto &c : g_records) {
        if (c.name == name) { out.push_back(c); }
    }
    return out;
}

inline size_t CountTransferCalls()
{
    return CountCalls("Write") + CountCalls("Read") + CountCalls("WriteReduce") + CountCalls("ReadReduce");
}

// ───────────── 数据 pattern ─────────────
inline std::vector<uint8_t> MakePattern(uint64_t bytes, uint8_t seed)
{
    std::vector<uint8_t> v(bytes);
    for (uint64_t i = 0; i < bytes; ++i) {
        v[i] = static_cast<uint8_t>(seed + i);
    }
    return v;
}

inline void ExpectBytesEq(const std::vector<uint8_t> &got, const std::vector<uint8_t> &want)
{
    ASSERT_EQ(got.size(), want.size()) << "byte size mismatch";
    EXPECT_EQ(memcmp(got.data(), want.data(), got.size()), 0) << "data content mismatch";
}

inline uint64_t DataTypeSize(HcclDataType dt)
{
    switch (dt) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8: return 1;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_FP16: return 2;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
        case HCCL_DATA_TYPE_FP32: return 4;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64: return 8;
        default: return 0;
    }
}

// ───────────── TransferContext 构造 ─────────────
class AiCpuEngineSendTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ClearMock();
        // 分配真实 buffer, 避免 stub captureBytes memcpy 虚构地址导致 segfault
        srcBuf_.assign(4 * 1024 * 1024, 0xAA);
        dstBuf_.assign(4 * 1024 * 1024, 0x55);
        srcBase_ = srcBuf_.data();
        dstBase_ = dstBuf_.data();
    }

    ChannelInfo MakeChannel(u32 rank, unsigned long handle)
    {
        ChannelInfo ch;
        ch.isValid = true;
        ch.remoteRank = rank;
        ch.protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
        ch.locationType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
        ch.handle = handle;
        return ch;
    }

    SlicesList MakeSlices(uint64_t n, uint64_t sizeElem, uint64_t countElem)
    {
        std::vector<DataSlice> src;
        std::vector<DataSlice> dst;
        for (uint64_t i = 0; i < n; ++i) {
            uint64_t off = i * sizeElem;
            src.emplace_back(srcBase_, off, sizeElem, countElem);
            dst.emplace_back(dstBase_, off, sizeElem, countElem);
        }
        return SlicesList(src, dst);
    }

    // readDir=true → direction=READ; false → WRITE
    TransferContext MakeCtx(bool hasTx, bool hasRx, bool readDir,
                            HcclReduceOp reduceOp = HCCL_REDUCE_RESERVED,
                            uint64_t sliceN = 1, uint64_t sizeElem = 16, uint64_t countElem = 4)
    {
        TransferContext ctx;
        ctx.enableRemoteMemAccess = readDir;
        ctx.buffType = readDir ? BufferType::OUTPUT : BufferType::HCCL_BUFFER;
        ctx.dataType = HCCL_DATA_TYPE_INT32;
        ctx.reduceOp = reduceOp;

        ctx.templateRes.threads.push_back(static_cast<ThreadHandle>(0x01));
        ctx.templateRes.channels[1].push_back(MakeChannel(1, 0xC1));
        ctx.templateRes.channels[2].push_back(MakeChannel(2, 0xC2));

        ctx.txRxSlicesList.dstRankId_ = 1;
        ctx.txRxSlicesList.srcRankId_ = 2;
        if (hasTx) {
            ctx.txRxSlicesList.txSlicesList_ = MakeSlices(sliceN, sizeElem, countElem);
        }
        if (hasRx) {
            ctx.txRxSlicesList.rxSlicesList_ = MakeSlices(sliceN, sizeElem, countElem);
        }
        return ctx;
    }

    AiCpuEngine engine_;
    std::vector<uint8_t> srcBuf_;
    std::vector<uint8_t> dstBuf_;
    void *srcBase_ = nullptr;
    void *dstBase_ = nullptr;
};

} // namespace testing
} // namespace ops_hccl
