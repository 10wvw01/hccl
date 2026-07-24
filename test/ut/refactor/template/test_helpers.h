/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Template UT helpers.
 */

#pragma once

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "hccl_algorithm.h"
#include "template/base_template.h"
#include "template/aicpu/aicpu_base_template.h"
#include "template/aicpu/allgather_mesh.h"
#include "template/aicpu/allgather_nhr.h"
#include "template/aicpu/reducescatter_mesh.h"
#include "template/aicpu/reducescatter_nhr.h"
#include "template/aicpu/scatter_mesh.h"
#include "template/aicpu/scatter_nhr.h"
#include "template/primitives/mesh_primitives.h"
#include "template/primitives/nhr_primitives.h"

namespace ops_hccl {
namespace testing {

// ───────────── mock 记录项 ─────────────
struct TemplateHcall {
    std::string name;       // LocalCopy / NotifyRecord / NotifyWait
    unsigned long thread = 0;
    unsigned long channel = 0;
    uint32_t idx = 0xFFFFFFFF;
    void *dst = nullptr;
    const void *src = nullptr;
    uint64_t len = 0;
};

extern std::vector<TemplateHcall> g_tmplRecords;
extern std::vector<std::vector<uint8_t>> g_tmplCaptured;
extern bool g_tmplFailNext;
extern int32_t g_tmplFailRet;
extern std::string g_tmplFailName;

inline void ClearTmplMock()
{
    g_tmplRecords.clear();
    g_tmplCaptured.clear();
    g_tmplFailNext = false;
    g_tmplFailRet = 0;
    g_tmplFailName.clear();
}

inline size_t CountTmplCalls(const std::string &name)
{
    size_t n = 0;
    for (const auto &c : g_tmplRecords) {
        if (c.name == name) { ++n; }
    }
    return n;
}

inline std::vector<TemplateHcall> FindTmplCalls(const std::string &name)
{
    std::vector<TemplateHcall> out;
    for (const auto &c : g_tmplRecords) {
        if (c.name == name) { out.push_back(c); }
    }
    return out;
}

// ───────────── Mesh primitives test fixture ─────────────
class MeshAllGatherTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        localCclMem_ = reinterpret_cast<void *>(0x10000000);
        remoteCclMemBase_ = 0x20000000;
    }

    TemplateDataParams MakeParams(const std::vector<u32> &ranksForInputData, u64 sliceCount = 4)
    {
        TemplateDataParams params;
        params.cclBufferPtr = localCclMem_;
        params.dataType = HCCL_DATA_TYPE_INT32;
        params.sliceCount = sliceCount;
        params.sliceOffset = 0;
        params.dataStride = sliceCount * sizeof(int32_t);
        params.scratchStride = sliceCount * sizeof(int32_t);
        params.ranksForInputData = ranksForInputData;
        return params;
    }

    TemplateResource MakeResource(const std::vector<u32> &ranks, u32 myRank)
    {
        TemplateResource resource;
        for (u32 rank : ranks) {
            if (rank == myRank) {
                continue;
            }
            ChannelInfo channel;
            channel.remoteRank = rank;
            channel.remoteCclMem.addr = reinterpret_cast<void *>(remoteCclMemBase_ + rank * 0x1000);
            channel.remoteCclMem.size = 0x10000;
            resource.channels[rank].emplace_back(channel);
        }
        return resource;
    }

    const DataSlice &TxSrc(const TxRxSlicesList &info, size_t idx = 0) const
    {
        return info.txSlicesList_.srcSlices_[idx];
    }

    const DataSlice &TxDst(const TxRxSlicesList &info, size_t idx = 0) const
    {
        return info.txSlicesList_.dstSlices_[idx];
    }

    const DataSlice &RxSrc(const TxRxSlicesList &info, size_t idx = 0) const
    {
        return info.rxSlicesList_.srcSlices_[idx];
    }

    const DataSlice &RxDst(const TxRxSlicesList &info, size_t idx = 0) const
    {
        return info.rxSlicesList_.dstSlices_[idx];
    }

    void *localCclMem_ = nullptr;
    uintptr_t remoteCclMemBase_ = 0;
};

// ───────────── NHR primitives test fixture ─────────────
class NhrAllGatherTest : public MeshAllGatherTest {};

// ───────────── AicpuBaseTemplate test fixture ─────────────
class AicpuBaseTemplateTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ClearTmplMock();
        // 分配真实 buffer 供 LocalCopy stub 拷贝
        buf_.assign(256 * 1024, 0xAA);
        outBuf_.assign(256 * 1024, 0x55);
        inputBuf_.assign(256 * 1024, 0x77);
        cclMem_ = buf_.data();
        outMem_ = outBuf_.data();
        inputMem_ = inputBuf_.data();
    }

    TemplateDesc MakeMeshDesc()
    {
        return TemplateDesc{HcclCMDType::HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
                            HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY};
    }

    TemplateDesc MakeNhrDesc()
    {
        return TemplateDesc{HcclCMDType::HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_NHR,
                            HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY};
    }

    TemplateDesc MakeReduceScatterMeshDesc()
    {
        return TemplateDesc{HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
                            HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY};
    }

    TemplateDesc MakeReduceScatterNhrDesc()
    {
        return TemplateDesc{HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclAlgoType::HCCL_ALGO_TYPE_NHR,
                            HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY};
    }

    TemplateDesc MakeScatterMeshDesc()
    {
        return TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
                            HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY};
    }

    TemplateDesc MakeScatterNhrDesc()
    {
        return TemplateDesc{HcclCMDType::HCCL_CMD_SCATTER, HcclAlgoType::HCCL_ALGO_TYPE_NHR,
                            HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY};
    }

    // Scatter 专用参数：root 持有全部 rank 数据，其他 rank 无输入。
    TemplateDataParams MakeScatterTmplParams(u32 root, const std::vector<u32> &ranks,
                                             u64 sliceCount = 4)
    {
        TemplateDataParams params;
        params.inputBufferPtr = inputMem_;
        params.outputBufferPtr = outMem_;
        params.cclBufferPtr = cclMem_;
        params.dataType = HCCL_DATA_TYPE_INT32;
        params.sliceCount = sliceCount;
        params.sliceOffset = 0;
        params.dataStride = sliceCount * sizeof(int32_t);
        params.scratchStride = sliceCount * sizeof(int32_t);
        params.root = root;
        // Scatter: 仅 root 持有全部输入；其他 rank 无输入。
        params.ranksForInputData = (/*myRankIsRoot*/ true) ? ranks : std::vector<u32>{};
        return params;
    }

    TemplateDataParams MakeTmplParams(const std::vector<u32> &ranksForInputData, u64 sliceCount = 4)
    {
        TemplateDataParams params;
        params.inputBufferPtr = inputMem_;
        params.outputBufferPtr = outMem_;
        params.cclBufferPtr = cclMem_;
        params.dataType = HCCL_DATA_TYPE_INT32;
        params.sliceCount = sliceCount;
        params.sliceOffset = 0;
        params.dataStride = sliceCount * sizeof(int32_t);
        params.scratchStride = sliceCount * sizeof(int32_t);
        params.ranksForInputData = ranksForInputData;
        return params;
    }

    TemplateResource MakeTmplResource(u32 threadNum = 1)
    {
        TemplateResource res;
        for (u32 i = 0; i < threadNum; ++i) {
            res.threads.push_back(static_cast<ThreadHandle>(0x10 + i));
        }
        return res;
    }

    TemplateResource MakeTmplResourceWithChannels(const std::vector<u32> &ranks, u32 myRank, u32 threadNum = 1)
    {
        TemplateResource res;
        for (u32 i = 0; i < threadNum; ++i) {
            res.threads.push_back(static_cast<ThreadHandle>(0x10 + i));
        }
        for (u32 rank : ranks) {
            if (rank == myRank) {
                continue;
            }
            ChannelInfo channel;
            channel.remoteRank = rank;
            // remoteCclMem.addr 必须指向真实可写内存，否则 DataTransferSend 走到
            // HcommReadOnThread/HcommWriteOnThread 的 stub 时 memcpy 假地址会段错误。
            channel.remoteCclMem.addr = buf_.data() + rank * 0x1000;
            channel.remoteCclMem.size = 0x10000;
            res.channels[rank].emplace_back(channel);
        }
        return res;
    }

    std::vector<uint8_t> buf_;
    std::vector<uint8_t> outBuf_;
    std::vector<uint8_t> inputBuf_;
    void *cclMem_ = nullptr;
    void *outMem_ = nullptr;
    void *inputMem_ = nullptr;
};

} // namespace testing
} // namespace ops_hccl
