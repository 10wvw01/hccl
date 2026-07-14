/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Mesh primitives UT helpers.
 */

#pragma once

#include <map>
#include <vector>
#include <gtest/gtest.h>

#include "hccl_algorithm.h"
#include "template/base_template.h"
#include "template/primitives/mesh_primitives.h"

namespace ops_hccl {
namespace testing {

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
        params.stride = sliceCount * sizeof(int32_t);
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

} // namespace testing
} // namespace ops_hccl
