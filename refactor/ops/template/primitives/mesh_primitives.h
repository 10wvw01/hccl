/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MESH_PRIMITIVES_H
#define MESH_PRIMITIVES_H

#include <vector>
#include "hccl_algorithm.h"

namespace ops_hccl {

struct TemplateDataParams;
struct TemplateResource;

struct DataSlice {
    void *addr_ = nullptr;
    u64 offset_{0};
    u64 size_{0};
    u64 count_{0};

    DataSlice(void *addr, u64 offset, u64 size, u64 count)
        : addr_(addr), offset_(offset), size_(size), count_(count)
    {
    }
};

struct SlicesList {
    std::vector<DataSlice> srcSlices_;
    std::vector<DataSlice> dstSlices_;

    SlicesList(const std::vector<DataSlice> &srcSlices, const std::vector<DataSlice> &dstSlices)
        : srcSlices_(srcSlices), dstSlices_(dstSlices)
    {
    }
};

struct TxRxChannels {
    ChannelInfo txChannel_;
    ChannelInfo rxChannel_;

    TxRxChannels(const ChannelInfo &txLink, const ChannelInfo &rxLink) : txChannel_(txLink), rxChannel_(rxLink)
    {
    }
};

struct TxRxSlicesList {
    SlicesList txSlicesList_;
    SlicesList rxSlicesList_;

    TxRxSlicesList(const SlicesList &txSlicesList, const SlicesList &rxSlicesList)
        : txSlicesList_(txSlicesList), rxSlicesList_(rxSlicesList)
    {
    }
};

struct SendRecvInfo {
    TxRxChannels sendRecvChannels_;
    TxRxSlicesList sendRecvSlices_;
    HcclDataType dataType_;

    SendRecvInfo(const TxRxChannels &sendRecvLinks, const TxRxSlicesList &sendRecvSlices, HcclDataType dataType)
        : sendRecvChannels_(sendRecvLinks), sendRecvSlices_(sendRecvSlices), dataType_(dataType)
    {
    }
};

// 构造 Mesh AllGather 的通信描述符，实际 SendRecv 由 template 执行。
HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            const std::vector<u32> &ranks, u32 myRank, std::vector<u32> &ranksForOutputData,
                            std::vector<SendRecvInfo> &sendRecvInfos);

} // namespace ops_hccl

#endif
