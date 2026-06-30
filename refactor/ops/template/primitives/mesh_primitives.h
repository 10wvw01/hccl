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

#include "template_utils.h"
#include "hccl_algorithm.h"

namespace ops_hccl {

enum class MeshAllGatherSliceMode {
    NORMAL_FIXED,
    VARIABLE_COUNT,
    OMNIPIPE_STEP,
    COMMON_CHANNEL_SPLIT,
    Z_AXIS_DETOUR,
    MESH_CHUNK,
};

struct ZAxisDetourConfig {
    u32 level0ChannelNumPerRank{0};
    u32 level1ChannelNumPerRank{0};
    double level0DataRatio{0.0};
};

struct MeshAllGatherPrimitiveOptions {
    MeshAllGatherSliceMode sliceMode{MeshAllGatherSliceMode::NORMAL_FIXED};
    bool hasZAxisDetourConfig{false};
    ZAxisDetourConfig zAxis;
};

enum class MeshAllGatherSendRecvMode {
    DMA_READ,
    BATCH_WRITE,
};

struct MeshAllGatherPeerChannelPlan {
    u32 peerRank{0};
    u32 peerAlgRank{0};
    u32 channelIdx{0};
    u32 threadIdx{0};
    const ChannelInfo *link{nullptr};
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
};

struct MeshAllGatherSlicePlan {
    MeshAllGatherSendRecvMode sendRecvMode{MeshAllGatherSendRecvMode::DMA_READ};
    std::vector<MeshAllGatherPeerChannelPlan> tasks;
};

HcclResult BuildMeshAllGatherSlicePlan(const TemplateDataParams &tempAlgParams,
                                       const TemplateResource &templateResource,
                                       const std::vector<u32> &ranks, u32 myRank,
                                       const MeshAllGatherPrimitiveOptions &options,
                                       MeshAllGatherSlicePlan &plan);

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            EngineType engineType, const std::vector<u32> &ranks, u32 myRank,
                            const MeshAllGatherPrimitiveOptions &options);

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            EngineType engineType, const std::vector<u32> &ranks, u32 myRank);

HcclResult RunMeshReduceScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                                EngineType engineType, const std::vector<u32> &ranks, u32 myRank);

HcclResult RunMeshScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                          EngineType engineType, const std::vector<u32> &ranks, u32 myRank);

HcclResult RunMeshGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                         EngineType engineType, const std::vector<u32> &ranks, u32 myRank);

HcclResult RunMeshAllToAll(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                           EngineType engineType, const std::vector<u32> &ranks, u32 myRank);

HcclResult RunMeshBarrier(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                          EngineType engineType, const std::vector<u32> &ranks, u32 myRank);

}

#endif
