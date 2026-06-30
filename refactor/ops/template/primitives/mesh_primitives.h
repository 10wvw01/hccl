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

// Describes the data slicing semantics selected by the upper algorithm description.
// This is intentionally finer-grained than Mesh/NHR or EngineType: different Mesh
// AllGather variants use different offset formulas even when they run on the same engine.
enum class MeshAllGatherSliceMode {
    NORMAL_FIXED,
    VARIABLE_COUNT,
    OMNIPIPE_STEP,
    COMMON_CHANNEL_SPLIT,
    Z_AXIS_DETOUR,
    MESH_CHUNK,
};

// Z-axis detour splits one peer's data across level0 and level1 channels with
// a fixed boundary/ratio. The boundary is known while planning channel resources;
// it cannot be recovered reliably after level0/level1 channels are merged.
struct ZAxisDetourConfig {
    u32 level0ChannelNumPerRank{0};
    u32 level1ChannelNumPerRank{0};
    double level0DataRatio{0.0};
};

// Primitive options carry only the slicing information that cannot be derived
// from TemplateDataParams or TemplateResource. In the final direct-call flow,
// sliceMode should come from TemplateDesc.variant and zAxis should come from
// resource-planning metadata.
struct MeshAllGatherPrimitiveOptions {
    MeshAllGatherSliceMode sliceMode{MeshAllGatherSliceMode::NORMAL_FIXED};
    bool hasZAxisDetourConfig{false};
    ZAxisDetourConfig zAxis;
};

// SendRecv mode is part of the executable plan so RunMeshAllGather does not
// need to inspect the variant-specific rules while executing each task.
enum class MeshAllGatherSendRecvMode {
    DMA_READ,
    BATCH_WRITE,
};

// One executable communication task for a peer/channel pair. The planner emits
// concrete DataSlice lists so the execution path does not need to know whether
// offsets came from fixed slices, variable-count displacements, OmniPipe steps,
// or Z-axis channel splitting.
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

// Complete plan consumed by RunMeshAllGather. This is the boundary between
// variant-specific data layout logic and the common SendRecv execution loop.
struct MeshAllGatherSlicePlan {
    MeshAllGatherSendRecvMode sendRecvMode{MeshAllGatherSendRecvMode::DMA_READ};
    std::vector<MeshAllGatherPeerChannelPlan> tasks;
};

// Build a slice plan without launching communication. This function is the
// intended extraction point for old template-specific offset/channel logic.
HcclResult BuildMeshAllGatherSlicePlan(const TemplateDataParams &tempAlgParams,
                                       const TemplateResource &templateResource,
                                       const std::vector<u32> &ranks, u32 myRank,
                                       const MeshAllGatherPrimitiveOptions &options,
                                       MeshAllGatherSlicePlan &plan);

// New explicit entry point. Callers that know the algorithm variant should use
// this overload so the primitive never guesses slicing semantics from EngineType
// or channel count.
HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            EngineType engineType, const std::vector<u32> &ranks, u32 myRank,
                            const MeshAllGatherPrimitiveOptions &options);

// Compatibility overload for the current refactor call sites. It preserves the
// existing extracted behavior until TemplateDesc.variant and metadata plumbing
// are wired through the executor.
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
