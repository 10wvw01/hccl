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

#include "hccl_algorithm.h"

namespace ops_hccl {

// Explicit Mesh transfer slice semantics. Do not infer this from EngineType or channel count.
enum class MeshTransferSliceMode {
    NORMAL_FIXED,
    VARIABLE_COUNT,
    OMNIPIPE_STEP,
    COMMON_CHANNEL_SPLIT,
    Z_AXIS_DETOUR,
    MESH_CHUNK,
};

// Z-axis detour metadata comes from resource planning.
struct ZAxisDetourConfig {
    u32 level0ChannelNumPerRank{0};
    u32 level1ChannelNumPerRank{0};
    double level0DataRatio{0.0};
};

// Only expose caller-provided variant options; plan/task types stay private in mesh_primitives.cc.
struct MeshPrimitiveOptions {
    MeshTransferSliceMode sliceMode{MeshTransferSliceMode::NORMAL_FIXED};
    bool hasZAxisDetourConfig{false};
    ZAxisDetourConfig zAxis;
};

// Full AICPU Mesh AllGather communication primitive entry.
// Caller-provided options cover current AICPU Mesh AllGather variants except MeshChunk.
HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            EngineType engineType, const std::vector<u32> &ranks, u32 myRank,
                            const MeshPrimitiveOptions &options);

// Temporary compatibility overload. It preserves the current extracted behavior only.
// New variant-aware callers should use the options overload above.
HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                            EngineType engineType, const std::vector<u32> &ranks, u32 myRank);

HcclResult RunMeshReduceScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                                EngineType engineType, const std::vector<u32> &ranks, u32 myRank,
                                const MeshPrimitiveOptions &options);

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

} // namespace ops_hccl

#endif
