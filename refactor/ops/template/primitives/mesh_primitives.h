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

#include <cstddef>
#include <vector>
#include "hccl_algorithm.h"
#include "alg_param.h"

namespace ops_hccl {

struct TemplateDataParams;

struct MeshSliceInfo {
    const TemplateDataParams &tempAlgParams;
    u64 sliceSize;
    u64 tailSize;
    u64 stride;
    u32 tailRankId;
};

struct MeshSlicePair {
    void *firstBufferPtr;
    void *secondBufferPtr;
    std::vector<DataSlice> &firstSlices;
    std::vector<DataSlice> &secondSlices;
};

struct MeshRsLayoutInfo {
    bool reuseCclBuffer{false};
    u32 rankSize{0};
    u32 myAlgRank{0};
    std::vector<u32> emptySlots;
};

void CollectEmptySlots(const std::vector<u32> &ranks, const std::vector<u32> &ranksForInputData,
                       u32 rankSize, std::vector<u32> &emptySlots);

HcclResult InitMeshRsLayoutInfo(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks,
                                u32 myRank, MeshRsLayoutInfo &layoutInfo);

u64 GetMeshRsInputOffset(const TemplateDataParams &tempAlgParams, size_t idx);

u64 GetMeshRsFinalCclOffset(const TemplateDataParams &tempAlgParams, const MeshRsLayoutInfo &layoutInfo,
                            size_t idx, u32 rank);

u64 GetMeshRsTempCclOffset(const TemplateDataParams &tempAlgParams, const MeshRsLayoutInfo &layoutInfo,
                           size_t idx, u32 algRank);

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks,
                            u32 myRank, std::vector<u32> &ranksForOutputData,
                            std::vector<TxRxSlicesList> &txRxSlicesLists);

HcclResult RunMeshScatter(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks,
                          u32 myRank, std::vector<u32> &ranksForOutputData,
                          std::vector<TxRxSlicesList> &txRxSlicesLists);

HcclResult RunMeshReduceScatter(const TemplateDataParams &tempAlgParams, const std::vector<u32> &ranks,
                                u32 myRank, std::vector<u32> &ranksForOutputData,
                                std::vector<TxRxSlicesList> &txRxSlicesLists);

} // namespace ops_hccl

#endif
