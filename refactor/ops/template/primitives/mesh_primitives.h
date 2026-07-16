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

// 构造 Mesh AllGather 的通信描述符，实际 SendRecv 由 template 执行。
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
