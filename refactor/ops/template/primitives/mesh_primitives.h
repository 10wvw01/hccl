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

// 收集 reuseCclBuffer 场景下的空位 slot：ccl buffer 按 rank 值寻址，空位 = 不在
// ranksForInputData 中的 rank。ranks 是 subComm 域 ranks，ranksForInputData 是本 template
// 要归约的 rank 列表。空位来源：
//   1. ranks 中不在 ranksForInputData 的 rank（subComm 域内空位）；
//   2. 不足时（ranksForInputData 包含 subComm 全部 rank）遍历 [0, ranks.back()+rankSize+1)
//      找不在 ranksForInputData 的 rank（其他 subComm 域的 slot）。
// 不越界：ReduceScatter 的 ccl buffer 覆盖全局 rankSize 个 slot
// （scratchMultiple_=1, scratchStride=sliceCount*dataTypeSize, cclBufferSize≥globalRankSize*scratchStride），
// 空位一定在 [0, globalRankSize) 内。
void CollectEmptySlots(const std::vector<u32> &ranks, const std::vector<u32> &ranksForInputData,
                       u32 rankSize, std::vector<u32> &emptySlots);

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
