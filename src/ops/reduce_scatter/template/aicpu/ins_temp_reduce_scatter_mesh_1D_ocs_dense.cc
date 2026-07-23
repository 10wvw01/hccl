/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D_ocs_dense.h"
#include "channel.h"

namespace ops_hccl {

InsTempReduceScatterMesh1DOcsDense::InsTempReduceScatterMesh1DOcsDense(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempReduceScatterMesh1DOcsSparse(param, rankId, subCommRanks)
{
}

InsTempReduceScatterMesh1DOcsDense::~InsTempReduceScatterMesh1DOcsDense()
{
}

u64 InsTempReduceScatterMesh1DOcsDense::GetRxSrcOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                                                  u32 myAlgRank, u32 channelIdx) const
{
    return tempAlgParam.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParam.inputRepeatStride +
           myAlgRank * tempAlgParam.inputSliceStride + elemOffset_[channelIdx];
}

u64 InsTempReduceScatterMesh1DOcsDense::GetRxDstOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                                                  u32 nextRank, u64 outputSliceStride, u32 channelIdx) const
{
    return templateRankSize_ * tempAlgParam.repeatNum * tempAlgParam.sliceSize + repeatIdx * tempAlgParam.outputRepeatStride +
           nextRank * outputSliceStride + elemOffset_[channelIdx];
}

// 在数据dense排布下临时数据搬运到input后位置放置
u64 InsTempReduceScatterMesh1DOcsDense::GetTxDstOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                                                  u32 myAlgRank, u64 outputSliceStride, u32 channelIdx) const
{
    return templateRankSize_ * tempAlgParam.repeatNum * tempAlgParam.sliceSize + repeatIdx * tempAlgParam.outputRepeatStride +
           myAlgRank * outputSliceStride + elemOffset_[channelIdx];
}

// ---- PostCopy 偏移计算（OCS 重载：else 分支，使用 (rank+1)%maxBlockNum 偏移公式） ----
u64 InsTempReduceScatterMesh1DOcsDense::GetPostCopySrcOffset(const TemplateDataParams &tempAlgParams, u32 repeatIdx,
                                                        u32 tmpRank, u64 buffSliceStride) const
{
    return templateRankSize_ * tempAlgParams.repeatNum * tempAlgParams.sliceSize + 
        repeatIdx * tempAlgParams.outputRepeatStride + tmpRank * buffSliceStride;
}

} // namespace ops_hccl
