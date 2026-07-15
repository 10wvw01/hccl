/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_MESH_1D_OCS_H
#define INS_TEMP_REDUCE_SCATTER_MESH_1D_OCS_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"
#include "ins_temp_reduce_scatter_mesh_1D.h"

namespace ops_hccl {

// 4层OCS拓扑最高层(net_layer_3, SuperNode级)的 ReduceScatter Mesh1D 模板。
// 与父类 InsTempReduceScatterMesh1D 的区别: CalcRes 时仅在指定的 net_layer(默认 layer3)
// 上建立 channel，避免遍历所有层误取低层(layer0/1/2)链路。
class InsTempReduceScatterMesh1DOcs : public InsTempReduceScatterMesh1D {
public:
    InsTempReduceScatterMesh1DOcs() = default;
    explicit InsTempReduceScatterMesh1DOcs(const OpParam& param, const u32 rankId,
                                           const std::vector<std::vector<u32>> &subCommRanks);

    ~InsTempReduceScatterMesh1DOcs() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter Mesh 1D OCS(layer3) with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
    // OCS 层到每个远端可能有多条链路(channelsPerRank_ > 1)，RunReduceScatter 会按
    // (templateRankSize_-1) * channelsPerRank_ 个从线程分发收发任务，故线程数需与之一致，
    // 与 InsTempReduceScatterMesh1DZAxisDetour::GetThreadNum 同口径。
    u64 GetThreadNum() const override;
    void SetHcclBufMaxBlockNum(u32 blockNum);

protected:
    // 重载搬移地址计算：按 myRank_ 取偏移，当前 rank 数据搬移到 (myRank_+1)%rankSize 的 block 位置
    u64 GetRxSrcOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                       u32 myAlgRank, u32 channelIdx) const override;
    u64 GetRxDstOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                       u32 nextRank, u64 outputSliceStride, u32 channelIdx) const override;
    u64 GetTxDstOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                       u32 myAlgRank, u64 outputSliceStride, u32 channelIdx) const override;
    u64 GetPostCopySrcOffset(const TemplateDataParams &tempAlgParams, u32 repeatIdx,
                             u32 tmpRank, u64 buffSliceStride) const override;

private:
    // OCS 层物理 net_layer 编号，4层拓扑下为 3
    u32 ocsNetLayer_{3};
    u32 hcclBufMaxBlockNum_= 0;
};

} // namespace ops_hccl

#endif // INS_TEMP_REDUCE_SCATTER_MESH_1D_OCS_H
