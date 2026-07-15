/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_MESH_1D_H
#define INS_TEMP_REDUCE_SCATTER_MESH_1D_H

#include "alg_v2_template_base.h"
#include "executor_v2_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempReduceScatterMesh1D : public InsAlgTemplateBase {
public:
    InsTempReduceScatterMesh1D() = default;
    explicit InsTempReduceScatterMesh1D(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks);
    
    ~InsTempReduceScatterMesh1D() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter Mesh with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& tempAlgParams,
                         TemplateResource& templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    HcclResult PostCopy(const OpParam& param, const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads); 
    
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;
protected:
    HcclResult RunReduceScatter(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                const std::vector<ThreadHandle> &threads,
                                const TemplateDataParams &tempAlgParam);

    // ---- RunReduceScatter 中 DataSlice 偏移计算（虚函数，子类可重载实现不同搬移策略） ----
    // 返回 rxSrcSlice 的偏移量（基类用 inBuffBaseOff，OCS 用 hcclBuffBaseOff）
    virtual u64 GetRxSrcOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                               u32 myAlgRank, u32 channelIdx) const;
    // 返回 rxDstSlice 的偏移量（基类用 hcclBuffBaseOff + nextRank * outputSliceStride，OCS 用偏移公式）
    virtual u64 GetRxDstOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                               u32 nextRank, u64 outputSliceStride, u32 channelIdx) const;
    // 返回 txDstSlice 的偏移量（基类用 hcclBuffBaseOff + myAlgRank * outputSliceStride，OCS 用偏移公式）
    virtual u64 GetTxDstOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                               u32 myAlgRank, u64 outputSliceStride, u32 channelIdx) const;

    // ---- PostCopy 中 LocalReduce srcSlice 偏移计算（虚函数，子类可重载实现不同搬移策略） ----
    // 返回 srcSlice 的偏移量（基类用 hcclBuffBaseOff + tmpRank * buffSliceStride，OCS 用偏移公式）
    virtual u64 GetPostCopySrcOffset(const TemplateDataParams &tempAlgParams, u32 repeatIdx,
                                     u32 tmpRank, u64 buffSliceStride) const;
    u64 processSize_{0};
    u64 count_{0};
    std::vector<u64> elemCountOut_;
    std::vector<u64> sizeOut_;
    std::vector<u64> elemOffset_;
};

} // namespace Hccl

#endif //OPEN_HCCL_INS_TEMP_REDUCE_SCATTER_MESH_H