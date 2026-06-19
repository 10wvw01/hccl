/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_MESH_H
#define ALL_GATHER_MESH_H

#include "alg_template_base.h"
#include "alg_data_trans_wrapper.h"
#include "alg_template_base_experimental.h"

namespace ops_hccl_experimental {
using ops_hccl::AlgTemplateBase;
using ops_hccl::Slice;
using ops_hccl::ChannelInfo;
using ops_hccl::NOTIFY_IDX_ACK;
using ops_hccl::CUSTOM_TIMEOUT;
using ops_hccl::AlgTemplateRegistry;
using ops_hccl::HCCL_MIN_SLICE_ALIGN_910B;
using ops_hccl::PostSyncInterThreads;
using ops_hccl::NOTIFY_IDX_DATA_SIGNAL;
using ops_hccl::RoundUpWithDivisor;
using ops_hccl::PreSyncInterThreads;
using ops_hccl::TemplateType;
using ops_hccl::DefaultTemplateCreator;

class AllGatherMesh : public AlgTemplateBaseExperimental {
public:
    explicit AllGatherMesh();

    ~AllGatherMesh() override;  

    // should be called soon after template AllGatherMesh instance created
    HcclResult Prepare(u32 interRank, u32 interRankSize) override;

    HcclResult Prepare(HcclMem &inputMem, HcclMem &outputMem, HcclMem &scratchMem,
                        const u64 count,
                        const HcclDataType dataType, ThreadHandle thread, const std::vector<ThreadHandle> &slaveThreads,
                        const HcclReduceOp reductionOp,
                        const u32 root, const std::vector<Slice> &slices, const u64 baseOffset,
                        const bool disableDMAReduce) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels) override;

protected:
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub, const u32 rankSize);
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain, const u32 rankSize);
    
    virtual HcclResult Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels);
    HcclResult PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const;

    u32 interRank_;       // comm内的rank排序
    u32 interRankSize_;   // 本comm内ranksize总数
    
    u64 localStrideSize;

    u32 unitSize;
    u64 reduceAttr_ = 0;
    ThreadHandle mainThread;
    std::vector<ThreadHandle> subThreads;
    std::vector<u32> notifyIdxMainToSub_;
    std::vector<u32> notifyIdxSubToMain_;

private:
};
}

#endif /* * REDUCE_SCATTER_BIRS_H */
