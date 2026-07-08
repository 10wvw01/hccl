/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_MINI_REDUCE_MESH_1D_H
#define INS_TEMP_MINI_REDUCE_MESH_1D_H

#include "alg_v2_template_base.h"
#include "executor_base.h"

namespace ops_hccl {

// MiniReduce Mesh 1D AICPU Template
// 算法: 所有非 root rank 将数据发送给 root rank, root rank 做本地求和
// 仅支持单 Server Mesh 拓扑 (level0 only)
class InsTempMiniReduceMesh1D : public InsAlgTemplateBase {
public:
    InsTempMiniReduceMesh1D() = default;
    explicit InsTempMiniReduceMesh1D(const OpParam &param, const u32 rankId,
                                     const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempMiniReduceMesh1D() override = default;

    std::string Describe() const override
    {
        std::string info = "Template of MiniReduce Mesh with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    // Root rank: 从所有非 root rank 接收数据并累加
    HcclResult RunRootRecv(const OpParam &param,
                           const TemplateDataParams &tempAlgParams,
                           const std::vector<ThreadHandle> &threads,
                           const std::map<u32, std::vector<ChannelInfo>> &channels);
    // Non-root rank: 发送本地数据到 root
    HcclResult RunNonRootSend(const OpParam &param,
                              const TemplateDataParams &tempAlgParams,
                              const std::vector<ThreadHandle> &threads,
                              const std::map<u32, std::vector<ChannelInfo>> &channels);
};

}  // namespace ops_hccl

#endif  // INS_TEMP_MINI_REDUCE_MESH_1D_H
