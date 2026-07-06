/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_MESH_1D_TWO_SHOT_Z_AXIS_DETOUR_H
#define REDUCE_MESH_1D_TWO_SHOT_Z_AXIS_DETOUR_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"
#include "reduce_mesh_1D_two_shot.h"

namespace ops_hccl {

class ReduceMesh1DTwoShotZAxisDetour : public ReduceMesh1DTwoShot {
public:
    ReduceMesh1DTwoShotZAxisDetour() = default;
    explicit ReduceMesh1DTwoShotZAxisDetour(const OpParam &param, const u32 rankId,
                                            const std::vector<std::vector<u32>> &subCommRanks);

    ~ReduceMesh1DTwoShotZAxisDetour() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce Mesh 1D Two Shot Z axis detour with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    u64 GetThreadNum() const override;
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;
    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;
    HcclResult CalcDataSplitByPortGroup(const u64 totalDataCount, const u64 dataTypeSize,
                                        const std::vector<ChannelInfo> &channels,
                                        std::vector<u64> &elemCountOut, std::vector<u64> &sizeOut,
                                        std::vector<u64> &elemOffset) override;
    HcclResult SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>> &channels) override;

private:
    HcclResult CalcSlice();
    HcclResult RunReduceScatter(const TemplateDataParams &tempAlgParam, const OpParam &param,
                                     const std::map<u32, std::vector<ChannelInfo>> &channels,
                                     const std::vector<ThreadHandle> &threads);
    HcclResult RunGatherToRoot(const TemplateDataParams &tempAlgParam,
                                    const std::map<u32, std::vector<ChannelInfo>> &channels,
                                    const std::vector<ThreadHandle> &threads);
    HcclResult SendRecvDataToPeers(const TemplateDataParams &tempAlgParam,
                                        const std::map<u32, std::vector<ChannelInfo>> &channels,
                                        const std::vector<ThreadHandle> &threads);
    HcclResult DoLocalReduce(const TemplateDataParams &tempAlgParam, const OpParam &param,
                                  const std::vector<ThreadHandle> &threads);
    HcclResult GatherLocalData(const TemplateDataParams &tempAlgParam,
                                    const std::vector<ThreadHandle> &threads) const;
    HcclResult GatherRemoteData(const TemplateDataParams &tempAlgParam,
                                     const std::map<u32, std::vector<ChannelInfo>> &channels,
                                     const std::vector<ThreadHandle> &threads);
    HcclResult SendToRoot(const TemplateDataParams &tempAlgParam,
                               const std::map<u32, std::vector<ChannelInfo>> &channels,
                               const std::vector<ThreadHandle> &threads);

    u32 level0ChannelNumPerRank_{1};
    u32 level1ChannelNumPerRank_{0};
    float level0DataRatio_{1.0f};
    u64 processSize_{0};
    u64 count_{0};
    u32 myIdx_{UINT32_MAX};
    std::vector<u32> notifyIdxMainToSub_;
    std::vector<u32> notifyIdxSubToMain_;
    std::vector<SplitSliceInfo> sliceInfoList_;
    std::vector<u32> rankList_;
    std::vector<u64> elemCountOut_;
    std::vector<u64> sizeOut_;
    std::vector<u64> elemOffset_;
};

}  // namespace ops_hccl

#endif  // REDUCE_MESH_1D_TWO_SHOT_Z_AXIS_DETOUR_H
