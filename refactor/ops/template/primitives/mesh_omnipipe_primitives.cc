/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mesh_omnipipe_primitives.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

namespace {

struct MeshOmniPipeTransferSlices {
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
};

HcclResult GetMyAlgRank(const std::vector<u32> &ranks, u32 myRank, u32 &myAlgRank)
{
    for (u32 i = 0; i < ranks.size(); ++i) {
        if (ranks[i] == myRank) {
            myAlgRank = i;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("[MeshOmniPipe] rank[%u] is not in ranks.", myRank);
    return HCCL_E_PARA;
}

HcclResult CheckStepRankSize(const StepSliceInfo &stepSliceInfo, u32 rankIdx)
{
    CHK_PRT_RET(stepSliceInfo.stepCount.size() <= rankIdx ||
                    stepSliceInfo.stepSliceSize.size() <= rankIdx ||
                    stepSliceInfo.stepInputSliceStride.size() <= rankIdx ||
                    stepSliceInfo.stepOutputSliceStride.size() <= rankIdx ||
                    stepSliceInfo.inputOmniPipeSliceStride.size() <= rankIdx ||
                    stepSliceInfo.outputOmniPipeSliceStride.size() <= rankIdx,
                HCCL_ERROR("[MeshOmniPipe] invalid rank slice info, rankIdx[%u].", rankIdx),
                HCCL_E_PARA);

    const u32 stepNum = static_cast<u32>(stepSliceInfo.inputOmniPipeSliceStride[rankIdx].size());
    CHK_PRT_RET(stepSliceInfo.stepCount[rankIdx].size() < stepNum ||
                    stepSliceInfo.stepSliceSize[rankIdx].size() < stepNum ||
                    stepSliceInfo.outputOmniPipeSliceStride[rankIdx].size() < stepNum,
                HCCL_ERROR("[MeshOmniPipe] invalid step slice info, rankIdx[%u].", rankIdx),
                HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult CheckStepPairSize(const StepSliceInfo &stepSliceInfo, u32 myAlgRank, u32 connectedAlgRank)
{
    CHK_RET(CheckStepRankSize(stepSliceInfo, myAlgRank));
    CHK_RET(CheckStepRankSize(stepSliceInfo, connectedAlgRank));
    const u32 stepNum = static_cast<u32>(stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size());
    CHK_PRT_RET(stepSliceInfo.stepCount[connectedAlgRank].size() < stepNum ||
                    stepSliceInfo.stepSliceSize[connectedAlgRank].size() < stepNum ||
                    stepSliceInfo.inputOmniPipeSliceStride[connectedAlgRank].size() < stepNum ||
                    stepSliceInfo.outputOmniPipeSliceStride[connectedAlgRank].size() < stepNum,
                HCCL_ERROR("[MeshOmniPipe] peer step size mismatch."), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult BuildMeshOmniPipeAllGatherSlices(const TemplateDataParams &tempAlgParams,
                                            const ChannelInfo &linkRemote, u32 myAlgRank,
                                            u32 connectedAlgRank, MeshOmniPipeTransferSlices &slices)
{
    const StepSliceInfo &step = tempAlgParams.stepSliceInfo;
    const u32 stepNum = static_cast<u32>(step.inputOmniPipeSliceStride[myAlgRank].size());

    for (u32 s = 0; s < stepNum; ++s) {
        const u64 txOff = tempAlgParams.buffInfo.inBuffBaseOff +
            step.inputOmniPipeSliceStride[myAlgRank][s] + step.stepInputSliceStride[myAlgRank];
        const u64 rxOff = tempAlgParams.buffInfo.outBuffBaseOff +
            step.outputOmniPipeSliceStride[connectedAlgRank][s] +
            step.stepOutputSliceStride[connectedAlgRank];
        slices.txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, txOff,
                                        step.stepSliceSize[myAlgRank][s], step.stepCount[myAlgRank][s]);
        slices.txDstSlices.emplace_back(linkRemote.remoteCclMem.addr, txOff,
                                        step.stepSliceSize[myAlgRank][s], step.stepCount[myAlgRank][s]);
        // 旧 OmniPipe 模板中 rx count 使用 stepSliceSize，这里保持兼容。
        slices.rxSrcSlices.emplace_back(linkRemote.remoteCclMem.addr, rxOff,
                                        step.stepSliceSize[connectedAlgRank][s],
                                        step.stepSliceSize[connectedAlgRank][s]);
        slices.rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, rxOff,
                                        step.stepSliceSize[connectedAlgRank][s],
                                        step.stepSliceSize[connectedAlgRank][s]);
    }
    return HCCL_SUCCESS;
}

HcclResult BuildMeshOmniPipeReduceScatterSlices(const TemplateDataParams &tempAlgParams,
                                                const ChannelInfo &linkRemote, u32 myAlgRank,
                                                u32 connectedAlgRank, MeshOmniPipeTransferSlices &slices)
{
    const StepSliceInfo &step = tempAlgParams.stepSliceInfo;
    const u32 stepNum = static_cast<u32>(step.inputOmniPipeSliceStride[myAlgRank].size());

    for (u32 s = 0; s < stepNum; ++s) {
        const u64 txSrcOff = tempAlgParams.buffInfo.inBuffBaseOff +
            step.stepInputSliceStride[connectedAlgRank] +
            step.inputOmniPipeSliceStride[connectedAlgRank][s];
        const u64 txDstOff = tempAlgParams.buffInfo.hcclBuffBaseOff +
            step.stepOutputSliceStride[myAlgRank] +
            step.outputOmniPipeSliceStride[myAlgRank][s];
        const u64 rxSrcOff = tempAlgParams.buffInfo.inBuffBaseOff +
            step.stepInputSliceStride[myAlgRank] +
            step.inputOmniPipeSliceStride[myAlgRank][s];
        const u64 rxDstOff = tempAlgParams.buffInfo.hcclBuffBaseOff +
            step.stepOutputSliceStride[connectedAlgRank] +
            step.outputOmniPipeSliceStride[connectedAlgRank][s];
        slices.txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, txSrcOff,
                                        step.stepSliceSize[connectedAlgRank][s],
                                        step.stepCount[connectedAlgRank][s]);
        slices.txDstSlices.emplace_back(linkRemote.remoteCclMem.addr, txDstOff,
                                        step.stepSliceSize[connectedAlgRank][s],
                                        step.stepCount[connectedAlgRank][s]);
        slices.rxSrcSlices.emplace_back(linkRemote.remoteCclMem.addr, rxSrcOff,
                                        step.stepSliceSize[myAlgRank][s],
                                        step.stepCount[myAlgRank][s]);
        slices.rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, rxDstOff,
                                        step.stepSliceSize[myAlgRank][s],
                                        step.stepCount[myAlgRank][s]);
    }
    return HCCL_SUCCESS;
}

}  // namespace

HcclResult RunMeshOmniPipeAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                                    EngineType engineType, const std::vector<u32> &ranks, u32 myRank)
{
    (void)engineType;
    const u32 rankSize = static_cast<u32>(ranks.size());
    if (rankSize <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }

    u32 myAlgRank = 0;
    CHK_RET(GetMyAlgRank(ranks, myRank, myAlgRank));
    const HcclDataType dataType = tempAlgParams.dataType;

    u32 threadIdx = 0;
    for (u32 i = 1; i < rankSize; ++i) {
        const u32 connectedAlgRank = (myAlgRank + i) % rankSize;
        const u32 connectedRank = ranks[connectedAlgRank];
        CHK_PRT_RET(templateResource.channels.count(connectedRank) == 0 ||
                        templateResource.channels.at(connectedRank).empty() ||
                        threadIdx >= templateResource.threads.size(),
                    HCCL_ERROR("[RunMeshOmniPipeAllGather] invalid task, connectedRank[%u], threadIdx[%u].",
                               connectedRank, threadIdx),
                    HCCL_E_PARA);
        CHK_RET(CheckStepPairSize(tempAlgParams.stepSliceInfo, myAlgRank, connectedAlgRank));

        const ChannelInfo &linkRemote = templateResource.channels.at(connectedRank)[0];
        MeshOmniPipeTransferSlices slices;
        CHK_RET(BuildMeshOmniPipeAllGatherSlices(tempAlgParams, linkRemote, myAlgRank,
                                                 connectedAlgRank, slices));
        SendRecvInfo sendRecvInfo{{linkRemote, linkRemote},
                                  {{slices.txSrcSlices, slices.txDstSlices},
                                   {slices.rxSrcSlices, slices.rxDstSlices}},
                                  dataType};
        CHK_RET(SendRecvWrite(sendRecvInfo, templateResource.threads[threadIdx]));
        ++threadIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshOmniPipeReduceScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                                        EngineType engineType, const std::vector<u32> &ranks, u32 myRank)
{
    (void)engineType;
    const u32 rankSize = static_cast<u32>(ranks.size());
    if (rankSize <= 1 || templateResource.channels.empty()) {
        return HCCL_SUCCESS;
    }

    u32 myAlgRank = 0;
    CHK_RET(GetMyAlgRank(ranks, myRank, myAlgRank));
    const HcclDataType dataType = tempAlgParams.dataType;

    u32 threadIdx = 0;
    for (u32 i = 1; i < rankSize; ++i) {
        const u32 connectedAlgRank = (myAlgRank + i) % rankSize;
        const u32 connectedRank = ranks[connectedAlgRank];
        CHK_PRT_RET(templateResource.channels.count(connectedRank) == 0 ||
                        templateResource.channels.at(connectedRank).empty() ||
                        threadIdx >= templateResource.threads.size(),
                    HCCL_ERROR("[RunMeshOmniPipeReduceScatter] invalid task, connectedRank[%u], threadIdx[%u].",
                               connectedRank, threadIdx),
                    HCCL_E_PARA);
        CHK_RET(CheckStepPairSize(tempAlgParams.stepSliceInfo, myAlgRank, connectedAlgRank));

        const ChannelInfo &linkRemote = templateResource.channels.at(connectedRank)[0];
        MeshOmniPipeTransferSlices slices;
        CHK_RET(BuildMeshOmniPipeReduceScatterSlices(tempAlgParams, linkRemote, myAlgRank,
                                                     connectedAlgRank, slices));
        SendRecvInfo sendRecvInfo{{linkRemote, linkRemote},
                                  {{slices.txSrcSlices, slices.txDstSlices},
                                   {slices.rxSrcSlices, slices.rxDstSlices}},
                                  dataType};
        CHK_RET(SendRecvWrite(sendRecvInfo, templateResource.threads[threadIdx]));
        ++threadIdx;
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
