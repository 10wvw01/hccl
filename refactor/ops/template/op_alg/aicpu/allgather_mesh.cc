/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_mesh.h"

namespace ops_hccl {

HcclResult AllGatherMeshTemplate::RunAlgorithm(TemplateResource &templateResource)
{
    HCCL_INFO("[AllGatherMeshTemplate][RunAlgorithm] start, myRank[%u], rankSize[%u].",
              myRank_, templateRankSize_);

    // RunMeshAllGather 构造每对 rank 的 SendRecvInfo 列表（仅规划，不执行传输）。
    std::vector<SendRecvInfo> sendRecvInfos;
    std::vector<u32> ranksForOutputData;
    CHK_RET(RunMeshAllGather(tempAlgParams_, templateResource, ranks, myRank_,
                             ranksForOutputData, sendRecvInfos));

    if (sendRecvInfos.empty()) {
        HCCL_INFO("[AllGatherMeshTemplate][RunAlgorithm] no sendRecv needed.");
        return HCCL_SUCCESS;
    }

    // Mesh 1D：每个对端 rank 对应一个 thread（threadIdx 与 sendRecvInfos 索引对应）。
    const bool isDmaRead = IsPcieProtocol(templateResource.channels);
    for (size_t i = 0; i < sendRecvInfos.size(); ++i) {
        const u32 threadIdx = static_cast<u32>(i);
        CHK_PRT_RET(threadIdx >= templateResource.threads.size(),
                    HCCL_ERROR("[AllGatherMeshTemplate][RunAlgorithm] threadIdx[%u] >= threads.size[%zu].",
                               threadIdx, templateResource.threads.size()),
                    HCCL_E_INTERNAL);
        CHK_RET(SendRecv(sendRecvInfos[i], templateResource.threads[threadIdx], isDmaRead));
    }

    HCCL_INFO("[AllGatherMeshTemplate][RunAlgorithm] end.");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
