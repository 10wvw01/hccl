/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_nhr.h"
#include "primitives/nhr_primitives.h"

namespace ops_hccl {

HcclResult AllGatherNhrTemplate::RunAlgorithm(TemplateResource &templateResource)
{
    HCCL_INFO("[AllGatherNhrTemplate][RunAlgorithm] start, myRank[%u], rankSize[%u].",
              myRank_, templateRankSize_);

    // RunNhrAllGather 构造每个 channel × step 的 SendRecvInfo 列表（仅规划，不执行传输）。
    std::vector<SendRecvInfo> sendRecvInfos;
    CHK_RET(RunNhrAllGather(tempAlgParams_, templateResource, ranks, myRank_, sendRecvInfos));

    if (sendRecvInfos.empty()) {
        HCCL_INFO("[AllGatherNhrTemplate][RunAlgorithm] no sendRecv needed.");
        return HCCL_SUCCESS;
    }

    // NHR：sendRecvInfos 按 channelIdx × step 排列，
    // 同一 channelIdx 的所有 step 共用一个 thread（threads[channelIdx]）。
    u32 nSteps = 0;
    for (u32 tmp = templateRankSize_ - 1; tmp != 0; tmp >>= 1, nSteps++) {}

    const bool isDmaRead = IsPcieProtocol(templateResource.channels);
    for (size_t i = 0; i < sendRecvInfos.size(); ++i) {
        const u32 channelIdx = static_cast<u32>(i / nSteps);
        CHK_PRT_RET(channelIdx >= templateResource.threads.size(),
                    HCCL_ERROR("[AllGatherNhrTemplate][RunAlgorithm] channelIdx[%u] >= threads.size[%zu].",
                               channelIdx, templateResource.threads.size()),
                    HCCL_E_INTERNAL);
        CHK_RET(SendRecv(sendRecvInfos[i], templateResource.threads[channelIdx], isDmaRead));
    }

    HCCL_INFO("[AllGatherNhrTemplate][RunAlgorithm] end.");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
