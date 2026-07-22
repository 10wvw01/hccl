/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reducescatter_nhr.h"
#include "primitives/nhr_primitives.h"

namespace ops_hccl {

HcclResult ReduceScatterNhrTemplate::RunAlgorithm(TemplateResource &templateResource,
                                                std::vector<TxRxSlicesList> &txRxSlicesLists,
                                                std::vector<u32> &ranksForOutputData)
{
    HCCL_INFO("[ReduceScatterNhrTemplate][RunAlgorithm] start, myRank[%u], rankSize[%u].",
              myRank_, templateRankSize_);

    std::vector<SendRecvInfo> sendRecvInfos;
    CHK_RET(RunNhrReduceScatter(tempAlgParams_, templateResource, ranks_, myRank_, ranksForOutputData,
                                sendRecvInfos));

    // RunNhrReduceScatter 返回 std::vector<SendRecvInfo>，基类 KernelRun 期望 txRxSlicesLists，
    // 从每个 SendRecvInfo 中提取 sendRecvSlices_（TxRxSlicesList）填入 txRxSlicesLists。
    for (auto &sendRecvInfo : sendRecvInfos) {
        txRxSlicesLists.push_back(sendRecvInfo.sendRecvSlices_);
    }

    HCCL_INFO("[ReduceScatterNhrTemplate][RunAlgorithm] end.");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
