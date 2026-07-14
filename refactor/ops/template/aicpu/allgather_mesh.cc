/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_mesh.h"

#include "primitives/mesh_primitives.h"

namespace ops_hccl {

HcclResult AllGatherMeshTemplate::RunAlgorithm(TemplateResource &templateResource,
                                                 std::vector<SendRecvInfo> &sendRecvInfos,
                                                 std::vector<u32> &ranksForOutputData)
{
    HCCL_INFO("[AllGatherMeshTemplate][RunAlgorithm] start, myRank[%u], rankSize[%u].",
              myRank_, templateRankSize_);

    std::vector<TxRxSlicesList> txRxSlicesLists;
    CHK_RET(RunMeshAllGather(tempAlgParams_, ranks_, myRank_, ranksForOutputData, txRxSlicesLists));
    CHK_RET(BuildSendRecvInfos(templateResource, txRxSlicesLists, sendRecvInfos));

    HCCL_INFO("[AllGatherMeshTemplate][RunAlgorithm] end.");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
