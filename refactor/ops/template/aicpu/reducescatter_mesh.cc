/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reducescatter_mesh.h"

#include "primitives/mesh_primitives.h"

namespace ops_hccl {

HcclResult ReduceScatterMeshTemplate::RunAlgorithm(TemplateResource &templateResource,
                                                 std::vector<TxRxSlicesList> &txRxSlicesLists,
                                                 std::vector<u32> &ranksForOutputData)
{
    (void)templateResource;
    HCCL_INFO("[ReduceScatterMeshTemplate][RunAlgorithm] start, myRank[%u], rankSize[%u].",
              myRank_, templateRankSize_);

    CHK_RET(RunMeshReduceScatter(tempAlgParams_, ranks_, myRank_, ranksForOutputData, txRxSlicesLists));

    HCCL_INFO("[ReduceScatterMeshTemplate][RunAlgorithm] end.");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
