/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MESH_OMNIPIPE_PRIMITIVES_H
#define MESH_OMNIPIPE_PRIMITIVES_H

#include "hccl_algorithm.h"

namespace ops_hccl {

// OmniPipe 的 step 切片形态和普通 Mesh 不同：一个 peer task 内会携带多组 step slice，
// 因此单独放在 OmniPipe primitive 中，避免污染普通 repeat/channel slice 主干。
HcclResult RunMeshOmniPipeAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                                    EngineType engineType, const std::vector<u32> &ranks, u32 myRank);

HcclResult RunMeshOmniPipeReduceScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                                        EngineType engineType, const std::vector<u32> &ranks, u32 myRank);

} // namespace ops_hccl

#endif
