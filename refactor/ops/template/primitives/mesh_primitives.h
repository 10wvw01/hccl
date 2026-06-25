/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MESH_PRIMITIVES_H
#define MESH_PRIMITIVES_H

#include "template_utils.h"
#include "hccl_algorithm.h"

namespace ops_hccl {

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams,
                            TemplateResource &templateResource, EngineType engineType);

HcclResult RunMeshReduceScatter(const TemplateDataParams &tempAlgParams,
                                TemplateResource &templateResource, EngineType engineType);

HcclResult RunMeshScatter(const TemplateDataParams &tempAlgParams,
                          TemplateResource &templateResource, EngineType engineType);

HcclResult RunMeshGather(const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource, EngineType engineType);

HcclResult RunMeshAllToAll(const TemplateDataParams &tempAlgParams,
                           TemplateResource &templateResource, EngineType engineType);

HcclResult RunMeshBarrier(const TemplateDataParams &tempAlgParams,
                          TemplateResource &templateResource, EngineType engineType);

}

#endif
