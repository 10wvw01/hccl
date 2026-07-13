/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NHR_PRIMITIVES_H
#define NHR_PRIMITIVES_H

#include <vector>
#include "hccl_algorithm.h"
#include "alg_param.h"

namespace ops_hccl {

struct TemplateDataParams;
struct TemplateResource;

// 构造 NHR AllGather 的通信描述符列表，实际 SendRecv 由 template 执行。
HcclResult RunNhrAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                           const std::vector<u32> &ranks, u32 myRank,
                           std::vector<SendRecvInfo> &sendRecvInfos);

// 构造 NHR ReduceScatter 的通信描述符列表，实际 SendRecv 由 template 执行。
HcclResult RunNhrReduceScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                               const std::vector<u32> &ranks, u32 myRank,
                               std::vector<SendRecvInfo> &sendRecvInfos);

// 构造 NHR Scatter 的通信描述符列表，实际 SendRecv 由 template 执行。
HcclResult RunNhrScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                         const std::vector<u32> &ranks, u32 myRank,
                         std::vector<SendRecvInfo> &sendRecvInfos);

// 构造 NHR Barrier 的通信描述符列表，实际 SendRecv 由 template 执行。
HcclResult RunNhrBarrier(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                         const std::vector<u32> &ranks, u32 myRank,
                         std::vector<SendRecvInfo> &sendRecvInfos);

}

#endif
