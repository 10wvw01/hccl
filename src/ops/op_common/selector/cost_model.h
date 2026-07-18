/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_SELECTOR_COST_MODEL
#define HCCLV2_COLL_ALG_SELECTOR_COST_MODEL

#include <string>

#include "alg_param.h"
#include "log.h"

namespace ops_hccl {

typedef struct {
    const char *algName;
    const char *executorName;
    const char **templateName;
    int templateNum;
    HcclCMDType opType;
} AlgElement;

typedef struct {
    AlgElement *algElements;
    int count;
    int capacity;
} AllAlgos;

AllAlgos *GetAllAlgos();

HcclResult AddAlgToAllAlgos(HcclCMDType opType, const char *algName, const char *executorName,
                            const char **templateName, int templateNum);

typedef struct {
    float A;  // 用来描述跨卡传输的时间随DataSize变化的趋势，会受到UB带宽利用率的影响
    float B;  // 用来描述本地传输的时间随DataSize变化的趋势，不受到UB带宽利用率的影响
    float C;  // 用来描述一些基本时延的常数项
} CostModelParam;

typedef struct {
    const char *algName;         //
    const CostModelParam *param; //
    int count;                //
} CostAlgoParams;

typedef struct {
    CostAlgoParams *costAlgoParams;
    int count;
} CostModel;

} // namespace ops_hccl

#endif
