/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cost_model.h"

#include <new>

namespace ops_hccl {

AllAlgos *GetAllAlgos()
{
    static AllAlgos globalAllAlgos{nullptr, 0, 0};
    return &globalAllAlgos;
}

HcclResult AddAlgToAllAlgos(HcclCMDType opType, const char *algName, const char *executorName,
                            const char **templateName, int templateNum)
{
    AllAlgos *allAlgos = GetAllAlgos();
    if (allAlgos->count >= allAlgos->capacity) {
        int newCapacity = (allAlgos->capacity == 0) ? 16 : allAlgos->capacity * 2;
        AlgElement *newElements = new (std::nothrow) AlgElement[newCapacity];
        if (newElements == nullptr) {
            HCCL_ERROR("[AllAlgos] alloc failed, newCapacity=%d.", newCapacity);
            return HcclResult::HCCL_E_PARA;
        }
        for (int i = 0; i < allAlgos->count; ++i) {
            newElements[i] = allAlgos->algElements[i];
        }
        delete[] allAlgos->algElements;
        allAlgos->algElements = newElements;
        allAlgos->capacity = newCapacity;
    }
    allAlgos->algElements[allAlgos->count] = {algName, executorName, templateName, templateNum, opType};
    ++allAlgos->count;
    HCCL_DEBUG("[AllAlgos] add algName=%s executorName=%s templateNum=%d opType=%d, total=%d.",
               algName, executorName, templateNum, opType, allAlgos->count);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
