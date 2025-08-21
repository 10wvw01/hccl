/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "stream_utils.h"
 #include <unordered_map>
 #include <functional>
 #include "log.h"
 #include "runtime/stream.h"
 #include "runtime/rt_model.h"
 #include "workflow_pub.h"
 
 HcclResult GetStreamCaptureInfo(rtStream_t stream, rtModel_t &rtModel, bool &isCapture)
 {
     return HCCL_SUCCESS;
 }

 HcclResult AddStreamToModel(rtStream_t stream, rtModel_t &rtModel)
{
    return HCCL_SUCCESS;
}

HcclResult GetModelId(rtModel_t &rtModel, u32 &modelId)
{
    return HCCL_SUCCESS;
}