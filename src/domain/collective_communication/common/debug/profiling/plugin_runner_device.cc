/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "plugin_runner.h"
#include "adapter_rts_common.h"
#include "externalinput_pub.h"
#include "runtime/rt_error_codes.h"

using namespace hccl;

HcclResult PluginRunner::isStreamCapture(rtStream_t stream, bool &isCapture) const
{
    HCCL_WARNING("[PluginRunner][isStreamCapture]Stream capture does not support!");
    return HCCL_SUCCESS;
}