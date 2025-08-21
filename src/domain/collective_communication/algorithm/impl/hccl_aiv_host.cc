/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mutex>
#include <vector>
#include <iostream>
#include <fstream>
#include "mmpa_api.h"
#include "adapter_rts_common.h"
#include "hccl_aiv.h"
#include "workflow_pub.h"
#include "ccl_buffer_manager.h"

namespace hccl {
void SetAivProfilingInfoBeginTime(AivProfilingInfo& aivProfilingInfo)
{
    HCCL_DEBUG("aiv aivProfilingInfo.beginTime host");
    aivProfilingInfo.beginTime = MsprofSysCycleTime();
    HCCL_DEBUG("aivProfilingInfo.beginTime:%lu",aivProfilingInfo.beginTime);
}

void SetAivProfilingInfoBeginTime(uint64_t& beginTime)
{
    HCCL_DEBUG("aiv beginTime host");
    beginTime = MsprofSysCycleTime();
    HCCL_DEBUG("beginTime:%lu",beginTime);
}
}