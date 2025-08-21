/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "env_config.h"
#include <algorithm>
#include <mutex>
#include <sstream>
#include <string>
#include "adapter_error_manager_pub.h"
#include "log.h"
#include "sal_pub.h"
#include "mmpa_api.h"
#include "config_log.h"

using namespace hccl;
HcclResult SetHcclAlgoConfig(const std::string &hcclAlgo)
{
    return HCCL_SUCCESS;
}
