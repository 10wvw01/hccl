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
    std::string algoConfig = hcclAlgo;
    algoConfig.erase(std::remove(algoConfig.begin(), algoConfig.end(), ' '), algoConfig.end());
    if (algoConfig.empty()) {
        HCCL_RUN_INFO("hccl algo config is empty, HCCL use built-in algo selection.");
        return HCCL_SUCCESS;
    }
    std::vector<std::string> algoPerOptype;
    CHK_RET(SplitHcclOpType(algoConfig, algoPerOptype));
    bool anyCommonConfig = false;
    bool anySpecificConfig = false;
    CHK_RET(CheckAlgoConfigValid(algoPerOptype, anyCommonConfig, anySpecificConfig));
    if (anyCommonConfig) {
        CHK_RET(SetCommonAlgType(algoPerOptype));
    } else {
        g_envConfig.specificAlgoMode = anySpecificConfig;
        CHK_RET(SetSpecificAlgType(algoPerOptype));
    }
    return HCCL_SUCCESS;
}
