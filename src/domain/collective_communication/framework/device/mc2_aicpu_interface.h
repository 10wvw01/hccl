/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MC2_AICPU_INTERFACE_H__
#define __MC2_AICPU_INTERFACE_H__

#include <cstdint>

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuKfcResInit(void *args);
__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvLaunch(void *args);
__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvGroupLaunch(void *args);
__attribute__((visibility("default"))) uint32_t RunAicpuKfcSrvLaunch(void *args[]);
__attribute__((visibility("default"))) uint32_t RunAicpuKfcResInitV2(void *args);
__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvLaunchV2(void *args);
}

#endif // __MC2_AICPU_INTERFACE_HPP__