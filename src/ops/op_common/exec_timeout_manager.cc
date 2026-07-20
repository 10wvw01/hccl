/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "exec_timeout_manager.h"
#include "log.h"
#include "hccl_common.h"
#include <cstdlib>
#include <cerrno>

namespace ops_hccl {

ExecTimeoutManager::ExecTimeoutManager() 
    : execTimeout_(HCCL_DEFAULT_TIMEOUT),
      timeoutSet_(false) {
    u32 envTimeout = LoadFromEnv();
    if (envTimeout > 0) {
        execTimeout_.store(envTimeout, std::memory_order_relaxed);
        timeoutSet_.store(true, std::memory_order_relaxed);
        HCCL_INFO("[ExecTimeoutManager] Initialized with env HCCL_EXEC_TIMEOUT: %u seconds", envTimeout);
    } else {
        HCCL_INFO("[ExecTimeoutManager] Initialized with default timeout: %u seconds", HCCL_DEFAULT_TIMEOUT);
    }
}

ExecTimeoutManager::~ExecTimeoutManager() {
}

ExecTimeoutManager& ExecTimeoutManager::Instance() {
    static ExecTimeoutManager instance;
    return instance;
}

void ExecTimeoutManager::SetExecTimeout(u32 execTimeout) {
    u32 timeoutValue = execTimeout;
    execTimeout_.store(timeoutValue, std::memory_order_relaxed);
    timeoutSet_.store(true, std::memory_order_relaxed);
    HCCL_INFO("[ExecTimeoutManager] Setting exec timeout to: %u seconds", timeoutValue);
}

u32 ExecTimeoutManager::GetExecTimeout() {
    bool isSet = timeoutSet_.load(std::memory_order_relaxed);
    u32 timeout = isSet ? execTimeout_.load(std::memory_order_relaxed) : HCCL_DEFAULT_TIMEOUT;
    HCCL_DEBUG("[ExecTimeoutManager] Getting exec timeout: %u seconds.", timeout);
    return timeout;
}

u32 ExecTimeoutManager::LoadFromEnv()
{
    const char* envVal = getenv("HCCL_EXEC_TIMEOUT");
    if (envVal == nullptr) {
        return 0;
    }
    char* endptr = nullptr;
    errno = 0;
    long val = strtol(envVal, &endptr, 10);
    if (errno == ERANGE || endptr == envVal || *endptr != '\0' || val <= 0 ||
        static_cast<u64>(val) > static_cast<u64>(UINT32_MAX)) {
        HCCL_WARNING("[ExecTimeoutManager] Invalid HCCL_EXEC_TIMEOUT value: %s, using default", envVal);
        return 0;
    }
    return static_cast<u32>(val);
}

} // namespace ops_hccl