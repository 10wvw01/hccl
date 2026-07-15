/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_VERSION_CACHE_H
#define HCOMM_VERSION_CACHE_H

#include <atomic>
#include <mutex>

class HcommVersionCache {
public:
    template <typename Query>
    int Get(Query query)
    {
        int version = version_.load(std::memory_order_acquire);
        if (version != 0) {
            return version;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        version = version_.load(std::memory_order_relaxed);
        if (version != 0) {
            return version;
        }

        if (!query(&version)) {
            return 0;
        }

        version_.store(version, std::memory_order_release);
        return version;
    }

private:
    std::atomic<int> version_{0};
    std::mutex mutex_;
};

#endif // HCOMM_VERSION_CACHE_H
