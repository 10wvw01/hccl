/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "hcomm_version_cache.h"

namespace {
constexpr int HCOMM_VERSION = 90100000;

TEST(HcommVersionCacheTest, CachesSuccessfulQuery)
{
    HcommVersionCache cache;
    int queryCount = 0;
    auto query = [&queryCount](int* version) {
        ++queryCount;
        *version = HCOMM_VERSION;
        return true;
    };

    EXPECT_EQ(cache.Get(query), HCOMM_VERSION);
    EXPECT_EQ(cache.Get(query), HCOMM_VERSION);
    EXPECT_EQ(queryCount, 1);
}

TEST(HcommVersionCacheTest, RetriesAfterQueryFailure)
{
    HcommVersionCache cache;
    int queryCount = 0;
    auto query = [&queryCount](int* version) {
        ++queryCount;
        if (queryCount == 1) {
            return false;
        }
        *version = HCOMM_VERSION;
        return true;
    };

    EXPECT_EQ(cache.Get(query), 0);
    EXPECT_EQ(cache.Get(query), HCOMM_VERSION);
    EXPECT_EQ(cache.Get(query), HCOMM_VERSION);
    EXPECT_EQ(queryCount, 2);
}

TEST(HcommVersionCacheTest, SerializesConcurrentInitialization)
{
    constexpr int THREAD_COUNT = 8;
    HcommVersionCache cache;
    std::atomic<int> queryCount{0};
    std::mutex startMutex;
    std::condition_variable startCondition;
    std::condition_variable readyCondition;
    int readyCount = 0;
    bool start = false;
    std::vector<int> results(THREAD_COUNT, 0);
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);

    auto query = [&queryCount](int* version) {
        queryCount.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        *version = HCOMM_VERSION;
        return true;
    };

    for (int index = 0; index < THREAD_COUNT; ++index) {
        threads.emplace_back([&, index] {
            {
                std::unique_lock<std::mutex> lock(startMutex);
                ++readyCount;
                readyCondition.notify_one();
                startCondition.wait(lock, [&start] { return start; });
            }
            results[index] = cache.Get(query);
        });
    }

    {
        std::unique_lock<std::mutex> lock(startMutex);
        readyCondition.wait(lock, [&readyCount] { return readyCount == THREAD_COUNT; });
        start = true;
    }
    startCondition.notify_all();

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(queryCount.load(std::memory_order_relaxed), 1);
    for (const int result : results) {
        EXPECT_EQ(result, HCOMM_VERSION);
    }
}
} // namespace

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
