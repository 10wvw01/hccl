/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>
#include <string>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                        \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                          \
    do {                                                                                        \
        if (ret != HCCL_SUCCESS) {                                                              \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                         \
        }                                                                                       \
    } while (0)

struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
    float *result_before;
    float *result_after;
    bool *success;
};

int RunAllReduce(void *arg) {
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    uint32_t device = ctx->device;
    uint64_t count = ctx->devCount;
    size_t mallocSize = count * sizeof(float);

    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    ACLCHECK(aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));

    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, mallocSize));
    float *tmpHostBuff = static_cast<float *>(hostBuf);
    for (uint64_t i = 0; i < count; ++i) {
        tmpHostBuff[i] = static_cast<float>(device);
    }
    ACLCHECK(aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostBuf));

    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    HCCLCHECK(HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream));
    ACLCHECK(aclrtSynchronizeStream(stream));

    void *resultBuff;
    ACLCHECK(aclrtMallocHost(&resultBuff, mallocSize));
    ACLCHECK(aclrtMemcpy(resultBuff, mallocSize, recvBuf, mallocSize, ACL_MEMCPY_DEVICE_TO_HOST));
    float *tmpResBuff = static_cast<float *>(resultBuff);

    float expected = 0.0f;
    for (uint32_t i = 0; i < ctx->devCount; ++i) {
        expected += static_cast<float>(i);
    }

    bool check_pass = true;
    for (uint64_t i = 0; i < count; ++i) {
        if (fabs(tmpResBuff[i] - expected) > 0.001f) {
            check_pass = false;
            break;
        }
    }

    if (ctx->result_before != nullptr) {
        for (uint64_t i = 0; i < count; ++i) {
            ctx->result_before[i] = tmpResBuff[i];
        }
    }
    if (ctx->result_after != nullptr) {
        for (uint64_t i = 0; i < count; ++i) {
            ctx->result_after[i] = tmpResBuff[i];
        }
    }
    *(ctx->success) = check_pass;

    printf("[Device %u] AllReduce: %s (expected=%.6f, got=%.6f)\n", 
           device, check_pass ? "PASS" : "FAIL", expected, tmpResBuff[0]);

    ACLCHECK(aclrtFreeHost(resultBuff));

    HCCLCHECK(HcclCommDestroy(hcclComm));
    ACLCHECK(aclrtFree(sendBuf));
    ACLCHECK(aclrtFree(recvBuf));
    ACLCHECK(aclrtDestroyStream(stream));
    return 0;
}

int main(int argc, char **argv) {
    std::string wait_file = "";
    std::string signal_file = "";
    
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--wait-file" && i + 1 < argc) {
            wait_file = argv[++i];
        } else if (std::string(argv[i]) == "--signal-file" && i + 1 < argc) {
            signal_file = argv[++i];
        }
    }

    ACLCHECK(aclInit(NULL));
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    printf("[HCCL] Found %u NPU device(s)\n", devCount);

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));

    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    printf("[HCCL] Running AllReduce BEFORE HIXL...\n");
    
    std::vector<float> result_before(devCount);
    std::vector<float> result_after(devCount);
    std::vector<bool> success_before(devCount, false);
    std::vector<bool> success_after(devCount, false);

    std::vector<std::thread> threads_before(devCount);
    std::vector<ThreadContext> args_before(devCount);
    for (uint32_t i = 0; i < devCount; ++i) {
        args_before[i].rootInfo = rootInfo;
        args_before[i].device = i;
        args_before[i].devCount = devCount;
        args_before[i].result_before = result_before.data();
        args_before[i].result_after = nullptr;
        args_before[i].success = success_before.data() + i;
        threads_before[i] = std::thread(RunAllReduce, (void *)&args_before[i]);
    }
    for (uint32_t i = 0; i < devCount; ++i) {
        threads_before[i].join();
    }

    bool all_before_pass = true;
    for (uint32_t i = 0; i < devCount; ++i) {
        if (!success_before[i]) {
            all_before_pass = false;
            break;
        }
    }

    if (!signal_file.empty()) {
        std::ofstream signal(signal_file);
        signal << "HCCL_BEFORE_DONE";
        signal.close();
        printf("[HCCL] Signal sent: %s\n", signal_file.c_str());
    }

    if (!wait_file.empty()) {
        printf("[HCCL] Waiting for HIXL to complete... (%s)\n", wait_file.c_str());
        while (!std::ifstream(wait_file).good()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        printf("[HCCL] HIXL completed, running AllReduce AFTER HIXL...\n");
    }

    std::vector<std::thread> threads_after(devCount);
    std::vector<ThreadContext> args_after(devCount);
    for (uint32_t i = 0; i < devCount; ++i) {
        args_after[i].rootInfo = rootInfo;
        args_after[i].device = i;
        args_after[i].devCount = devCount;
        args_after[i].result_before = nullptr;
        args_after[i].result_after = result_after.data();
        args_after[i].success = success_after.data() + i;
        threads_after[i] = std::thread(RunAllReduce, (void *)&args_after[i]);
    }
    for (uint32_t i = 0; i < devCount; ++i) {
        threads_after[i].join();
    }

    bool all_after_pass = true;
    for (uint32_t i = 0; i < devCount; ++i) {
        if (!success_after[i]) {
            all_after_pass = false;
            break;
        }
    }

    printf("[HCCL] === Result Comparison ===\n");
    bool consistent = true;
    for (uint32_t i = 0; i < devCount; ++i) {
        if (fabs(result_before[i] - result_after[i]) > 0.001f) {
            printf("[HCCL] Device %u: BEFORE=%.6f, AFTER=%.6f -> INCONSISTENT\n", 
                   i, result_before[i], result_after[i]);
            consistent = false;
        } else {
            printf("[HCCL] Device %u: BEFORE=%.6f, AFTER=%.6f -> CONSISTENT\n", 
                   i, result_before[i], result_after[i]);
        }
    }

    ACLCHECK(aclrtFreeHost(rootInfoBuf));
    ACLCHECK(aclFinalize());

    if (all_before_pass && all_after_pass && consistent) {
        printf("[HCCL] === ALL TESTS PASSED ===\n");
        return 0;
    } else {
        printf("[HCCL] === TEST FAILED ===\n");
        return 1;
    }
}
