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

#include <acl/acl_rt.h>
#include <hccl/hccl_types.h>
#include <hccl_custom_allreduce.h>

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
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    void *hostBuf = nullptr;
    void *resultBuff = nullptr;
    aclrtStream stream = nullptr;
    HcclComm hcclComm = nullptr;
    float *tmpHostBuff = nullptr;
    float *tmpResBuff = nullptr;
    int32_t ret = 0;
    uint32_t device = ctx->device;
    uint64_t count = 1U;
    size_t sendSize = count * sizeof(float);
    size_t recvSize = count * sizeof(float);
    // 设置当前线程操作的设备
    ret = aclrtSetDevice(static_cast<int32_t>(device));
    if (ret != ACL_SUCCESS) {
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);
        goto cleanup;
    }

    // 初始化集合通信域
    ret = HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm);
    if (ret != HCCL_SUCCESS) {
        printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);
        goto cleanup;
    }

    // 创建任务流
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);
        goto cleanup;
    }

    // 申请集合通信操作的 Device 内存
    ret = aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
    if (ret != ACL_SUCCESS) {
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);
        goto cleanup;
    }
    ret = aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);
    if (ret != ACL_SUCCESS) {
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);
        goto cleanup;
    }

    // 申请 Host 内存用于存放输入数据，并将内容初始化为 Device ID
    ret = aclrtMallocHost(&hostBuf, sendSize);
    if (ret != ACL_SUCCESS) {
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);
        goto cleanup;
    }
    tmpHostBuff = static_cast<float *>(hostBuf);
    for (uint64_t i = 0; i < count; ++i) {
        tmpHostBuff[i] = static_cast<float>(device);
    }
    std::cout << "rankId: " << device << ", input: [";
    for (uint64_t i = 0; i < count; ++i) {
        std::cout << " " << tmpHostBuff[i];
    }
    std::cout << " ]" << std::endl;

    // 将 Host 侧输入数据拷贝到 Device 侧
    ret = aclrtMemcpy(sendBuf, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);
        goto cleanup;
    }
    // 释放 Host 侧内存
    aclrtFreeHost(hostBuf);
    hostBuf = nullptr;

    // 执行 AllReduce，将通信域内所有 rank 的 sendBuf 按指定归约操作（SUM）进行归约，
    // 结果发送到所有 rank 的 recvBuf。recvBuf 大小与 sendBuf 相同。
    ret = HcclAllReduceCustom(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
    if (ret != HCCL_SUCCESS) {
        printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);
        goto cleanup;
    }
    // 阻塞等待任务流中的集合通信任务执行完成
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);
        goto cleanup;
    }

    // 将 Device 侧集合通信任务结果拷贝到 Host，并打印结果
    std::this_thread::sleep_for(std::chrono::seconds(ctx->device));
    ret = aclrtMallocHost(&resultBuff, recvSize);
    if (ret != ACL_SUCCESS) {
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);
        goto cleanup;
    }
    ret = aclrtMemcpy(resultBuff, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);
        goto cleanup;
    }
    tmpResBuff = static_cast<float *>(resultBuff);
    std::cout << "rankId: " << ctx->device << ", output: [";
    for (uint64_t i = 0; i < count; ++i) {
        std::cout << " " << tmpResBuff[i];
    }
    std::cout << " ]" << std::endl;
    aclrtFreeHost(resultBuff);
    resultBuff = nullptr;

cleanup:
    // 释放资源（不提前返回，确保所有已分配资源都被释放）
    if (resultBuff != nullptr) {
        aclrtFreeHost(resultBuff);
    }
    if (hcclComm != nullptr) {
        HcclCommDestroy(hcclComm);  // 销毁通信域
    }
    if (sendBuf != nullptr) {
        aclrtFree(sendBuf);      // 释放 Device 侧内存
    }
    if (recvBuf != nullptr) {
        aclrtFree(recvBuf);      // 释放 Device 侧内存
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);  // 销毁任务流
    }
    if (hostBuf != nullptr) {
        aclrtFreeHost(hostBuf);
    }
    aclrtResetDevice(device);    // 重置设备
    return ret;
}

int main()
{
    // 设备资源初始化
    ACLCHECK(aclInit(NULL));
    // 查询设备数量
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    // 生成 Root 节点信息，各线程使用同一份 RootInfo
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    // 启动线程执行集合通信操作
    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    for (uint32_t i = 0; i < devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = devCount;
        threads[i] = std::thread(Sample, (void *)&args[i]);
    }
    for (uint32_t i = 0; i < devCount; i++) {
        threads[i].join();
    }

    // 释放资源
    ACLCHECK(aclrtFreeHost(rootInfoBuf));  // 释放 Host 内存
    ACLCHECK(aclrtFinalize());             // 设备去初始化
    return 0;
}
