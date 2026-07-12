/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include <thread>
#include "alg_env_config.h"
#include "dlsym_common.h"
#include "alg_param.h"

extern "C" unsigned int HcclLaunchP2pAicpuKernel(void *args);
extern "C" void ResetP2pCleanupStub();
extern "C" void SetP2pCommStatusStub(HcclResult statusRet, HcclCommStatus commStatus);
extern "C" void SetP2pCleanupRetStub(int32_t batchStartRet, int32_t batchEndRet, int32_t releaseRet);
extern "C" void GetP2pCleanupStubCount(uint32_t *acquireCount, uint32_t *releaseCount,
    uint32_t *batchStartCount, uint32_t *batchEndCount);
extern "C" void ResetP2pDfxProfilingStub();
extern "C" void SetP2pDfxProfilingStub(HcclResult dfxRegRet, HcclResult profilingStartRet,
    HcclResult profilingEndRet, HcclResult profilingDeviceRet);

constexpr uint32_t DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t),
    sizeof(int16_t),
    sizeof(int32_t),
    2,
    sizeof(float),
    sizeof(int64_t),
    sizeof(uint64_t),
    sizeof(uint8_t),
    sizeof(uint16_t),
    sizeof(uint32_t),
    8,
    2,
    16,
    2,
    1,
    1,
    1,
    1};

using namespace HcclSim;
using namespace ops_hccl;

class DPU_SEND_RECV_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        ResetP2pCleanupStub();
        ResetP2pDfxProfilingStub();
    }

    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }

    static void SetUpTestCase()
    {}

    static void TearDownTestCase()
    {}

    /**
     * 构造最小P2P AICPU参数并直接执行内核入口
     * @param caseTag 用于隔离反序列化缓存的用例标签
     * @return P2P AICPU内核入口的执行结果
     */
    unsigned int RunP2pCleanupCase(const char *caseTag)
    {
        AlgResourceCtxSerializable resCtx;
        resCtx.threads = {1};
        std::vector<char> resCtxData = resCtx.Serialize();

        alignas(OpParam) HcclP2pKernelParam kernelParam{};
        OpParam *param = new (kernelParam.opParams) OpParam();
        param->opType = HcclCMDType::HCCL_CMD_SEND;
        param->deviceType = DevType::DEV_TYPE_950;
        param->resCtx = resCtxData.data();
        param->ctxSize = resCtxData.size();
        (void)strncpy_s(param->algName, sizeof(param->algName), "opv2_p2p_cleanup_test",
            sizeof(param->algName) - 1);
        (void)strncpy_s(param->algTag, sizeof(param->algTag), caseTag, sizeof(param->algTag) - 1);
        (void)strncpy_s(param->commName, sizeof(param->commName), caseTag, sizeof(param->commName) - 1);

        unsigned int result = HcclLaunchP2pAicpuKernel(&kernelParam);
        param->~OpParam();
        return result;
    }
};

TEST_F(DPU_SEND_RECV_TEST, dpu_p2p_cleanup_when_status_api_fails)
{
    SetP2pCommStatusStub(HCCL_E_INTERNAL, HCCL_COMM_STATUS_INVALID);
    EXPECT_EQ(RunP2pCleanupCase("status_api_fail"), 1U);
    uint32_t acquire = 0, release = 0, batchStart = 0, batchEnd = 0;
    GetP2pCleanupStubCount(&acquire, &release, &batchStart, &batchEnd);
    EXPECT_EQ(acquire, 1U);
    EXPECT_EQ(release, 1U);
    EXPECT_EQ(batchStart, 0U);
    EXPECT_EQ(batchEnd, 0U);
}

TEST_F(DPU_SEND_RECV_TEST, dpu_p2p_cleanup_when_status_is_not_ready)
{
    SetP2pCommStatusStub(HCCL_SUCCESS, HCCL_COMM_STATUS_RESERVED);
    EXPECT_EQ(RunP2pCleanupCase("status_not_ready"), 1U);
    uint32_t acquire = 0, release = 0, batchStart = 0, batchEnd = 0;
    GetP2pCleanupStubCount(&acquire, &release, &batchStart, &batchEnd);
    EXPECT_EQ(acquire, 1U);
    EXPECT_EQ(release, 1U);
    EXPECT_EQ(batchStart, 0U);
    EXPECT_EQ(batchEnd, 0U);
}

TEST_F(DPU_SEND_RECV_TEST, dpu_p2p_cleanup_when_batch_mode_start_fails)
{
    SetP2pCleanupRetStub(HCCL_E_INTERNAL, HCCL_SUCCESS, HCCL_SUCCESS);
    EXPECT_EQ(RunP2pCleanupCase("batch_start_fail"), 1U);
    uint32_t acquire = 0, release = 0, batchStart = 0, batchEnd = 0;
    GetP2pCleanupStubCount(&acquire, &release, &batchStart, &batchEnd);
    EXPECT_EQ(acquire, 1U);
    EXPECT_EQ(release, 1U);
    EXPECT_EQ(batchStart, 1U);
    EXPECT_EQ(batchEnd, 0U);
}

TEST_F(DPU_SEND_RECV_TEST, dpu_p2p_cleanup_when_dfx_registration_fails)
{
    SetP2pDfxProfilingStub(HCCL_E_INTERNAL, HCCL_SUCCESS, HCCL_SUCCESS, HCCL_SUCCESS);
    EXPECT_EQ(RunP2pCleanupCase("dfx_reg_fail"), 1U);
    uint32_t acquire = 0, release = 0, batchStart = 0, batchEnd = 0;
    GetP2pCleanupStubCount(&acquire, &release, &batchStart, &batchEnd);
    EXPECT_EQ(acquire, 1U);
    EXPECT_EQ(release, 1U);
    EXPECT_EQ(batchStart, 1U);
    EXPECT_EQ(batchEnd, 1U);
}

TEST_F(DPU_SEND_RECV_TEST, dpu_p2p_cleanup_when_profiling_start_fails)
{
    SetP2pDfxProfilingStub(HCCL_SUCCESS, HCCL_E_INTERNAL, HCCL_SUCCESS, HCCL_SUCCESS);
    EXPECT_EQ(RunP2pCleanupCase("profiling_start_fail"), 1U);
    uint32_t acquire = 0, release = 0, batchStart = 0, batchEnd = 0;
    GetP2pCleanupStubCount(&acquire, &release, &batchStart, &batchEnd);
    EXPECT_EQ(acquire, 1U);
    EXPECT_EQ(release, 1U);
    EXPECT_EQ(batchStart, 1U);
    EXPECT_EQ(batchEnd, 1U);
}

TEST_F(DPU_SEND_RECV_TEST, dpu_p2p_release_comm_when_batch_mode_end_fails)
{
    SetP2pCleanupRetStub(HCCL_SUCCESS, HCCL_E_INTERNAL, HCCL_SUCCESS);
    EXPECT_EQ(RunP2pCleanupCase("batch_end_fail"), 1U);
    uint32_t acquire = 0, release = 0, batchStart = 0, batchEnd = 0;
    GetP2pCleanupStubCount(&acquire, &release, &batchStart, &batchEnd);
    EXPECT_EQ(acquire, 1U);
    EXPECT_EQ(release, 1U);
    EXPECT_EQ(batchStart, 1U);
    EXPECT_EQ(batchEnd, 1U);
}

using RankId = uint32_t;

void DPUSendRecvTest(
    TopoMeta &topoMeta, const std::map<RankId, RankId> &sendRecvMap, u32 dataCount, HcclDataType dataType)
{
    u32 rankSize = 0;
    for (const auto &superPod : topoMeta)
        for (const auto &server : superPod)
            rankSize += server.size();
    // 仿真模型初始化
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    // 设置展开模式
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 多线程运行send&recv算子
    std::vector<std::thread> threads;
    for (const auto &kv : sendRecvMap) {
        RankId srcRankId = kv.first;
        RankId dstRankId = kv.second;
        threads.emplace_back([=]() {
            // 1.SetDevice
            aclrtSetDevice(srcRankId);

            // 2.创建流
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            // 3.初始化通信域
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", srcRankId, &comm));

            u64 bufSize = dataCount * DATATYPE_SIZE_TABLE[dataType];  // 数据量转化为字节数

            // 4.算子下发
            void *sendBuf = nullptr;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, bufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            CHK_RET(HcclSend(sendBuf, dataCount, dataType, dstRankId, comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
        threads.emplace_back([=]() {
            // 1.SetDevice
            aclrtSetDevice(dstRankId);

            // 2.创建流
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            // 3.初始化通信域
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", dstRankId, &comm));

            u64 bufSize = dataCount * DATATYPE_SIZE_TABLE[dataType];  // 数据量转化为字节数

            // 4.算子下发
            void *recvBuf = nullptr;
            aclrtMalloc(&recvBuf, bufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            CHK_RET(HcclRecv(recvBuf, dataCount, dataType, srcRankId, comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    // 等待多线程执行完成
    for (auto &thread : threads) {
        thread.join();
    }

    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    for (const auto &kv : sendRecvMap) {
        RankId srcRankId = kv.first;
        RankId dstRankId = kv.second;
        HcclResult sendRes = CheckSend(taskQueues, rankSize, dataType, dataCount, srcRankId, dstRankId);
        EXPECT_TRUE(sendRes == HCCL_SUCCESS);
        HcclResult recvRes = CheckRecv(taskQueues, rankSize, dataType, dataCount, srcRankId, dstRankId);
        EXPECT_TRUE(recvRes == HCCL_SUCCESS);
    }

    // 资源清理
    SimWorld::Global()->Deinit();
};

TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_int32)
{
    std::cout << "here 1" << endl;
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1}, {2, 3}, {4, 5}, {6, 7}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 1}, {2, 5}, {3, 4}, {6, 7}};

    // 算子执行参数设置
    auto dataCount = 600 * 1024 * 1024;                  // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    std::cout << "func:" << endl;
    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_int16)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 2}, {1, 3}};

    // 算子执行参数设置
    auto dataCount = 100;                                // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_count1)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 2}, {1, 3}};

    auto dataCount = 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 单卡单机两超节点100个int32
TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_uint16)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0}, {1}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 1}};

    // 算子执行参数设置
    auto dataCount = 100;                                 // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT16;  // 数据类型

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 八卡单机256M个int32
TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_uint8)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1, 2}, {3, 4, 5}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 5}, {3, 1}, {2, 4}};

    // 算子执行参数设置
    auto dataCount = 256 * 1024 * 1024;                  // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;  // 数据类型

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个int32
TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_fp8e5m2)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {1, 6}, {2, 3}, {5, 7}};

    // 算子执行参数设置
    auto dataCount = 600 * 1024 * 1024;                    // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2;  // 数据类型

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_fp8e8m0)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};

    // 算子执行参数设置
    auto dataCount = 256 * 1024 * 1024;                    // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0;  // 数据类型

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_bfp16)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};

    // 算子执行参数设置
    auto dataCount = 256 * 1024 * 1024;                  // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;  // 数据类型

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_fp16)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};

    // 算子执行参数设置
    auto dataCount = 256 * 1024 * 1024;                 // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;  // 数据类型

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_fp32)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};

    // 算子执行参数设置
    auto dataCount = 256 * 1024 * 1024;                 // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个HIF8
TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_hif8)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};

    // 算子执行参数设置
    auto dataCount = 256 * 1024 * 1024;                 // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_HIF8;  // 数据类型

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个FP8E4M3
TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_fp8e4m3)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};

    // 算子执行参数设置
    auto dataCount = 256 * 1024 * 1024;                    // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E4M3;  // 数据类型

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个FP8E5M2
TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_int8)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};

    // 算子执行参数设置
    auto dataCount = 256 * 1024 * 1024;                 // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型

    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个FP8E8M0
TEST_F(DPU_SEND_RECV_TEST, dpu_send_recv_test_fp64)
{
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};

    // 算子执行参数设置
    auto dataCount = 256 * 1024 * 1024;                 // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP64;  // 数据类型
    DPUSendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}
