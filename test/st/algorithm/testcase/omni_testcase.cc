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
#include <future>
#include "alg_env_config.h"
#include "omni_parser.h"

using namespace HcclSim;
using namespace ops_hccl;
namespace checker {
class ST_OMNI_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        (void)setenv("HCCL_ALL_TO_ALL_V_ALGO", "OMNI", 1);
    }
    void TearDown() override
    {
        (void)unsetenv("HCCL_ALL_TO_ALL_V_ALGO");
        // 取消设置环境变量
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {
        omni::BinaryParser parser{};
        for (uint32_t i = 0; i < 4; ++i) {
            std::ostringstream oss;
            oss << "/srv/workspace/yhb/test/bin-parser/bin_output/rank_" << i << ".bin";
            std::string fileName = oss.str();
            (void)parser.SetFile(fileName, i);
            (void)parser.Parse(xmlInfo_[i]);
        }
    }
    static void TearDownTestCase()
    {}
    static omni::XmlInfo xmlInfo_[4];

    HcclResult CheckOmni(const std::vector<void *> &sendBufs, const std::vector<void *> &recvBufs,
        std::vector<void *> &checkRecvBufs, uint32_t rankSize)
    {
        // TODO run omni with sendBufs and checkRecvBufs
 
        // TODO compare recvBufs and checkRecvBufs
        return HCCL_SUCCESS;
    }

    void RunOmniTest(TopoMeta &topoMeta, uint32_t rankSize, HcclDataType dataType, std::vector<u64> &sendCountMatrix)
    {
        SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
        // 设置展开模式为HOST_TS
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("HCCL_BUFFSIZE", "200", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);

        std::vector<std::shared_ptr<std::promise<std::tuple<void*, void*, void*>>>> promises;
        std::vector<std::thread> threads;
        for (auto rankId = 0; rankId < rankSize; ++rankId) {
            auto promise = std::make_shared<std::promise<std::tuple<void *, void *, void *>>>();
            promises.push_back(promise);
            threads.emplace_back([=]() {
                // 1.SetDevice
                aclrtSetDevice(rankId);

                // 2.创建流
                aclrtStream stream = nullptr;
                aclrtCreateStream(&stream);

                // 3.初始化通信域
                HcclComm comm = nullptr;
                CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

                // 构造数据
                std::vector<u64> sendCounts(rankSize, 0);
                std::vector<u64> recvCounts(rankSize, 0);
                std::vector<u64> sdispls(rankSize, 0);
                std::vector<u64> rdispls(rankSize, 0);

                u64 sendDataCount = 0;
                for (u64 i = 0; i < rankSize; i++) {
                    sendCounts[i] = sendCountMatrix[rankId * rankSize + i];
                    sdispls[i] = sendDataCount;
                    sendDataCount += sendCounts[i];
                }

                u64 recvDataCount = 0;
                for (u64 i = 0; i < rankSize; i++) {
                    recvCounts[i] = sendCountMatrix[i * rankSize + rankId];
                    rdispls[i] = recvDataCount;
                    recvDataCount += recvCounts[i];
                }

                void *sendBuf = nullptr;
                void *recvBuf = nullptr;
                void *checkRecvBuf = nullptr;
                // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
                aclrtMalloc(&sendBuf, sendDataCount * SIZE_TABLE[dataType], static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
                aclrtMalloc(&recvBuf, recvDataCount * SIZE_TABLE[dataType], static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
                aclrtMalloc(&checkRecvBuf, recvDataCount * SIZE_TABLE[dataType], static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

                // 将 buffer 返回主线程
                promise->set_value(std::make_tuple(sendBuf, recvBuf, checkRecvBuf));

                // 4.算子下发
                CHK_RET(HcclAlltoAllV(sendBuf,
                    sendCounts.data(),
                    sdispls.data(),
                    dataType,
                    recvBuf,
                    recvCounts.data(),
                    rdispls.data(),
                    dataType,
                    comm,
                    stream));

                // 5.销毁通信域
                CHK_RET(HcclCommDestroy(comm));
                return HCCL_SUCCESS;
            });
        }

        // 等待多线程执行完成
        for (auto& thread : threads) {
            thread.join();
        }
        
        // 用于收集各个 rank 的 sendBuf / recvBuf
        std::vector<void*> sendBufs(rankSize, nullptr);
        std::vector<void*> recvBufs(rankSize, nullptr);
        std::vector<void*> checkRecvBufs(rankSize, nullptr);
        for (uint32_t i = 0; i < rankSize; ++i) {
            auto buffers = promises[i]->get_future().get();
            sendBufs[i] = std::get<0>(buffers);
            recvBufs[i] = std::get<1>(buffers);
            checkRecvBufs[i] = std::get<2>(buffers);
        }

        HcclResult verifyRes = CheckOmni(sendBufs, recvBufs, checkRecvBufs, rankSize);
        EXPECT_EQ(verifyRes, HCCL_SUCCESS);

        // 资源清理
        SimWorld::Global()->Deinit();
    }
};

omni::XmlInfo ST_OMNI_TEST::xmlInfo_[4];

TEST_F(ST_OMNI_TEST, st_omni_1)
{
    (void)setenv("HCCL_ALL_TO_ALL_V_ALGO", "OMNI", 1);
    TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 4;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    u64 dataSize = 13107200 * 8;
    std::vector<u64> sendCountMatrix = {
        dataSize, dataSize, dataSize, dataSize,
        dataSize, dataSize, dataSize, dataSize,
        dataSize, dataSize, dataSize, dataSize,
        dataSize, dataSize, dataSize, dataSize,
    };
    RunOmniTest(topoMeta, rankSize, dataType, sendCountMatrix);
    (void)unsetenv("HCCL_ALL_TO_ALL_V_ALGO");
}
}
