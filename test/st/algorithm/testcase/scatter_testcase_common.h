#ifndef SCATTER_TEST_COMMON_H
#define SCATTER_TEST_COMMON_H

#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"

#include <thread>
#include <vector>

using namespace HcclSim;
using namespace ops_hccl;


constexpr uint32_t DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};


inline void RunScatterTest(int root, TopoMeta &topoMeta, int dataCount, HcclDataType dataType) 
{
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    // 算子执行参数设置
    auto rankSize = CalRankSize(topoMeta);  // 参与集合通信的卡数(同topoMeta卡数一致)
    auto recvCount = dataCount;  // 接收数据量
    // auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型

    // 多线程运行SCATTER算子
    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            // 1.SetDevice
            aclrtSetDevice(rankId);

            // 2.创建流
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            // 3.初始化通信域
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void *sendBuf = nullptr;
            void *recvBuf = nullptr;
            u64 sendDataSize = recvCount * DATATYPE_SIZE_TABLE[dataType] * rankSize;
            u64 recvDataSize = recvCount * DATATYPE_SIZE_TABLE[dataType];
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendDataSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvDataSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            // 4.算子下发
            CHK_RET(HcclScatter(sendBuf, recvBuf, recvCount, dataType, root, comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    // 等待多线程执行完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckScatter(taskQueues, rankSize, dataType, recvCount, root);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit(); 
}


#endif