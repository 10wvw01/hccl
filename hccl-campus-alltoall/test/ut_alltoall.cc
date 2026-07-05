/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_types.h>
#include <acl/acl_rt.h>

#include "hccl.h"
#include "common.h"

// 测试类
class HcclAlltoAllTest : public testing::Test {
protected:
    void SetUp() override
    {
        std::cout << "Setting up test..." << std::endl;
    }

    void TearDown() override
    {
        std::cout << "Tearing down test..." << std::endl;
    }
};

// 正常用例
TEST_F(HcclAlltoAllTest, Ut_HcclAlltoAll_When_ValidParams_Expect_Success)
{
    // 构造入参
    uint64_t count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    // 申请 devcie 内存
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    size_t size = count * SIZE_TABLE.at(dataType);
    EXPECT_EQ(aclrtMalloc(&sendBuf, size, ACL_MEM_MALLOC_HUGE_ONLY), ACL_SUCCESS);
    EXPECT_EQ(aclrtMalloc(&recvBuf, size, ACL_MEM_MALLOC_HUGE_ONLY), ACL_SUCCESS);
    // 创建 stream 资源
    aclrtStream stream;
    EXPECT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    // 初始化通信域
    HcclComm comm;
    EXPECT_EQ(HcclCommInitClusterInfo(nullptr, 8, &comm), HCCL_SUCCESS);

    // 调用算子
    HcclResult result = HcclAlltoAll(sendBuf, count, dataType, recvBuf, count, dataType, comm, stream);

    // 检查结果
    EXPECT_EQ(result, HCCL_SUCCESS);
}
