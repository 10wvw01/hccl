/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ExecTimeoutManager unit test.
 */

#include "test_helpers.h"

using namespace ops_hccl;
using ops_hccl::testing::OpCommonTest;

// SetExecTimeout changes the value returned by GetExecTimeout
TEST_F(OpCommonTest, SetExecTimeoutReturnsSetValue)
{
    ExecTimeoutManager::Instance().SetExecTimeout(500);
    EXPECT_EQ(ExecTimeoutManager::Instance().GetExecTimeout(), 500u);
}

// SetExecTimeout to 0 (disabled)
TEST_F(OpCommonTest, SetExecTimeoutToZero)
{
    ExecTimeoutManager::Instance().SetExecTimeout(0);
    EXPECT_EQ(ExecTimeoutManager::Instance().GetExecTimeout(), 0u);
}

// SetExecTimeout to a large value
TEST_F(OpCommonTest, SetExecTimeoutLargeValue)
{
    ExecTimeoutManager::Instance().SetExecTimeout(9999);
    EXPECT_EQ(ExecTimeoutManager::Instance().GetExecTimeout(), 9999u);
}

// SetExecTimeout multiple times, last one wins
TEST_F(OpCommonTest, SetExecTimeoutMultipleLastWins)
{
    ExecTimeoutManager::Instance().SetExecTimeout(100);
    EXPECT_EQ(ExecTimeoutManager::Instance().GetExecTimeout(), 100u);

    ExecTimeoutManager::Instance().SetExecTimeout(200);
    EXPECT_EQ(ExecTimeoutManager::Instance().GetExecTimeout(), 200u);

    ExecTimeoutManager::Instance().SetExecTimeout(300);
    EXPECT_EQ(ExecTimeoutManager::Instance().GetExecTimeout(), 300u);
}

// After SetExecTimeout, value is not the default CUSTOM_TIMEOUT
TEST_F(OpCommonTest, SetExecTimeoutOverridesDefault)
{
    ExecTimeoutManager::Instance().SetExecTimeout(42);
    EXPECT_NE(ExecTimeoutManager::Instance().GetExecTimeout(), CUSTOM_TIMEOUT);
    EXPECT_EQ(ExecTimeoutManager::Instance().GetExecTimeout(), 42u);
}
