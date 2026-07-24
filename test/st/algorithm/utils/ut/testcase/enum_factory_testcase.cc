/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "enum_factory.h"

MAKE_ENUM(TestEnum, FIRST, SECOND)

TEST(EnumFactoryTest, DescribeReturnsNameForValidValue)
{
    EXPECT_EQ(TestEnum(TestEnum::FIRST).Describe(), "TestEnum::FIRST");
    EXPECT_EQ(TestEnum(TestEnum::SECOND).Describe(), "TestEnum::SECOND");
}

TEST(EnumFactoryTest, DescribeReturnsInvalidForCountSentinel)
{
    EXPECT_EQ(TestEnum(TestEnum::__COUNT__).Describe(), "TestEnum::Invalid");
}

TEST(EnumFactoryTest, DescribeReturnsInvalidForInvalidValue)
{
    EXPECT_EQ(TestEnum(TestEnum::INVALID).Describe(), "TestEnum::Invalid");
}

TEST(EnumFactoryTest, DescribeReturnsInvalidForOutOfRangeValue)
{
    auto value = static_cast<TestEnum::Value>(TestEnum::INVALID + 1);
    EXPECT_EQ(TestEnum(value).Describe(), "TestEnum::Invalid");
}
