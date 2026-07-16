/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * DPU kernel launch unit tests.
 * Covers: HcclLaunchDPUKernel error paths and stub return.
 */

#include "test_helpers.h"
#include "dpu/kernel_launch.h"

namespace ops_hccl {
namespace testing {

TEST(DpuKernelLaunchTest, NullPtrReturnsEPtr)
{
    EXPECT_EQ(HcclLaunchDPUKernel(0, 64), static_cast<int32_t>(HCCL_E_PTR));
}

TEST(DpuKernelLaunchTest, ZeroSizeReturnsEPtr)
{
    uint64_t ptr = 0x1000;
    EXPECT_EQ(HcclLaunchDPUKernel(ptr, 0), static_cast<int32_t>(HCCL_E_PTR));
}

TEST(DpuKernelLaunchTest, NegativeSizeReturnsEPtr)
{
    uint64_t ptr = 0x1000;
    EXPECT_EQ(HcclLaunchDPUKernel(ptr, -1), static_cast<int32_t>(HCCL_E_PTR));
}

TEST(DpuKernelLaunchTest, ValidArgsReturnsNotSupport)
{
    uint64_t ptr = 0x1000;
    EXPECT_EQ(HcclLaunchDPUKernel(ptr, 64), static_cast<int32_t>(HCCL_E_NOT_SUPPORT));
}

} // namespace testing
} // namespace ops_hccl
