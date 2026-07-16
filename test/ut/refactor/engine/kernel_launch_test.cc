/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * kernel_launch free function unit tests.
 * Covers: IsOpsV2, IsResCtxCacheReusable, RestoreVarData*.
 */

#include "test_helpers.h"
#include "kernel_launch.h"

#include <new>
#include <cstring>

namespace ops_hccl {
// Forward declaration — IsOpsV2 is implemented in kernel_launch.cc but not declared in the header.
bool IsOpsV2(const char *algName, DevType deviceType);
} // namespace ops_hccl

namespace ops_hccl {
namespace testing {

// ═══════════════════════════════════════════════════════════════════
// IsOpsV2
// ═══════════════════════════════════════════════════════════════════

TEST(KernelLaunchTest, IsOpsV2PrefixReturnsTrue)
{
    EXPECT_TRUE(IsOpsV2("opv2_somealg", DevType::DEV_TYPE_COUNT));
}

TEST(KernelLaunchTest, IsOpsV2NullptrAlgNameWithNonMatchingDeviceReturnsFalse)
{
    EXPECT_FALSE(IsOpsV2(nullptr, DevType::DEV_TYPE_COUNT));
}

TEST(KernelLaunchTest, IsOpsV2NonPrefixWith91095ReturnsTrue)
{
#ifdef MACRO_DEV_TYPE_NEW
    EXPECT_TRUE(IsOpsV2("somealg", DevType::DEV_TYPE_950));
#else
    EXPECT_TRUE(IsOpsV2("somealg", DevType::DEV_TYPE_910_95));
#endif
}

TEST(KernelLaunchTest, IsOpsV2NonPrefixWithOtherDeviceReturnsFalse)
{
    EXPECT_FALSE(IsOpsV2("somealg", DevType::DEV_TYPE_COUNT));
}

// ═══════════════════════════════════════════════════════════════════
// IsResCtxCacheReusable
// ═══════════════════════════════════════════════════════════════════

TEST(KernelLaunchTest, IsResCtxCacheReusableMatchReturnsTrue)
{
    AlgResourceCtxSerializable resCtx;
    resCtx.commInfoPtr = reinterpret_cast<void *>(0x1000);

    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 64];
    memset(buf, 0, sizeof(buf));
    OpParam *param = new (buf) OpParam();
    param->cacheValid = true;
    param->hcclComm = reinterpret_cast<void *>(0x1000);

    EXPECT_TRUE(IsResCtxCacheReusable(resCtx, *param));
}

TEST(KernelLaunchTest, IsResCtxCacheReusableCacheInvalidReturnsFalse)
{
    AlgResourceCtxSerializable resCtx;
    resCtx.commInfoPtr = reinterpret_cast<void *>(0x1000);

    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 64];
    memset(buf, 0, sizeof(buf));
    OpParam *param = new (buf) OpParam();
    param->cacheValid = false;
    param->hcclComm = reinterpret_cast<void *>(0x1000);

    EXPECT_FALSE(IsResCtxCacheReusable(resCtx, *param));
}

TEST(KernelLaunchTest, IsResCtxCacheReusableCommMismatchReturnsFalse)
{
    AlgResourceCtxSerializable resCtx;
    resCtx.commInfoPtr = reinterpret_cast<void *>(0x1000);

    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 64];
    memset(buf, 0, sizeof(buf));
    OpParam *param = new (buf) OpParam();
    param->cacheValid = true;
    param->hcclComm = reinterpret_cast<void *>(0x2000);

    EXPECT_FALSE(IsResCtxCacheReusable(resCtx, *param));
}

// ═══════════════════════════════════════════════════════════════════
// RestoreVarDataBatchSendRecv
// ═══════════════════════════════════════════════════════════════════

TEST(KernelLaunchTest, RestoreVarDataBatchSendRecvValidSucceeds)
{
    const u32 itemNum = 3;
    u64 varMemSize = itemNum * sizeof(HcclSendRecvItem);

    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    memset(buf, 0, sizeof(buf));
    OpParam *param = new (buf) OpParam();
    param->batchSendRecvDataDes.itemNum = itemNum;
    param->varMemSize = varMemSize;

    EXPECT_EQ(RestoreVarDataBatchSendRecv(*param), HCCL_SUCCESS);
    EXPECT_EQ(param->batchSendRecvDataDes.sendRecvItemsPtr,
              reinterpret_cast<HcclSendRecvItem *>(param->varData));
}

TEST(KernelLaunchTest, RestoreVarDataBatchSendRecvInvalidSizeReturnsPara)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    memset(buf, 0, sizeof(buf));
    OpParam *param = new (buf) OpParam();
    param->batchSendRecvDataDes.itemNum = 3;
    param->varMemSize = 1; // wrong size

    EXPECT_EQ(RestoreVarDataBatchSendRecv(*param), HCCL_E_PARA);
}

// ═══════════════════════════════════════════════════════════════════
// RestoreVarDataAlltoAllV
// ═══════════════════════════════════════════════════════════════════

TEST(KernelLaunchTest, RestoreVarDataAlltoAllVValidSucceeds)
{
    const u32 rankSize = 4;
    u64 varMemSize = ALL_TO_ALL_V_VECTOR_NUM * rankSize * sizeof(u64);

    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    memset(buf, 0, sizeof(buf));
    OpParam *param = new (buf) OpParam();
    param->varMemSize = varMemSize;

    AlgResourceCtxSerializable resCtx;
    resCtx.topoInfo.userRankSize = rankSize;

    EXPECT_EQ(RestoreVarDataAlltoAllV(*param, resCtx), HCCL_SUCCESS);
    EXPECT_EQ(param->all2AllVDataDes.sendCounts,
              reinterpret_cast<void *>(param->varData));
    EXPECT_EQ(param->all2AllVDataDes.recvCounts,
              reinterpret_cast<u64 *>(param->varData) + 1 * rankSize);
    EXPECT_EQ(param->all2AllVDataDes.sdispls,
              reinterpret_cast<u64 *>(param->varData) + 2 * rankSize);
    EXPECT_EQ(param->all2AllVDataDes.rdispls,
              reinterpret_cast<u64 *>(param->varData) + 3 * rankSize);
}

TEST(KernelLaunchTest, RestoreVarDataAlltoAllVInvalidSizeReturnsPara)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    memset(buf, 0, sizeof(buf));
    OpParam *param = new (buf) OpParam();
    param->varMemSize = 1; // wrong size

    AlgResourceCtxSerializable resCtx;
    resCtx.topoInfo.userRankSize = 4;

    EXPECT_EQ(RestoreVarDataAlltoAllV(*param, resCtx), HCCL_E_PARA);
}

// ═══════════════════════════════════════════════════════════════════
// RestoreVarDataReduceScatterV
// ═══════════════════════════════════════════════════════════════════

TEST(KernelLaunchTest, RestoreVarDataReduceScatterVValidSucceeds)
{
    const u32 rankSize = 4;
    u64 varMemSize = REDUCE_SCATTER_V_VECTOR_NUM * rankSize * sizeof(u64);

    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    memset(buf, 0, sizeof(buf));
    OpParam *param = new (buf) OpParam();
    param->varMemSize = varMemSize;

    AlgResourceCtxSerializable resCtx;
    resCtx.topoInfo.userRankSize = rankSize;

    EXPECT_EQ(RestoreVarDataReduceScatterV(*param, resCtx), HCCL_SUCCESS);
    EXPECT_EQ(param->vDataDes.counts,
              reinterpret_cast<void *>(param->varData));
    EXPECT_EQ(param->vDataDes.displs,
              reinterpret_cast<u64 *>(param->varData) + rankSize);
}

TEST(KernelLaunchTest, RestoreVarDataReduceScatterVInvalidSizeReturnsPara)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    memset(buf, 0, sizeof(buf));
    OpParam *param = new (buf) OpParam();
    param->varMemSize = 1; // wrong size

    AlgResourceCtxSerializable resCtx;
    resCtx.topoInfo.userRankSize = 4;

    EXPECT_EQ(RestoreVarDataReduceScatterV(*param, resCtx), HCCL_E_PARA);
}

// ═══════════════════════════════════════════════════════════════════
// RestoreVarDataAllGatherV
// ═══════════════════════════════════════════════════════════════════

TEST(KernelLaunchTest, RestoreVarDataAllGatherVValidSucceeds)
{
    const u32 rankSize = 4;
    u64 varMemSize = ALL_GATHER_V_VECTOR_NUM * rankSize * sizeof(u64);

    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    memset(buf, 0, sizeof(buf));
    OpParam *param = new (buf) OpParam();
    param->varMemSize = varMemSize;

    AlgResourceCtxSerializable resCtx;
    resCtx.topoInfo.userRankSize = rankSize;

    EXPECT_EQ(RestoreVarDataAllGatherV(*param, resCtx), HCCL_SUCCESS);
    EXPECT_EQ(param->vDataDes.counts,
              reinterpret_cast<void *>(param->varData));
    EXPECT_EQ(param->vDataDes.displs,
              reinterpret_cast<u64 *>(param->varData) + rankSize);
}

TEST(KernelLaunchTest, RestoreVarDataAllGatherVInvalidSizeReturnsPara)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    memset(buf, 0, sizeof(buf));
    OpParam *param = new (buf) OpParam();
    param->varMemSize = 1; // wrong size

    AlgResourceCtxSerializable resCtx;
    resCtx.topoInfo.userRankSize = 4;

    EXPECT_EQ(RestoreVarDataAllGatherV(*param, resCtx), HCCL_E_PARA);
}

} // namespace testing
} // namespace ops_hccl
