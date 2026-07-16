/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * DFX task exception function unit tests.
 * Covers: CreateScatter, GetScatterOpInfo, GetHcclDfxOpInfoDataType, ConvertToHcclDfxOpInfo.
 */

#include "test_helpers.h"
#include "dfx/task_exception_fun.h"

#include <new>
#include <cstring>
#include <string>

namespace ops_hccl {
namespace testing {

// Helper: create a valid OpParam via placement new on a buffer large enough for varData.
static OpParam *MakeOpParam(uint8_t *buf, size_t bufSize)
{
    memset(buf, 0, bufSize);
    return new (buf) OpParam();
}

// ═══════════════════════════════════════════════════════════════════
// CreateScatter
// ═══════════════════════════════════════════════════════════════════

TEST(DfxTaskExceptionTest, CreateScatterValidSetsFields)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    OpParam *param = MakeOpParam(buf, sizeof(buf));
    strncpy(param->algTag, "test_alg_tag", ALG_TAG_LENGTH);
    strncpy(param->commName, "test_comm", COMM_INDENTIFIER_MAX_LENGTH);
    param->DataDes.count = 100;
    param->DataDes.dataType = HCCL_DATA_TYPE_INT32;
    param->opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param->root = 2;
    param->inputPtr = reinterpret_cast<void *>(0x2000);
    param->outputPtr = reinterpret_cast<void *>(0x3000);

    ScatterOpInfo opInfo;
    EXPECT_EQ(CreateScatter(param, &opInfo), HCCL_SUCCESS);
    EXPECT_STREQ(opInfo.algTag, "test_alg_tag");
    EXPECT_STREQ(opInfo.commName, "test_comm");
    EXPECT_EQ(opInfo.count, 100u);
    EXPECT_EQ(opInfo.dataType, HCCL_DATA_TYPE_INT32);
    EXPECT_EQ(opInfo.opType, HcclCMDType::HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(opInfo.root, 2u);
    EXPECT_EQ(opInfo.inputPtr, reinterpret_cast<void *>(0x2000));
    EXPECT_EQ(opInfo.outputPtr, reinterpret_cast<void *>(0x3000));
}

TEST(DfxTaskExceptionTest, CreateScatterNullParamReturnsEPtr)
{
    ScatterOpInfo opInfo;
    EXPECT_EQ(CreateScatter(nullptr, &opInfo), HCCL_E_PTR);
}

TEST(DfxTaskExceptionTest, CreateScatterNullOpInfoReturnsEPtr)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    OpParam *param = MakeOpParam(buf, sizeof(buf));
    EXPECT_EQ(CreateScatter(param, nullptr), HCCL_E_PTR);
}

// ═══════════════════════════════════════════════════════════════════
// GetScatterOpInfo
// ═══════════════════════════════════════════════════════════════════

TEST(DfxTaskExceptionTest, GetScatterOpInfoContainsExpectedFields)
{
    ScatterOpInfo opInfo;
    memset(&opInfo, 0, sizeof(opInfo));
    strncpy(opInfo.algTag, "my_tag", ALG_TAG_LENGTH);
    strncpy(opInfo.commName, "my_group", COMM_INDENTIFIER_MAX_LENGTH);
    opInfo.count = 42;
    opInfo.dataType = HCCL_DATA_TYPE_INT32;
    opInfo.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    opInfo.root = 1;
    opInfo.inputPtr = reinterpret_cast<void *>(0x4000);
    opInfo.outputPtr = reinterpret_cast<void *>(0x5000);

    char outPut[1024] = {0};
    GetScatterOpInfo(&opInfo, outPut, sizeof(outPut));
    std::string result(outPut);
    EXPECT_NE(result.find("tag:"), std::string::npos);
    EXPECT_NE(result.find("my_tag"), std::string::npos);
    EXPECT_NE(result.find("group:"), std::string::npos);
    EXPECT_NE(result.find("my_group"), std::string::npos);
    EXPECT_NE(result.find("count:"), std::string::npos);
    EXPECT_NE(result.find("42"), std::string::npos);
    EXPECT_NE(result.find("dataType:"), std::string::npos);
    EXPECT_NE(result.find("opType:"), std::string::npos);
    EXPECT_NE(result.find("rootId:"), std::string::npos);
    EXPECT_NE(result.find("dstAddr:"), std::string::npos);
    EXPECT_NE(result.find("srcAddr:"), std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════
// GetHcclDfxOpInfoDataType
// ═══════════════════════════════════════════════════════════════════

TEST(DfxTaskExceptionTest, GetDataTypeReduceScatterVReturnsVDataDesType)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    OpParam *param = MakeOpParam(buf, sizeof(buf));
    param->opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    param->vDataDes.dataType = HCCL_DATA_TYPE_FP16;

    uint32_t dataType = 0;
    EXPECT_EQ(GetHcclDfxOpInfoDataType(*param, dataType), HCCL_SUCCESS);
    EXPECT_EQ(dataType, static_cast<uint32_t>(HCCL_DATA_TYPE_FP16));
}

TEST(DfxTaskExceptionTest, GetDataTypeAllGatherVReturnsVDataDesType)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    OpParam *param = MakeOpParam(buf, sizeof(buf));
    param->opType = HcclCMDType::HCCL_CMD_ALLGATHER_V;
    param->vDataDes.dataType = HCCL_DATA_TYPE_INT64;

    uint32_t dataType = 0;
    EXPECT_EQ(GetHcclDfxOpInfoDataType(*param, dataType), HCCL_SUCCESS);
    EXPECT_EQ(dataType, static_cast<uint32_t>(HCCL_DATA_TYPE_INT64));
}

TEST(DfxTaskExceptionTest, GetDataTypeAlltoallReturnsSendType)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    OpParam *param = MakeOpParam(buf, sizeof(buf));
    param->opType = HcclCMDType::HCCL_CMD_ALLTOALL;
    param->all2AllVDataDes.sendType = HCCL_DATA_TYPE_UINT8;

    uint32_t dataType = 0;
    EXPECT_EQ(GetHcclDfxOpInfoDataType(*param, dataType), HCCL_SUCCESS);
    EXPECT_EQ(dataType, static_cast<uint32_t>(HCCL_DATA_TYPE_UINT8));
}

TEST(DfxTaskExceptionTest, GetDataTypeAlltoallVReturnsSendType)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    OpParam *param = MakeOpParam(buf, sizeof(buf));
    param->opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    param->all2AllVDataDes.sendType = HCCL_DATA_TYPE_UINT32;

    uint32_t dataType = 0;
    EXPECT_EQ(GetHcclDfxOpInfoDataType(*param, dataType), HCCL_SUCCESS);
    EXPECT_EQ(dataType, static_cast<uint32_t>(HCCL_DATA_TYPE_UINT32));
}

TEST(DfxTaskExceptionTest, GetDataTypeAlltoallVCReturnsSendType)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    OpParam *param = MakeOpParam(buf, sizeof(buf));
    param->opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
    param->all2AllVDataDes.sendType = HCCL_DATA_TYPE_UINT16;

    uint32_t dataType = 0;
    EXPECT_EQ(GetHcclDfxOpInfoDataType(*param, dataType), HCCL_SUCCESS);
    EXPECT_EQ(dataType, static_cast<uint32_t>(HCCL_DATA_TYPE_UINT16));
}

TEST(DfxTaskExceptionTest, GetDataTypeDefaultReturnsDataDesType)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    OpParam *param = MakeOpParam(buf, sizeof(buf));
    param->opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param->DataDes.dataType = HCCL_DATA_TYPE_FP32;

    uint32_t dataType = 0;
    EXPECT_EQ(GetHcclDfxOpInfoDataType(*param, dataType), HCCL_SUCCESS);
    EXPECT_EQ(dataType, static_cast<uint32_t>(HCCL_DATA_TYPE_FP32));
}

TEST(DfxTaskExceptionTest, GetDataTypeBatchSendRecvWithItemNumZeroReturnsSuccess)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    OpParam *param = MakeOpParam(buf, sizeof(buf));
    param->opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    param->batchSendRecvDataDes.itemNum = 0;

    uint32_t dataType = 99;
    EXPECT_EQ(GetHcclDfxOpInfoDataType(*param, dataType), HCCL_SUCCESS);
    EXPECT_EQ(dataType, 0u);
}

// ═══════════════════════════════════════════════════════════════════
// ConvertToHcclDfxOpInfo
// ═══════════════════════════════════════════════════════════════════

TEST(DfxTaskExceptionTest, ConvertToHcclDfxOpInfoValidSetsFields)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    OpParam *param = MakeOpParam(buf, sizeof(buf));
    strncpy(param->algTag, "dfx_alg_tag", ALG_TAG_LENGTH);
    param->opMode = OpMode::OPBASE;
    param->opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param->reduceType = HCCL_REDUCE_SUM;
    param->DataDes.dataType = HCCL_DATA_TYPE_INT32;
    param->dataCount = 1024;
    param->root = 3;
    param->engine = CommEngine::COMM_ENGINE_AICPU;
    param->opThread = 0x1234;
    param->aicpuRecordCpuIdx = 5;
    param->inputPtr = reinterpret_cast<void *>(0x6000);
    param->inputSize = 4096;
    param->outputPtr = reinterpret_cast<void *>(0x7000);
    param->outputSize = 8192;

    HcclDfxOpInfoCompat dfxInfo;
    EXPECT_EQ(ConvertToHcclDfxOpInfo(param, &dfxInfo), HCCL_SUCCESS);
    EXPECT_EQ(dfxInfo.opMode, static_cast<u32>(OpMode::OPBASE));
    EXPECT_EQ(dfxInfo.opType, static_cast<u32>(HcclCMDType::HCCL_CMD_ALLREDUCE));
    EXPECT_EQ(dfxInfo.reduceOp, static_cast<u32>(HCCL_REDUCE_SUM));
    EXPECT_EQ(dfxInfo.dataType, static_cast<uint32_t>(HCCL_DATA_TYPE_INT32));
    EXPECT_EQ(dfxInfo.dataCount, 1024u);
    EXPECT_EQ(dfxInfo.root, 3u);
    EXPECT_EQ(dfxInfo.engine, CommEngine::COMM_ENGINE_AICPU);
    EXPECT_EQ(dfxInfo.cpuTsThread, 0x1234u);
    EXPECT_EQ(dfxInfo.cpuWaitAicpuNotifyIdx, 5u);
    EXPECT_EQ(dfxInfo.inputMemAddr, static_cast<uint64_t>(0x6000));
    EXPECT_EQ(dfxInfo.inputMemSize, 4096u);
    EXPECT_EQ(dfxInfo.outputMemAddr, static_cast<uint64_t>(0x7000));
    EXPECT_EQ(dfxInfo.outputMemSize, 8192u);
    EXPECT_STREQ(dfxInfo.algTag, "dfx_alg_tag");
}

TEST(DfxTaskExceptionTest, ConvertToHcclDfxOpInfoNullParamReturnsEPtr)
{
    HcclDfxOpInfoCompat dfxInfo;
    EXPECT_EQ(ConvertToHcclDfxOpInfo(nullptr, &dfxInfo), HCCL_E_PTR);
}

TEST(DfxTaskExceptionTest, ConvertToHcclDfxOpInfoNullDfxInfoReturnsEPtr)
{
    alignas(alignof(OpParam)) uint8_t buf[sizeof(OpParam) + 256];
    OpParam *param = MakeOpParam(buf, sizeof(buf));
    EXPECT_EQ(ConvertToHcclDfxOpInfo(param, nullptr), HCCL_E_PTR);
}

} // namespace testing
} // namespace ops_hccl
