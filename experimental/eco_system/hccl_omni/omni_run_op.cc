/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "omni_run_op.h"
#include "param_check.h"

#include "op_common_ops.h"
#include <algorithm>
#include <cstdint>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

// ---------------------------------------------------------------------------
// OpParamBlob: C++ view of the Python-serialized opParam binary layout.
//
// Layout (4104 bytes, little-endian, matches Python op_param.py):
//   byte 0:    op_name (4b low) + reduce_op (4b high)
//   bytes 1-3: reserved
//   bytes 4-7: data_count (uint32 LE)
//   bytes 8-1031:   send_counts (int64[128] LE)
//   bytes 1032-2055: recv_counts (int64[128] LE)
//   bytes 2056-3079: sdispls     (int64[128] LE)
//   bytes 3080-4103: rdispls     (int64[128] LE)
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct OpParamBlob {
    uint8_t opNameReduceOp; // low 4 bits = op_name, high 4 bits = reduce_op
    uint8_t reserved[3];
    uint32_t dataCount; // little-endian
    int64_t sendCounts[128];
    int64_t recvCounts[128];
    int64_t sdispls[128];
    int64_t rdispls[128];
};
#pragma pack(pop)
static_assert(sizeof(OpParamBlob) == 4104, "OpParamBlob must be 4104 bytes");

constexpr uint64_t OP_PARAM_BLOB_SIZE = sizeof(OpParamBlob);

// ---------------------------------------------------------------------------
// Local helper functions (file-scoped, no link conflict with all_to_all_v)
// Migrated from all_to_all_v_op.cc to decouple omni_run from all_to_all_v.
// ---------------------------------------------------------------------------
namespace {

constexpr u64 SEND_COUNT_IDX = 0;
constexpr u64 RECV_COUNT_IDX = 1;
constexpr u64 SEND_DISPL_IDX = 2;
constexpr u64 RECV_DISPL_IDX = 3;

HcclResult CalcInputOutputSize(const u64 *sendCountsData, const u64 *recvCountsData, const u64 *sdisplsData,
    const u64 *rdisplsData, const u32 userRankSize, u64 &inputSize, u64 &outputSize)
{
    for (u64 i = 0; i < userRankSize; i++) {
        u64 tmpInputSize = sdisplsData[i] + sendCountsData[i];
        u64 tmpOutputSize = rdisplsData[i] + recvCountsData[i];
        if (tmpInputSize > inputSize) {
            inputSize = tmpInputSize;
        }
        if (tmpOutputSize > outputSize) {
            outputSize = tmpOutputSize;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ContructVarData(const u64 *sendCountsData, const u64 *recvCountsData, const u64 *sdisplsData,
    const u64 *rdisplsData, const u32 userRankSize, const u32 rankSize, OpParam &param)
{
    CHK_PTR_NULL(param.varData);
    u64 *data = reinterpret_cast<u64 *>(param.varData);
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * userRankSize; i++) {
        u64 val = i / rankSize;
        switch (val) {
            case SEND_COUNT_IDX:
                data[i] = sendCountsData[i % rankSize];
                break;
            case RECV_COUNT_IDX:
                data[i] = recvCountsData[i % rankSize];
                break;
            case SEND_DISPL_IDX:
                data[i] = sdisplsData[i % rankSize];
                break;
            case RECV_DISPL_IDX:
                data[i] = rdisplsData[i % rankSize];
                break;
            default:
                break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CheckBufNullptr(
    const u64 *countsData, u32 rankSize, const void *buf, const std::string funcName, const std::string bufName)
{
    bool zeroFlag = true;
    for (u32 i = 0; i < rankSize; i++) {
        if (countsData[i] != 0) {
            zeroFlag = false;
            break;
        }
    }
    if (zeroFlag) {
        return HCCL_SUCCESS;
    }
    RPT_INPUT_ERR(buf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({funcName, "nullptr", bufName, "non-null pointer"}));
    CHK_PTR_NULL(buf);
    return HCCL_SUCCESS;
}

HcclResult CheckOmniRunInputPara(const HcclComm comm, const void *sendBuf, const void *sendCounts, const void *sdispls,
    const HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls,
    const HcclDataType recvType, const aclrtStream stream)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclOmniRun", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclOmniRun", "nullptr", "sendCounts", "non-null pointer"}));
    CHK_PTR_NULL(sendCounts);
    RPT_INPUT_ERR(sdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclOmniRun", "nullptr", "sdispls", "non-null pointer"}));
    CHK_PTR_NULL(sdispls);
    RPT_INPUT_ERR(recvCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclOmniRun", "nullptr", "recvCounts", "non-null pointer"}));
    CHK_PTR_NULL(recvCounts);
    RPT_INPUT_ERR(rdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclOmniRun", "nullptr", "rdispls", "non-null pointer"}));
    CHK_PTR_NULL(rdispls);
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclOmniRun", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    return HCCL_SUCCESS;
}

HcclResult OmniRunConstructOpParam(const void *sendBuf, const void *sendCounts, const void *sdispls,
    const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclComm comm,
    aclrtStream stream, const std::string &tag, HcclCMDType opType, u32 rankSize, OpMode opMode, u64 varMemSize,
    OpParam &param, const char *xmlPath, u32 omniDataCount)
{
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
    param.opMode = opMode;
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    param.deviceType = deviceType;

    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }

    param.inputPtr = const_cast<void *>(sendBuf);
    param.outputPtr = const_cast<void *>(recvBuf);
    param.varMemSize = varMemSize;
    param.dataCount = omniDataCount;
    param.all2AllVDataDes.sendType = dataType;
    param.all2AllVDataDes.recvType = dataType;

    const u64 *sendCountsData = static_cast<const u64 *>(sendCounts);
    const u64 *recvCountsData = static_cast<const u64 *>(recvCounts);
    const u64 *sdisplsData = static_cast<const u64 *>(sdispls);
    const u64 *rdisplsData = static_cast<const u64 *>(rdispls);
    u64 inputSize = 0;
    u64 outputSize = 0;
    CHK_RET(
        CalcInputOutputSize(sendCountsData, recvCountsData, sdisplsData, rdisplsData, rankSize, inputSize, outputSize));
    param.inputSize = inputSize;
    param.outputSize = outputSize;

    param.enableDetour = false;
    param.opType = opType;

    CHK_RET(ContructVarData(sendCountsData, recvCountsData, sdisplsData, rdisplsData, rankSize, rankSize, param));
    u64 *data = reinterpret_cast<u64 *>(param.varData);
    param.all2AllVDataDes.sendCounts = data;
    param.all2AllVDataDes.recvCounts = data + RECV_COUNT_IDX * rankSize;
    param.all2AllVDataDes.sdispls = data + SEND_DISPL_IDX * rankSize;
    param.all2AllVDataDes.rdispls = data + RECV_DISPL_IDX * rankSize;

    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize; i++) {
        HCCL_INFO("[OmniRunConstructOpParam] varData[%u] is [%u]", i, data[i]);
    }
    HCCL_INFO("[OmniRunConstructOpParam] DATATYPE_SIZE_TABLE[dataType] is [%u]", DATATYPE_SIZE_TABLE[dataType]);

    return HCCL_SUCCESS;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// OmniRunOutPlaceCommon: dedicated execution flow for HcclOmniRun.
// Migrated from AlltoAllVOutPlaceCommon, with the following differences:
// - No useInnerOp / HcclAlltoAllV fallback (OmniRun is standalone)
// - Uses OmniRunAutoSelector (priority 17) via the standard Selector() path
// ---------------------------------------------------------------------------
namespace ops_hccl {

HcclResult OmniRunOutPlaceCommon(const void *sendBuf, const void *sendCounts, const void *sdispls, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string &tag, HcclCMDType opType, u32 rankSize, OpMode opMode, const ResPackGraphMode &resPack,
    const char *xmlPath, u32 omniDataCount)
{
    u64 varMemSize = ALL_TO_ALL_V_VECTOR_NUM * rankSize * sizeof(u64);
    void *paramMem = malloc(sizeof(OpParam) + varMemSize);
    if (!paramMem) {
        HCCL_ERROR("[OmniRunOutPlaceCommon] malloc OpParam failed!");
        return HCCL_E_INTERNAL;
    }
    OpParam *tmpParamPtr = new (paramMem) OpParam();
    auto deleter = [](OpParam *p) {
        if (p) {
            p->~OpParam();
            free(p);
        }
    };
    std::unique_ptr<OpParam, decltype(deleter)> paramPtr(tmpParamPtr, deleter);
    OpParam &param = *paramPtr;

    CHK_RET(OmniRunConstructOpParam(sendBuf, sendCounts, sdispls, recvBuf, recvCounts, rdispls, dataType, comm, stream,
        tag, opType, rankSize, opMode, varMemSize, param, xmlPath, omniDataCount));

    CHK_RET(HcclGetOpExpansionMode(comm, param));

    CcuFastLaunchCtx *ccuFastLaunchCtx = nullptr;
    if (ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
        return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);
    }

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_PTR_NULL(topoInfo);
    CHK_RET(Selector(comm, param, topoInfo, algName));

    HCCL_INFO("[OmniRunOutPlaceCommon] Selector picked algName=%s", algName.c_str());

    if (rankSize == 1) {
        HCCL_WARNING("[%s] rankSize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HCCL_SUCCESS;
    }

    // 解析xml
    if (xmlPath[0] != '\0') {
        omni::BinaryParser parser{};
        std::string fileName(xmlPath);
        HCCL_INFO("[%s] parsing OMNI config from xmlPath: %s", __func__, fileName.c_str());
        CHK_RET(parser.SetFile(fileName, topoInfo->userRank));
        CHK_RET(parser.Parse(topoInfo->xmlInfo));
        // topoInfo->xmlInfo.toString();
        std::cout << topoInfo->xmlInfo.toString() << std::endl;
    }

    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    return HCCL_SUCCESS;
}

} // namespace ops_hccl

// ---------------------------------------------------------------------------
// HcclOmniRun: C API entry point (extern "C" for dlsym/dlopen resolution)
// ---------------------------------------------------------------------------
extern "C" HcclResult HcclOmniRun(const void *sendBuf, const void *recvBuf, HcclDataType sendType,
    HcclDataType recvType, const char *xmlPath, const void *opParam, uint64_t opParamSize, HcclComm comm,
    aclrtStream stream)
{
    // Validate opParam blob
    if (opParam == nullptr || opParamSize < OP_PARAM_BLOB_SIZE) {
        HCCL_ERROR(
            "[HcclOmniRun] opParam is null or too small (size=%lu, required=%lu)", opParamSize, OP_PARAM_BLOB_SIZE);
        return HCCL_E_PARA;
    }

    const OpParamBlob *blob = reinterpret_cast<const OpParamBlob *>(opParam);
    const void *sendCounts = blob->sendCounts;
    const void *recvCounts = blob->recvCounts;
    const void *sdisplsPtr = blob->sdispls;
    const void *rdisplsPtr = blob->rdispls;
    uint8_t opName = blob->opNameReduceOp & 0x0F;

    HCCL_INFO("[HcclOmniRun] Start. xmlPath=%s opName=%u dataCount=%u opParamSize=%lu",
        (xmlPath != nullptr) ? xmlPath : "(null)", opName, blob->dataCount, opParamSize);

    // Device and version compatibility checks
    if (GetHcommVersion() < 90000000) {
        HCCL_ERROR("[HcclOmniRun] hcomm version too old, not supported");
        return HCCL_E_PARA;
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
#ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
#else
    if (deviceType != DevType::DEV_TYPE_910_95) {
#endif
        HCCL_ERROR("[HcclOmniRun] device type not supported");
        return HCCL_E_PARA;
    }

    HcclUs startut = TIME_NOW();
    CHK_RET(InitEnvConfig());

    // Parameter validation
    CHK_RET(CheckOmniRunInputPara(
        comm, sendBuf, sendCounts, sdisplsPtr, sendType, recvBuf, recvCounts, rdisplsPtr, recvType, stream));

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));

    // Use OMNIRUN_ prefix — OmniRunAutoSelector (priority 17) matches this
    const string tag = "OMNIRUN_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));

    // Buffer null checks
    CHK_RET(CheckBufNullptr(reinterpret_cast<const u64 *>(sendCounts), rankSize, sendBuf, string(__func__), "sendBuf"));
    CHK_RET(CheckBufNullptr(reinterpret_cast<const u64 *>(recvCounts), rankSize, recvBuf, string(__func__), "recvBuf"));

    // Count and type validation
    u64 maxSendRecvCount = 0;
    for (u64 i = 0; i < rankSize; i++) {
        maxSendRecvCount = max(maxSendRecvCount, static_cast<const u64 *>(sendCounts)[i]);
        maxSendRecvCount = max(maxSendRecvCount, static_cast<const u64 *>(recvCounts)[i]);
    }
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET(CheckCount(maxSendRecvCount));
    CHK_RET(CheckDataType(recvType, false));

    // Execute via dedicated OmniRun flow (not AlltoAllVOutPlaceCommon)
    CHK_RET_AND_PRINT_IDE(ops_hccl::OmniRunOutPlaceCommon(sendBuf, sendCounts, sdisplsPtr, recvBuf, recvCounts,
                              rdisplsPtr, recvType, comm, stream, tag, HcclCMDType::HCCL_CMD_ALLTOALLV, rankSize,
                              OpMode::OPBASE, ResPackGraphMode(), xmlPath, blob->dataCount),
        tag.c_str());

    CHK_RET(LogHcclExit("HcclOmniRun", tag.c_str(), startut));

    HCCL_INFO("[HcclOmniRun] Exit success");
    return HCCL_SUCCESS;
}
