/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_op.h"
#include "../op_common/op_common_ops.h"
#include "../selector/execute_selector.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclReduceScatter(
    void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclReduceScatter");

    if (GetHcommVersion() < 90000000) { // compat handle
        return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
#ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
#else
    if (deviceType != DevType::DEV_TYPE_910_95) {
#endif
        return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return reduce scatter success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    std::string opTag;
    CHK_RET(ReduceScatterInitAndCheck(comm, sendBuf, recvBuf, recvCount, dataType, op, stream, opTag));

    CHK_RET(ReduceScatterEntryLog(sendBuf, recvBuf, recvCount, dataType, op, stream, opTag, "HcclReduceScatter"));

    // 执行ReduceScatter
    CHK_RET_AND_PRINT_IDE(
        ReduceScatterOutPlace(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, opTag), opTag.c_str());

    CHK_RET(LogHcclExit("HcclReduceScatter", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatterGraphMode(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, const char *group, aclrtStream stream, const char *tag, void **streams, size_t streamCount,
    void *scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclReduceScatterGraphMode");
    // 根据group获取通信域
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclReduceScatterGraphMode] get group name: %s", group);
    CHK_RET(HcomGetCommHandleByGroup(group, &comm));
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return reduce scatter success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    std::string opTag;
    CHK_RET(ReduceScatterInitAndCheck(comm, sendBuf, recvBuf, recvCount, dataType, op, stream, opTag));

    // 检查tag有效性
    CHK_RET(HcclCheckTag(tag));

    // 拼装ResPackGraphMode
    ResPackGraphMode resPack;
    // 设置tag
    if (strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1) != 0) {
        HCCL_ERROR("failed to fill resPack.tag");
        return HCCL_E_INTERNAL;
    }
    // 设置streams
    if (streams != nullptr && streamCount > 0) {
        for (size_t i = 0; i < streamCount; i++) {
            resPack.streams.push_back(static_cast<aclrtStream>(streams[i]));
        }
    }
    // 设置scratchMem
    resPack.scratchMemAddr = scratchMemAddr;
    resPack.scratchMemSize = scratchMemSize;
    std::string tagStr = tag;

    CHK_RET(ReduceScatterEntryLog(sendBuf, recvBuf, recvCount, dataType, op, stream, opTag, "HcclReduceScatterGraphMode"));

    // 执行ReduceScatter
    CHK_RET_AND_PRINT_IDE(
        ReduceScatterOutPlaceGraphMode(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, tagStr, resPack),
        tagStr.c_str());

    CHK_RET(LogHcclExit("HcclReduceScatterGraphMode", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}
namespace ops_hccl {

HcclResult ReduceScatterInitAndCheck(HcclComm comm, void *sendBuf, void *recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, aclrtStream stream, std::string &opTag)
{
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    CHK_RET(InitEnvConfig());
    // 参数校验等工作
    // 检查入参指针有效性
    CHK_RET(CheckReduceScatterInputPara(comm, sendBuf, recvBuf, stream));
    // tag有效性,是否过长
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    opTag = "ReduceScatter_" + string(commName);
    CHK_RET(HcclCheckTag(opTag.c_str()));
    // 检查recvCount是否合法(超出系统上限)
    CHK_RET(CheckCount(recvCount));
    // 检查数据类型是否支持
    CHK_RET(CheckDataType(dataType, false));
    // 检查reduce op是否支持
    CHK_RET(CheckReduceOp(dataType, op));
    // 检查rank有效性，是否超出rankSize
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult CheckReduceScatterInputPara(
    const HcclComm comm, const void *sendBuf, const void *recvBuf, const aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclReduceScatter", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclReduceScatter", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclReduceScatter", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclReduceScatter", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOutPlaceCommon(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag, OpMode opMode,
    const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute ReduceScatterOutPlaceCommon");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;      // 每个rank上结果为recvCount个数据
    u64 inputSize = outputSize * userRankSize;     // 输入包含所有rank的数据

    OpParam param;
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
    param.opMode = opMode;
    param.reduceType = op;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    // topoInfo的tag，所有相同的算子可以共享
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }

    // 参数准备
    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.DataDes.count = recvCount;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    param.enableDetour = false;
    param.deviceType = deviceType;

    CHK_RET(HcclGetOpExpansionMode(comm, param));

    HcclAlgorithm alg;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, alg));
    if (ShouldUseInnerOp(param.opExecuteConfig)) {
        return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] rankSize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, alg, resPack));
    HCCL_INFO("Execute ReduceScatterOutPlaceCommon success.");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOutPlaceGraphMode(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute ReduceScatterOutPlaceGraphMode");
    CHK_RET(ReduceScatterOutPlaceCommon(
        sendBuf, recvBuf, recvCount, dataType, op, comm, stream, tag, OpMode::OFFLOAD, resPack));
    HCCL_INFO("Execute ReduceScatterOutPlaceGraphMode success.");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOutPlace(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute ReduceScatterOutPlace");
    CHK_RET(ReduceScatterOutPlaceCommon(
        sendBuf, recvBuf, recvCount, dataType, op, comm, stream, tag, OpMode::OPBASE, ResPackGraphMode()));
    HCCL_INFO("Execute ReduceScatterOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterEntryLog(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, aclrtStream stream, const std::string &tag, const std::string &opName)
{
    /* 接口交互信息日志 */
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], reduceOp[%s], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(),
            GetReduceOpEnumStr(op).c_str(), streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
