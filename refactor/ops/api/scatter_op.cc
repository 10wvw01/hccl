/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_op.h"
#include "../op_common/op_common_ops.h"
#include "../selector/execute_selector.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclScatter");

    if (GetHcommVersion() < 90000000) { // compat handle
        return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
#ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
#else
    if (deviceType != DevType::DEV_TYPE_910_95) {
#endif
        return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    }
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return scatter success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    std::string opTag;
    CHK_RET(ScatterInitAndCheck(comm, sendBuf, recvBuf, recvCount, dataType, root, stream, opTag));

    CHK_RET(ScatterEntryLog(sendBuf, recvBuf, recvCount, dataType, root, stream, opTag, "HcclScatter"));

    // 执行Scatter
    CHK_RET_AND_PRINT_IDE(
        ScatterOutPlace(sendBuf, recvBuf, recvCount, dataType, root, comm, stream, opTag), opTag.c_str());

    CHK_RET(LogHcclExit("HcclScatter", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}
namespace ops_hccl {

HcclResult ScatterInitAndCheck(HcclComm comm, void *sendBuf, void *recvBuf, uint64_t recvCount,
    HcclDataType dataType, uint32_t root, aclrtStream stream, std::string &opTag)
{
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    CHK_RET(InitEnvConfig());
    // 参数校验等工作
    // 检查入参指针有效性
    CHK_RET(CheckScatterInputPara(comm, sendBuf, recvBuf, root));
    // tag有效性,是否过长
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    opTag = "Scatter_" + string(commName);
    CHK_RET(HcclCheckTag(opTag.c_str()));
    // 检查recvCount是否合法(超出系统上限)
    CHK_RET(CheckCount(recvCount));
    // 检查数据类型是否支持
    CHK_RET(CheckDataType(dataType, false));
    // 检查rank有效性，是否超出rankSize
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET(HcomCheckUserRank(rankSize, root));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult CheckScatterInputPara(
    const HcclComm comm, const void *sendBuf, const void *recvBuf, uint32_t root)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclScatter", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclScatter", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);
    // root 节点的 sendBuf 不允许为空；非 root 节点 sendBuf 可为空
    u32 userRank = INVALID_VALUE_RANKID;
    if (HcclGetRankId(comm, &userRank) == HCCL_SUCCESS && userRank == root) {
        RPT_INPUT_ERR(sendBuf == nullptr, "EI0003",
            std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
            std::vector<std::string>({"HcclScatter", "nullptr", "sendBuf", "non-null pointer for root rank"}));
        CHK_PTR_NULL(sendBuf);
    }
    (void)root;
    return HCCL_SUCCESS;
}

HcclResult ScatterOutPlace(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute ScatterOutPlace");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;      // 每个rank上结果为recvCount个数据
    u64 inputSize = outputSize * userRankSize;    // 输入包含所有rank的数据（root 持有全部）

    OpParam param;
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
    param.opMode = OpMode::OPBASE;
    param.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED; // scatter 不做归约

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
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    param.enableDetour = false;
    param.deviceType = deviceType;

    CHK_RET(HcclGetOpExpansionMode(comm, param));

    HcclAlgorithm alg;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, alg));
    if (ShouldUseInnerOp(param.opExecuteConfig)) {
        return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    }
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] rankSize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, alg, ResPackGraphMode()));
    HCCL_INFO("Execute ScatterOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult ScatterEntryLog(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    aclrtStream stream, const std::string &tag, const std::string &opName)
{
    /* 接口交互信息日志 */
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], root[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), root, streamId,
            deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
