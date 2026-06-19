/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_op_experimental.h"
#include "reduce_scatter_op_experimental.h"
#include "topo_host.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>
#include "op_common_experimental.h"
#include "param_check.h"
#include "all_gather_op.h"
#include "load_kernel.h"
#include "hcomm_dlsym.h"
#include "hcomm_host_profiling_dl.h"

using namespace std;

extern "C" HcclResult HcclAllGatherInner(void *sendBuf, void *recvBuf, uint64_t count,
    HcclDataType dataType, HcclComm comm, aclrtStream stream);

namespace ops_hccl_experimental {
using ops_hccl::AllGatherInitAndCheck;
using ops_hccl::AllGatherEntryLog;
using ops_hccl::LogHcclExit;
using ops_hccl::OpMode;
using ops_hccl::DATATYPE_SIZE_TABLE;
using ops_hccl::LoadAICPUKernel;

HcclResult AllGatherExperimental(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                         HcclComm comm, aclrtStream stream)
{
    if (!MatchBIRS()) {
        return HcclAllGatherInner(sendBuf, recvBuf, sendCount, dataType, comm, stream);
    }

    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input sendCount is 0, return all gather success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    std::string opTag;
    CHK_RET(AllGatherInitAndCheck(comm, sendBuf, recvBuf, sendCount, dataType, stream, opTag));

    CHK_RET(AllGatherEntryLog(sendBuf, recvBuf, sendCount, dataType, stream, opTag, "HcclAllGather"));

    CHK_RET(AllGatherOutPlaceCustom(sendBuf, recvBuf, sendCount, dataType, comm, stream, opTag));

    CHK_RET(LogHcclExit("HcclAllGather", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}

static HcclResult PrepareAllGatherParam(OpParam &param, void *sendBuf, void *recvBuf, uint64_t sendCount,
    HcclDataType dataType, HcclComm comm, aclrtStream stream, u32 userRankSize,
 	OpMode opMode)
{
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 inputSize = sendCount * perDataSize;    // all gather 每个rank上一份数据
    u64 outputSize = inputSize * userRankSize;  // 每个卡上结果为rankSize份数据

    param.stream = stream;
    param.opMode = opMode;

    if (param.commName[0] == '\0') {
        CHK_RET(HcclGetCommName(comm, param.commName));
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.DataDes.count = sendCount;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    param.enableDetour = false;
    param.deviceType = deviceType;

    return HCCL_SUCCESS;
}

HcclResult AllGatherOutPlaceCustom(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                                      aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute AllGatherOutPlaceCustom");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    OpParam param;
    CHK_RET(PrepareAllGatherParam(param, sendBuf, recvBuf, sendCount, dataType, comm, stream, userRankSize,
 	    OpMode::OPBASE));
    
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }

    if (IsAiCpuMode(param.deviceType, userRankSize)) {
        HCCL_DEBUG("is aicpu mode");
        CHK_RET(LoadAICPUKernel());
        param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    } else {
        HCCL_DEBUG("is host mode");
        param.engine = CommEngine::COMM_ENGINE_CPU_TS;
    }

    uint64_t beginTime;
    if (HcommIsProfilingSupported()) {
        beginTime = HcommGetProfilingSysCycleTime();
    }
    
    CHK_RET(ProcessA3(comm, param, beginTime));

    HCCL_INFO("Execute AllGatherOutPlaceCustom success.");
    return HCCL_SUCCESS;
}

}