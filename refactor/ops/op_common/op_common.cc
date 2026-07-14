/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_common.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
#include <sstream>

#include "ops_executor.h"
#include "hccl_algorithm.h"
#include "base_engine.h"
#include "log.h"
#include "topo/topo_host.h"
#include "exec_timeout_manager.h"
#include "binary_stream.h"

namespace ops_hccl {

/**
 * 算子执行入口。
 */
HcclResult HcclExecOp(HcclComm comm, OpParam &param,
                      std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
                      HcclAlgorithm &alg, const ResPackGraphMode &resPack)
{
    auto engine = alg.GetEngine();
    auto executor = alg.GetExecutor(param);

    CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo.get()));

    AlgResourceCtxSerializable &resCtx = engine->GetResCtx();
    resCtx.algHierarchyInfo = executor->GetAlgHierarchyInfo();

    AlgResourceRequest resReq;
    CHK_RET(executor->CalcRes(resCtx.algHierarchyInfo, resReq));

    CHK_RET(engine->CreateRes(comm, resReq));

    // 将 HcclAlgorithm 序列化到 resCtx，供 device 侧重建 executor
    BinaryStream algoBs;
    alg.SerializeTo(algoBs);
    std::vector<char> algoSerialData;
    algoBs.Dump(algoSerialData);
    resCtx.algoSerialData = std::move(algoSerialData);

    CHK_RET(engine->LaunchKernel(param, resCtx));

    return HCCL_SUCCESS;
}

HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo)
{
    HCCL_INFO("[%s] HcclCalcTopoInfo start.", __func__);
    uint64_t size = 0;
    void *ctx = nullptr;
    HcclResult ret = HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size);
    if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA) {
        CHK_RET(InitRankInfo(comm, topoInfo.get()));
        std::vector<char> seq = topoInfo->Serialize();
        size = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
        CHK_SAFETY_FUNC_RET(memcpy_s(ctx, size, seq.data(), size));
        return HCCL_SUCCESS;
    }
    char *ctxTemp = reinterpret_cast<char*>(ctx);
    std::vector<char> seq(ctxTemp, ctxTemp + size);
    TopoInfoWithNetLayerDetails topoInfoTemp;
    topoInfoTemp.DeSerialize(seq);
    topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>(std::move(topoInfoTemp));
    HCCL_INFO("[%s] HcclCalcTopoInfo end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CheckAsymmetricTopoSupport(HcclCMDType opType, const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (topoInfo->topoLevelNums > 1 && topoInfo->multiModuleDiffDeviceNumMode) {
        bool isSupportedOp = (opType == HcclCMDType::HCCL_CMD_ALLGATHER ||
                             opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
                             opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
                             opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
                             opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
                             opType == HcclCMDType::HCCL_CMD_ALLTOALLVC);
        if (!isSupportedOp) {
            HCCL_ERROR("[CheckAsymmetricTopoSupport] OpType[%d] does not support asymmetric topology", opType);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

// ============================================================
// 参数校验函数
// ============================================================

HcclResult CheckCount(const u64 count)
{
    if (UNLIKELY(count > SYS_MAX_COUNT)) {
        HCCL_ERROR("[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
                    HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

static std::string GetSupportDataType(bool needReduce)
{
    std::vector<HcclDataType> supportList = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32,
                                             HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32};
    if (needReduce) {
        supportList.insert(supportList.end(), {HCCL_DATA_TYPE_BFP16, HCCL_DATA_TYPE_UINT64,
                                               HCCL_DATA_TYPE_FP64});
    } else {
        supportList.insert(supportList.end(), {HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16,
                                               HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64,
                                               HCCL_DATA_TYPE_HIF8, HCCL_DATA_TYPE_FP8E4M3,  HCCL_DATA_TYPE_FP8E5M2,
                                               HCCL_DATA_TYPE_FP8E8M0});
        supportList.push_back(HCCL_DATA_TYPE_BFP16);
    }

    std::string supportInfo = "";
    for (u32 i = 0; i < supportList.size(); i++) {
        if (i != 0) {
            supportInfo += ", ";
        }
        supportInfo += GetDataTypeEnumStr(supportList[i]);
    }

    return supportInfo;
}

HcclResult CheckDataType(const HcclDataType dataType, bool needReduce)
{
    if (needReduce) {
        if ((dataType == HCCL_DATA_TYPE_UINT8)   || (dataType == HCCL_DATA_TYPE_UINT16)  ||
            (dataType == HCCL_DATA_TYPE_UINT32)  || (dataType == HCCL_DATA_TYPE_INT128)  ||
            (dataType == HCCL_DATA_TYPE_HIF8)    || (dataType == HCCL_DATA_TYPE_FP8E4M3) ||
            (dataType == HCCL_DATA_TYPE_FP8E5M2) || (dataType == HCCL_DATA_TYPE_FP8E8M0) ||
            (dataType == HCCL_DATA_TYPE_RESERVED)) {
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                        GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else {
        if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8) ||
            (dataType == HCCL_DATA_TYPE_INT128)) {
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                        GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCheckTag(const char *tag)
{
    CHK_PTR_NULL(tag);

    u32 tagLen = strnlen(tag, TAG_MAX_LEN + 1);
    if (UNLIKELY((tagLen == (TAG_MAX_LEN + 1) || tagLen == 0))) {
        HCCL_ERROR("[Check][Tag]errNo[0x%016llx] tag is too long", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

// ============================================================
// 引擎开关检查（用于 ShouldUseInnerOp）
// ============================================================

bool HcclCheckAicpuEnableOpen()
{
    const char* envValue = std::getenv("HCCL_ENABLE_OPEN_AICPU");
    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return false;
    }
    return true;
}

bool HcclCheckCcuEnableOpen()
{
    const char* envValue = std::getenv("HCCL_CCU_CUSTOM_OP_MODE");
    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return true;
    }
    return false;
}

bool HcclCheckAivEnableOpen()
{
    const char* envValue = std::getenv("HCCL_ENABLE_OPEN_AIV");
    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return false;
    }
    return true;
}

bool ShouldUseInnerOp(OpExecuteConfig opExecuteConfig)
{
    bool isAicpuOrHostMode = (opExecuteConfig == OpExecuteConfig::AICPU_TS ||
                              opExecuteConfig == OpExecuteConfig::HOSTCPU);
    bool isCcuMode = (opExecuteConfig == OpExecuteConfig::CCU_MS ||
                      opExecuteConfig == OpExecuteConfig::CCU_SCHED);
    bool isAivMode = (opExecuteConfig == OpExecuteConfig::AIV ||
                      opExecuteConfig == OpExecuteConfig::AIV_ONLY);

    if (isAicpuOrHostMode) {
        return !HcclCheckAicpuEnableOpen();
    } else if (isCcuMode) {
        return !HcclCheckCcuEnableOpen();
    } else if (isAivMode) {
        return !HcclCheckAivEnableOpen();
    }

    return false;
}

// ============================================================
// DFX / 日志
// ============================================================

HcclResult LogHcclExit(const std::string &opName, const char *tag, HcclUs startut, bool forceLog)
{
    if (forceLog) {
        HcclUs endut = TIME_NOW();
        std::string endInfo = opName + ":success,take time: " +
            std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + tag;
        HCCL_RUN_INFO("%s", endInfo.c_str());
    }
    return HCCL_SUCCESS;
}

// ============================================================
// 配置类函数（桩实现）
// ============================================================

HcclResult SingleRankProc(HcclComm comm, OpParam &param)
{
    // 仅在 rankSize==1 时调用，直接返回成功
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SetExecTimeout(const OpParam &param)
{
    return HCCL_SUCCESS;
}

HcclResult SetMultipleDimensionSplitRatio(const OpParam &param)
{
    return HCCL_SUCCESS;
}

HcclResult CheckHostDPUOnly(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool &hostDPUOnly)
{
    hostDPUOnly = false;
    return HCCL_SUCCESS;
}

HcclResult RegisterKernel()
{
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
