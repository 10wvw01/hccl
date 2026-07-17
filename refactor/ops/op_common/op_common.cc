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
#include "adapter_error_manager_pub.h"
#include "topo/topo_host.h"
#include "exec_timeout_manager.h"
#include "binary_stream.h"
#include "engine/aicpu/dfx/task_exception_fun.h"
#include "c_adaptor/common/alg_env_config.h"
#include "hcomm_host_profiling_dl.h"

namespace ops_hccl {
constexpr u32 HOST_WAIT_AICPU_NOTIFYIDX = 0;// host主流wait aicpu流的notify idx
constexpr u32 HOST_NOTIFY_TIMEOUT_OFFSET = 27;  // host等待Device通知的超时时间偏移量
constexpr u32 KERNEL_TIMEOUT_OFFSET = 25;       // kernel启动超时时间偏移量
/**
 * 算子执行入口。
 */
HcclResult HcclExecOp(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    HcclAlgorithm &alg, const ResPackGraphMode &resPack)
{
    auto engine = alg.GetEngine();
    auto executor = alg.GetExecutor(param);
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo.get(), algHierarchyInfo));

    AlgResourceRequest resReq;
    CHK_RET(executor->CalcRes(comm, resReq));

    CHK_RET(engine->CreateRes(comm, param, alg, algHierarchyInfo, resReq, *topoInfo));

    CHK_RET(engine->LaunchKernel(param));

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
    char *ctxTemp = reinterpret_cast<char *>(ctx);
    std::vector<char> seq(ctxTemp, ctxTemp + size);
    TopoInfoWithNetLayerDetails topoInfoTemp;
    topoInfoTemp.DeSerialize(seq);
    topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>(std::move(topoInfoTemp));
    HCCL_INFO("[%s] HcclCalcTopoInfo end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CheckAsymmetricTopoSupport(HcclCMDType opType, const TopoInfoWithNetLayerDetails *topoInfo)
{
    if (topoInfo->topoLevelNums > 1 && topoInfo->multiModuleDiffDeviceNumMode) {
        bool isSupportedOp
            = (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_ALLREDUCE
                || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || opType == HcclCMDType::HCCL_CMD_ALLTOALL
                || opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC);
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
        supportList.insert(supportList.end(), {HCCL_DATA_TYPE_BFP16, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64});
    } else {
        supportList.insert(
            supportList.end(), {HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16, HCCL_DATA_TYPE_UINT32,
                                   HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64, HCCL_DATA_TYPE_HIF8,
                                   HCCL_DATA_TYPE_FP8E4M3, HCCL_DATA_TYPE_FP8E5M2, HCCL_DATA_TYPE_FP8E8M0});
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
        if ((dataType == HCCL_DATA_TYPE_UINT8) || (dataType == HCCL_DATA_TYPE_UINT16)
            || (dataType == HCCL_DATA_TYPE_UINT32) || (dataType == HCCL_DATA_TYPE_INT128)
            || (dataType == HCCL_DATA_TYPE_HIF8) || (dataType == HCCL_DATA_TYPE_FP8E4M3)
            || (dataType == HCCL_DATA_TYPE_FP8E5M2) || (dataType == HCCL_DATA_TYPE_FP8E8M0)
            || (dataType == HCCL_DATA_TYPE_RESERVED)) {
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else {
        if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8)
            || (dataType == HCCL_DATA_TYPE_INT128)) {
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

std::string GetReduceProdSupportDataType()
{
    std::vector<HcclDataType> supportList = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64,
                                             HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP64};
    std::string supportInfo = "";
    for (u32 i = 0; i < supportList.size(); i++) {
        if (i != 0) {
            supportInfo += ", ";
        }
        supportInfo += GetDataTypeEnumStr(supportList[i]);
    }

    return supportInfo;
}

HcclResult CheckReduceOp(const HcclDataType dataType, const HcclReduceOp op)
{
    std::vector<HcclDataType> prodSupportList = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64,
                                                 HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP64};
    const std::vector<std::string> infoTitle({"ccl_op", "value", "parameter", "expect"});
    if (op == HcclReduceOp::HCCL_REDUCE_PROD) {
        if (std::find(prodSupportList.begin(), prodSupportList.end(), dataType) == prodSupportList.end()) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, std::vector<std::string>({"CheckReduceDataType", GetDataTypeEnumStr(dataType), "dataType",
                GetReduceProdSupportDataType()}));
            HCCL_ERROR("[Check][ReduceOp][DataType]errNo[0x%016llx] reduceop is [%s] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetReduceOpEnumStr(op).c_str(), GetDataTypeEnumStr(dataType).c_str(),
                        GetReduceProdSupportDataType().c_str());
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
    const char *envValue = std::getenv("HCCL_ENABLE_OPEN_AICPU");
    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return false;
    }
    return true;
}

bool HcclCheckCcuEnableOpen()
{
    const char *envValue = std::getenv("HCCL_CCU_CUSTOM_OP_MODE");
    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return true;
    }
    return false;
}

bool HcclCheckAivEnableOpen()
{
    const char *envValue = std::getenv("HCCL_ENABLE_OPEN_AIV");
    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return false;
    }
    return true;
}

bool ShouldUseInnerOp(OpExecuteConfig opExecuteConfig)
{
    bool isAicpuOrHostMode
        = (opExecuteConfig == OpExecuteConfig::AICPU_TS || opExecuteConfig == OpExecuteConfig::HOSTCPU);
    bool isCcuMode = (opExecuteConfig == OpExecuteConfig::CCU_MS || opExecuteConfig == OpExecuteConfig::CCU_SCHED);
    bool isAivMode = (opExecuteConfig == OpExecuteConfig::AIV || opExecuteConfig == OpExecuteConfig::AIV_ONLY);

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
        std::string endInfo = opName + ":success,take time: " + std::to_string(DURATION_US(endut - startut).count())
                              + " us, tag: " + tag;
        HCCL_RUN_INFO("%s", endInfo.c_str());
    }
    return HCCL_SUCCESS;
}

// ============================================================
// 配置类函数（桩实现）
// ============================================================

HcclResult GetHcclDfxOpInfoDataCount(const OpParam &param, const u32 &rankSize, uint64_t &sendCount) {
    sendCount = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
        sendCount += *(reinterpret_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts)); // 非v类算子，只上报入参里的count
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
        for (u64 i = 0; i < rankSize; i++) { // v类算子上报累加的count
            sendCount += *(reinterpret_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts) + i);
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        CHK_PTR_NULL(param.varData);
        for (u64 i = 0; i < rankSize; i++) {
            sendCount += *(reinterpret_cast<const uint64_t*>(param.varData) + i);
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        CHK_PTR_NULL(param.varData);
        for (u64 i = rankSize; i < 2 * rankSize; i++) {
            sendCount += *(reinterpret_cast<const uint64_t*>(param.varData) + i);
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        for (u32 idx = 0; idx < param.batchSendRecvDataDes.itemNum; idx++) {
            HcclSendRecvItem* item = param.batchSendRecvDataDes.sendRecvItemsPtr + idx;
            CHK_PRT_RET(item == nullptr, HCCL_ERROR("[%s]fail, item is nullptr, idx[%u], itemNum[%u], tag[%s]",
                __func__, idx, param.batchSendRecvDataDes.itemNum, param.tag), HCCL_E_PTR);
            sendCount += item->count;
        }
    } else {
        sendCount = param.DataDes.count;
    }
    HCCL_INFO("[%s]tag[%s], sendCount[%u], opType[%u], rankSize[%u]",
        __func__, param.tag, sendCount, param.opType, rankSize);
    return HCCL_SUCCESS;
}

HcclResult SingleRankProc(HcclComm comm, OpParam &param)
{
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY) {
        HCCL_ERROR("[SingleRankProc] opType[%d] currently do not select aiv mode, aiv only not support, "
            "please ensure rankNum is greater than one", static_cast<int>(param.opType));
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.opType == HcclCMDType::HCCL_CMD_SEND || param.opType == HcclCMDType::HCCL_CMD_RECEIVE) {
        HCCL_WARNING("[%s] ranksize == 1 is not support BATCHSENDRECV SEND RECV", __func__);
        return HcclResult::HCCL_SUCCESS;
    }
    if (param.inputPtr == param.outputPtr) {
        HCCL_WARNING("[%s] sendBuf == recvBuf, return success", __func__);
        return HcclResult::HCCL_SUCCESS;
    }
    u64 len{0};
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        len = DATATYPE_SIZE_TABLE[param.all2AllVDataDes.sendType] * *(static_cast<const u64 *>(param.all2AllVDataDes.sendCounts));
    } else if (param.opType == HCCL_CMD_ALLGATHER_V || param.opType == HCCL_CMD_REDUCE_SCATTER_V) {
        len = DATATYPE_SIZE_TABLE[param.vDataDes.dataType] * *(static_cast<const u64 *>(param.vDataDes.counts));
    } else {
        len = DATATYPE_SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;
    }

    HCCL_INFO("[%s] sendBuf[%p], recvBuf[%p], len[%llu]", __func__, param.inputPtr, param.outputPtr, len);
    if (len > 0) {
        ThreadHandle cpuTsThread{0};
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &cpuTsThread));
        // Op注册
        HcclDfxOpInfoCompat hcclDfxOpInfo{};
        hcclDfxOpInfo.opMode = static_cast<u32>(param.opMode);
        hcclDfxOpInfo.opType = static_cast<u32>(param.opType);
        hcclDfxOpInfo.reduceOp = static_cast<u32>(param.reduceType);
        CHK_RET(GetHcclDfxOpInfoDataType(param, hcclDfxOpInfo.dataType));

        // rankSize获取指定算子的dataCount
        u32 userRankSize{0};
        CHK_RET(HcclGetRankSize(comm, &userRankSize));
        CHK_RET(GetHcclDfxOpInfoDataCount(param, userRankSize, hcclDfxOpInfo.dataCount));
        hcclDfxOpInfo.root = param.root;
        hcclDfxOpInfo.engine = param.engine;
        hcclDfxOpInfo.cpuTsThread = cpuTsThread;
        hcclDfxOpInfo.cpuWaitAicpuNotifyIdx = HOST_WAIT_AICPU_NOTIFYIDX;
        CHK_RET(SetOpParamAlgTag(param, "SingleRankProc"));
        s32 sRet = strncpy_s(hcclDfxOpInfo.algTag, ALG_TAG_LENGTH, param.algTag, ALG_TAG_LENGTH);
        CHK_PRT_RET(sRet != EOK, HCCL_ERROR("%s call strncpy_s failed, param.algTag %s,  return %d.",
            __func__, param.algTag, sRet), HCCL_E_MEMORY);

        CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void*>(&hcclDfxOpInfo)));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(cpuTsThread, param.outputPtr, param.inputPtr, len)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SetExecTimeout(OpParam &param) {
    double execTimeoutValue = 0;
    if (!GetExternalInputExecTimeout(execTimeoutValue)) {
        param.opConfig.execTimeout = CUSTOM_TIMEOUT;
        HCCL_INFO("[OpCommon] Exec timeout is not set, use default value: %u seconds", CUSTOM_TIMEOUT);
    } else {
        // 验证转换后的值是否合理
        if (execTimeoutValue < 0 || execTimeoutValue > UINT32_MAX) {
            HCCL_WARNING("[OpCommon] Exec timeout value %.2f out of range, use default: %u seconds", 
                         execTimeoutValue, CUSTOM_TIMEOUT);
            param.opConfig.execTimeout = CUSTOM_TIMEOUT;
        } else {
            param.opConfig.execTimeout = static_cast<uint32_t>(execTimeoutValue);
            if (param.opConfig.execTimeout == 0) {
                HCCL_INFO("[OpCommon] Exec timeout is disabled (never timeout).");
            } else {
                HCCL_INFO("[OpCommon] Set exec timeout to: %u seconds", param.opConfig.execTimeout);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult SetMultipleDimensionSplitRatio(OpParam &param) {
    double ratioValue = 0;
    const double DEFAULT_MULT_RATIO = 0.5;
    if (!GetExternalInputMultipleDimensionSplitRatio(ratioValue)) {
        param.opConfig.multipleDimensionSplitRatio = DEFAULT_MULT_RATIO;
        HCCL_INFO("[OpCommon] Ratio is not set, use default value: %f", DEFAULT_MULT_RATIO);
    } else {
        // 验证转换后的值是否合理
        if (ratioValue < 0 || ratioValue > 1) {
            HCCL_WARNING("[OpCommon] Ratio value %.2f out of range, use default value: %f", 
                        ratioValue, DEFAULT_MULT_RATIO);
            param.opConfig.multipleDimensionSplitRatio = DEFAULT_MULT_RATIO;
        } else {
            param.opConfig.multipleDimensionSplitRatio = ratioValue;
            HCCL_INFO("[OpCommon] Set ratio to: %f", param.opConfig.multipleDimensionSplitRatio);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CheckHostDPUOnly(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool &hostDPUOnly)
{
    hostDPUOnly = false;
    HCCL_INFO("Start CheckHostDPUOnly");
    // 只有一个server，不使用DPU
    if (topoInfo->serverNum == 1) {
        HCCL_INFO("Not using hostdpu because serverNum is 1");
        return HCCL_SUCCESS;
    }

    // 只有一层topo，不使用DPU
    if (topoInfo->topoLevelNums == 1) {
        HCCL_INFO("Not using hostdpu because topoLevelNums is 1");
        return HCCL_SUCCESS;
    }

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    if ((netLayers == nullptr) || (netLayerNum == 0)) {
        HCCL_WARNING("HcclRankGraphGetLayers fail");
        return HCCL_E_INTERNAL;
    }

    bool hostDPU = false;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; layerIdx++) {
        uint32_t netLayer = netLayers[layerIdx];
        // 只校验最后一个level
        if (netLayer < (topoInfo->topoLevelNums - 1)) {
            HCCL_INFO("Skip checking layer[%u], topoLevelNums is [%u]", netLayer, topoInfo->topoLevelNums);
            continue;
        }
        uint32_t *topoInsts = nullptr;
        uint32_t topoInsNum = 0;
        CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInsNum));
        if ((topoInsts == nullptr) || (topoInsNum == 0)) {
            HCCL_WARNING("HcclRankGraphGetTopoInstsByLayer fail, netLayer[%u]", netLayer);
            return HCCL_E_INTERNAL;
        }
        for (uint32_t topoInsIdx = 0; topoInsIdx < topoInsNum; topoInsIdx++) {
            uint32_t topoInstId = topoInsts[topoInsIdx];
            HCCL_INFO("Start checking topoInstId[%u]", topoInstId);
            CommTopo topoType;
            CHK_RET(HcclRankGraphGetTopoType(comm, netLayer, topoInstId, &topoType));
            if (topoType != COMM_TOPO_CLOS) {
                HCCL_INFO("Not using hostdpu because topo type is not COMM_TOPO_CLOS");
                continue;
            }
            uint32_t *ranks = nullptr;
            uint32_t rankNum = 0;
            CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInstId, &ranks, &rankNum));
            // 校验当前rank与其他所有rank连通
            if (rankNum != topoInfo->userRankSize) {
                HCCL_INFO("Not using hostdpu because current rank is not fully connected to all other ranks");
                continue;
            }
            uint32_t endPointNums = 0;
            CHK_RET(HcclRankGraphGetEndpointNum(comm, netLayer, topoInstId, &endPointNums));
            EndpointDesc endPointDescs[endPointNums];
            CHK_RET(HcclRankGraphGetEndpointDesc(comm, netLayer, topoInstId, &endPointNums, endPointDescs));
            for (uint32_t endPointIdx = 0; endPointIdx < endPointNums; endPointIdx++) {
                EndpointDesc endPointDesc = endPointDescs[endPointIdx];
                if (endPointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                    HCCL_INFO("Not using hostdpu because there is links on device in netLayer[%u] in endPointIdx[%u]",
                        netLayer, endPointIdx);
                    return HCCL_SUCCESS;
                } else if (endPointDesc.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
                    HCCL_INFO("Found a host endPoint in netLayer[%u] endPointIdx[%u]", netLayer, endPointIdx);
                    hostDPU = true;
                }
            }
        }
    }
    if (hostDPU) {
        HCCL_INFO("Using host dpu trans.");
        hostDPUOnly = true;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterKernel()
{
    return HCCL_SUCCESS;
}

static HcclResult BuildCcuExtraTag(const OpParam &param, std::string &ccuExtraTag)
{
    HcclDataType tmpDataType;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        tmpDataType = param.all2AllVDataDes.sendType;
    } else if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V ||
               param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        tmpDataType = param.vDataDes.dataType;
    } else {
        tmpDataType = param.DataDes.dataType;
    }
    ccuExtraTag = "_" + HCOM_DATA_TYPE_STR_MAP.at(tmpDataType);

    if (param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
        param.opType == HcclCMDType::HCCL_CMD_REDUCE ||
        param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
        param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        ccuExtraTag += "_" + HCOM_REDUCE_OP_STR_MAP.at(param.reduceType);
    }

    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE || param.opType == HcclCMDType::HCCL_CMD_SCATTER ||
        param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        ccuExtraTag += "_r" + std::to_string(param.root);
    }
    return HCCL_SUCCESS;
}

HcclResult SetOpParamAlgTag(OpParam &param, const std::string &algName)
{
    std::string temp = algName; // 创建algName的副本

    const char* launchMode = (((param.engine == CommEngine::COMM_ENGINE_AICPU) ||
                                (param.engine == CommEngine::COMM_ENGINE_AICPU_TS)) ? "device" : "host");
    int len;
    // 图模式下去掉param.tag前缀，避免tag不同导致algTag不同而无法复用资源
    if (param.opMode == OpMode::OFFLOAD && param.engine == CommEngine::COMM_ENGINE_CCU) {
        len = snprintf_s(param.algTag, sizeof(param.algTag), sizeof(param.algTag), "Graph_%s_%s", temp.c_str(), launchMode);
    } else {
        len = snprintf_s(param.algTag, sizeof(param.algTag), sizeof(param.algTag), "%s_%s_%s", param.tag, temp.c_str(), launchMode);
    }
    if (len < 0|| len >= sizeof(param.algTag)) {
        HCCL_ERROR("failed to fill param.algTag");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // ccu模式，考虑kernel是否能复用，需要添加dataType和reduceType
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        try{
            std::string ccuExtraTag;
            CHK_RET(BuildCcuExtraTag(param, ccuExtraTag));
            size_t remainBytes = sizeof(param.algTag) - len;

            int len_ccu = snprintf_s(param.algTag + len, remainBytes, remainBytes, "%s", ccuExtraTag.c_str());
            CHK_PRT_RET((len_ccu < 0 || len_ccu >= sizeof(param.algTag) - len),
                HCCL_ERROR("failed to fill alg tag with ccu dataType"), HCCL_E_INTERNAL);
        }
        catch (const std::out_of_range& e) {
            HCCL_ERROR("[SetOpParamAlgTag] dataType or reduceType out of range: %s", e.what());
            return HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}



} // namespace ops_hccl
