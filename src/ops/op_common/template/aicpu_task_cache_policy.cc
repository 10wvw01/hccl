/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_task_cache_policy.h"
#include "alg_env_config.h"
#include "log.h"

namespace ops_hccl {

HcclResult AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(const OpParam &param, const TopoInfoWithNetLayerDetails &topoInfo, 
    const AlgResourceCtxSerializable &resCtxHost, bool isCapture, bool &isCacheEnable)
{
    isCacheEnable = false;
    if (!GetExternalInputHcclAicpuCacheEnable()) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] AICPU_CacheDisable is not supported");
        return HCCL_SUCCESS;
    }

    // MC2算子不支持
    if (param.isMc2) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] mc2 is not supported");
        return HCCL_SUCCESS;
    }

    // 图模式不支持
    if (param.opMode == OpMode::OFFLOAD) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] OpMode::OFFLOAD is not supported");
        return HCCL_SUCCESS;
    }

    // aclgraph 不支持
    if (isCapture) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] aclgraph is not supported");
        return HCCL_SUCCESS;
    }

    // 校验算子类型
    if (!IsOpTypeSupported(param.opType)) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] opType[%d] is not supported", param.opType);
        return HCCL_SUCCESS;
    }

    // 屏蔽inplace场景
    bool isInplace = false;
    CHK_RET(IsInplaceForCache(param, isInplace, topoInfo));
    if (isInplace) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] inplace case is not supported for operator unfolding "
                  "cache");
        return HCCL_SUCCESS;
    }

    if(!IsTopoSupported(resCtxHost)) {
        return HCCL_SUCCESS;
    }

    // 所有使能判断都通过是返回true
    isCacheEnable = true;
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCachePolicy::IsInplaceForCache(const OpParam &param, bool &isInplace,
    const TopoInfoWithNetLayerDetails &topoInfo)
{
    // 准备input/output size
    HcclDataType sendType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    HcclDataType recvType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CHK_RET(ParseOpParamForCache(param, sendType, recvType, inputSize, outputSize, topoInfo));
    UNUSED_PARAM(sendType);
    UNUSED_PARAM(recvType);

    // 注意: A3下alltoall/alltoallv/alltoallvc可能存在inputSize/outputSize为0的情况, 导致不分配user input/output
    //     但会使用tinySendRecvMem_更新algResource.paramInput/OutputMem用于建链, 导致cache无法区分给定地址字段的地址类型
    //     参考aicpu_communicator.cc中的SetAlltoAllInputAndOutPutMem
    // 注意: 这里继承A3, 不支持同时为0的场景
    if (inputSize == 0 && outputSize == 0) {
        isInplace = true;
        HCCL_INFO("[AicpuTaskCachePolicy][IsInplace] inputSize[%u] is overlapping with outputSize[%u] -> isInplace[%d]",
            inputSize, outputSize, isInplace);
        return HCCL_SUCCESS;
    }

    // 注意: 如果inputSize和outputSize只有一个为0, 则一定是outplace场景
    if (inputSize == 0 || outputSize == 0) {
        isInplace = false;
        HCCL_INFO("[AicpuTaskCachePolicy][IsInplace] inputSize[%u] is not overlapping with outputSize[%u] -> isInplace[%d]",
            inputSize, outputSize, isInplace);
        return HCCL_SUCCESS;
    }

    const uint64_t inputStart = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t inputEnd = inputStart + inputSize - 1;
    const uint64_t outputStart = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t outputEnd = outputStart + outputSize - 1;

    // 对于broadcast算子, UserInput与UserOutput完全重叠, 需要按照outplace场景特殊处理, 正常使能cache
    if (inputStart == outputStart && inputEnd == outputEnd && param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        isInplace = false;
        HCCL_INFO("[AicpuTaskCachePolicy][IsInplace] input==output[0x%016llx, 0x%016llx] for opType[%d] -> isInplace[%d]",
            inputStart, inputEnd, param.opType, isInplace);
        return HCCL_SUCCESS;
    }

    if (inputStart <= outputEnd && outputStart <= inputEnd) {
        isInplace = true;
        HCCL_INFO("[AicpuTaskCachePolicy][IsInplace] input[0x%016llx, 0x%016llx] is overlapping with output[0x%016llx, "
                  "0x%016llx] -> isInplace[%d]",
            inputStart, inputEnd, outputStart, outputEnd, isInplace);
    } else {
        isInplace = false;
        HCCL_INFO("[AicpuTaskCachePolicy][IsInplace] input[0x%016llx, 0x%016llx] is not overlapping with "
                  "output[0x%016llx, 0x%016llx] -> isInplace[%d]",
            inputStart, inputEnd, outputStart, outputEnd, isInplace);
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCachePolicy::ParseOpParamForCache(const OpParam &param, HcclDataType &sendType, HcclDataType &recvType,
    uint64_t &inputSize, uint64_t &outputSize, const TopoInfoWithNetLayerDetails &topoInfo)
{
    // 注意: 由于ParseOpParamForCache前已经做过IsAicpuTaskCacheEnable检查, 这里不再做重复检验

    const HcclCMDType opType = param.opType;
    const uint32_t rankSize = topoInfo.userRankSize;

    // 准备data type和count
    // NOTE: 非V类算子 (DataRes), V类算子 (VDataDes), All2All类算子 (All2AllDataDes), batch类算子
    // (BatchSendRecvDataDes/BatchWriteDataDes)
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) { // alltoall算子
        // 注意: sendType和recvType一定相同
        sendType = param.all2AllDataDes.sendType;
        recvType = param.all2AllDataDes.recvType;

        // 注意: 对于alltoall算子, inputSize和outputSize一定相同 (但不能直接使用param.input/outputSize,
        // alltoall算子不会设置这两个字段)
        const uint64_t sendCount = *(reinterpret_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts));
        inputSize = sendCount * rankSize * SIZE_TABLE[sendType];

        // 注意: 不能使用param.All2AllDataDes.recvCount * rankSize * SIZE_TABLE[recvType],
        // 因为alltoall使用sendCount来表示send/recvCount, 而recvCount本身为0
        outputSize = inputSize;
                                
        HCCL_DEBUG("[AicpuTaskCachePolicy][ParseOpParamForCache] opType[%u] rankSize[%u] sendType[%u] recvType[%u] "
            "inputSize[%llu] outputSize[%llu] sendCount[%llu] dataTypeSize[%u]",
            opType, rankSize, sendType, recvType, inputSize, outputSize, sendCount, SIZE_TABLE[sendType]);
    } else {
        sendType = param.DataDes.dataType;
        recvType = param.DataDes.dataType;
        inputSize = param.inputSize;
        outputSize = param.outputSize;

        HCCL_DEBUG("[AicpuTaskCachePolicy][ParseOpParamForCache] opType[%u] rankSize[%u] sendType[%u] recvType[%u] "
            "inputSize[%llu] outputSize[%llu]",
            opType, rankSize, sendType, recvType, inputSize, outputSize);
    }

    return HCCL_SUCCESS;
}

bool AicpuTaskCachePolicy::IsTopoSupported(const AlgResourceCtxSerializable &resCtxHost)
{
    for (const auto& levelChannels : resCtxHost.channels) {
        for (const auto& channel : levelChannels) {
            if (!channel.isValid) {
                continue;
            }

            // 只支持基于UB的跨卡通信 (当前aicpu仅支持UBC_CTP/UBC_TP/UBOE)
            // 不支持其他通信方式, 例如基于外置网卡的跨超RDMA, 基于PCIe的跨卡P2P等
            if (channel.protocol != CommProtocol::COMM_PROTOCOL_UBC_CTP &&
                channel.protocol != CommProtocol::COMM_PROTOCOL_UBC_TP &&
                channel.protocol != CommProtocol::COMM_PROTOCOL_UBOE) {
                HCCL_INFO("[AicpuTaskCachePolicy][IsTopoSupported] found channel protocol[%] not supported",
                    channel.protocol);
                return false;
            }
        }
    }
    return true;
}

bool AicpuTaskCachePolicy::IsOpTypeSupported(HcclCMDType opType)
{
    // 目前V类算子、batch类型算子、以及send/recv不考虑动态缓存 (使用白名单而非黑名单管理, 避免非预期算子进入cache机制)
    // 注意: 当前hccl不支持HcclGather
    if (opType == HcclCMDType::HCCL_CMD_BROADCAST || opType == HcclCMDType::HCCL_CMD_REDUCE
        || opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
        || opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_SCATTER
        || opType == HcclCMDType::HCCL_CMD_ALLREDUCE) { // 非V类算子
        HCCL_INFO("[AicpuTaskCachePolicy][IsOpTypeSupported] opType[%d] is supported for operator unfolding cache",
            opType);
        return true;
    }
    return false;
}

} // namespace ops_hccl
