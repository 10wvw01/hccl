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
#include "aicpu_task_cache_utils.h"

namespace ops_hccl {

HcclResult AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, bool &isCacheEnable)
{
    isCacheEnable = false;
    if (!param.aicpuCacheEnable) {
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
    if (param.isCapture) {
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
    CHK_RET(IsInplaceForCache(param, resCtx.topoInfo.userRankSize, isInplace));
    if (isInplace) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] inplace case is not supported for operator unfolding "
                  "cache");
        return HCCL_SUCCESS;
    }

    if(!IsTopoSupported(resCtx)) {
        return HCCL_SUCCESS;
    }

    // 所有使能判断都通过是返回true
    isCacheEnable = true;
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCachePolicy::IsInplaceForCache(const OpParam &param, const uint32_t rankSize, bool &isInplace)
{
    // 准备input/output size
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;

    CHK_RET(AicpuTaskCacheUtils::GetInputOutputInfoForCache(param, rankSize, inputSize, outputSize));

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

    // 对于broadcast算子, UserInput与UserOutput完全重叠, 需要按(照outplace场景特殊处理, 正常使能cache
    if (param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        CHK_PRT_RET(!(inputStart == outputStart && inputSize == outputSize),
            HCCL_INFO("[AicpuTaskCachePolicy][IsInplace] broadcast shoud input==output[0x%016llx, 0x%016llx] "
                      "inputSize==outputSize[%u,%u]",
                inputStart, outputStart, inputSize, outputSize),
            HCCL_E_PARA);
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

bool AicpuTaskCachePolicy::IsTopoSupported(const AlgResourceCtxSerializable &resCtx)
{
    for (const auto& levelChannels : resCtx.channels) {
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
