/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ms_res.h"

#include <hccl/hccl_comm.h>
#include "alg_param.h"
#include "hccl_ccu_res_dl.h"
#include "ccu_launch_dl.h"
#include "ccu_log.h"
#include "log.h"

namespace ops_hccl {

HcclResult CcuMsGetChannelForCcu(HcclComm comm, AlgResourceRequest &res)
{
    // 以 kernel 为粒度申请 channel（迁移自 op_common.cc:1564-1599 HcclGetChannelForCcu）
    for (CcuKernelInfo &kernelInfo : res.ccuKernelInfos) {
        std::vector<HcclChannelDesc> &kernelChannelRequest = kernelInfo.channels;

        u32 channelNum = static_cast<u32>(kernelChannelRequest.size());
        std::vector<ChannelHandle> kernelChannels(channelNum);

        if (channelNum > 0) {
            // CCU_MS 引擎统一使用 CommEngine::COMM_ENGINE_CCU（CCU_MS 与 CCU_SCHE 共用同一底层 CommEngine）
            // AddExchangeInfo 暂省略（原 op_common.cc:1576 注册一致性校验信息），待 CreateRes 接口扩展 param 后补全
            auto ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, kernelChannelRequest.data(),
                                          channelNum, kernelChannels.data());
            if (ret == HCCL_E_UNAVAIL) {
                HCCL_WARNING("[CcuMsGetChannelForCcu] channel unavailable, channel num[%u].", channelNum);
                return HCCL_E_UNAVAIL;
            }
            CHK_RET(ret);
        }
        auto *kernelArgBase = static_cast<CcuKernelArgBase *>(kernelInfo.kernelArg);
        if (kernelArgBase == nullptr) {
            HCCL_ERROR("[CcuMsGetChannelForCcu] kernelArg ptr is err.");
            return HCCL_E_INTERNAL;
        }
        for (u32 i = 0; i < channelNum; ++i) {
            kernelArgBase->channels[i] = kernelChannels[i];
        }
        kernelArgBase->channelCount = channelNum;
        HCCL_INFO("[CcuMsGetChannelForCcu] get [%u] channels", channelNum);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuMsGetCcuKernel(HcclComm comm, AlgResourceRequest &res, AlgResourceCtxSerializable &resCtx)
{
    // 迁移自 op_common.cc:1601-1664 HcclGetCcuKernel
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
                HCCL_ERROR("[CcuMsGetCcuKernel] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
                HCCL_E_INTERNAL);

    u32 totalKernelNum = 0;
    for (auto t : res.ccuKernelNum) {
        totalKernelNum += t;
    }
    CHK_PRT_RET(totalKernelNum != res.ccuKernelInfos.size(),
                HCCL_ERROR("[CcuMsGetCcuKernel] ccuKernel num not match! totalKernelNum[%u] ccuKernelInfos[%zu]",
                           totalKernelNum, res.ccuKernelInfos.size()),
                HCCL_E_INTERNAL);

    // 按照 resGroup 进行注册
    u32 currentResGroup = 0;
    u32 maxResGroup = 0;
    resCtx.ccuKernels.resize(totalKernelNum);

    while (currentResGroup <= maxResGroup) {
        CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
        if (regStartRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuMsGetCcuKernel] ccu kernel register start failed: ccuRet -> %d", regStartRet);
            return ConvertCcuToHccl(regStartRet);
        }
        for (u32 i = 0; i < totalKernelNum; i++) {
            CcuKernelInfo &kernelInfo = res.ccuKernelInfos[i];
            if (kernelInfo.resGroup > maxResGroup) {
                maxResGroup = kernelInfo.resGroup;
            }
            if (kernelInfo.resGroup != currentResGroup) {
                continue;
            }

            HCCL_DEBUG("[CcuMsGetCcuKernel] kernelFuncName[%s]", kernelInfo.kernelFuncName);
            CcuKernelHandle kernelHandle;
            const void *kernelArgs[] = {kernelInfo.kernelArg};

            constexpr uint32_t dieId = 0;  // 预留接口，暂无含义
            constexpr uint32_t kernelArgNum = 1;
            CcuResult regRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
                                                      reinterpret_cast<void *>(kernelInfo.kernelFunc),
                                                      kernelArgs, kernelArgNum, &kernelHandle);
            if (regRet != CCU_SUCCESS) {
                HCCL_ERROR("[CcuMsGetCcuKernel] ccu kernel register failed: ccuRet -> %d", regRet);
                return ConvertCcuToHccl(regRet);
            }
            resCtx.ccuKernels[i] = kernelHandle;
        }
        CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
        if (regEndRet == CCU_E_UNAVAIL) {
            HCCL_WARNING("[CcuMsGetCcuKernel] ccu kernel register end unavailable, try to fallback.");
            return HCCL_E_UNAVAIL;
        } else if (regEndRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuMsGetCcuKernel] ccu kernel register end failed: ccuRet -> %d", regEndRet);
            return ConvertCcuToHccl(regEndRet);
        }
        currentResGroup++;
    }
    resCtx.ccuKernelNum = res.ccuKernelNum;
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
