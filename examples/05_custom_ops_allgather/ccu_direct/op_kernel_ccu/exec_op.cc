/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <memory>
#include <vector>
#include <hccl/hcomm_primitives.h>
#include "log.h"
#include "ccu_launch.h"
#include "ccu_res.h"
#include "hccl_ccu_res.h"
#include "common.h"
#include "ccu_kernel.h"
#include "utils.h"
using namespace ops_hccl_ag;

namespace ops_hccl_ag {
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128; // 地址对齐
constexpr uint64_t UB_MAX_DATA_SIZE = 256*1024*1024; // UB协议一次传输最大size，单位Byte

typedef struct {
    uint32_t numBlocks = 1; // misson个数
    uint32_t reserved; // 对齐预留
    uint64_t phyDieMask = 0x00; // 指定在哪个物理die上运行，每次只起1个die
}HcommCcuSchd;

typedef struct {
    HcommCcuSchd ccuSchd;
    CcuInsHandle ccuIns = 0;
    void* attrs = nullptr;
    aclrtStream stream = nullptr;
} HcommLaunchKernelCfg;

constexpr uint32_t HCOMM_HOST_MAX_TASK_ARG_NUM = 10;

struct HcommHostKernelArgs {
    uint64_t taskArgs[HCOMM_HOST_MAX_TASK_ARG_NUM];
    uint32_t taskArgNum;
    void *kernelArg;
};

struct HcommLaunchKernelAttrs {
    const char *kernelName;
    // void *kernelArg;
    ThreadHandle thread;
    CcuKernelHandle *kernelHandle;
};

typedef void(__CcuHostKernelFunc)(void * args);
// HcclResult HcommCcuHostKernelLaunch(__CcuHostKernelFunc kernelFunc, HcommLaunchKernelCfg *cfg, void *args,
//                                     size_t argLen)
HcclResult HcommCcuHostKernelLaunch(__CcuHostKernelFunc kernelFunc, HcommLaunchKernelCfg *cfg, void *args) // args包含了taskargs和kernelarg
{
    CHK_PTR_NULL(kernelFunc);
    CHK_PTR_NULL(cfg);
    CHK_PTR_NULL(cfg->attrs);
    CHK_PTR_NULL(args);

    auto *attrs = static_cast<HcommLaunchKernelAttrs *>(cfg->attrs);
    auto *launchArgs = static_cast<HcommHostKernelArgs *>(args);
    CHK_PTR_NULL(attrs->kernelName);
    CHK_PTR_NULL(attrs->kernelHandle);
    CHK_PTR_NULL(launchArgs->kernelArg);

    CcuResult regStartRet = HcommCcuKernelRegisterStart(cfg->ccuIns);
    if (regStartRet != CCU_SUCCESS) {
        HCCL_ERROR("ccu kernel register start failed: ccuRet -> %d", regStartRet);
        return ConvertCcuToHccl(regStartRet);
    }

    const void *kernelArgs[] = {launchArgs->kernelArg};

    constexpr uint32_t dieId = 0; // 预留接口，暂无含义
    constexpr uint32_t kernelArgNum = 1;
    CcuResult regRet = HcommCcuKernelRegister(cfg->ccuIns, dieId, attrs->kernelName,
                                                reinterpret_cast<void*>(kernelFunc),
                                                kernelArgs, kernelArgNum, attrs->kernelHandle); // 注册kernel

    if (regRet != CCU_SUCCESS) {
        HCCL_ERROR("ccu kernel register failed: ccuRet -> %d", regRet);
        return ConvertCcuToHccl(regRet);
    }

    CcuResult regEndRet = HcommCcuKernelRegisterEnd(cfg->ccuIns);
    if (regEndRet != CCU_SUCCESS) {
        HCCL_ERROR("ccu kernel register start failed: ccuRet -> %d", regEndRet);
        return ConvertCcuToHccl(regEndRet);
    }

    // Todo: 根据kernelHandle获取taskArgs参数个数 当前 case taskargLen 是10
    // uint32_t sqeArgsNums = HcommCcuArgsNumGet(*attrs->kernelHandle);

    CcuResult launchRet = HcommCcuKernelLaunch(attrs->thread, *attrs->kernelHandle,
                                                launchArgs->taskArgs, launchArgs->taskArgNum);

    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllGatherMesh1DMem2Mem::ExecOp] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    return HCCL_SUCCESS;
}


static HcclResult LaunchCcuKernel(HcclComm comm, const OpParam &param, AlgResourceCtxSerializable &resCtx,
                               const std::vector<ChannelHandle> &kernelChannels, CcuKernelInfo &kernelInfo,
                               uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t dataSize,
                               uint64_t sliceCount, uint64_t dataTypeSize)
{
    // 设置kernel函数名和函数指针
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllGatherMesh1DMem2MemKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllGatherMesh1DMem2MemKernel);

    auto kernelArg = std::make_shared<CcuKernelArgAllGatherMesh1DMem2Mem>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    // kernelArg->taskArgBaseAddr = param.taskArgBaseAddr;
    kernelArg->varHandle = param.varHandle;
    kernelArg->eventHandle = param.eventHandle;

    kernelInfo.setKernelArg(kernelArg);

    auto* kernelArgBase = static_cast<CcuKernelArgBase*>(kernelInfo.kernelArg);
    if (!kernelArgBase) {
        HCCL_ERROR("[LaunchCcuKernel] kernelArg ptr is err.");
        return HCCL_E_INTERNAL;
    }

    for (uint32_t i = 0; i < kernelChannels.size(); ++i) {
        kernelArgBase->channels[i] = kernelChannels[i];  // 将channelhandle保存在kernelArgBase中
    }
    kernelArgBase->channelCount = static_cast<uint32_t>(kernelChannels.size());

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[LaunchCcuKernel] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
        HCCL_E_INTERNAL);

    uint64_t sliceSize = sliceCount * dataTypeSize;

    uint64_t currentRankSliceInputOffset = 0;
    uint64_t currentRankSliceOutputOffset = dataSize * param.myRank;

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;
    auto goSize         = CalGoSize(sliceSize, config);

    //Todo：第一阶段 HcommCcuHostKernelLaunch---------------纯通信场景-------------------
    HcommLaunchKernelCfg cfg;
    cfg.ccuSchd = {1,0,0x01};
    cfg.ccuIns = insHandle;
    cfg.stream = param.stream;

    resCtx.ccuKernels.resize(1); // 只注册1个kernel
    HcommLaunchKernelAttrs attrs {
        kernelInfo.kernelFuncName,
        resCtx.threads[0],
        &resCtx.ccuKernels[0],
    };
    cfg.attrs = &attrs;

    HcommHostKernelArgs launchArgs = {
        {},
        0,
        kernelInfo.kernelArg,
    };


    CHK_RET(HcommCcuHostKernelLaunch(reinterpret_cast<__CcuHostKernelFunc *>(kernelInfo.kernelFunc), &cfg,
                                     &launchArgs));

    //Todo：第二阶段 <<<>>> 纯通信场景，有taskargs参数
    // CcuAllGatherMesh1DMem2MemKernel<<<{1,0,0x01}, insHandle, param.stream>>>(
            // inputAddr,
            // outputAddr,
            // token,
            // currentRankSliceInputOffset,
            // currentRankSliceOutputOffset,
            // sliceSize,
            // goSize[0],
            // goSize[1],
            // goSize[2],
            // goSize[3],
    //     kernelInfo.kernelArg
    //     );



    /* 第一阶段 通算融合场景，没有taskargs
    HcommLaunchKernelCfg cfg;
    cfg.ccuSchd = {1,0,0x01};
    cfg.ccuIns = insHandle;
    cfg.stream = param.stream;

    resCtx.ccuKernels.resize(1); // 只注册1个kernel
    HcommLaunchKernelAttrs attrs {
        kernelInfo.kernelFuncName,
        kernelInfo.kernelArg,
        resCtx.threads[0],
        &resCtx.ccuKernels[0],
    };
    cfg.attrs = &attrs;
    // 此处没有taskArgs，在前面aicore kernel中设置taskArags
    HcommHostKernelArgs launchArgs = {
        nullptr,
        0,
        kernelInfo.kernelArg,
    };
    CHK_RET(HcommCcuHostKernelLaunch(reinterpret_cast<__CcuHostKernelFunc *>(kernelInfo.kernelFunc), &cfg, launchArgs);
    
    
    // 第二阶段 通算融合场景，没有taskargs
    // CcuAllGatherMesh1DMem2MemKernel<<<{1,0,0x01}, insHandle, param.stream>>>(kernelInfo.kernelArg);
    */



    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param, AlgResourceCtxSerializable &resCtx)
{
    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::ExecOp] start");

    uint32_t dataTypeSize = SIZE_TABLE[param.dataType];
    uint64_t dataSize = param.count * dataTypeSize;
    uint64_t count = param.count;

    if (count == 0) { // 数据量为0，直接返回
        HCCL_INFO("[CcuTempAllGatherMesh1DMem2Mem] DataCount == 0, ExecOp Run Ends.");
        return HcclResult::HCCL_SUCCESS;
    }

    if (param.rankSize == 1) { // 单卡，直接本地拷贝
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize)));
        HCCL_INFO("[CcuTempAllGatherMesh1DMem2Mem] RankSize == 1, ExecOp Run Ends.");
        return HCCL_SUCCESS;
    }

    uint64_t token = 0;
    uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    if (param.inputPtr != nullptr) {
        HcommCcuGetMemToken(baseInputAddr, static_cast<uint64_t>(dataSize), &token);
    } else if (param.outputPtr != nullptr) {
        HcommCcuGetMemToken(baseOutputAddr, static_cast<uint64_t>(dataSize), &token);
    }

    HcclComm comm = static_cast<HcclComm>(param.hcclComm);
    CHK_PTR_NULL(comm);

    CcuKernelInfo kernelInfo;
    CHK_RET(LaunchCcuKernel(comm, param, resCtx, resCtx.kernelChannels, kernelInfo,
                         baseInputAddr, baseOutputAddr, token, dataSize, count, dataTypeSize));


    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::ExecOp] end");
    return HCCL_SUCCESS;
}
} // namespace ops_hccl_ag
