/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_comm.h>
#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "log.h"
#include "common.h"
#include "exec_ccu_op.h"
#include "ccu_kernel.h"
#include "alltoall_ccu.h"

using namespace std;

namespace {
HcclResult GetChannelForCcu(HcclComm comm, const OpParam &param, std::vector<ChannelHandle> &kernelChannels) {
    uint32_t channelNum = param.rankSize - 1;
    kernelChannels.resize(channelNum);

    uint32_t channelIndex = 0;
    for(uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
        if (remoteRank == param.myRank) {
            continue;
        }

        uint32_t netLayer = 0, listSize = 0;
        CommLink *linkList = nullptr;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &linkList,
                                      &listSize)); // 获取srcRank和dstRank间link信息

        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
        bool protocolExists = false;
        for (uint32_t idx = 0; idx < listSize; idx++) {
            CommLink link = linkList[idx];
            if (link.linkAttr.linkProtocol == protocol) {
                desc.remoteRank = remoteRank;
                desc.notifyNum = 3;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                protocolExists = true;
                break;
            }
        }
        if (!protocolExists) {
            HCCL_ERROR("[GetChannelForCcu] Protocol %d not found between rank %u and rank %u",
                protocol, param.myRank, remoteRank);
            return HCCL_E_NOT_FOUND;
        }
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &kernelChannels[channelIndex])); // 获取channelhandle
        channelIndex++;
    }

    return HCCL_SUCCESS;
}

HcclResult GetCcuKernel(HcclComm comm, const OpParam &param, AlgResourceCtxSerializable &resCtxHost, 
                        const std::vector<ChannelHandle> &kernelChannels, CcuKernelInfo &kernelInfo) {
    
    // 设置kernel函数名和函数指针
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAlltoAllMesh1DMem2MemKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_ccu::CcuAlltoAllMesh1DMem2MemKernel);

    auto kernelArg = std::make_shared<ops_ccu::CcuKernelArgAllToAllMesh1DMem2Mem>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelInfo.setKernelArg(kernelArg);

    auto* kernelArgBase = static_cast<CcuKernelArgBase*>(kernelInfo.kernelArg);
    if (!kernelArgBase) {
        HCCL_ERROR("[GetCcuKernel] kernelArg ptr is err.");
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
        HCCL_ERROR("[GetCcuKernel] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
        HCCL_E_INTERNAL);

    resCtxHost.ccuKernels.resize(1); // 只注册1个kernel

    CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
    if (regStartRet != CCU_SUCCESS) {
        HCCL_ERROR("ccu kernel register start failed: ccuRet -> %d", regStartRet);
        return ConvertCcuToHccl(regStartRet);
    }

    CcuKernelHandle kernelHandle;
    const void *kernelArgs[] = {kernelInfo.kernelArg};

    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;

    // ==============================================
    // 注册ccu kernel, 翻译得到ccu微码指令, 写入到ccu中
    // ==============================================
    CcuResult regRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
                                                reinterpret_cast<void*>(kernelInfo.kernelFunc),
                                                kernelArgs, kernelArgNum, &kernelHandle); // 注册kernel
    if (regRet != CCU_SUCCESS) {
        HCCL_ERROR("ccu kernel register failed: ccuRet -> %d", regRet);
        return ConvertCcuToHccl(regRet);
    }
    resCtxHost.ccuKernels[0] = kernelHandle;

    CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
    if (regEndRet != CCU_SUCCESS) {
        HCCL_ERROR("ccu kernel register start failed: ccuRet -> %d", regEndRet);
        return ConvertCcuToHccl(regEndRet);
    }
    resCtxHost.ccuKernelNum = {1};
    
    return HCCL_SUCCESS;
}

HcclResult AllocAlgResource(HcclComm comm, const OpParam &param, aclrtStream stream, AlgResourceCtxSerializable &resCtxHost) {
    HCCL_INFO("Start to execute AllocAlgResourceCCU.");
    void *cclBufferAddr;
    uint64_t cclBufferSize;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize)); // 从通信域获取CCL buffer
    resCtxHost.cclMem = CommBuffer{cclBufferAddr, cclBufferSize};

    // ==============================================
    // STEP 2.1: 申请Thread资源
    // 将用户传入的 stream 转换为 thread，并申请 Notify
    // ==============================================
    ThreadHandle thread;
    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, 0, &thread));
    resCtxHost.threads.push_back(thread);

    // ==============================================
    // STEP 2.1: 申请Channel资源
    // ==============================================
    std::vector<ChannelHandle> kernelChannels;
    CHK_RET(GetChannelForCcu(comm, param, kernelChannels)); // 申请channel资源
    
    // ==============================================
    // STEP 2.2: 注册ccu kernel, 翻译得到ccu微码指令, 写入到ccu中
    // ==============================================
    CcuKernelInfo kernelInfo;
    CHK_RET(GetCcuKernel(comm, param, resCtxHost, kernelChannels, kernelInfo)); // 注册kernel

    HCCL_INFO("End to execute AllocAlgResourceCCU success.");
    return HCCL_SUCCESS;
}

static HcclResult InitAlgResourceCtx(HcclComm comm, OpParam &param, aclrtStream stream,
                                      std::unique_ptr<AlgResourceCtxSerializable> &resCtxHost)
{
    void *ctx = nullptr;
    uint64_t size = 0;

    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &size) == HCCL_SUCCESS) {
        // CCU模式通信资源已存在, 复用资源
        HCCL_INFO("Engine context already exists");
        param.ctxSize = size;
        char *resCtxSequence = static_cast<char *>(ctx);
        std::vector<char> ctxData(resCtxSequence, resCtxSequence + param.ctxSize);
        resCtxHost->DeSerialize(ctxData);
    } else {
        // CCU模式通信资源不存在, 资源构建
        HCCL_INFO("Creating engine context");
        HcclResult ret = AllocAlgResource(comm, param, stream, *resCtxHost);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("failed to alloc alg resource.");
            return ret;
        }
        std::vector<char> seq = resCtxHost->Serialize();
        uint64_t ctxSize = seq.size();

        void *newCtx = nullptr;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU, ctxSize, &newCtx));
        memcpy(newCtx, seq.data(), ctxSize);
        param.ctxSize = ctxSize;
        HCCL_INFO("Execute GetAlgResCCU success.");
    }

    return HCCL_SUCCESS;
}
}

namespace ops_ccu {
HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to execute HcclAlltoAll");

    // 1.校验参数是否为空
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);

    // 2.获取算子参数信息
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_alltoall");
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.inputPtr = const_cast<void *>(sendBuf);
    param.outputPtr = const_cast<void *>(recvBuf);
    param.count = sendCount;
    param.dataType = sendType;
    param.opType = HcclCMDType::HCCL_CMD_ALLTOALL;
    
    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================

    HcclDfxOpInfo dfxInfo;
    CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void *>(&dfxInfo)));

    std::unique_ptr<AlgResourceCtxSerializable> resCtxHost = std::make_unique<AlgResourceCtxSerializable>();
    CHK_RET(InitAlgResourceCtx(comm, param, stream, resCtxHost));

    // 4.下发 CCU 任务
    CHK_RET(ExecOp(param, *resCtxHost));

    HCCL_INFO("HcclAlltoAll executed successfully");
    return HCCL_SUCCESS;
}
}
