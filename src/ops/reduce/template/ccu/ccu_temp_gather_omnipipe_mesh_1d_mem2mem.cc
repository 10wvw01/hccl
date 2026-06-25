/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "ccu_kernel_gather_omnipipe_mesh_1d_mem2mem.h"
#include "ccu_temp_gather_omnipipe_mesh_1d_mem2mem.h"
#include "alg_data_trans_wrapper.h" 
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempGatherOmniPipeMesh1DMem2Mem::CcuTempGatherOmniPipeMesh1DMem2Mem(const OpParam& param, const u32 rankId,
                                                                        const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    rankId_ = rankId;
    // 子通信域的root卡号
    auto rootIt = std::find(ranks.begin(), ranks.end(), param.root);
    if (rootIt != ranks.end()) {
        subCommRootId_ = std::distance(ranks.begin(), rootIt);
    }

    ifRealRoot_ = (rankId == param.root);
    // HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem] mySubCommRank_=%u, subCommRootId_=%u, rankId=%u",
    //            mySubCommRank_, subCommRootId_, rankId);
}

CcuTempGatherOmniPipeMesh1DMem2Mem::~CcuTempGatherOmniPipeMesh1DMem2Mem()
{
}


u64 CcuTempGatherOmniPipeMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1); 

    return HCCL_SUCCESS;
}

uint32_t CcuTempGatherOmniPipeMesh1DMem2Mem::RemoteRankId2RankId(const uint32_t remoteRankId) const
{
    uint32_t subCommRankId = 0;
    std::vector<u32> ranks = subCommRanks_[0];
    auto it = std::find(ranks.begin(), ranks.end(), remoteRankId);
    if (it != ranks.end()) {
        subCommRankId = std::distance(ranks.begin(), it);
    }
    return subCommRankId;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param,
                                                        const TopoInfoWithNetLayerDetails* topoInfo,
                                                        AlgResourceRequest& resourceRequest)
{
    GetRes(resourceRequest);
    resourceRequest.ccuKernelNum.push_back(1);

    HCCL_DEBUG("[%s]notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    CHK_SAFETY_FUNC_RET(strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuGatherOmniPipeMesh1DMem2MemKernel"));
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuGatherOmniPipeMesh1DMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1DFullMesh(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for (auto channel : channelDescs) {
            if (channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
                HCCL_ERROR("[CcuTempGatherOmniPipeMesh1DMem2Mem][%s] channel.channelProtocol[%u]", __func__, channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }

    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem][%s] Get Mesh channels Success.", __func__);
    std::map<u32, u32> subRankIdx2RankIdx;
    for (u32 i=0; i< channelDescs.size(); i++) {
        remoteRank = channelDescs[i].remoteRank;
        subRankIdx = RemoteRankId2RankId(remoteRank);
        subRankIdx2RankIdx[subRankIdx] = remoteRank;

        HCCL_DEBUG("[%s] myRank_[%u]  remoteRank[%u] ", __func__, myRank_, remoteRank);
    }
    subRankIdx2RankIdx[mySubCommRank_] = myRank_;

    auto kernelArg = std::make_shared<CcuKernelArgGatherOmniPipeMesh1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->rootId = subCommRootId_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelArg->subRankIdx2RankIdx = subRankIdx2RankIdx;
    kernelArg->ifRealRoot = ifRealRoot_;
    kernelArg->myrealrank = myRank_;

    kernelInfo.setKernelArg(kernelArg);
    
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    HCCL_DEBUG("[%s]channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu", __func__, channelDescs.size(),
        subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());
    HCCL_DEBUG("[%s] myRank_[%u] mySubCommRank_[%u] remoteRank[%u] localAddr[%u] remoteAddr[%u]", __func__, myRank_,
        mySubCommRank_, channelDescs[0].remoteRank, channelDescs[0].localEndpoint.commAddr.addr,
        channelDescs[0].remoteEndpoint.commAddr.addr);

    return HcclResult::HCCL_SUCCESS;
}


HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                          const TemplateDataParams& templateDataParams,
                                                          TemplateResource& templateResource)
{
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] start1");

    buffInfo_ = templateDataParams.buffInfo;
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);
    uint64_t inBuffBaseOff = buffInfo_.inBuffBaseOff;
    uint64_t outBuffBaseOff = buffInfo_.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff; //基址 + loop偏移
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff; //基址 + loop偏移
    
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] start2");

    if (localCopyFlag == 0) {
        HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] start3");
        // 当前卡就是root卡，receive其他所有卡
        // 当前卡不是root卡，直接结束
        // uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
        uint64_t inputSliceStride = 0;
        uint64_t outputSliceStride = 0;
        HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] start5:%d",stepSliceInfo.inputOmniPipeSliceStride.size());
        // uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();
        uint64_t sliceSize;
        uint64_t inputOmniPipeSliceStride;
        uint64_t outputOmniPipeSliceStride;
        // HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] repeatNum=%u", repeatNum);
        // 遍历peer对端的卡
        auto inputOmniPipeSliceStrides = stepSliceInfo.inputOmniPipeSliceStride;
        HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] peerIdSize=%u", inputOmniPipeSliceStrides.size()); // 这里子通信域里的第几个卡
        // inputOmniPipeSliceStrides[peerId][rpt] 第peerId个卡要发到root的第rpt个数据片
        for (uint32_t peerId = 0; peerId < inputOmniPipeSliceStrides.size(); ++peerId) { // size其实是子通信域卡数
            // 判断是不是本端自己的卡
            HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] peerId=%u size=%d", peerId, inputOmniPipeSliceStrides[peerId].size());
            for (uint32_t rpt = 0; rpt < inputOmniPipeSliceStrides[peerId].size(); ++rpt) { // 子通信域的第peerId个卡，要发的第几个数据片
                sliceSize = stepSliceInfo.stepSliceSize[peerId][rpt];
                // if (isLastStep_==false && peerId==1) {//不搬运1卡
                //     sliceSize = 0;
                // }
                // if (isLastStep_==false && peerId==2) {//不搬运2卡
                //     sliceSize = 0;
                // }
                // if (isLastStep_ && peerId==1) {//不搬运4卡
                //     sliceSize = 0;
                // }
                // if (isLastStep_&& peerId==2) {//不搬运5卡
                //     sliceSize = 0;
                // }
                inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[peerId][rpt]; // 远端的输入offset
                outputOmniPipeSliceStride = stepSliceInfo.outputOmniPipeSliceStride[peerId][rpt]; // 远端的输出offset

                HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] sliceSize=%u", sliceSize);
                // 自己是逻辑root卡 && 自己不和自己通信
                bool ifNewRoot = (subRoot == mySubCommRank_ && peerId != subRoot); // 判断是不是root，需不需要做搬运  
                std::vector<uint64_t> taskArgs = {
                    inputAddr, 
                    outputAddr,
                    token, 
                    localCopyFlag, 
                    sliceSize, 
                    inputOmniPipeSliceStride, 
                    outputOmniPipeSliceStride, 
                    isStepOne_, 
                    isLastStep_, 
                    ifNewRoot
                };
                // if (ifNewRoot && sliceSize!=0) {
                HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] rpt=%u inputAddr=%llu outputAddr=%llu  inBuffBaseOff=%llu outBuffBaseOff=%llu"
                            " sliceSize=%llu localCopyFlag=%llu inputOmniPipeSliceStride=%llu outputOmniPipeSliceStride=%llu ifNewRoot=%llu isloopOne_t=%llu isStepOne_=%llu isLastStep_=%llu  myRank[%u]  subroot[%d]",
                            rpt, inputAddr, outputAddr, inBuffBaseOff, outBuffBaseOff, sliceSize, localCopyFlag, inputOmniPipeSliceStride,outputOmniPipeSliceStride, ifNewRoot, isloopOne_, isStepOne_, isLastStep_, myRank_, subRoot);
            
                // }
                uint64_t argSize = taskArgs.size();
                CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), argSize);
                if (launchRet != CCU_SUCCESS) {
                    HCCL_ERROR("[%s] myRank[%u] HcommCcuKernelLaunch failed, ccuRet is:[%d]", __func__, myRank_, launchRet);
                    return ConvertCcuToHccl(launchRet);
                }
            }
        }
    } 
    else if (localCopyFlag == 1) {
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy start", __func__, myRank_);
        DataSlice srcSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);

        HCCL_DEBUG("[%s]buffInfo_.inputPtr[%u] buffInfo_.inBuffBaseOff[%llu] templateDataParams.sliceSize[%llu] templateDataParams.count[%llu]",
            __func__, PointerToAddr(buffInfo_.inputPtr), buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);

         HCCL_DEBUG("[%s]buffInfo_.outputPtr[%u] buffInfo_.outBuffBaseOff[%llu] templateDataParams.sliceSize[%llu] templateDataParams.count[%llu]",
            __func__, PointerToAddr(buffInfo_.outputPtr), buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
            
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy inputAddrBase[%llu] inputAddrOffset[%llu] outputAddrBase[%llu]"
                   "outputAddrOffset[%llu] sliceSize[%llu]",
            __func__, myRank_, inputAddrBase, buffInfo_.inBuffBaseOff, outputAddrBase, buffInfo_.outBuffBaseOff,
            templateDataParams.sliceSize);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy end", __func__, myRank_);
    }

    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempGatherOmniPipeMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

} // namespace ops_hccl