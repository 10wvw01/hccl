/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_reduce_scatter_concurrent_mesh_nhr.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

constexpr u32 CLOS_PORT_NUM_SERVER_V2_CC = 8;

CcuTempReduceScatterConcurrentMeshNHR::CcuTempReduceScatterConcurrentMeshNHR(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    if (subCommRanks.size() >= 1) {
        meshGroup_ = subCommRanks[0];
    }
    if (subCommRanks.size() >= 2) {
        nhrGroup_ = subCommRanks[1];
    }
    rankSize_ = meshGroup_.size();
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    meshAlg_ = std::make_shared<CcuTempReduceScatterMesh1D>(param, rankId, std::vector<std::vector<u32>>{meshGroup_});
    nhrAlg_ = std::make_shared<CcuTempReduceScatterNHR1DMem2Mem>(param, rankId, std::vector<std::vector<u32>>{nhrGroup_});
}

CcuTempReduceScatterConcurrentMeshNHR::~CcuTempReduceScatterConcurrentMeshNHR() {}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(topoInfo == nullptr,
        HCCL_ERROR("[CcuTempReduceScatterConcurrentMeshNHR][CalcRes] topoInfo is nullptr"), HCCL_E_PARA);

    AlgResourceRequest meshReq;
    AlgResourceRequest nhrReq;
    CHK_RET(meshAlg_->CalcRes(comm, param, topoInfo, meshReq));
    CHK_RET(nhrAlg_->CalcRes(comm, param, topoInfo, nhrReq));

    resourceRequest.slaveThreadNum = meshReq.slaveThreadNum + nhrReq.slaveThreadNum + 1;
    resourceRequest.notifyNumOnMainThread = meshReq.notifyNumOnMainThread + 1;
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                               meshReq.notifyNumPerThread.begin(), meshReq.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(nhrReq.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                               nhrReq.notifyNumPerThread.begin(), nhrReq.notifyNumPerThread.end());

    resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(),
                                        meshReq.ccuKernelNum.begin(), meshReq.ccuKernelNum.end());
    resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(),
                                        nhrReq.ccuKernelNum.begin(), nhrReq.ccuKernelNum.end());
    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          meshReq.ccuKernelInfos.begin(), meshReq.ccuKernelInfos.end());
    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          nhrReq.ccuKernelInfos.begin(), nhrReq.ccuKernelInfos.end());

    mergedReq_ = resourceRequest;
    HCCL_INFO("[CcuTempReduceScatterConcurrentMeshNHR][CalcRes] success, meshKernel[%zu], nhrKernel[%zu]",
        meshReq.ccuKernelInfos.size(), nhrReq.ccuKernelInfos.size());
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest = mergedReq_;
    return HCCL_SUCCESS;
}

u64 CcuTempReduceScatterConcurrentMeshNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    u64 meshScratch = meshAlg_ != nullptr ? meshAlg_->CalcScratchMultiple(inBuffType, outBuffType) : 0;
    u64 nhrScratch = nhrAlg_ != nullptr ? nhrAlg_->CalcScratchMultiple(inBuffType, outBuffType) : 0;
    return std::max(meshScratch, nhrScratch);
}

u64 CcuTempReduceScatterConcurrentMeshNHR::GetThreadNum() const
{
    u64 meshNum = meshAlg_ != nullptr ? meshAlg_->GetThreadNum() : 0;
    u64 nhrNum = nhrAlg_ != nullptr ? nhrAlg_->GetThreadNum() : 0;
    return meshNum + nhrNum;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::FastLaunch(const OpParam& param,
    const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    (void)tempFastLaunchCtx;
    HCCL_INFO("[CcuTempReduceScatterConcurrentMeshNHR][FastLaunch] not supported, fallback to KernelRun");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::CalcDataSplit(const OpParam& param,
    const TemplateDataParams& templateDataParams, TemplateDataParams& meshParams,
    TemplateDataParams& nhrParams, u64& meshCount, u64& nhrCount) const
{
    u64 portNum0 = (rankSize_ > 0) ? (rankSize_ - 1) : 0;
    u64 portNum = CLOS_PORT_NUM_SERVER_V2_CC;

    const u64 sliceAlignCount = (dataTypeSize_ > 0) ? (HCCL_MIN_SLICE_ALIGN / dataTypeSize_) : 1;
    const u64 dataCount = templateDataParams.count;
    meshCount = (portNum0 + portNum) > 0 ? (dataCount * portNum0 / (portNum0 + portNum)) : (dataCount / 2);
    meshCount = meshCount / sliceAlignCount * sliceAlignCount;
    nhrCount = (dataCount > meshCount) ? (dataCount - meshCount) : 0;

    meshParams = templateDataParams;
    meshParams.count = meshCount;
    meshParams.buffInfo.inBuffBaseOff = templateDataParams.buffInfo.inBuffBaseOff;
    meshParams.buffInfo.outBuffBaseOff = templateDataParams.buffInfo.outBuffBaseOff;
    meshParams.sliceSize = meshCount * dataTypeSize_;
    meshParams.tailSize = meshParams.sliceSize;
    meshParams.outputSliceStride = 0;

    nhrParams = templateDataParams;
    nhrParams.count = nhrCount;
    u64 nhrDataOff = templateDataParams.buffInfo.inBuffBaseOff + meshCount * dataTypeSize_;
    nhrParams.buffInfo.inBuffBaseOff = nhrDataOff;
    nhrParams.buffInfo.outBuffBaseOff = nhrDataOff;
    nhrParams.sliceSize = nhrCount * dataTypeSize_;
    nhrParams.tailSize = nhrParams.sliceSize;
    nhrParams.outputSliceStride = 0;

    HCCL_INFO("[CcuTempReduceScatterConcurrentMeshNHR][CalcDataSplit] dataCount[%llu], meshCount[%llu], nhrCount[%llu]",
        dataCount, meshCount, nhrCount);
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::KernelRun(const OpParam& param,
    const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    // 1.切分线程：前 meshThreadNum 个给 mesh 路，其余给 nhr 路
    u64 meshThreadNum = meshAlg_->GetThreadNum();
    CHK_PRT_RET(meshThreadNum > templateResource.threads.size(),
        HCCL_ERROR("[CcuTempReduceScatterConcurrentMeshNHR][KernelRun] meshThreadNum[%llu] > threads[%zu]",
            meshThreadNum, templateResource.threads.size()), HCCL_E_PARA);
    std::vector<ThreadHandle> meshThreads(templateResource.threads.begin(),
                                          templateResource.threads.begin() + meshThreadNum);
    std::vector<ThreadHandle> nhrThreads(templateResource.threads.begin() + meshThreadNum,
                                         templateResource.threads.end());
    CHK_PRT_RET(meshThreads.empty() || nhrThreads.empty(),
        HCCL_ERROR("[CcuTempReduceScatterConcurrentMeshNHR][KernelRun] meshThreads or nhrThreads is empty"), HCCL_E_INTERNAL);
    ThreadHandle meshMain = meshThreads.at(0);
    ThreadHandle nhrMain = nhrThreads.at(0);

    // 2.校验 ccuKernel 数量充足（mesh 用第0个、nhr 用第1个）
    CHK_PRT_RET(templateResource.ccuKernels.size() < 2,
        HCCL_ERROR("[CcuTempReduceScatterConcurrentMeshNHR][KernelRun] ccuKernels size[%zu] < 2",
            templateResource.ccuKernels.size()), HCCL_E_INTERNAL);

    // 3.为两路子模板分别组装资源：线程 + 对应的 ccuKernel
    TemplateResource meshRes;
    meshRes.threads = meshThreads;
    meshRes.ccuKernels.push_back(templateResource.ccuKernels[0]);
    TemplateResource nhrRes;
    nhrRes.threads = nhrThreads;
    nhrRes.ccuKernels.push_back(templateResource.ccuKernels[1]);

    // 4.数据切分：按带宽比把当前数据块切成 mesh 半 + nhr 半，填好两路参数
    TemplateDataParams meshParams;
    TemplateDataParams nhrParams;
    u64 meshCount = 0;
    u64 nhrCount = 0;
    CHK_RET(CalcDataSplit(param, templateDataParams, meshParams, nhrParams, meshCount, nhrCount));

    // 5.前同步：mesh 主流通知 nhr 主流，保证两路同时开始
    std::vector<ThreadHandle> subThreads = {nhrMain};
    std::vector<u32> notifyIdxMainToSub = {static_cast<u32>(nhrThreads.size() - 1)};
    CHK_RET(PreSyncInterThreads(meshMain, subThreads, notifyIdxMainToSub));

    // 6.并行执行两路：各自下发到自己的线程组，异步并行执行
    if (meshCount > 0 && meshParams.sliceSize > 0) {
        CHK_RET(meshAlg_->KernelRun(param, meshParams, meshRes));
    }
    if (nhrCount > 0 && nhrParams.sliceSize > 0) {
        CHK_RET(nhrAlg_->KernelRun(param, nhrParams, nhrRes));
    }

    // 7.后同步：等两路都完成，再返回
    std::vector<u32> notifyIdxSubToMain = {static_cast<u32>(meshThreads.size() - 1)};
    CHK_RET(PostSyncInterThreads(meshMain, subThreads, notifyIdxSubToMain));

    HCCL_INFO("[CcuTempReduceScatterConcurrentMeshNHR][KernelRun] done, meshCount[%llu], nhrCount[%llu]",
        meshCount, nhrCount);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
