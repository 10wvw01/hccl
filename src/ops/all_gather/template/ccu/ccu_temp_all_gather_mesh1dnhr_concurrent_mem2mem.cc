/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_gather_mesh1dnhr_concurrent_mem2mem.h"
#include "alg_data_trans_wrapper.h"
#include "alg_template_base.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

// mesh 链路与 CLOS 链路的带宽比，用于按比例切分数据
constexpr u32 CONCURRENT_MESH_BW = 11;
constexpr u32 CONCURRENT_CLOS_BW = 10;

// mesh 主流与 NHR 主流之间同步使用的 notify 索引
// threads[0](mesh 主流/executor 主流): notifyNumOnMainThread=1, 索引 0 用于 PostSync
// threads[1](NHR 主流/从流): notifyNumPerThread=2, 索引 0 用于 NHR 内部, 索引 1 用于 PreSync
constexpr u32 NOTIFY_IDX_PRE_SYNC = 1;   // PreSync: mainThread 向 NHR 主流发 record
constexpr u32 NOTIFY_IDX_POST_SYNC = 0;  // PostSync: NHR 主流向 mainThread 发 record

CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::CcuTempAllGatherMesh1DNHRConcurrentMem2Mem(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    // 两路均为全量 rank, 取 subCommRanks_[0] 的 size 作为 templateRankSize_
    if (!subCommRanks.empty() && !subCommRanks[0].empty()) {
        templateRankSize_ = subCommRanks[0].size();
        auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
        if (it != subCommRanks[0].end()) {
            mySubCommRank_ = static_cast<uint32_t>(std::distance(subCommRanks[0].begin(), it));
        }
    }
}

HcclResult CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    // 构造 mesh 子 template (subCommRanks_[0]) 和 NHR 子 template (subCommRanks_[1])
    std::vector<std::vector<u32>> meshSubCommRanks = {subCommRanks_[0]};
    std::vector<std::vector<u32>> nhrSubCommRanks = {subCommRanks_[1]};
    CcuTempAllGatherMesh1DMem2Mem meshSub(param, myRank_, meshSubCommRanks);
    CcuTempAllGatherNHR1DMem2Mem nhrSub(param, myRank_, nhrSubCommRanks);

    AlgResourceRequest meshReq;
    AlgResourceRequest nhrReq;
    CHK_RET(meshSub.CalcRes(comm, param, topoInfo, meshReq));
    CHK_RET(nhrSub.CalcRes(comm, param, topoInfo, nhrReq));

    // 聚合资源: 线程叠加 (+1: NHR 主流作为 executor 从流), notify 叠加, kernel 叠加
    resourceRequest.slaveThreadNum = meshReq.slaveThreadNum + nhrReq.slaveThreadNum + 1;
    resourceRequest.notifyNumOnMainThread = meshReq.notifyNumOnMainThread + 1;

    // mesh 子 template 的从流 notify
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              meshReq.notifyNumPerThread.begin(),
                                              meshReq.notifyNumPerThread.end());
    // NHR 主流需要与 mesh 主流通信 (+1), 再加上 NHR 自身的从流 notify
    resourceRequest.notifyNumPerThread.emplace_back(nhrReq.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              nhrReq.notifyNumPerThread.begin(),
                                              nhrReq.notifyNumPerThread.end());

    // CCU kernel: mesh 1 个 + NHR 1 个
    resourceRequest.ccuKernelNum.emplace_back(meshReq.ccuKernelNum[0]);
    resourceRequest.ccuKernelNum.emplace_back(nhrReq.ccuKernelNum[0]);
    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          meshReq.ccuKernelInfos.begin(), meshReq.ccuKernelInfos.end());
    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          nhrReq.ccuKernelInfos.begin(), nhrReq.ccuKernelInfos.end());

    HCCL_INFO("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem][CalcRes] rank[%u] slaveThreadNum[%u], "
              "notifyNumOnMainThread[%u], ccuKernelNum[%zu]",
              myRank_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread,
              resourceRequest.ccuKernelNum.size());
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.slaveThreadNum = 2;  // NHR 主流(1) + NHR 从流(1)
    resourceRequest.notifyNumOnMainThread = 1;  // mesh 主流与 NHR 主流同步
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    // NHR 主流需要额外 1 个 notify 与 mesh 主流通信
    resourceRequest.notifyNumPerThread[0] = 2;
    return HCCL_SUCCESS;
}

u64 CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::GetThreadNum() const
{
    return 3;  // mesh 主流(1) + NHR 主流(1) + NHR 从流(1)
}

u64 CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;  // 两路均 mem2mem, 不用 cclBuff
}

void CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::CalcDataSplit(
    u64 totalCount, u64 dataTypeSize, u64 &meshCount, u64 &closCount) const
{
    double splitRatio = static_cast<double>(CONCURRENT_MESH_BW) / (CONCURRENT_MESH_BW + CONCURRENT_CLOS_BW);
    u64 sliceAlignCount = HCCL_MIN_SLICE_ALIGN / dataTypeSize;
    if (sliceAlignCount == 0) {
        sliceAlignCount = 1;
    }
    meshCount = static_cast<u64>(std::floor(splitRatio * static_cast<double>(totalCount)));
    meshCount = meshCount / sliceAlignCount * sliceAlignCount;
    closCount = totalCount - meshCount;
    HCCL_INFO("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem][CalcDataSplit] totalCount[%llu], meshCount[%llu], "
              "closCount[%llu], splitRatio[%.4f]", totalCount, meshCount, closCount, splitRatio);
}

void CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::GenMeshParams(
    const TemplateDataParams &src, u64 meshCount, u64 meshSize, TemplateDataParams &dst) const
{
    dst = src;
    dst.count = meshCount;
    dst.sliceSize = meshSize;
    dst.tailSize = meshSize;
    // inBuffBaseOff / outBuffBaseOff 保持与 chunk 起始一致
    // outputSliceStride 保持 dataSize_ (全量 per-rank stride)
}

void CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::GenNhrParams(
    const TemplateDataParams &src, u64 closCount, u64 closSize, u64 meshSize, TemplateDataParams &dst) const
{
    dst = src;
    dst.count = closCount;
    dst.sliceSize = closSize;
    dst.tailSize = closSize;
    // NHR 份在 chunk 内偏移 meshSize
    dst.buffInfo.inBuffBaseOff = src.buffInfo.inBuffBaseOff + meshSize;
    dst.buffInfo.outBuffBaseOff = src.buffInfo.outBuffBaseOff + meshSize;
    // outputSliceStride 保持 dataSize_ (全量 per-rank stride)
}

HcclResult CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::KernelRun(
    const OpParam &param, const TemplateDataParams &templateDataParams, TemplateResource &templateResource)
{
    HCCL_INFO("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem][KernelRun] rank[%u] start.", myRank_);

    u64 dataTypeSize = DataTypeSizeGet(param.DataDes.dataType);
    u64 chunkCount = templateDataParams.count;

    // 1. 按带宽比切分当前 chunk
    u64 meshCount = 0;
    u64 closCount = 0;
    CalcDataSplit(chunkCount, dataTypeSize, meshCount, closCount);
    u64 meshSize = meshCount * dataTypeSize;
    u64 closSize = closCount * dataTypeSize;

    if (meshCount == 0 && closCount == 0) {
        HCCL_INFO("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem][KernelRun] both meshCount and closCount are 0, skip.");
        return HCCL_SUCCESS;
    }

    // 2. 切分线程和 kernel
    // threads[0] -> mesh 主流, threads[1] -> NHR 主流, threads[2] -> NHR 从流
    std::vector<ThreadHandle> meshThreads = {templateResource.threads[0]};
    std::vector<ThreadHandle> nhrThreads;
    for (size_t i = 1; i < templateResource.threads.size(); i++) {
        nhrThreads.push_back(templateResource.threads[i]);
    }
    // ccuKernels[0] -> mesh, ccuKernels[1] -> NHR
    u32 meshKernelNum = templateResource.ccuKernels.size() > 0 ? 1 : 0;
    u32 nhrKernelNum = templateResource.ccuKernels.size() > meshKernelNum ?
                       static_cast<u32>(templateResource.ccuKernels.size()) - meshKernelNum : 0;

    // 3. 构造子 template 参数
    TemplateDataParams meshParams;
    GenMeshParams(templateDataParams, meshCount, meshSize, meshParams);

    TemplateDataParams nhrParams;
    GenNhrParams(templateDataParams, closCount, closSize, meshSize, nhrParams);

    // 4. 构造子 template 资源
    TemplateResource meshRes;
    meshRes.threads = meshThreads;
    if (meshKernelNum > 0) {
        meshRes.ccuKernels.assign(templateResource.ccuKernels.begin(),
                                  templateResource.ccuKernels.begin() + meshKernelNum);
    }

    TemplateResource nhrRes;
    nhrRes.threads = nhrThreads;
    if (nhrKernelNum > 0) {
        nhrRes.ccuKernels.assign(templateResource.ccuKernels.begin() + meshKernelNum,
                                 templateResource.ccuKernels.end());
    }

    // 5. 构造子 template 实例
    std::vector<std::vector<u32>> meshSubCommRanks = {subCommRanks_[0]};
    std::vector<std::vector<u32>> nhrSubCommRanks = {subCommRanks_[1]};
    CcuTempAllGatherMesh1DMem2Mem meshSub(param, myRank_, meshSubCommRanks);
    CcuTempAllGatherNHR1DMem2Mem nhrSub(param, myRank_, nhrSubCommRanks);

    // 6. 同步 + 双路并发下发
    // PreSync: mesh 主流(threads[0]) 向 NHR 主流(threads[1]) 发起跑信号
    CHK_RET(PreSyncInterThreads(templateResource.threads[0],
                                {templateResource.threads[1]}, {NOTIFY_IDX_PRE_SYNC}));

    if (meshCount > 0 && meshKernelNum > 0) {
        CHK_RET(meshSub.KernelRun(param, meshParams, meshRes));
    }
    if (closCount > 0 && nhrKernelNum > 0) {
        CHK_RET(nhrSub.KernelRun(param, nhrParams, nhrRes));
    }

    // PostSync: mesh 主流(threads[0]) 等待 NHR 主流(threads[1]) 完成
    CHK_RET(PostSyncInterThreads(templateResource.threads[0],
                                 {templateResource.threads[1]}, {NOTIFY_IDX_POST_SYNC}));

    // 回填 sub-template 的 submitInfos 到 templateResource,供 FastLaunchSaveCtx 读取。
    // 顺序: mesh 在前, NHR 在后, 与 ccuKernels 切分顺序一致。
    templateResource.submitInfos.insert(templateResource.submitInfos.end(),
        meshRes.submitInfos.begin(), meshRes.submitInfos.end());
    templateResource.submitInfos.insert(templateResource.submitInfos.end(),
        nhrRes.submitInfos.begin(), nhrRes.submitInfos.end());

    HCCL_INFO("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem][KernelRun] rank[%u] end.", myRank_);
    return HCCL_SUCCESS;
}

// FastLaunch: 复用 KernelRun 保存的 cachedArgs, 仅重算与 input/output 地址相关的 arg, 然后
// 按 KernelRun 的同步时序回放 kernel launch。arg 索引约定见设计文档 9.3 节, 来源于
// CcuTempAllGatherMesh1DMem2Mem::PrepareLaunchArgs 与 CcuTempAllGatherNHR1DMem2Mem::PrepareLaunchArgs。
HcclResult CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch(
    const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    u32 totalKernelNum = static_cast<u32>(tempFastLaunchCtx.ccuKernelSubmitInfos.size());
    if (totalKernelNum == 0) {
        HCCL_INFO("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    if (tempFastLaunchCtx.threads.size() < 1) {
        HCCL_ERROR("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] thread num is 0.");
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] start, totalKernelNum[%u], threadNum[%zu]",
               totalKernelNum, tempFastLaunchCtx.threads.size());

    // submitInfos 顺序: mesh 在前(1 个), NHR 在后(1 或 2 个, 与 KernelRun 回填顺序一致)
    u32 meshKernelNum = 1;
    u32 nhrKernelNum = (totalKernelNum > meshKernelNum) ? (totalKernelNum - meshKernelNum) : 0;
    bool hasMesh = (meshKernelNum > 0);
    bool hasNhr = (nhrKernelNum > 0);

    // 1. 更新 mesh kernel args
    //    mesh args 布局(来自 CcuTempAllGatherMesh1DMem2Mem): argSize=15, cachedArgs 存 17 个
    //    [0]=inputAddr [1]=outputAddr [10]=isInputOutputEqual
    //    [15]=inBuffBaseOff(用作 input offset) [16]=outBuffBaseOff(用作 output offset)
    if (hasMesh) {
        uint64_t *meshArgs = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
        constexpr u32 meshInputIdx = 0;
        constexpr u32 meshOutputIdx = 1;
        constexpr u32 meshCurrentRankSliceInputOffsetIdx = 3;
        constexpr u32 meshCurrentRankSliceOutputOffsetIdx = 4;
        constexpr u32 meshIsInputOutputEqualIdx = 10;
        constexpr u32 meshInputOffsetIdx = 15;
        constexpr u32 meshOutputOffsetIdx = 16;
        constexpr u32 meshArgSize = 15;

        uint64_t meshInputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + meshArgs[meshInputOffsetIdx];
        uint64_t meshOutputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + meshArgs[meshOutputOffsetIdx];
        uint64_t meshCurrentRankSliceInputOffset = meshArgs[meshCurrentRankSliceInputOffsetIdx];
        uint64_t meshCurrentRankSliceOutputOffset = meshArgs[meshCurrentRankSliceOutputOffsetIdx];
        bool meshInputOutputEqual = (meshInputAddr + meshCurrentRankSliceInputOffset ==
                                     meshOutputAddr + meshCurrentRankSliceOutputOffset);
        meshArgs[meshInputIdx] = meshInputAddr;
        meshArgs[meshOutputIdx] = meshOutputAddr;
        meshArgs[meshIsInputOutputEqualIdx] = static_cast<uint64_t>(meshInputOutputEqual);
        (void)meshArgSize;
    }

    // 2. 更新 NHR kernel args
    //    NHR args 布局(来自 CcuTempAllGatherNHR1DMem2Mem): argSize=13, cachedArgs 存 16 个
    //    [0]=inputAddr [1]=outputAddr [6]=inputSliceStride [7]=outputSliceStride
    //    [10]=isInputOutputEqual [13]=inBuffBaseOff [14]=outBuffBaseOff [15]=mySubCommRank
    uint64_t nhrArgSize = 13;
    if (hasNhr) {
        uint64_t *nhrArgs = const_cast<uint64_t*>(
            tempFastLaunchCtx.ccuKernelSubmitInfos[meshKernelNum].cachedArgs);
        constexpr u32 nhrInputIdx = 0;
        constexpr u32 nhrOutputIdx = 1;
        constexpr u32 nhrInputSliceStrideIdx = 6;
        constexpr u32 nhrOutputSliceStrideIdx = 7;
        constexpr u32 nhrIsInputOutputEqualIdx = 10;
        constexpr u32 nhrInputOffsetIdx = 13;
        constexpr u32 nhrOutputOffsetIdx = 14;
        constexpr u32 nhrMySubCommRankIdx = 15;

        uint64_t nhrInputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + nhrArgs[nhrInputOffsetIdx];
        uint64_t nhrOutputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + nhrArgs[nhrOutputOffsetIdx];
        uint64_t nhrInputSliceStride = nhrArgs[nhrInputSliceStrideIdx];
        uint64_t nhrOutputSliceStride = nhrArgs[nhrOutputSliceStrideIdx];
        uint64_t nhrMySubCommRank = nhrArgs[nhrMySubCommRankIdx];
        bool nhrInputOutputEqual = (nhrInputAddr + nhrInputSliceStride * nhrMySubCommRank ==
                                    nhrOutputAddr + nhrOutputSliceStride * nhrMySubCommRank);
        nhrArgs[nhrInputIdx] = nhrInputAddr;
        nhrArgs[nhrOutputIdx] = nhrOutputAddr;
        nhrArgs[nhrIsInputOutputEqualIdx] = static_cast<uint64_t>(nhrInputOutputEqual);
    }

    // 3. outer PreSync: threads[0](mesh 主流) -> threads[1](NHR 主流), notifyIdx=NOTIFY_IDX_PRE_SYNC
    if (hasNhr && tempFastLaunchCtx.threads.size() >= 2) {
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0],
            {tempFastLaunchCtx.threads[1]}, {NOTIFY_IDX_PRE_SYNC}));
    }

    // 4. launch mesh kernel
    if (hasMesh) {
        uint64_t *meshArgs = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
        constexpr u32 meshArgSize = 15;
        void *taskArgs = reinterpret_cast<void*>(meshArgs);
        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
            tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgs, meshArgSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] mesh kernel launch failed, ccuRet -> %d",
                       launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    // 5. NHR 内部 PreSync(若 NHR 有 2 个 kernel, threads[1] -> threads[2])
    if (hasNhr && nhrKernelNum > 1 && tempFastLaunchCtx.threads.size() >= 3) {
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[1],
            {tempFastLaunchCtx.threads[2]}, {0}));
    }

    // 6. launch NHR kernel(s)
    if (hasNhr) {
        uint64_t *nhrArgs = const_cast<uint64_t*>(
            tempFastLaunchCtx.ccuKernelSubmitInfos[meshKernelNum].cachedArgs);
        void *taskArgs = reinterpret_cast<void*>(nhrArgs);
        // NHR 主流 threads[1] 下发第一个 NHR kernel
        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[1],
            tempFastLaunchCtx.ccuKernelSubmitInfos[meshKernelNum].kernelHandle, taskArgs,
            static_cast<uint32_t>(nhrArgSize));
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] nhr kernel0 launch failed, ccuRet -> %d",
                       launchRet);
            return ConvertCcuToHccl(launchRet);
        }
        // NHR 第二个 kernel(die1) 由 threads[2] 下发
        if (nhrKernelNum > 1 && tempFastLaunchCtx.threads.size() >= 3) {
            launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[2],
                tempFastLaunchCtx.ccuKernelSubmitInfos[meshKernelNum + 1].kernelHandle, taskArgs,
                static_cast<uint32_t>(nhrArgSize));
            if (launchRet != CCU_SUCCESS) {
                HCCL_ERROR("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] nhr kernel1 launch failed, ccuRet -> %d",
                           launchRet);
                return ConvertCcuToHccl(launchRet);
            }
        }
    }

    // 7. NHR 内部 PostSync(若 NHR 有 2 个 kernel, threads[1] <- threads[2])
    if (hasNhr && nhrKernelNum > 1 && tempFastLaunchCtx.threads.size() >= 3) {
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[1],
            {tempFastLaunchCtx.threads[2]}, {0}));
    }

    // 8. outer PostSync: threads[0](mesh 主流) <- threads[1](NHR 主流), notifyIdx=NOTIFY_IDX_POST_SYNC
    if (hasNhr && tempFastLaunchCtx.threads.size() >= 2) {
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0],
            {tempFastLaunchCtx.threads[1]}, {NOTIFY_IDX_POST_SYNC}));
    }

    HCCL_DEBUG("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] end");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
