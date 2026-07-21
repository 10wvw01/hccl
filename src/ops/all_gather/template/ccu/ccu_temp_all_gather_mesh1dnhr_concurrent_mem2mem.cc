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
#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_utils.h"
#include "template_utils.h"

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

void CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::CalcNhrDieSplit(
    u64 sliceSize, u64 typeSize, u64 &die0Size, u64 &die1Size) const
{
    // 搬自 CcuTempAllGatherNHR1DMem2Mem::SplitDataFor2Dies,纯数学,不依赖子 template 类
    constexpr u64 MULTIPLIER = 4;
    u64 dataCount = sliceSize / typeSize;
    if (dataCount <= templateRankSize_ * MULTIPLIER) {  // 数据量极小,不划分 die
        die0Size = dataCount * typeSize;
        die1Size = 0;
        return;
    }
    u8 die0PortGroupSize = 6;
    u8 die1PortGroupSize = 2;
    die0Size = (dataCount * die0PortGroupSize / (die0PortGroupSize + die1PortGroupSize)) * typeSize;
    die1Size = sliceSize - die0Size;
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

    // 1.1 按带宽比切分 tail(最后 rank 的 slice),sole executor 中 tailSize == sliceSize
    u64 meshTailCount = 0;
    u64 closTailCount = 0;
    CalcDataSplit(templateDataParams.tailSize / dataTypeSize, dataTypeSize, meshTailCount, closTailCount);
    u64 meshTailSize = meshTailCount * dataTypeSize;
    u64 closTailSize = closTailCount * dataTypeSize;

    if (meshCount == 0 && closCount == 0) {
        HCCL_INFO("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem][KernelRun] both meshCount and closCount are 0, skip.");

    }

    // 2. 切分 kernel: ccuKernels[0] -> mesh, ccuKernels[1..] -> NHR
    u32 meshKernelNum = templateResource.ccuKernels.size() > 0 ? 1 : 0;
    u32 nhrKernelNum = templateResource.ccuKernels.size() > meshKernelNum ?
                       static_cast<u32>(templateResource.ccuKernels.size()) - meshKernelNum : 0;
    bool hasMesh = (meshCount > 0 && meshKernelNum > 0);
    bool hasNhr = (closCount > 0 && nhrKernelNum > 0);

    // 3. 装填 mesh kernel 的 taskArgs(顺序按 CcuAllGatherMesh1DMem2MemArgLayout)
    std::vector<uint64_t> meshTaskArgs;
    u64 meshArgSize = 0;
    if (hasMesh) {
        const BuffInfo &meshBuff = templateDataParams.buffInfo;
        uint64_t inputAddr = PointerToAddr(meshBuff.inputPtr) + meshBuff.inBuffBaseOff;
        uint64_t outputAddr = PointerToAddr(meshBuff.outputPtr) + meshBuff.outBuffBaseOff;
        uint64_t token;
        CHK_RET(GetToken(meshBuff, token));
        uint64_t inputSliceStride = templateDataParams.inputSliceStride;
        uint64_t outputSliceStride = templateDataParams.outputSliceStride;
        uint32_t repeatNum = templateDataParams.repeatNum;
        uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
        uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
        // mesh 份的 slice 大小必须缩到 meshSize, 不能用全量 sliceSize/tailSize
        uint64_t normalSliceSize = meshSize;
        uint64_t lastSliceSize = meshTailSize;
        if (mySubCommRank_ == templateRankSize_ - 1) {
            normalSliceSize = meshTailSize;
        }
        bool inputOutputEqual = (inputAddr + inputSliceStride * mySubCommRank_ ==
                                 outputAddr + outputSliceStride * mySubCommRank_);
        uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);
        uint64_t currentRankSliceInputOffset = inputSliceStride * mySubCommRank_;
        uint64_t currentRankSliceOutputOffset = outputSliceStride * mySubCommRank_;
        uint64_t tmpRepeatNum = UINT64_MAX - repeatNum;
        LoopGroupConfig config{};
        config.msInterleave = CCU_MS_INTERLEAVE;
        config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
        config.memSlice = CCU_MS_SIZE * LOCAL_COPY_MS_PER_LOOP;
        auto goSize = CalGoSize(normalSliceSize, config, GetCcuVersion());
        meshTaskArgs = {inputAddr, outputAddr, token, currentRankSliceInputOffset,
                        currentRankSliceOutputOffset, tmpRepeatNum, inputRepeatStride,
                        outputRepeatStride, normalSliceSize, lastSliceSize,
                        isInputOutputEqual, goSize[0], goSize[1], goSize[2], goSize[3]};
        meshArgSize = CcuAllGatherMesh1DMem2MemArgLayout::ARG_SIZE;
    }

    // 4. 装填 NHR kernel 的 taskArgs(顺序按 CcuAllGatherNHR1DMem2MemArgLayout)
    std::vector<uint64_t> nhrTaskArgs;
    u64 nhrArgSize = 0;
    u64 nhrDie0Size = 0;
    u64 nhrDie1Size = 0;
    if (hasNhr) {
        const BuffInfo &nhrBuff = templateDataParams.buffInfo;
        // NHR 份在 chunk 内偏移 meshSize
        uint64_t inputAddr = PointerToAddr(nhrBuff.inputPtr) + nhrBuff.inBuffBaseOff + meshSize;
        uint64_t outputAddr = PointerToAddr(nhrBuff.outputPtr) + nhrBuff.outBuffBaseOff + meshSize;
        uint64_t token;
        CHK_RET(GetToken(nhrBuff, token));
        if (nhrKernelNum == 2) {
            CalcNhrDieSplit(closSize, dataTypeSize, nhrDie0Size, nhrDie1Size);
        } else {
            nhrDie0Size = closSize;
        }
        uint64_t repeatNum = UINT64_MAX - templateDataParams.repeatNum;
        uint64_t inputSliceStride = templateDataParams.inputSliceStride;
        uint64_t outputSliceStride = templateDataParams.outputSliceStride;
        uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
        uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
        // NHR 份的 last slice 大小必须从 closTailSize 按 die 切分, 不能用全量 tailSize
        u64 nhrDie0LastSize = 0;
        u64 nhrDie1LastSize = 0;
        if (nhrKernelNum == 2) {
            CalcNhrDieSplit(closTailSize, dataTypeSize, nhrDie0LastSize, nhrDie1LastSize);
        } else {
            nhrDie0LastSize = closTailSize;
        }
        bool inputOutputEqual = (inputAddr + inputSliceStride * mySubCommRank_ ==
                                 outputAddr + outputSliceStride * mySubCommRank_);
        uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);
        nhrTaskArgs = {inputAddr, outputAddr, token, nhrDie0Size, nhrDie1Size, repeatNum,
                       inputSliceStride, outputSliceStride, inputRepeatStride,
                       outputRepeatStride, isInputOutputEqual, nhrDie0LastSize, nhrDie1LastSize};
        nhrArgSize = CcuAllGatherNHR1DMem2MemArgLayout::ARG_SIZE;
    }

    // 5. outer PreSync: threads[0](mesh 主流) -> threads[1](NHR 主流)
    if (hasNhr && templateResource.threads.size() >= 2) {
        CHK_RET(PreSyncInterThreads(templateResource.threads[0],
            {templateResource.threads[1]}, {NOTIFY_IDX_PRE_SYNC}));
    }

    // 6. launch mesh kernel
    if (hasMesh) {
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0],
            templateResource.ccuKernels[0], meshTaskArgs.data(),
            static_cast<uint32_t>(meshArgSize));
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::KernelRun] mesh kernel launch failed, ccuRet -> %d",
                       launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    // 7. NHR 内部 PreSync(若 nhrKernelNum > 1): threads[1] -> threads[2]
    if (hasNhr && nhrKernelNum > 1 && templateResource.threads.size() >= 3) {
        CHK_RET(PreSyncInterThreads(templateResource.threads[1],
            {templateResource.threads[2]}, {0}));
    }

    // 8. launch NHR kernel(s),两 kernel 共享同一份 taskArgs
    if (hasNhr) {
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[1],
            templateResource.ccuKernels[meshKernelNum], nhrTaskArgs.data(),
            static_cast<uint32_t>(nhrArgSize));
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::KernelRun] nhr kernel0 launch failed, ccuRet -> %d",
                       launchRet);
            return ConvertCcuToHccl(launchRet);
        }
        if (nhrKernelNum > 1 && templateResource.threads.size() >= 3) {
            launchRet = HcommCcuKernelLaunch(templateResource.threads[2],
                templateResource.ccuKernels[meshKernelNum + 1], nhrTaskArgs.data(),
                static_cast<uint32_t>(nhrArgSize));
            if (launchRet != CCU_SUCCESS) {
                HCCL_ERROR("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::KernelRun] nhr kernel1 launch failed, ccuRet -> %d",
                           launchRet);
                return ConvertCcuToHccl(launchRet);
            }
        }
    }

    // 9. NHR 内部 PostSync(若 nhrKernelNum > 1): threads[1] <- threads[2]
    if (hasNhr && nhrKernelNum > 1 && templateResource.threads.size() >= 3) {
        CHK_RET(PostSyncInterThreads(templateResource.threads[1],
            {templateResource.threads[2]}, {0}));
    }

    // 10. outer PostSync: threads[0] <- threads[1]
    if (hasNhr && templateResource.threads.size() >= 2) {
        CHK_RET(PostSyncInterThreads(templateResource.threads[0],
            {templateResource.threads[1]}, {NOTIFY_IDX_POST_SYNC}));
    }

    // 11. 存 submitInfos.cachedArgs,顺序 mesh 在前、NHR 在后
    if (hasMesh) {
        CcuKernelSubmitInfo meshSubmit;
        meshSubmit.kernelHandle = templateResource.ccuKernels[0];
        CHK_RET(FillCachedArgs(meshSubmit,
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::INPUT],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::OUTPUT],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::TOKEN],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::CUR_RANK_SLICE_IN_OFF],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::CUR_RANK_SLICE_OUT_OFF],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::TMP_REPEAT_NUM],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::INPUT_REPEAT_STRIDE],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::OUTPUT_REPEAT_STRIDE],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::NORMAL_SLICE_SIZE],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::LAST_SLICE_SIZE],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::IS_INPUT_OUTPUT_EQUAL],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::GO_SIZE_ADDR_OFFSET],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::GO_SIZE_LOOP_PARAM],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::GO_SIZE_PARALLEL_PARAM],
            meshTaskArgs[CcuAllGatherMesh1DMem2MemArgLayout::GO_SIZE_RESIDUAL],
            templateDataParams.buffInfo.inBuffBaseOff,
            templateDataParams.buffInfo.outBuffBaseOff));
        templateResource.submitInfos.push_back(meshSubmit);
    }
    if (hasNhr) {
        CcuKernelSubmitInfo nhrSubmit;
        nhrSubmit.kernelHandle = templateResource.ccuKernels[meshKernelNum];
        CHK_RET(FillCachedArgs(nhrSubmit,
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::INPUT],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::OUTPUT],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::TOKEN],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::DIE0_SIZE],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::DIE1_SIZE],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::REPEAT_NUM],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::INPUT_SLICE_STRIDE],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::OUTPUT_SLICE_STRIDE],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::INPUT_REPEAT_STRIDE],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::OUTPUT_REPEAT_STRIDE],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::IS_INPUT_OUTPUT_EQUAL],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::DIE0_LAST_SIZE],
            nhrTaskArgs[CcuAllGatherNHR1DMem2MemArgLayout::DIE1_LAST_SIZE],
            templateDataParams.buffInfo.inBuffBaseOff + meshSize,
            templateDataParams.buffInfo.outBuffBaseOff + meshSize,
            mySubCommRank_));
        templateResource.submitInfos.push_back(nhrSubmit);
        // NHR 双 die 时第二个 kernel 共用同一份 cachedArgs,单独存一份 submitInfo(不同 kernelHandle)
        if (nhrKernelNum > 1) {
            CcuKernelSubmitInfo nhrSubmit1 = nhrSubmit;
            nhrSubmit1.kernelHandle = templateResource.ccuKernels[meshKernelNum + 1];
            templateResource.submitInfos.push_back(nhrSubmit1);
        }
    }

    HCCL_INFO("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem][KernelRun] rank[%u] end.", myRank_);
    return HCCL_SUCCESS;
}

// FastLaunch: 从 KernelRun 保存的 cachedArgs 按 ArgLayout 常量重算 input/output 地址,再回放 launch。
// 不依赖子 template 类,全部索引引用 ArgLayout 常量。
HcclResult CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch(
    const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    u32 totalKernelNum = static_cast<u32>(tempFastLaunchCtx.ccuKernelSubmitInfos.size());
    if (totalKernelNum == 0) {
        HCCL_INFO("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] ccu kernel num is 0, just success.");

    }
    if (tempFastLaunchCtx.threads.size() < 1) {
        HCCL_ERROR("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] thread num is 0.");
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] start, totalKernelNum[%u], threadNum[%zu]",
               totalKernelNum, tempFastLaunchCtx.threads.size());

    u32 meshKernelNum = 1;
    u32 nhrKernelNum = (totalKernelNum > meshKernelNum) ? (totalKernelNum - meshKernelNum) : 0;
    bool hasMesh = (meshKernelNum > 0);
    bool hasNhr = (nhrKernelNum > 0);

    // 1. 更新 mesh kernel args
    if (hasMesh) {
        uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
        uint64_t inputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) +
            args[CcuAllGatherMesh1DMem2MemArgLayout::IN_BUFF_BASE_OFF];
        uint64_t outputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) +
            args[CcuAllGatherMesh1DMem2MemArgLayout::OUT_BUFF_BASE_OFF];
        uint64_t curRankSliceInOff = args[CcuAllGatherMesh1DMem2MemArgLayout::CUR_RANK_SLICE_IN_OFF];
        uint64_t curRankSliceOutOff = args[CcuAllGatherMesh1DMem2MemArgLayout::CUR_RANK_SLICE_OUT_OFF];
        bool inputOutputEqual = (inputAddr + curRankSliceInOff == outputAddr + curRankSliceOutOff);
        args[CcuAllGatherMesh1DMem2MemArgLayout::INPUT] = inputAddr;
        args[CcuAllGatherMesh1DMem2MemArgLayout::OUTPUT] = outputAddr;
        args[CcuAllGatherMesh1DMem2MemArgLayout::IS_INPUT_OUTPUT_EQUAL] = static_cast<uint64_t>(inputOutputEqual);
    }

    // 2. 更新 NHR kernel args
    if (hasNhr) {
        uint64_t *args = const_cast<uint64_t*>(
            tempFastLaunchCtx.ccuKernelSubmitInfos[meshKernelNum].cachedArgs);
        uint64_t inputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) +
            args[CcuAllGatherNHR1DMem2MemArgLayout::IN_BUFF_BASE_OFF];
        uint64_t outputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) +
            args[CcuAllGatherNHR1DMem2MemArgLayout::OUT_BUFF_BASE_OFF];
        uint64_t inputSliceStride = args[CcuAllGatherNHR1DMem2MemArgLayout::INPUT_SLICE_STRIDE];
        uint64_t outputSliceStride = args[CcuAllGatherNHR1DMem2MemArgLayout::OUTPUT_SLICE_STRIDE];
        uint64_t mySubCommRank = args[CcuAllGatherNHR1DMem2MemArgLayout::MY_SUB_COMM_RANK];
        bool inputOutputEqual = (inputAddr + inputSliceStride * mySubCommRank ==
                                 outputAddr + outputSliceStride * mySubCommRank);
        args[CcuAllGatherNHR1DMem2MemArgLayout::INPUT] = inputAddr;
        args[CcuAllGatherNHR1DMem2MemArgLayout::OUTPUT] = outputAddr;
        args[CcuAllGatherNHR1DMem2MemArgLayout::IS_INPUT_OUTPUT_EQUAL] = static_cast<uint64_t>(inputOutputEqual);
    }

    // 3. outer PreSync: threads[0] -> threads[1]
    if (hasNhr && tempFastLaunchCtx.threads.size() >= 2) {
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0],
            {tempFastLaunchCtx.threads[1]}, {NOTIFY_IDX_PRE_SYNC}));
    }

    // 4. launch mesh kernel
    if (hasMesh) {
        uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
        void *taskArgs = reinterpret_cast<void*>(args);
        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
            tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgs,
            CcuAllGatherMesh1DMem2MemArgLayout::ARG_SIZE);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] mesh kernel launch failed, ccuRet -> %d",
                       launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    // 5. NHR 内部 PreSync(若 nhrKernelNum > 1): threads[1] -> threads[2]
    if (hasNhr && nhrKernelNum > 1 && tempFastLaunchCtx.threads.size() >= 3) {
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[1],
            {tempFastLaunchCtx.threads[2]}, {0}));
    }

    // 6. launch NHR kernel(s),两 kernel 共享同一份 cachedArgs
    if (hasNhr) {
        uint64_t *args = const_cast<uint64_t*>(
            tempFastLaunchCtx.ccuKernelSubmitInfos[meshKernelNum].cachedArgs);
        void *taskArgs = reinterpret_cast<void*>(args);
        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[1],
            tempFastLaunchCtx.ccuKernelSubmitInfos[meshKernelNum].kernelHandle, taskArgs,
            CcuAllGatherNHR1DMem2MemArgLayout::ARG_SIZE);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] nhr kernel0 launch failed, ccuRet -> %d",
                       launchRet);
            return ConvertCcuToHccl(launchRet);
        }
        if (nhrKernelNum > 1 && tempFastLaunchCtx.threads.size() >= 3) {
            launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[2],
                tempFastLaunchCtx.ccuKernelSubmitInfos[meshKernelNum + 1].kernelHandle, taskArgs,
                CcuAllGatherNHR1DMem2MemArgLayout::ARG_SIZE);
            if (launchRet != CCU_SUCCESS) {
                HCCL_ERROR("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] nhr kernel1 launch failed, ccuRet -> %d",
                           launchRet);
                return ConvertCcuToHccl(launchRet);
            }
        }
    }

    // 7. NHR 内部 PostSync(若 nhrKernelNum > 1): threads[1] <- threads[2]
    if (hasNhr && nhrKernelNum > 1 && tempFastLaunchCtx.threads.size() >= 3) {
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[1],
            {tempFastLaunchCtx.threads[2]}, {0}));
    }

    // 8. outer PostSync: threads[0] <- threads[1]
    if (hasNhr && tempFastLaunchCtx.threads.size() >= 2) {
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0],
            {tempFastLaunchCtx.threads[1]}, {NOTIFY_IDX_POST_SYNC}));
    }

    HCCL_DEBUG("[CcuTempAllGatherMesh1DNHRConcurrentMem2Mem::FastLaunch] end");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
