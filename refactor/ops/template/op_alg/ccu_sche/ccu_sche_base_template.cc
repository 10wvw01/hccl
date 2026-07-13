/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_sche_base_template.h"

#include "alg_data_trans_wrapper.h"
#include "ccu_alg_utils.h"
#include "ccu_kernel_alg_base.h"
#include "ccu_launch_dl.h"
#include "log.h"

namespace ops_hccl {

CcuScheBaseTemplate::CcuScheBaseTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc,
                                         std::unique_ptr<IDieStrategy> dieStrategy)
    : BaseTemplate(myRank, std::move(ranks), HcclAlgEngineType::CCU_SCHED, templateDesc),
      dieStrategy_(std::move(dieStrategy))
{
    if (dieStrategy_ == nullptr) {
        HCCL_WARNING("[CcuScheBaseTemplate] dieStrategy is nullptr, die decision will fail at runtime.");
    }
}

HcclResult CcuScheBaseTemplate::DecideDie(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs,
                                          u64 sliceSize, u32 dataTypeSize, DieDecision &decision)
{
    if (dieStrategy_ == nullptr) {
        HCCL_ERROR("[CcuScheBaseTemplate::DecideDie] dieStrategy_ is nullptr.");
        return HCCL_E_INTERNAL;
    }
    return dieStrategy_->Decide(comm, myRank_, channelDescs, sliceSize, dataTypeSize,
                                templateRankSize_, decision);
}

HcclResult CcuScheBaseTemplate::BuildKernelInfo(HcclComm comm, AlgResourceRequest &res,
                                                 const DieDecision &dieDecision)
{
    (void)comm;
    void *kernelFunc = nullptr;
    const char *kernelName = nullptr;
    CHK_RET(GetKernelEntry(kernelFunc, kernelName));
    if (kernelFunc == nullptr || kernelName == nullptr) {
        HCCL_ERROR("[CcuScheBaseTemplate::BuildKernelInfo] GetKernelEntry returned null, func=%p name=%p",
                   kernelFunc, kernelName);
        return HCCL_E_INTERNAL;
    }

    for (u32 dieIdx = 0; dieIdx < dieDecision.dieNum; ++dieIdx) {
        if (dieIdx >= dieDecision.channelsPerDie.size()) {
            HCCL_ERROR("[CcuScheBaseTemplate::BuildKernelInfo] channelsPerDie size[%zu] < dieNum[%u]",
                       dieDecision.channelsPerDie.size(), dieDecision.dieNum);
            return HCCL_E_INTERNAL;
        }
        if (dieDecision.channelsPerDie[dieIdx].empty()) {
            HCCL_DEBUG("[CcuScheBaseTemplate::BuildKernelInfo] die[%u] has no channel, skip.", dieIdx);
            continue;
        }

        CcuKernelInfo kernelInfo;
        errno_t ret = strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), kernelName);
        if (ret != EOK) {
            HCCL_ERROR("[CcuScheBaseTemplate::BuildKernelInfo] strcpy_s fail, ret=%d", ret);
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = kernelFunc;

        std::shared_ptr<CcuKernelArgBaseSche> arg;
        CHK_RET(BuildKernelArg(arg));
        if (arg == nullptr) {
            HCCL_ERROR("[CcuScheBaseTemplate::BuildKernelInfo] BuildKernelArg returned nullptr.");
            return HCCL_E_INTERNAL;
        }
        // 2Die peer 集分片：每 die 的 ifHandleSelfRank 互补（迁移自 ccu_temp_all_gather_2dies_mesh_1D.cc:94-106）
        if (dieDecision.dieNum > 1 && dieIdx < dieDecision.rankIdGroupPerDie.size()) {
            arg->ifHandleSelfRank = (dieIdx == 0) ? dieDecision.ifHandleSelfRank
                                                   : !dieDecision.ifHandleSelfRank;
        }
        kernelInfo.setKernelArg(arg);
        kernelInfo.channels = dieDecision.channelsPerDie[dieIdx];
        res.ccuKernelInfos.push_back(std::move(kernelInfo));
    }

    res.ccuKernelNum.push_back(static_cast<u32>(res.ccuKernelInfos.size()));
    // SCHE 2Die 时 slaveThreadNum=1（迁移自 ccu_temp_all_gather_2dies_mesh_1D.cc:67-70）
    res.slaveThreadNum = (dieDecision.dieNum > 1) ? (dieDecision.dieNum - 1) : 0;
    res.notifyNumOnMainThread = (dieDecision.dieNum > 1) ? 1 : 0;
    res.notifyNumPerThread.assign(res.slaveThreadNum, 1);

    HCCL_INFO("[CcuScheBaseTemplate::BuildKernelInfo] dieNum=%u, ccuKernelInfos.size=%zu, slaveThreadNum=%u",
              dieDecision.dieNum, res.ccuKernelInfos.size(), res.slaveThreadNum);
    return HCCL_SUCCESS;
}

HcclResult CcuScheBaseTemplate::KernelRun(const TemplateDataParams &params, TemplateResource &templateResource,
                                          std::vector<u32> &ranksForOutputData)
{
    HCCL_INFO("[CcuScheBaseTemplate::KernelRun] start, sliceSize=%llu, tailSize=%llu",
              params.sliceSize, params.tailSize);

    buffInfo_ = params.buffInfo;
    const u32 dataTypeSize = DataTypeSizeGet(params.dataType);
    const u64 dataCount = (dataTypeSize == 0) ? 0 : (params.sliceSize / dataTypeSize);
    if (dataCount == 0 && params.tailSize == 0) {
        HCCL_INFO("[CcuScheBaseTemplate::KernelRun] DataCount == 0, Template Run Ends.");
        ranksForOutputData = ranks;
        return HCCL_SUCCESS;
    }

    // 1. Die 决策（SCHE 仅用 peer 集分片，2Die 时每 die 独立 kernel）
    DieDecision dieDecision;
    CHK_RET(DecideDie(nullptr, channels, params.sliceSize, dataTypeSize, dieDecision));

    // 2. 构造 LoopGroupConfig + CalGoSize（迁移自 ccu_temp_all_gather_mesh_1D.cc:135-139）
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_DEFAULT_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE;
    const std::vector<uint64_t> goSize = CalGoSize(params.sliceSize, config);

    // 3. 构造 taskArgs（8 字段，子类决定具体布局）
    std::vector<uint64_t> taskArgs;
    uint64_t argSize = 0;
    CHK_RET(BuildTaskArgs(params, dieDecision, goSize, taskArgs, argSize));

    const u32 kernelNum = static_cast<u32>(templateResource.ccuKernels.size());
    if (kernelNum == 0) {
        HCCL_ERROR("[CcuScheBaseTemplate::KernelRun] ccuKernels is empty.");
        return HCCL_E_INTERNAL;
    }

    // 4. 多 die 时前流同步（迁移自 ccu_temp_all_gather_2dies_mesh_1D.cc:155-157）
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                                              templateResource.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    // 5. 遍历 die 下发 kernel（迁移自 ccu_temp_all_gather_2dies_mesh_1D.cc:160-168）
    for (u32 dieIdx = 0; dieIdx < kernelNum; ++dieIdx) {
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[dieIdx],
                                                    templateResource.ccuKernels[dieIdx],
                                                    taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuScheBaseTemplate::KernelRun] kernel launch failed, dieIdx=%u, ccuRet=%d",
                       dieIdx, launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    // 6. 多 die 时后流同步（迁移自 ccu_temp_all_gather_2dies_mesh_1D.cc:171-172）
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                                              templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    // 7. 保存 SubmitInfo 供 FastLaunch 复用
    CHK_RET(SaveSubmitInfo(taskArgs, templateResource));

    // 8. 默认填充 ranksForOutputData（AllGather 语义：输出对应全部 rank）
    ranksForOutputData = ranks;

    HCCL_INFO("[CcuScheBaseTemplate::KernelRun] end, kernelNum=%u", kernelNum);
    return HCCL_SUCCESS;
}

HcclResult CcuScheBaseTemplate::FastLaunch(const OpParam &param, const TemplateFastLaunchCtx &ctx)
{
    (void)param;
    HCCL_INFO("[CcuScheBaseTemplate::FastLaunch] start.");

    if (ctx.ccuKernelSubmitInfos.empty()) {
        HCCL_INFO("[CcuScheBaseTemplate::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }

    // cachedArgs 布局约定（迁移自 ccu_temp_all_gather_mesh_1D.cc:81-109）：
    //   [0]           = inputAddr
    //   [1]           = outputAddr
    //   [argSize]     = inBuffBaseOff
    //   [argSize+1]   = outBuffBaseOff
    // SCHE 无 scratchAddr，末尾仅 2 个 offset 字段
    const uint64_t argSize = GetTaskArgSize();
    constexpr u32 INPUT_IDX = 0;
    constexpr u32 OUTPUT_IDX = 1;
    const u32 inputOffsetIdx = static_cast<u32>(argSize);
    const u32 outputOffsetIdx = static_cast<u32>(argSize) + 1;

    uint64_t *args = const_cast<uint64_t *>(ctx.ccuKernelSubmitInfos[0].cachedArgs);

    args[INPUT_IDX] = ccu_alg_utils::PointerToAddr(ctx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[OUTPUT_IDX] = ccu_alg_utils::PointerToAddr(ctx.buffInfo.outputPtr) + args[outputOffsetIdx];

    void *taskArgs = reinterpret_cast<void *>(args);
    CcuResult launchRet = HcommCcuKernelLaunch(ctx.threads[0],
                                                ctx.ccuKernelSubmitInfos[0].kernelHandle,
                                                taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuScheBaseTemplate::FastLaunch] kernel launch failed, ccuRet=%d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_INFO("[CcuScheBaseTemplate::FastLaunch] end, argSize=%llu.", argSize);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
