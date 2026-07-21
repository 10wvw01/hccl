/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_v2_template_base.h"
#include "exec_timeout_manager.h"

namespace ops_hccl {

InsAlgTemplateBase::InsAlgTemplateBase(
    const OpParam &param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : opMode_(param.opMode), root_(param.root), myRank_(rankId),
      subCommRanks_(subCommRanks), reduceOp_(param.reduceType), enableDetour_(param.enableDetour)
{
    if (subCommRanks.size() > 1) {
        templateRankSize_ = subCommRanks[0].size() * subCommRanks[1].size();
    } else {
        templateRankSize_ = subCommRanks[0].size();
    }
}

InsAlgTemplateBase::~InsAlgTemplateBase()
{
}

HcclResult InsAlgTemplateBase::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    (void)tempFastLaunchCtx;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of InsAlgTemplateBase::FastLaunch!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::KernelRun(const OpParam& param,
                                         const TemplateDataParams& tempAlgParams,
                                         TemplateResource& templateResource)
{
    (void)param;
    (void)tempAlgParams;
    (void)templateResource;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of kernel run!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::KernelRunCommon(const OpParam& param,
    const TemplateDataParams& tempAlgParams, TemplateResource& templateResource,
    const std::string& tag)
{
    HCCL_INFO("[%s][KernelRun] Start, threadNum[%u], count[%llu], "
        "dataType[%u], deterministicStrict[%d]", tag.c_str(), threadNum_, count_, dataType_, deterministicStrict_);

    // 步骤1: 执行预处理本地拷贝（将本rank对应的数据从用户输入拷贝到临时缓冲区）
    CHK_RET(PreLocalCopy(tempAlgParams, templateResource.threads));

    // 多线程同步：如果线程数大于1，等待子线程就绪，为all2all做准备
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // 步骤2: 执行AllToAll操作
    CHK_RET(RunAllToAll(templateResource.channels, templateResource.threads, tempAlgParams));

    // 多线程同步：如果线程数大于1，需要在操作完成后同步，等待子线程完成
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    if (dataType_ == HCCL_DATA_TYPE_FP64 || reduceOp_ == HcclReduceOp::HCCL_REDUCE_PROD) {
        // 必须确保所有通信任务完成，因为接下来的 AICPU Reduce 运行在 CPU 上，不感知任务队列同步
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto &thread : templateResource.threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, ExecTimeoutManager::Instance().GetExecTimeout())));
        }
    }

    // 步骤3: 执行本地归约操作（将收到的所有数据在本地进行归约）
    CHK_RET(RunLocalReduce(templateResource.threads, tempAlgParams));

    // 步骤4: 执行后处理拷贝（将归约结果从临时缓冲区拷贝到用户输出缓冲区）
    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));

    HCCL_INFO("[%s][KernelRun] End", tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult InsAlgTemplateBase::DPUKernelRun(const TemplateDataParams& tempAlgParam,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
    (void)tempAlgParam;
    (void)channels;
    (void)myRank;
    (void)subCommRanks;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of dpu kernel run!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::PreLocalCopy(const TemplateDataParams &tempAlgParams,
    const std::vector<ThreadHandle> &threads)
{
    (void)tempAlgParams;
    (void)threads;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of PreLocalCopy!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::RunAllToAll(const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    (void)channels;
    (void)threads;
    (void)tempAlgParams;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of RunAllToAll!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::RunLocalReduce(const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    (void)threads;
    (void)tempAlgParams;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of RunLocalReduce!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::PostCopy(const TemplateDataParams &tempAlgParams,
    const std::vector<ThreadHandle> &threads)
{
    (void)tempAlgParams;
    (void)threads;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of PostCopy!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                       AlgResourceRequest& resourceRequest)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)resourceRequest;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::GetRes(AlgResourceRequest& resourceRequest) const
{
    (void)resourceRequest;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

u64 InsAlgTemplateBase::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 0;
}

u64 InsAlgTemplateBase::GetThreadNum() const
{
    return 0;
}

bool InsAlgTemplateBase::IsPcieProtocol(const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    for (auto it = channels.begin(); it != channels.end(); it++) {
        if ((it->second).at(0).protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
            HCCL_DEBUG("[IsPcieProtocol] the protocol of channel is PCIE");
            return true;
        }
    }
    HCCL_DEBUG("[IsPcieProtocol] the protocol of channel is Non-PCIE");
    return false;
}

}