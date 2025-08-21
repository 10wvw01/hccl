 
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "asymmetric_hierarchical_concatenate_alg_template_base.h"
 
#include <iostream>
#include <fstream>
 
namespace hccl {
 
AHCAlgTemplateBase::AHCAlgTemplateBase(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher), rankSize_(1), extendFlag_(false), fftsPhase_(1) 
{
}
 
AHCAlgTemplateBase::~AHCAlgTemplateBase()
{
}

HcclResult AHCAlgTemplateBase::Prepare(u64 reduceAttrBitMap, HcomCollOpInfo *opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

HcclResult AHCAlgTemplateBase::Prepare(u64 totalCount, const std::vector<std::vector<std::vector<u32>>> &globalSubGroups,
    std::map<AHCConcOpType, TemplateType> &ahcAlgOption, bool extendFlag, AHCExtendPreparePara extendPara)
{
    globalSubGroups_ = globalSubGroups;
    totalCount_ = totalCount;
    ahcAlgOption_ = ahcAlgOption;
    extendFlag_ = extendFlag;
    ahcExtendPreparePara_ = extendPara;
    return HCCL_SUCCESS;
}

HcclResult AHCAlgTemplateBase::DisposeSubGroups(const u32 rank)
{
    return HCCL_SUCCESS;
}
 
HcclResult AHCAlgTemplateBase::CommAHCInfoInit()
{
    return HCCL_SUCCESS;
}
 
HcclResult AHCAlgTemplateBase::GetFftsPhase(u32 &fftsPhase) const
{
    fftsPhase = fftsPhase_;
    return HCCL_SUCCESS;
}
 
HcclResult AHCAlgTemplateBase::SetFftsPhase(u32 &fftsPhase)
{
    fftsPhase_ = fftsPhase;
    return HCCL_SUCCESS;
}
 
HcclResult AHCAlgTemplateBase::PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    CHK_PRT_RET(!outputMem_ || !inputMem_,
        HCCL_ERROR("[AHCAlgTemplateBase][PrepareRunAsync]rank[%u] run_async inputmem or outputmem is null", rank), HCCL_E_PTR);
 
    HCCL_INFO("AHCAlgTemplateBase run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);
 
    CHK_PRT_RET(links.size() < rankSize, HCCL_ERROR("[AHCAlgTemplateBase][PrepareRunAsync]rank[%u] linksize[%llu] is less "\
        "than rankSize[%u]", rank, links.size(), rankSize), HCCL_E_INTERNAL);
 
    // 如果ranksize为1, inline reduce和普通跨片reduce操作一致，从input->output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AHCAlgTemplateBase][PrepareRunAsync]rank[%u] memcpy async failed", rank), ret);
        }
        return ret;
    }
 
    DisposeSubGroups(rank);
 
    rankSize_ = rankSize;
 
    CommAHCInfoInit();
 
    // 检查、并清空slices_
    if (slices_.size() != 0) {
        HCCL_WARNING("[AHCAlgTemplateBase][PrepareRunAsync]AHCAlgTemplateBase not supported passing in parameter slice_, "\
            "otherwise will be cleared");
        slices_.clear();
    }
 
    return HCCL_SUCCESS;
}

HcclResult AHCAlgTemplateBase::PrepareAlgTemplate(std::unique_ptr<AlgTemplateBase> &tempAlg, const std::vector<Slice> &slices, AHCOpType opType)
{
    HcclResult ret = HCCL_SUCCESS;
    switch (opType) {
        case AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER: {
            ret = tempAlg->Prepare(inputMem_, inputMem_, scratchMem_, count_, dataType_,
                stream_, reductionOp_, root_, slices, baseOffset_);
            break;
        }
        case AHCOpType::AHC_OP_TYPE_ALLGATHER: {
            ret = tempAlg->Prepare(outputMem_, outputMem_, scratchMem_, count_, dataType_,
                stream_, reductionOp_, root_, slices, baseOffset_);
            break;
        }
        case AHCOpType::AHC_OP_TYPE_ALLREDUCE: {
            ret = tempAlg->Prepare(inputMem_, outputMem_, scratchMem_, count_, dataType_,
                stream_, reductionOp_, root_, slices, baseOffset_);
            break;
        }
        case AHCOpType::AHC_OP_TYPE_RESERVED:{
            // 其他算子不支持，直接返回
            ret = HCCL_E_PARA;
            break;
        }
    }

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AHCAlgTemplateBase][PrepareAlgTemplate] prepare step error"), ret);
     
    return ret;
}
 
HcclResult AHCAlgTemplateBase::MemcpyForSingleOp(const u32 rank, AHCOpType opType)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 commRank = commAHCBaseInfo_->GetCommRank(rank);
    HCCL_DEBUG("[AHCAlgTemplateBase][MemcpyForSingleOp] rank[%u] commRank[%u]", rank, commRank);
    switch (opType) {
        case AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER:{
            u64 srcSize = (inputMem_.size() - commRank * count_ * DataUnitSize(dataType_)) > count_ * DataUnitSize(dataType_) ?
                count_ * DataUnitSize(dataType_) : (inputMem_.size() - commRank * count_ * DataUnitSize(dataType_)); 
            DeviceMem srcMem = inputMem_.range(commRank * count_ * DataUnitSize(dataType_), srcSize);
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, srcMem, stream_);
            break;
        }
        case AHCOpType::AHC_OP_TYPE_ALLGATHER:{
            u64 dstSize = (outputMem_.size() - commRank * count_ * DataUnitSize(dataType_)) > count_ * DataUnitSize(dataType_) ?
                count_ * DataUnitSize(dataType_) : (outputMem_.size() - commRank * count_ * DataUnitSize(dataType_));
            DeviceMem dstMem = outputMem_.range(commRank * count_ * DataUnitSize(dataType_), dstSize);
            ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, inputMem_, stream_);
            break;
        }
        case AHCOpType::AHC_OP_TYPE_ALLREDUCE:{
            // 使用 RS+AG 实现 AR 时，需要在 RS 完成时进行一次额外的数据搬运
            ret = HcclD2DMemcpyAsync(dispatcher_, inputMem_, outputMem_, stream_);
            break;
        }
        case AHCOpType::AHC_OP_TYPE_RESERVED:{
            // 其他算子不支持，无需copy，直接返回
            break;
        }
    }
    return ret;
}
 
HcclResult AHCAlgTemplateBase::RunInstance(const u32 rank, const std::vector<LINK> &links, const std::vector<Slice> &slices,
        std::unique_ptr<AlgTemplateBase> &tempAlg, AHCOpType opType)
{
    HcclResult ret = HCCL_SUCCESS;
 
    // 判断是否关闭reducescatter的barrier
    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }
    // 调用算法执行
    ret = PrepareAlgTemplate(tempAlg, slices, opType);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AHCAlgTemplateBase][RunInstance]rank[%u] prepare optype[%d] failed", rank, opType), ret);
  
    ret = tempAlg->RegisterProfiler(
        profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AHCAlgTemplateBase][RunInstance]rank[%u] registerProfiler optype[%d] failed", rank, opType), ret);
    
    ret = tempAlg->RunAsync(rank, links.size(), links);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AHCAlgTemplateBase][RunInstance]rank[%u] run optype[%d] failed", rank, opType), ret);
 
    return ret;
}
 
ReduceScatterAHCBase::ReduceScatterAHCBase(const HcclDispatcher dispatcher)
    : AHCAlgTemplateBase(dispatcher)
{
}
 
ReduceScatterAHCBase::~ReduceScatterAHCBase()
{
}
 
HcclResult ReduceScatterAHCBase::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<LINK> &links)
{
    HCCL_INFO("[ReduceScatterAHCBase][RunAsync] start rank[%u] rankSize[%u]", rank, rankSize);
 
    HcclResult ret = HCCL_SUCCESS;
    ret = PrepareRunAsync(rank, rankSize, links);
    HCCL_DEBUG("[ReduceScatterAHCBase][RunAsync] inputmem.size[%llu] outputmem.size[%llu] count[%llu]", inputMem_.size(), outputMem_.size(), count_);
 
    commAHCBaseInfo_->SetTotalSize(inputMem_.size());
 
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterAHCBase][RunAsync]rank[%u] count[%llu] failed in PrepareRunAsync step", rank, count_), ret);
 
    CHK_PRT_RET(rankSize == 1, HCCL_INFO("[ReduceScatterAHCBase][RunAsync] rankSize[%u], do nothing.",
        rankSize), HCCL_SUCCESS);
 
    HCCL_DEBUG("[ReduceScatterAHCBase][RunAsync] rank[%u] begin intra rs", rank);
 
    // 做组内 reduce-scatter
    ret = RunIntraReduceScatter(rank, links, commAHCBaseInfo_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceScatterAHCBase][RunAsync]rank[%u] count[%llu] failed in "\
        "RunIntraReduceScatter  step", rank, count_), ret);
 
    HCCL_DEBUG("[ReduceScatterAHCBase][RunAsync] rank[%u] end intra rs begin inter", rank);
 
    // 做组间 reduce-scatter
    ret = RunInterReduceScatter(rank, links, commAHCBaseInfo_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceScatterAHCBase][RunAsync]rank[%u] count[%llu] failed in "\
        "RunInterReduceScatter step", rank, count_), ret);
 
    // 对于单独的 Reduce-scatter 算子，在运算结束时进行数据搬运
    if (inputMem_ != outputMem_) {
        ret = MemcpyForSingleOp(rank, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceScatterAHCBase][RunAsync]rank[%u] memcpy failed", rank), ret);
    }
 
    HCCL_DEBUG("[ReduceScatterAHCBase][RunAsync] rank[%u] end inter rs", rank);
 
    HCCL_INFO("[ReduceScatterAHCBase][RunAsync] finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
 
HcclResult ReduceScatterAHCBase::RunIntraReduceScatter(const u32 rank, const std::vector<LINK> &links,
    const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo)
{
    // 获取当前rank的组内rank
    HcclResult ret = HCCL_SUCCESS;
    HCCL_INFO("[ReduceScatterAHC][RunIntraReduceScatter] begin intra reduce scatter rank[%u] count[%llu]", rank, count_);
 
    u32 intraRank = commAHCBaseInfo->GetIntraRank(rank);
 
    // 创建执行算子实列
    std::unique_ptr<AlgTemplateBase> tempAlg;
    commAHCBaseInfo->GetIntraAlgTemplateOpInstance(AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER, tempAlg, dispatcher_, reduceAttr_,
        extendFlag_, ahcExtendPreparePara_);
 
    std::vector<std::vector<Slice>> intraSlicesVector;
    std::vector<std::vector<LINK>> intraLinksVector;
    CHK_RET(commAHCBaseInfo->CalcIntraSlicesAndLinks(rank, DataUnitSize(dataType_), count_, links, intraLinksVector, intraSlicesVector));
 
    HCCL_DEBUG("[ReduceScatterAHCBase][RunIntraReduceScatter] run inst rank[%u] intraRank[%u]",
        rank, intraRank);
 
    for (u32 i = 0; i < intraLinksVector.size(); i++) {
        std::vector<Slice> intraSlices = intraSlicesVector[i];
        std::vector<LINK> intraLinks = intraLinksVector[i];
        if (intraLinks.size() <= 1 ) {
            continue;
        }
        CHK_RET(RunInstance(intraRank, intraLinks, intraSlices, tempAlg, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER));
    }
 
    HCCL_DEBUG("[ReduceScatterAHCBase][RunIntraReduceScatter] end intra reduce scatter rank[%u]", rank);
 
    return ret;
}
 
AllGatherAHCBase::AllGatherAHCBase(const HcclDispatcher dispatcher)
    : AHCAlgTemplateBase(dispatcher)
{
}
 
AllGatherAHCBase::~AllGatherAHCBase()
{
}
 
HcclResult AllGatherAHCBase::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<LINK> &links)
{
    HCCL_INFO("[AllGatherAHCBase][RunAsync] start rank[%u] rankSize[%u]", rank, rankSize);
 
    HcclResult ret = HCCL_SUCCESS;
    ret = PrepareRunAsync(rank, rankSize, links);
 
    commAHCBaseInfo_->SetTotalSize(outputMem_.size());
 
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherAHCBase][RunAsync]rank[%u] count[%llu] failed in PrepareRunAsync step", rank, count_), ret);
 
    CHK_PRT_RET(rankSize == 1, HCCL_INFO("[AllGatherAHCBase][RunAsync] rankSize[%u], do nothing.",
        rankSize), HCCL_SUCCESS);
 
    HCCL_DEBUG("[AllGatherAHCBase][RunAsync] rank[%u] begin intra ag", rank);
 
    // 对于单独的 All-gather 算子，在运算开始时进行数据搬运
    if (inputMem_ != outputMem_) {
        ret = MemcpyForSingleOp(rank, AHCOpType::AHC_OP_TYPE_ALLGATHER);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[AllGatherAHCBase][RunAsync]rank[%u] memcpy failed", rank), ret);
    }
 
    // 做组间 all-gather
    ret = RunInterAllGather(rank, links, commAHCBaseInfo_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherAHCBase][RunAsync]rank[%u] count[%llu] failed in "\
        "RunInterAllGather step", rank, count_), ret);
 
    HCCL_DEBUG("[AllGatherAHCBase][RunAsync] rank[%u] end inter ag", rank);
 
    // 做组内 allgather
    ret = RunIntraAllGather(rank, links, commAHCBaseInfo_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherAHCBase][RunAsync]rank[%u] count[%llu] failed in "\
        "RunIntraAllGather  step", rank, count_), ret);
 
    HCCL_DEBUG("[AllGatherAHCBase][RunAsync] rank[%u] end intra ag begin inter", rank);
 
    HCCL_INFO("[AllGatherAHCBase][RunAsync] finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
 
HcclResult AllGatherAHCBase::RunIntraAllGather(const u32 rank, const std::vector<LINK> &links,
    const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo)
{
    // 获取当前rank的组内rank
    HCCL_INFO("[AllGatherAHCBase][RunIntraAllGather] begin intra all gather rank[%u]", rank);
 
    u32 intraRank = commAHCBaseInfo->GetIntraRank(rank);
 
    // 创建执行算子实列
    std::unique_ptr<AlgTemplateBase> tempAlg;
    commAHCBaseInfo->GetIntraAlgTemplateOpInstance(AHCOpType::AHC_OP_TYPE_ALLGATHER, tempAlg, dispatcher_, reduceAttr_,
        extendFlag_, ahcExtendPreparePara_);
 
    std::vector<std::vector<Slice>> intraSlicesVector;
    std::vector<std::vector<LINK>> intraLinksVector;
    CHK_RET(commAHCBaseInfo->CalcIntraSlicesAndLinks(rank, DataUnitSize(dataType_), count_, links, intraLinksVector, intraSlicesVector));
 
    HCCL_DEBUG("[AllGatherAHCBase][RunIntraAllGather] run inst rank[%u] intraRank[%u]",
        rank, intraRank);
 
    for (u32 i = 0; i < intraLinksVector.size(); i++) {
        std::vector<Slice> intraSlices = intraSlicesVector[i];
        std::vector<LINK> intraLinks = intraLinksVector[i];
        if (intraLinks.size() <= 1) {
            continue;
        }
        CHK_RET(RunInstance(intraRank, intraLinks, intraSlices, tempAlg, AHCOpType::AHC_OP_TYPE_ALLGATHER));
    }
 
    HCCL_DEBUG("[AllGatherAHCBase][RunIntraAllGather] end intra all gather rank[%u]", rank);
 
    return HCCL_SUCCESS;
}

AllReduceAHCBase::AllReduceAHCBase(const HcclDispatcher dispatcher)
    : AHCAlgTemplateBase(dispatcher)
{
}
 
AllReduceAHCBase::~AllReduceAHCBase()
{
}
 
HcclResult AllReduceAHCBase::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<LINK> &links)
{  
    HCCL_INFO("[AllReduceAHCBase][RunAsync] start rank[%u] rankSize[%u]", rank, rankSize);
 
    HcclResult ret = HCCL_SUCCESS;
    ret = PrepareRunAsync(rank, rankSize, links);
 
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceAHCBase][RunAsync]rank[%u] count[%llu] failed in PrepareRunAsync step", rank, count_), ret);
 
    CHK_PRT_RET(rankSize == 1, HCCL_INFO("[AllReduceAHCBase][RunAsync] rankSize[%u], do nothing.",
        rankSize), HCCL_SUCCESS);
 
    CHK_PRT_RET(count_ == 0, HCCL_INFO("[AllReduceAHCBase][RunAsync] count_[%u], do nothing.", count_), HCCL_SUCCESS);

    HCCL_DEBUG("[AllReduceAHCBase][RunAsync] rank[%u] begin intra rs", rank);
 
    ret = RunIntraReduceScatter(rank, links, commAHCBaseInfo_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceAHCBase][RunAsync]rank[%u] count[%llu] failed in "\
        "RunIntraReduceScatter  step", rank, count_), ret);
 
    HCCL_DEBUG("[AllReduceAHCBase][RunAsync] rank[%u] end intra rs begin inter", rank);
 
    // 垂直方向做allreduce ring
    ret = RunInterAllReduce(rank, links, commAHCBaseInfo_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceAHCBase][RunAsync]rank[%u] count[%llu] failed in "\
        "RunInterAllReduce step", rank, count_), ret);
 
    HCCL_DEBUG("[AllReduceAHCBase][RunAsync] rank[%u] end inter begin intra ag", rank);
 
    // 水平方向做broken allgather ring
    ret = RunIntraAllGather(rank, links, commAHCBaseInfo_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceAHCBase][RunAsync]rank[%u] count[%llu] failed in "\
        "RunIntraAllGather step", rank, count_), ret);
 
    HCCL_DEBUG("[AllReduceAHCBase][RunAsync] rank[%u] end intra ag", rank);
 
    HCCL_INFO("[AllReduceAHCBase][RunAsync] finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
 
HcclResult AllReduceAHCBase::RunIntraReduceScatter(const u32 rank, const std::vector<LINK> &links,
    const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo)
{
    // 获取当前rank的组内rank
    HCCL_INFO("[AllReduceAHCBase][RunIntraReduceScatter] begin intra reduce scatter rank[%u]", rank);
 
    u32 intraRank = commAHCBaseInfo->GetIntraRank(rank);
 
    // 创建执行算子实列
    std::unique_ptr<AlgTemplateBase> tempAlg;
    commAHCBaseInfo->GetIntraAlgTemplateOpInstance(AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER, tempAlg, dispatcher_, reduceAttr_,
        extendFlag_, ahcExtendPreparePara_);
 
    std::vector<Slice> intraSlices;
    std::vector<LINK> intraLinks;
    CHK_RET(commAHCBaseInfo->CalcIntraSlicesAndLinks(rank, DataUnitSize(dataType_), count_, links, intraLinks, intraSlices));
 
    // 长度不足2，直接跳过
    if (intraLinks.size() <= 1) {
        return HCCL_SUCCESS;
    }
 
    HCCL_DEBUG("[AllReduceAHCBase][RunIntraReduceScatter] run inst rank[%u] intraRank[%u], IntraSize=%u",
        rank, intraRank, intraLinks.size());
 
    CHK_RET(RunInstance(intraRank, intraLinks, intraSlices, tempAlg, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER));
 
    HCCL_DEBUG("[AllReduceAHCBase][RunIntraReduceScatter] end intra reduce scatter rank[%u]", rank);
 
    return HCCL_SUCCESS;
}
 
HcclResult AllReduceAHCBase::RunIntraAllGather(const u32 rank, const std::vector<LINK> &links,
    const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo)
{
    HCCL_INFO("[AllReduceAHCBase][RunIntraAllGather] begin intra allgather rank[%u]", rank);
 
    // 获取当前rank的组内rank
    u32 intraRank = commAHCBaseInfo->GetIntraRank(rank);
 
    // 创建执行算子实列
    std::unique_ptr<AlgTemplateBase> tempAlg;
    commAHCBaseInfo->GetIntraAlgTemplateOpInstance(AHCOpType::AHC_OP_TYPE_ALLGATHER, tempAlg, dispatcher_, reduceAttr_,
        extendFlag_, ahcExtendPreparePara_);
 
    std::vector<Slice> intraSlices;
    std::vector<LINK> intraLinks;
 
    CHK_RET(commAHCBaseInfo->CalcIntraSlicesAndLinks(rank, DataUnitSize(dataType_), count_, links, intraLinks, intraSlices));
 
    // 长度不足2，直接跳过
    if (intraLinks.size() <= 1) {
        return HCCL_SUCCESS;
    }
 
    HCCL_DEBUG("[AllReduceAHCBase][RunIntraAllGather] run inst rank[%u] intraRank[%u], IntraSize=%u",
        rank, intraRank, intraLinks.size());
 
    CHK_RET(RunInstance(intraRank, intraLinks, intraSlices, tempAlg, AHCOpType::AHC_OP_TYPE_ALLGATHER));
 
    HCCL_DEBUG("[AllReduceAHCBase][RunIntraAllGather] end intra allgather rank[%u]", rank);
    return HCCL_SUCCESS;
}
 
}   // ~~ namespace hccl