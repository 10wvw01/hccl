/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_sequence_executor_aicpu_3level.h"
#include "ins_temp_scatter_mesh_1D_intra.h"
#include "ins_temp_scatter_nhr_dpu_inter.h"
#include "ins_temp_allgather_nhr_dpu_inter.h"
#include "ins_temp_allgather_mesh_1D_intra.h"
#include "ins_temp_scatter_mesh_1D.h"
#include "ins_temp_scatter_nhr.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::InsV2BroadcastSequenceExecutorAicpu3Level() {}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::InitCommInfo(HcclComm comm, const OpParam &param,
        const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][InitCommInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]",
        myRank_,
        rankSize_,
        dataTypeSize_);
    return HCCL_SUCCESS;
}

// 实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    rankSizeLevel0_ = algHierarchyInfo.infos[0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1].size();
    rankSizeLevel2_ = algHierarchyInfo.infos[2].size();
    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][CalcRes] rankSizeLevel0 [%u], rankSizeLevel1 [%u], rankSizeLevel2 [%u]", rankSizeLevel0_,
        rankSizeLevel1_, rankSizeLevel2_);

    std::shared_ptr<InsAlgTemplate0> ScatterL0TempAlg =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> ScatterL1TempAlg =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate1> ScatterL2TempAlg =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[2]);
    std::shared_ptr<InsAlgTemplate2> agL2TempAlg =
        std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[2]);
    std::shared_ptr<InsAlgTemplate2> agL1TempAlg =
        std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate3> agL0TempAlg =
        std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqScatterL0;
    AlgResourceRequest resReqScatterL1;
    AlgResourceRequest resReqScatterL2;
    AlgResourceRequest resReqAGL2;
    AlgResourceRequest resReqAGL1;
    AlgResourceRequest resReqAGL0;

    CHK_RET(ScatterL0TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL0));
    CHK_RET(ScatterL1TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL1));
    CHK_RET(ScatterL2TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL2));
    CHK_RET(agL2TempAlg->CalcRes(comm, param, topoInfo, resReqAGL2));
    CHK_RET(agL1TempAlg->CalcRes(comm, param, topoInfo, resReqAGL1));
    CHK_RET(agL0TempAlg->CalcRes(comm, param, topoInfo, resReqAGL0));

    // step1在完成后，完成后同步后展开step2，因此slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum = std::max({resReqScatterL0.slaveThreadNum,
        resReqScatterL1.slaveThreadNum,
        resReqScatterL2.slaveThreadNum,
        resReqAGL2.slaveThreadNum,
        resReqAGL1.slaveThreadNum,
        resReqAGL0.slaveThreadNum});
    resourceRequest.notifyNumPerThread = std::max({resReqScatterL0.notifyNumPerThread,
        resReqScatterL1.notifyNumPerThread,
        resReqScatterL2.notifyNumPerThread,
        resReqAGL2.notifyNumPerThread,
        resReqAGL1.notifyNumPerThread,
        resReqAGL0.notifyNumPerThread});
    resourceRequest.notifyNumOnMainThread = std::max({resReqScatterL0.notifyNumOnMainThread,
        resReqScatterL1.notifyNumOnMainThread,
        resReqScatterL2.notifyNumOnMainThread,
        resReqAGL2.notifyNumOnMainThread,
        resReqAGL1.notifyNumOnMainThread,
        resReqAGL0.notifyNumOnMainThread});

    u64 channelsSize = 3;
    resourceRequest.channels.resize(channelsSize);
    resourceRequest.channels[0] = resReqScatterL0.channels[0];
    resourceRequest.channels[1] = resReqScatterL1.channels[0];
    resourceRequest.channels[2] = resReqScatterL2.channels[0];

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][Orchestrate] Orchestrate Start");
    // 参数填充
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    CHK_RET(InitExecutorInfo(param, resCtx));
    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2BroadcastSequenceExecutorAicpu3Level][Orchestrate]errNo[0x%016llx] Broadcast excutor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::InitExecutorInfo(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = (myRank_ / algHierarchyInfo_.infos[0][0].size()) % algHierarchyInfo_.infos[1][0].size();

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
        
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][InitExecutorInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]",
        +myRank_,
        rankSize_,
        dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][OrchestrateLoop] Start");

    // scatter L0
    TemplateDataParams tempAlgParamsScatterL0;
    tempAlgParamsScatterL0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsScatterL0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL0.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate0> algTemplateScatterL0 =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);

    // scatter L1
    TemplateDataParams tempAlgParamsScatterL1;
    tempAlgParamsScatterL1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL1.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate1> algTemplateScatterL1 =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);

    // scatter L2
    TemplateDataParams tempAlgParamsScatterL2;
    tempAlgParamsScatterL2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL2.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate2> algTemplateScatterL2 =
        std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);

    // AG L2
    TemplateDataParams tempAlgParamsAllGatherL2;
    tempAlgParamsAllGatherL2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL2.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate3> algTemplateAllGatherL2 =
        std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[2]);

    // AG L1
    TemplateDataParams tempAlgParamsAllGatherL1;
    tempAlgParamsAllGatherL1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL1.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate4> algTemplateAllGatherL1 =
        std::make_shared<InsAlgTemplate4>(param, myRank_, algHierarchyInfo_.infos[1]);

    // AG L0
    TemplateDataParams tempAlgParamsAllGatherL0;
    tempAlgParamsAllGatherL0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsAllGatherL0.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate5> algTemplateAllGatherL0 =
        std::make_shared<InsAlgTemplate5>(param, myRank_, algHierarchyInfo_.infos[0]);

    // 构造L0 template资源
    TemplateResource templateResourceL0;
    templateResourceL0.channels = remoteRankToChannelInfo_[0];
    templateResourceL0.threads = resCtx.threads;
    templateResourceL0.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceL0.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造L1 template资源
    TemplateResource templateResourceL1;
    templateResourceL1.channels = remoteRankToChannelInfo_[1];
    templateResourceL1.threads = resCtx.threads;
    templateResourceL1.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceL1.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造L2 template资源
    TemplateResource templateResourceL2;
    templateResourceL2.channels = remoteRankToChannelInfo_[2];
    templateResourceL2.threads = resCtx.threads;
    templateResourceL2.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceL2.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    u64 dataCount_ = param.DataDes.count;
    // 当前公式是未优化版本
    u64 maxCountPerLoop = tempAlgParamsScatterL0.buffInfo.hcclBuff.size / HCCL_MIN_SLICE_ALIGN *
                          HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    // 已处理的元素数
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        // 本轮实际处理的元素数量
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        // L0 Scatter的数据偏移和搬运量计算
        tempAlgParamsScatterL0.count = currDataCount;
        // 输入偏移
        tempAlgParamsScatterL0.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        // 输出偏移
        tempAlgParamsScatterL0.buffInfo.outBuffBaseOff = 0;
        // ccl buffer基地址偏移
        tempAlgParamsScatterL0.buffInfo.hcclBuffBaseOff = 0;

        CHK_RET(SplitData(
            currDataCount, rankSizeLevel0_, tempAlgParamsScatterL0));  // 计算每个卡对应位置的offset,count,size
        CHK_PRT_RET(tempAlgParamsScatterL0.allRankSliceSize.size() != rankSizeLevel0_,
            HCCL_ERROR("[InsV2BroadcastSequenceExecutorAicpu3Level][tempAlgParamsScatterL0] slice num[%u] is not equal to rank "
                       "size[%u].",
                tempAlgParamsScatterL0.allRankSliceSize.size(),
                rankSizeLevel0_),
            HcclResult::HCCL_E_INTERNAL);

        tempAlgParamsScatterL0.sliceSize = 0;
        tempAlgParamsScatterL0.tailSize = 0;
        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsScatterL0.inputSliceStride = 0;
        tempAlgParamsScatterL0.outputSliceStride = 0;

        // 不需要重复
        tempAlgParamsScatterL0.repeatNum = 1;
        tempAlgParamsScatterL0.inputRepeatStride = 0;
        tempAlgParamsScatterL0.outputRepeatStride = 0;
        CHK_RET(algTemplateScatterL0->KernelRun(param, tempAlgParamsScatterL0, templateResourceL0));

        // ----------- L1 Scatter数据搬运 -----------
        // L1 Scatter的数据偏移和搬运计算
        tempAlgParamsScatterL1.count = tempAlgParamsScatterL0.allRankProcessedDataCount.at(rankIdxLevel0_);
        tempAlgParamsScatterL1.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsScatterL1.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsScatterL1.buffInfo.hcclBuffBaseOff = tempAlgParamsScatterL0.allRankDispls.at(rankIdxLevel0_);
        // 计算L1 Scatter本地根
        tempAlgParamsScatterL1.root = (param.root / rankSizeLevel0_) * rankSizeLevel0_ + (myRank_ % rankSizeLevel0_);

        tempAlgParamsScatterL1.sliceSize = 0;
        tempAlgParamsScatterL1.tailSize = 0;

        CHK_RET(SplitData(tempAlgParamsScatterL1.count, rankSizeLevel1_, tempAlgParamsScatterL1));
        CHK_PRT_RET(tempAlgParamsScatterL1.allRankSliceSize.size() != rankSizeLevel1_,
            HCCL_ERROR("[InsV2BroadcastSequenceExecutorAicpu3Level][tempAlgParamsScatterL1] slice num[%u] is not equal to rank "
                       "size[%u].",
                tempAlgParamsScatterL1.allRankSliceSize.size(),
                rankSizeLevel1_),
            HcclResult::HCCL_E_INTERNAL);
        HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][SplitData][tempAlgParamsScatterL1] count[%u] slicenum[%u]",
            tempAlgParamsScatterL1.count, rankSizeLevel1_);

        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsScatterL1.inputSliceStride = 0;
        tempAlgParamsScatterL1.outputSliceStride = 0;

        HCCL_DEBUG("[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsScatterL1.inputSliceStride [%u],"
                  "tempAlgParamsScatterL1.outputSliceStride [%u] tempAlgParamsScatterL1.sliceSize [%u]",
            loop,
            tempAlgParamsScatterL1.inputSliceStride,
            tempAlgParamsScatterL1.outputSliceStride,
            tempAlgParamsScatterL1.sliceSize);
        HCCL_DEBUG("[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsScatterL1.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParamsScatterL1.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParamsScatterL1.buffInfo.inBuffBaseOff,
            tempAlgParamsScatterL1.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsScatterL1.repeatNum = 1;
        tempAlgParamsScatterL1.inputRepeatStride = 0;
        tempAlgParamsScatterL1.outputRepeatStride = 0;
        if (tempAlgParamsScatterL1.count != 0) {
            CHK_RET(algTemplateScatterL1->KernelRun(param, tempAlgParamsScatterL1, templateResourceL1));
        }

        // pod间scatter的数据搬运
        tempAlgParamsScatterL2.count = tempAlgParamsScatterL1.allRankProcessedDataCount.at(rankIdxLevel1_);
        tempAlgParamsScatterL2.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsScatterL2.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsScatterL2.buffInfo.hcclBuffBaseOff = 
        tempAlgParamsScatterL0.allRankDispls.at(rankIdxLevel0_) + tempAlgParamsScatterL1.allRankDispls.at(rankIdxLevel1_);
        // 计算pod间本地根
        u64 rankNumsPerPod = rankSizeLevel0_ * rankSizeLevel1_;
        tempAlgParamsScatterL2.root = (param.root / rankNumsPerPod) * rankNumsPerPod + (myRank_ % rankNumsPerPod);

        tempAlgParamsScatterL2.sliceSize = 0;
        tempAlgParamsScatterL2.tailSize = 0;

        CHK_RET(SplitData(tempAlgParamsScatterL2.count, rankSizeLevel2_, tempAlgParamsScatterL2));
        CHK_PRT_RET(tempAlgParamsScatterL2.allRankSliceSize.size() != rankSizeLevel2_,
            HCCL_ERROR("[InsV2BroadcastSequenceExecutorAicpu3Level][tempAlgParamsScatterL2] slice num[%u] is not equal to rank "
                       "size[%u].",
                tempAlgParamsScatterL2.allRankSliceSize.size(),
                rankSizeLevel2_),
            HcclResult::HCCL_E_INTERNAL);
        HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][SplitData][tempAlgParamsScatterL2] count[%u] slicenum[%u]",
            tempAlgParamsScatterL2.count, rankSizeLevel2_);

        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsScatterL2.inputSliceStride = 0;
        tempAlgParamsScatterL2.outputSliceStride = 0;

        HCCL_DEBUG("[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsScatterL2.inputSliceStride [%u],"
                  "tempAlgParamsScatterL2.outputSliceStride [%u] tempAlgParamsScatterL2.sliceSize [%u]",
            loop,
            tempAlgParamsScatterL2.inputSliceStride,
            tempAlgParamsScatterL2.outputSliceStride,
            tempAlgParamsScatterL2.sliceSize);
        HCCL_DEBUG("[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsScatterL2.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParamsScatterL2.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParamsScatterL2.buffInfo.inBuffBaseOff,
            tempAlgParamsScatterL2.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsScatterL2.repeatNum = 1;
        tempAlgParamsScatterL2.inputRepeatStride = 0;
        tempAlgParamsScatterL2.outputRepeatStride = 0;
        if (tempAlgParamsScatterL2.count != 0) {
            CHK_RET(algTemplateScatterL2->KernelRun(param, tempAlgParamsScatterL2, templateResourceL2));
        }


        tempAlgParamsAllGatherL2.count = tempAlgParamsScatterL1.allRankProcessedDataCount.at(rankIdxLevel1_);
        tempAlgParamsAllGatherL2.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsAllGatherL2.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsAllGatherL2.buffInfo.hcclBuffBaseOff =
        tempAlgParamsScatterL0.allRankDispls.at(rankIdxLevel0_) + tempAlgParamsScatterL1.allRankDispls.at(rankIdxLevel1_);


        tempAlgParamsAllGatherL2.allRankDispls = tempAlgParamsScatterL2.allRankDispls;
        tempAlgParamsAllGatherL2.allRankSliceSize = tempAlgParamsScatterL2.allRankSliceSize;
        tempAlgParamsAllGatherL2.allRankProcessedDataCount = tempAlgParamsScatterL2.allRankProcessedDataCount;

        tempAlgParamsAllGatherL2.sliceSize = 0;
        tempAlgParamsAllGatherL2.tailSize = 0;
        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsAllGatherL2.inputSliceStride = 0;
        tempAlgParamsAllGatherL2.outputSliceStride = 0;

        HCCL_DEBUG("[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsAllGatherL2.inputSliceStride [%u],"
                  "tempAlgParamsAllGatherL2.outputSliceStride [%u] tempAlgParamsAllGatherL2.sliceSize [%u]",
            loop,
            tempAlgParamsAllGatherL2.inputSliceStride,
            tempAlgParamsAllGatherL2.outputSliceStride,
            tempAlgParamsAllGatherL2.sliceSize);
        HCCL_DEBUG("[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsAllGatherL2.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParamsAllGatherL2.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParamsAllGatherL2.buffInfo.inBuffBaseOff,
            tempAlgParamsAllGatherL2.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsAllGatherL2.repeatNum = 1;
        tempAlgParamsAllGatherL2.inputRepeatStride = 0;
        tempAlgParamsAllGatherL2.outputRepeatStride = 0;
        HCCL_DEBUG(
            "[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsAllGatherL2.repeatNum [%u],"
            "tempAlgParamsAllGatherL2.inputRepeatStride [%u], tempAlgParamsAllGatherL2.outputRepeatStride [%u]",
            loop,
            tempAlgParamsAllGatherL2.repeatNum,
            tempAlgParamsAllGatherL2.inputRepeatStride,
            tempAlgParamsAllGatherL2.outputRepeatStride);
        CHK_RET(algTemplateAllGatherL2->KernelRun(param, tempAlgParamsAllGatherL2, templateResourceL2));

        
        // ----------- server间AllGather数据搬运 -----------
        // server间的数据偏移和搬运计算
        tempAlgParamsAllGatherL1.count = tempAlgParamsScatterL0.allRankProcessedDataCount.at(rankIdxLevel0_);  // 沿用框间Scatter的切片结果
        tempAlgParamsAllGatherL1.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsAllGatherL1.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsAllGatherL1.buffInfo.hcclBuffBaseOff = tempAlgParamsScatterL0.allRankDispls.at(rankIdxLevel0_); // 将框内的切片偏移传到框间

        tempAlgParamsAllGatherL1.allRankDispls = tempAlgParamsScatterL1.allRankDispls;
        tempAlgParamsAllGatherL1.allRankSliceSize = tempAlgParamsScatterL1.allRankSliceSize;
        tempAlgParamsAllGatherL1.allRankProcessedDataCount = tempAlgParamsScatterL1.allRankProcessedDataCount;

        tempAlgParamsAllGatherL1.sliceSize = 0;
        tempAlgParamsAllGatherL1.tailSize = 0;
        tempAlgParamsAllGatherL1.inputSliceStride = 0;
        tempAlgParamsAllGatherL1.outputSliceStride = 0;

        HCCL_DEBUG("[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsAllGatherL1.inputSliceStride [%u],"
                  "tempAlgParamsAllGatherL1.outputSliceStride [%u] tempAlgParamsAllGatherL1.sliceSize [%u]",
            loop,
            tempAlgParamsAllGatherL1.inputSliceStride,
            tempAlgParamsAllGatherL1.outputSliceStride,
            tempAlgParamsAllGatherL1.sliceSize);
        HCCL_DEBUG("[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsAllGatherL1.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParamsAllGatherL1.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParamsAllGatherL1.buffInfo.inBuffBaseOff,
            tempAlgParamsAllGatherL1.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsAllGatherL1.repeatNum = 1;
        tempAlgParamsAllGatherL1.inputRepeatStride = 0;
        tempAlgParamsAllGatherL1.outputRepeatStride = 0;
        HCCL_DEBUG(
            "[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsAllGatherL1.repeatNum [%u],"
            "tempAlgParamsAllGatherL1.inputRepeatStride [%u], tempAlgParamsAllGatherL1.outputRepeatStride [%u]",
            loop,
            tempAlgParamsAllGatherL1.repeatNum,
            tempAlgParamsAllGatherL1.inputRepeatStride,
            tempAlgParamsAllGatherL1.outputRepeatStride);
        CHK_RET(algTemplateAllGatherL1->KernelRun(param, tempAlgParamsAllGatherL1, templateResourceL1));

        // ----------- server内AllGather数据搬运 -----------
        // server内的数据偏移和搬运计算
        tempAlgParamsAllGatherL0.count = currDataCount;
        tempAlgParamsAllGatherL0.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsAllGatherL0.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsAllGatherL0.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParamsAllGatherL0.allRankDispls = tempAlgParamsScatterL0.allRankDispls;  // 沿用框内Scatter的切片结果
        tempAlgParamsAllGatherL0.allRankSliceSize = tempAlgParamsScatterL0.allRankSliceSize;
        tempAlgParamsAllGatherL0.allRankProcessedDataCount = tempAlgParamsScatterL0.allRankProcessedDataCount;

        tempAlgParamsAllGatherL0.sliceSize = 0;
        tempAlgParamsAllGatherL0.tailSize = 0;

        tempAlgParamsAllGatherL0.inputSliceStride = 0;
        tempAlgParamsAllGatherL0.outputSliceStride = 0;

        HCCL_DEBUG("[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsAllGatherL0.inputSliceStride [%u],"
                  "tempAlgParamsAllGatherL0.outputSliceStride [%u] tempAlgParamsAllGatherL0.sliceSize [%u]",
            loop,
            tempAlgParamsAllGatherL0.inputSliceStride,
            tempAlgParamsAllGatherL0.outputSliceStride,
            tempAlgParamsAllGatherL0.sliceSize);
        HCCL_DEBUG("[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsAllGatherL0.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParamsAllGatherL0.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParamsAllGatherL0.buffInfo.inBuffBaseOff,
            tempAlgParamsAllGatherL0.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsAllGatherL0.repeatNum = 1;
        tempAlgParamsAllGatherL0.inputRepeatStride = 0;
        tempAlgParamsAllGatherL0.outputRepeatStride = 0;
        HCCL_DEBUG(
            "[InsV2BroadcastSequenceExecutorAicpu3Level] loop [%u] tempAlgParamsAllGatherL0.repeatNum [%u],"
            "tempAlgParamsAllGatherL0.inputRepeatStride [%u], tempAlgParamsAllGatherL0.outputRepeatStride [%u]",
            loop,
            tempAlgParamsAllGatherL0.repeatNum,
            tempAlgParamsAllGatherL0.inputRepeatStride,
            tempAlgParamsAllGatherL0.outputRepeatStride);
        CHK_RET(algTemplateAllGatherL0->KernelRun(param, tempAlgParamsAllGatherL0, templateResourceL0));

        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2BroadcastSequenceExecutorAicpu3Level][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::SplitData(const u64 &dataCount, const uint64_t &rankSize, TemplateDataParams &tempAlgParams)
{
    u32 sliceNum = rankSize;
    u64 offsetCount = 0;
    u64 offsetSize = 0;
    tempAlgParams.allRankSliceSize.clear();
    tempAlgParams.allRankDispls.clear();
    tempAlgParams.allRankProcessedDataCount.clear();
    tempAlgParams.allRankSliceSize.reserve(sliceNum);
    tempAlgParams.allRankDispls.reserve(sliceNum);
    tempAlgParams.allRankProcessedDataCount.reserve(sliceNum);

    u64 sliceCount = RoundUp(dataCount, sliceNum);
    u64 sliceSize = sliceCount * dataTypeSize_;

    for (u32 sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        if (dataCount - offsetCount >= sliceCount) {
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            tempAlgParams.allRankSliceSize.emplace_back(sliceSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(sliceCount);
            offsetCount += sliceCount;
            offsetSize = offsetCount * dataTypeSize_;
        } else {
            u64 curSliceCount = dataCount - offsetCount;
            u64 curSliceSize = curSliceCount * dataTypeSize_;
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            tempAlgParams.allRankSliceSize.emplace_back(curSliceSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(curSliceCount);
            offsetCount = dataCount;
            offsetSize = offsetCount * dataTypeSize_;
        }
    }

    for (u32 i = 0; i < tempAlgParams.allRankSliceSize.size(); ++i) {
        HCCL_DEBUG("[InsV2BroadcastSequenceExecutorAicpu3Level] SliceInfo: offset[%u] size[%u] count[%u]",
            tempAlgParams.allRankDispls.at(i),
            tempAlgParams.allRankSliceSize.at(i),
            tempAlgParams.allRankProcessedDataCount.at(i));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
u64 InsV2BroadcastSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3, InsAlgTemplate4, InsAlgTemplate5>::RoundUp(const u64 dividend, const u64 divisor)
{
    if (divisor == 0) {
        HCCL_WARNING("[InsV2BroadcastSequenceExecutorAicpu3Level][RoundUp] divisor is 0.");
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_BROADCAST,
    InsBroadcastSequenceMesh1DNHRNHR,
    InsV2BroadcastSequenceExecutorAicpu3Level,
    TopoMatchMultilevel,
    InsTempScatterMesh1D,      // Scatter L0 (框内)
    InsTempScatterNHR,          // Scatter L1 (框间)
    InsTempScatterNHR,          // Scatter L2 (跨超节点)
    InsTempAllGatherNHR,        // AllGather L2 (跨超节点)
    InsTempAllGatherNHR,        // AllGather L1 (框间)
    InsTempAllGatherMesh1D1DZAxisDetour);  // AllGather L0 (框内, Z 轴绕路)
}  // namespace ops_hccl