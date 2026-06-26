/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_sequence_executor_4level_ocs.h"
#include <cmath>
#include "alg_data_trans_wrapper.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_mesh_1D_ocs.h"

#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "topo_match_pcie_mix.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
InsV2AllGatherSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::InsV2AllGatherSequenceExecutor4LevelOCS()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2AllGatherSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2AllGatherSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    HCCL_DEBUG("[InsV2AllGatherSequenceExecutor4LevelOCS][CalcRes] myRank[%u] start", topoInfo->userRank);
    if (algHierarchyInfo.infos.size() != SEQUENCE_EXECUTOR_4_LEVEL_NUM_OCS) {
        HCCL_ERROR("[InsV2AllGatherSequenceExecutor4LevelOCS] algHierarchyInfo size %zu should be %u",
            algHierarchyInfo.infos.size(), SEQUENCE_EXECUTOR_4_LEVEL_NUM_OCS);
        return HCCL_E_INTERNAL;
    }
    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1][0].size();
    rankSizeLevel2_ = algHierarchyInfo.infos[2][0].size();
    rankSizeLevel3_ = algHierarchyInfo.infos[3][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);
    skipLevel2_ = (rankSizeLevel2_ == 1);
    skipLevel3_ = (rankSizeLevel3_ == 1);

    // 构建template
    InsAlgTemplate0 Level0TempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    InsAlgTemplate1 Level1TempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[1]);
    InsAlgTemplate2 Level2TempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[2]);
    InsAlgTemplate3 Level3TempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[3]);

    // 调用计算资源的函数
    AlgResourceRequest Level0TempRequest, Level1TempRequest, Level2TempRequest, Level3TempRequest;
    CHK_RET(Level0TempAlg.CalcRes(comm, param, topoInfo, Level0TempRequest));
    if (skipLevel1_) {
        HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS][CalcRes] myRank[%u] level1 rankSize is 1, skip level1 CalcRes",
            topoInfo->userRank);
    } else {
        CHK_RET(Level1TempAlg.CalcRes(comm, param, topoInfo, Level1TempRequest));
    }
    if (skipLevel2_) {
        HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS][CalcRes] myRank[%u] level2 rankSize is 1, skip level2 CalcRes",
            topoInfo->userRank);
    } else {
        CHK_RET(Level2TempAlg.CalcRes(comm, param, topoInfo, Level2TempRequest));
    }
    if (skipLevel3_) {
        HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS][CalcRes] myRank[%u] level3 rankSize is 1, skip level3 CalcRes",
            topoInfo->userRank);
    } else {
        CHK_RET(Level3TempAlg.CalcRes(comm, param, topoInfo, Level3TempRequest));
    }

    resourceRequest.notifyNumOnMainThread = Level0TempRequest.notifyNumOnMainThread;
    resourceRequest.slaveThreadNum = Level0TempRequest.slaveThreadNum;
    if (!skipLevel1_) {
        resourceRequest.notifyNumOnMainThread = std::max(resourceRequest.notifyNumOnMainThread, Level1TempRequest.notifyNumOnMainThread);
        resourceRequest.slaveThreadNum = std::max(resourceRequest.slaveThreadNum, Level1TempRequest.slaveThreadNum);
    }
    if (!skipLevel2_) {
        resourceRequest.notifyNumOnMainThread = std::max(resourceRequest.notifyNumOnMainThread, Level2TempRequest.notifyNumOnMainThread);
        resourceRequest.slaveThreadNum = std::max(resourceRequest.slaveThreadNum, Level2TempRequest.slaveThreadNum);
    }
    if (!skipLevel3_) {
        resourceRequest.notifyNumOnMainThread = std::max(resourceRequest.notifyNumOnMainThread, Level3TempRequest.notifyNumOnMainThread);
        resourceRequest.slaveThreadNum = std::max(resourceRequest.slaveThreadNum, Level3TempRequest.slaveThreadNum);
    }
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              Level0TempRequest.notifyNumPerThread.begin(),
                                              Level0TempRequest.notifyNumPerThread.end());
    if (!skipLevel1_) {
        resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                                  Level1TempRequest.notifyNumPerThread.begin(),
                                                  Level1TempRequest.notifyNumPerThread.end());
    }
    if (!skipLevel2_) {
        resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                                  Level2TempRequest.notifyNumPerThread.begin(),
                                                  Level2TempRequest.notifyNumPerThread.end());
    }
    if (!skipLevel3_) {
        resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                                  Level3TempRequest.notifyNumPerThread.begin(),
                                                  Level3TempRequest.notifyNumPerThread.end());
    }
    CHK_PRT_RET(Level0TempRequest.channels.empty(),
                     HCCL_ERROR("[InsV2AllGatherSequenceExecutor4LevelOCS][CalcRes] Level0TempRequest has empty channels."),
                     HcclResult::HCCL_E_INTERNAL);
    resourceRequest.channels.resize(SEQUENCE_EXECUTOR_4_LEVEL_NUM_OCS);
    resourceRequest.channels[0] = Level0TempRequest.channels[0];
    if (!skipLevel1_) {
        CHK_PRT_RET(Level1TempRequest.channels.empty(),
                         HCCL_ERROR("[InsV2AllGatherSequenceExecutor4LevelOCS][CalcRes] Level1TempRequest has empty channels."),
                         HcclResult::HCCL_E_INTERNAL);
        resourceRequest.channels[1] = Level1TempRequest.channels[0];
    }
    if (!skipLevel2_) {
        CHK_PRT_RET(Level2TempRequest.channels.empty(),
                         HCCL_ERROR("[InsV2AllGatherSequenceExecutor4LevelOCS][CalcRes] Level2TempRequest has empty channels."),
                         HcclResult::HCCL_E_INTERNAL);
        resourceRequest.channels[2] = Level2TempRequest.channels[0];
    }
    if (!skipLevel3_) {
        CHK_PRT_RET(Level3TempRequest.channels.empty(),
                         HCCL_ERROR("[InsV2AllGatherSequenceExecutor4LevelOCS][CalcRes] Level3TempRequest has empty channels."),
                         HcclResult::HCCL_E_INTERNAL);
        resourceRequest.channels[3] = Level3TempRequest.channels[0];
    }
    HCCL_DEBUG("[InsV2AllGatherSequenceExecutor4LevelOCS][CalcRes] notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG("[InsV2AllGatherSequenceExecutor4LevelOCS][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", i,
                   resourceRequest.notifyNumPerThread[i]);
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2AllGatherSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;  // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    myRank_ = resCtx.topoInfo.userRank;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    rankSizeLevel0_ = resCtx.algHierarchyInfo.infos[0][0].size();
    rankSizeLevel1_ = resCtx.algHierarchyInfo.infos[1][0].size();
    rankSizeLevel2_ = resCtx.algHierarchyInfo.infos[2][0].size();
    rankSizeLevel3_ = resCtx.algHierarchyInfo.infos[3][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);
    skipLevel2_ = (rankSizeLevel2_ == 1);
    skipLevel3_ = (rankSizeLevel3_ == 1);
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;                                    // level0 组内偏移
    rankIdxLevel1_ = myRank_ % (rankSizeLevel0_ * rankSizeLevel1_);                // level1 组编号
    if (skipLevel1_) {
        HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS] [Orchestrate] myRank[%u] level1 rankSize is 1, skip level1",
            myRank_);
    }
    if (skipLevel2_) {
        HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS] [Orchestrate] myRank[%u] level2 rankSize is 1, skip level2",
            myRank_);
    }
    if (skipLevel3_) {
        HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS] [Orchestrate] myRank[%u] level3 rankSize is 1, skip level3",
            myRank_);
    }
    HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS] [Orchestrate] myRank_[%u] rankIdxLevel0_[%u] "
        "rankIdxLevel1_[%u] rankSizeLevel0_[%u] rankSizeLevel1_[%u] rankSizeLevel2_[%u] rankSizeLevel3_[%u]",
        myRank_, rankIdxLevel0_, rankIdxLevel1_,
        rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_, rankSizeLevel3_);

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherSequenceExecutor4LevelOCS][Orchestrate]errNo[0x%016llx] All Gather excutor kernel run failed",
                   HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
template <typename InsAlgTemplate>
HcclResult InsV2AllGatherSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::GenTempResource
    (const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
    const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempResource) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR("[InsV2AllGatherSequenceExecutor4LevelOCS][GenTempResource] myRank[%u] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%zu]", myRank_, channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2AllGatherSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS] AlgTemplate Level0 server is [%s]",
        "InsTempAllGatherMesh1D1DZAxisDetour");
    HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS] AlgTemplate Level1 is [%s]", "InsTempAllGatherNHR");
    HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS] AlgTemplate Level2 is [%s]", "InsTempAllGatherNHR");
    HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS] AlgTemplate Level3 is [%s]", "InsTempAllGatherMesh1DOcs");

    // level0 intra (HCCL -> OUTPUT): 最后执行，写入 OUTPUT
    TemplateDataParams tempAlgParamsLevel0;
    tempAlgParamsLevel0.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsLevel0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsLevel0.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate0> algTemplateLevel0 =
        std::make_shared<InsAlgTemplate0>(param, myRank_, resCtx.algHierarchyInfo.infos[0]);
    if (rankSizeLevel0_ > 1) {
        CHK_RET(algTemplateLevel0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));
    }

    // level1 inter (HCCL -> HCCL)
    TemplateDataParams tempAlgParamsLevel1;
    tempAlgParamsLevel1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate1> algTemplateLevel1 =
        std::make_shared<InsAlgTemplate1>(param, myRank_, resCtx.algHierarchyInfo.infos[1]);
    if (!skipLevel1_) {
        CHK_RET(algTemplateLevel1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    // level2 inter (HCCL -> HCCL)
    TemplateDataParams tempAlgParamsLevel2;
    tempAlgParamsLevel2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel2.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate2> algTemplateLevel2 =
        std::make_shared<InsAlgTemplate2>(param, myRank_, resCtx.algHierarchyInfo.infos[2]);
    if (!skipLevel2_) {
        CHK_RET(algTemplateLevel2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));
    }

    // level3 inter (INPUT -> HCCL): 最先执行，读 INPUT
    TemplateDataParams tempAlgParamsLevel3;
    tempAlgParamsLevel3.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsLevel3.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel3.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel3.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsLevel3.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel3.buffInfo.hcclBuff = resCtx.cclMem;
    std::shared_ptr<InsAlgTemplate3> algTemplateLevel3 =
        std::make_shared<InsAlgTemplate3>(param, myRank_, resCtx.algHierarchyInfo.infos[3]);
    if (!skipLevel3_) {
        CHK_RET(algTemplateLevel3->SetchannelsPerRank(remoteRankToChannelInfo_[3]));
    }

    u32 templateScratchMultiplierLevel0 = algTemplateLevel0->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);
    u32 templateScratchMultiplierLevel1 = skipLevel1_ ? 1 :
        algTemplateLevel1->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplierLevel2 = skipLevel2_ ? 1 :
        algTemplateLevel2->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplierLevel3 = skipLevel3_ ? 1 :
        algTemplateLevel3->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 totalScratchMultiple = templateScratchMultiplierLevel0 * templateScratchMultiplierLevel1 *
        templateScratchMultiplierLevel2 * templateScratchMultiplierLevel3;

    TemplateResource Level3TempAlgRes, Level2TempAlgRes, Level1TempAlgRes, Level0TempAlgRes;
    if (!skipLevel3_) {
        CHK_RET(GenTempResource(resCtx, 3, algTemplateLevel3, Level3TempAlgRes));
    }
    if (!skipLevel2_) {
        CHK_RET(GenTempResource(resCtx, 2, algTemplateLevel2, Level2TempAlgRes));
    }
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(resCtx, 1, algTemplateLevel1, Level1TempAlgRes));
    }
    CHK_RET(GenTempResource(resCtx, 0, algTemplateLevel0, Level0TempAlgRes));

    u64 scratchMemBlockSize = maxTmpMemSize_;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / HCCL_MIN_SLICE_ALIGN / totalScratchMultiple) * HCCL_MIN_SLICE_ALIGN;
        scratchMemBlockSize = std::min(scratchMemBlockSize, transportBoundDataSize);
    }
    u64 maxCountPerLoop =
        (std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_ / 10) * 10;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR("[InsV2AllGatherSequenceExecutor4LevelOCS] myRank[%u] maxCountPerLoop is 0, "
            "scratchMultiplier[%u] too large for cclBuffSize[%llu]",
            myRank_, totalScratchMultiple, scratchMemBlockSize);
        return HCCL_E_INTERNAL;
    }
    // level3 (Mesh1D) 把 s3 份 gather 结果写到 [above, above + s3*slice):
    //   - skipLevel2_=false 时 above = s1*s2*s3*slice (越过累积区, 由 level2 搬到 [0,…));
    //   - skipLevel2_=true  时 above = 0 (无层搬运, 直接写 [0,…))。
    // 校验最坏情形(maxCountPerLoop)下该区间不越界 cclMem。
    if (!skipLevel3_) {
        const u64 maxSliceSize = maxCountPerLoop * dataTypeSize_;
        const u64 level3Above = skipLevel2_ ? 0 :
            (rankSizeLevel1_ * rankSizeLevel2_ * rankSizeLevel3_ * maxSliceSize);
        const u64 level3RegionEnd = level3Above + rankSizeLevel3_ * maxSliceSize;
        CHK_PRT_RET(level3RegionEnd > maxTmpMemSize_,
            HCCL_ERROR("[InsV2AllGatherSequenceExecutor4LevelOCS] myRank[%u] level3 out-region [above, above+s3*slice] "
                "exceeds cclMem: need[%llu] > cclMemSize[%llu] (skipLevel2[%d] s1[%llu] s2[%llu] s3[%llu] maxSliceSize[%llu])",
                myRank_, level3RegionEnd, maxTmpMemSize_, skipLevel2_,
                rankSizeLevel1_, rankSizeLevel2_, rankSizeLevel3_, maxSliceSize),
            HCCL_E_INTERNAL);
    }
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex == loopTimes - 1) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        u64 dataOffset = loopIndex * maxCountPerLoop * dataTypeSize_;

        // 执行序: level3(外, INPUT->HCCL) -> level2(HCCL->HCCL) -> level1(HCCL->HCCL) -> level0(内, HCCL->OUTPUT)
        if (!skipLevel3_) {
            GenTemplateAlgParamsLevel3(param, resCtx, currCount, dataOffset, tempAlgParamsLevel3);
            CHK_RET(algTemplateLevel3->KernelRun(param, tempAlgParamsLevel3, Level3TempAlgRes));
        }

        if (!skipLevel2_) {
            GenTemplateAlgParamsLevel2(param, resCtx, currCount, dataOffset, tempAlgParamsLevel2);
            CHK_RET(algTemplateLevel2->KernelRun(param, tempAlgParamsLevel2, Level2TempAlgRes));
        }

        if (!skipLevel1_) {
            GenTemplateAlgParamsLevel1(param, resCtx, currCount, dataOffset, tempAlgParamsLevel1);
            CHK_RET(algTemplateLevel1->KernelRun(param, tempAlgParamsLevel1, Level1TempAlgRes));
        }

        GenTemplateAlgParamsLevel0(param, resCtx, currCount, dataOffset, tempAlgParamsLevel0);
        CHK_RET(algTemplateLevel0->KernelRun(param, tempAlgParamsLevel0, Level0TempAlgRes));
    }

    HCCL_INFO("[InsV2AllGatherSequenceExecutor4LevelOCS][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
}

// level3 inter (INPUT -> HCCL): 最外层最先执行，读 INPUT。
// 本层是 Mesh1D 模板(非 NHR)，gather 结果落到 out 区(outputPtr=cclMem)，每 rank 一份，按 sliceSize 间隔，
// 基址置于累积区之后(above = s1*s2*s3*slice)，与 level2 的 inBuffBaseOff 对齐。
// 注意: Mesh1D 在 OPBASE(enableRemoteMemAccess_=false)下，PostLocalCopy 因 outBuffType=HCCL_BUFFER 被跳过，
//       结果由 ring 步直接写入 out 区；因此 outBuffBaseOff + outputSliceStride*algRank 必须给每个 rank 独立 slot，
//       不能像 NHR 那样 outputSliceStride=0(否则所有 rank 写同一 offset 互相覆盖)。
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2AllGatherSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::GenTemplateAlgParamsLevel3(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 curCount, const u64 dataOffset,
    TemplateDataParams &tempAlgParamsLevel3) const
{
    const u64 sliceSize = curCount * dataTypeSize_;
    // level3 产物落点基址。设计契约: level2 负责把 level3 产物从 [above] 搬运到累积区 [0,…)。
    // - level2 活动 (rankSizeLevel2_>1): level3 写到 [above, above+s3*slice) (越过累积区, 不与 ring 步冲突),
    //   由 level2 (inBuffBaseOff2=above) 读入并搬到 [0,…)。
    // - level2 跳过 (rankSizeLevel2_==1, skipLevel2_): 没有层搬运 above→0, level3 必须直接写到 [0, s3*slice),
    //   供 level1 (若活动, 读[0]) 或 level0 (读[0]) 直接读。否则 level0 读 [0] 是空的 → 全零。
    const u64 above = skipLevel2_ ? 0 : (rankSizeLevel1_ * rankSizeLevel2_ * rankSizeLevel3_ * sliceSize);

    tempAlgParamsLevel3.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsLevel3.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel3.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsLevel3.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsLevel3.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel3.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel3.buffInfo.inputSize = param.inputSize;
    tempAlgParamsLevel3.buffInfo.outputSize = param.outputSize;

    tempAlgParamsLevel3.buffInfo.inBuffBaseOff = dataOffset;
    // out 区基址 = above: Mesh1D 把 s3 份 gather 结果写到这里(mesh_1D.cc 的 outBuffBaseOff/outOffset)
    tempAlgParamsLevel3.buffInfo.outBuffBaseOff = above;
    // scratch 镜像与 out 同基址: Mesh1D 在 OPBASE 下 ring 走 remoteCclBuff, 镜像供 TX 读自身数据;
    // 与 out 共基址无冲突(TX 读 [above+slice*myRank], RX 写 [above+slice*connectedRank], 因 stride=slice 不重叠)
    tempAlgParamsLevel3.buffInfo.hcclBuffBaseOff = above;
    tempAlgParamsLevel3.sliceSize = sliceSize;
    tempAlgParamsLevel3.count = curCount;
    tempAlgParamsLevel3.tailSize = sliceSize;

    tempAlgParamsLevel3.inputSliceStride = 0;
    // 关键修复: 必须用 sliceSize 间隔, 给每个 rank 独立 slot, 避免 Mesh1D ring 各步写同一 offset 覆盖。
    // (旧值 0 是从 3 级 NHR 层照搬的, 对 NHR 有效但对 Mesh1D 致命)
    tempAlgParamsLevel3.outputSliceStride = sliceSize;
    tempAlgParamsLevel3.repeatNum = 1;
    tempAlgParamsLevel3.inputRepeatStride = 0;
    tempAlgParamsLevel3.outputRepeatStride = 0;
    tempAlgParamsLevel3.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    HCCL_DEBUG("[InsV2AllGatherSequenceExecutor4LevelOCS][GenTemplateAlgParamsLevel3] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu]",
               myRank_, tempAlgParamsLevel3.buffInfo.inBuffBaseOff, tempAlgParamsLevel3.buffInfo.outBuffBaseOff,
               tempAlgParamsLevel3.buffInfo.hcclBuffBaseOff, tempAlgParamsLevel3.sliceSize,
               tempAlgParamsLevel3.outputSliceStride);
    return;
}

// level2 inter (HCCL -> HCCL): 读 level3 产物(越过累积区位置)，repeatNum = s3，写 [0,…) 累积区
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2AllGatherSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::GenTemplateAlgParamsLevel2(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 curCount, const u64 dataOffset,
    TemplateDataParams &tempAlgParamsLevel2) const
{
    tempAlgParamsLevel2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel2.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsLevel2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.inputSize = param.inputSize;
    tempAlgParamsLevel2.buffInfo.outputSize = param.outputSize;

    tempAlgParamsLevel2.buffInfo.inBuffBaseOff =
        rankSizeLevel1_ * rankSizeLevel2_ * rankSizeLevel3_ * curCount * dataTypeSize_;
    tempAlgParamsLevel2.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsLevel2.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParamsLevel2.sliceSize = curCount * dataTypeSize_;
    tempAlgParamsLevel2.count = curCount;
    tempAlgParamsLevel2.tailSize = tempAlgParamsLevel2.sliceSize;

    tempAlgParamsLevel2.inputSliceStride = 0;
    tempAlgParamsLevel2.outputSliceStride = 0;
    tempAlgParamsLevel2.repeatNum = rankSizeLevel3_;
    tempAlgParamsLevel2.inputRepeatStride = curCount * dataTypeSize_;
    tempAlgParamsLevel2.outputRepeatStride = 0;
    tempAlgParamsLevel2.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    HCCL_DEBUG("[InsV2AllGatherSequenceExecutor4LevelOCS][GenTemplateAlgParamsLevel2] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] "
               "outputRepeatStride[%llu]",
               myRank_, tempAlgParamsLevel2.buffInfo.inBuffBaseOff, tempAlgParamsLevel2.buffInfo.outBuffBaseOff,
               tempAlgParamsLevel2.buffInfo.hcclBuffBaseOff, tempAlgParamsLevel2.sliceSize,
               tempAlgParamsLevel2.outputSliceStride, tempAlgParamsLevel2.outputRepeatStride);
    return;
}

// level1 inter (HCCL -> HCCL): 读 [0,…), repeatNum = s2*s3，写 [0,…) 累积区
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2AllGatherSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::GenTemplateAlgParamsLevel1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 curCount, const u64 dataOffset,
    TemplateDataParams &tempAlgParamsLevel1) const
{
    tempAlgParamsLevel1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsLevel1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsLevel1.buffInfo.outputSize = param.outputSize;

    tempAlgParamsLevel1.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsLevel1.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsLevel1.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParamsLevel1.sliceSize = curCount * dataTypeSize_;
    tempAlgParamsLevel1.count = curCount;
    tempAlgParamsLevel1.tailSize = tempAlgParamsLevel1.sliceSize;

    tempAlgParamsLevel1.inputSliceStride = 0;
    tempAlgParamsLevel1.outputSliceStride = 0;
    tempAlgParamsLevel1.repeatNum = rankSizeLevel2_ * rankSizeLevel3_;
    tempAlgParamsLevel1.inputRepeatStride = curCount * dataTypeSize_;
    tempAlgParamsLevel1.outputRepeatStride = 0;
    tempAlgParamsLevel1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    HCCL_DEBUG("[InsV2AllGatherSequenceExecutor4LevelOCS][GenTemplateAlgParamsLevel1] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] "
               "outputRepeatStride[%llu]",
               myRank_, tempAlgParamsLevel1.buffInfo.inBuffBaseOff, tempAlgParamsLevel1.buffInfo.outBuffBaseOff,
               tempAlgParamsLevel1.buffInfo.hcclBuffBaseOff, tempAlgParamsLevel1.sliceSize,
               tempAlgParamsLevel1.outputSliceStride, tempAlgParamsLevel1.outputRepeatStride);
    return;
}

// level0 intra (HCCL -> OUTPUT): 最后执行，repeatNum = s1*s2*s3，写到 OUTPUT
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2AllGatherSequenceExecutor4LevelOCS<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2, InsAlgTemplate3>::GenTemplateAlgParamsLevel0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 curCount, const u64 dataOffset,
    TemplateDataParams &tempAlgParamsLevel0) const
{
    tempAlgParamsLevel0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsLevel0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsLevel0.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsLevel0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.inputSize = param.inputSize;
    tempAlgParamsLevel0.buffInfo.outputSize = param.outputSize;

    tempAlgParamsLevel0.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsLevel0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsLevel0.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParamsLevel0.sliceSize = curCount * dataTypeSize_;
    tempAlgParamsLevel0.count = curCount;
    tempAlgParamsLevel0.tailSize = tempAlgParamsLevel0.sliceSize;

    tempAlgParamsLevel0.inputSliceStride = 0;
    tempAlgParamsLevel0.outputSliceStride = dataSize_;
    tempAlgParamsLevel0.repeatNum = rankSizeLevel1_ * rankSizeLevel2_ * rankSizeLevel3_;
    tempAlgParamsLevel0.inputRepeatStride = curCount * dataTypeSize_;
    tempAlgParamsLevel0.outputRepeatStride = rankSizeLevel0_ * dataSize_;
    tempAlgParamsLevel0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    HCCL_DEBUG(
        "[InsV2AllGatherSequenceExecutor4LevelOCS][GenTemplateAlgParamsLevel0] rank[%d] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] levels_[0].rankSize[%llu] "
        "levels_[1].rankSize[%llu] rankIdxLevel0[%llu] rankIdxLevel1[%llu]",
        myRank_, tempAlgParamsLevel0.buffInfo.inBuffBaseOff, tempAlgParamsLevel0.buffInfo.outBuffBaseOff,
        tempAlgParamsLevel0.buffInfo.hcclBuffBaseOff, tempAlgParamsLevel0.sliceSize,
        tempAlgParamsLevel0.outputSliceStride, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    return;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER,
    InsAllGatherSequenceNHRNHRMesh1DOcs,
    InsV2AllGatherSequenceExecutor4LevelOCS,
    TopoMatchMultilevel,
    InsTempAllGatherMesh1D1DZAxisDetour,
    InsTempAllGatherNHR,
    InsTempAllGatherNHR,
    InsTempAllGatherMesh1DOcs);
}
// 算法注册
