/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_sequence_omnipipe_executor.h"
#include <algorithm>
#include "alg_data_trans_wrapper.h"
#include "alg_param.h"
#include "topo_match_ubx.h"
#include "ins_temp_all_gather_omnipipe_mesh_1D.h"
#include "ins_temp_all_gather_omnipipe_nhr.h"
#include "ins_temp_all_gather_nhr_dpu_no_localcopy.h"
#include "omnipipe_template_utils.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2AllGatherSequenceOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                       InsAlgTemplate2>::InsV2AllGatherSequenceOmniPipeExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    opMode_ = param.opMode;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2AllGatherSequenceOmniPipeExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], "
              "dataType [%u] dataTypeSize [%u]",
              myRank_, rankSize_, devType_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2>::BuildSubCommAndTempMap(
    const OpParam& param,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    std::vector<std::vector<u32>>& subCommRanks0,
    std::vector<std::vector<u32>>& subCommRanks1,
    std::vector<std::vector<u32>>& subCommRanks2,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    HCCL_INFO("[BuildSubCommAndTempMap] topoUBX");
    // UBX拓扑: Level0=X(Mesh), Level1=Y(CLOS), Level2=Z(机间)
    std::vector<u32> closRanks;
    if (!algHierarchyInfo_.infos[0].empty() && !algHierarchyInfo_.infos[0][0].empty()) {
        subCommRanks0 = {algHierarchyInfo_.infos[0][0]};
        u32 meshSize = algHierarchyInfo_.infos[0][0].size();
        if (!algHierarchyInfo_.infos[0][1].empty()) {
            for (auto rank : algHierarchyInfo_.infos[0][1]) {
                if (rank % meshSize == topoInfo->userRank % meshSize) {
                    closRanks.push_back(rank);
                }
            }
        }
    }
    subCommRanks1 = {closRanks};
    if (!algHierarchyInfo_.infos[1].empty()) {
        subCommRanks2 = algHierarchyInfo_.infos[1];
    } else {
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }

    rankSizeLevel_[OMNIPIPE_LEVEL0] = subCommRanks0[0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL1] = subCommRanks1[0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL2] = subCommRanks2[0].size();
    tempMap.clear();
    if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
        tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
        tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
    }
    // Z轴使用InsTempAllGatherNHRDPU（非OmniPipe），放入tempMap以分配线程资源
    if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
        tempMap[OMNIPIPE_LEVEL2] = std::make_shared<InsTempAllGatherNHRDPUNoLocalCopy>(param, myRank_, subCommRanks2);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherSequenceOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    InitCommInfo(param, topoInfo, algHierarchyInfo);
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    rankSizeLevel_.resize(OMNIPIPE_LEVEL_NUM);
    rankIdxLevel_.resize(OMNIPIPE_LEVEL_NUM);

    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo,
            subCommRanks0, subCommRanks1, subCommRanks2, tempMap, topoInfo));

    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] = myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]) /
                                      rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = myRank_ / (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]);

    // Z轴使用InsTempAllGatherNHRDPU，已通过tempMap计算资源
    for (auto& temp : tempMap) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, temp.second, resourceRequest));
    }
    HCCL_INFO("[InsV2AllGatherSequenceOmniPipeExecutor][CalcRes]");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherSequenceOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2>::CalcResLevel(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    std::shared_ptr<InsAlgTemplateBase> tempAlg, AlgResourceRequest& resourceRequest) const
{
    AlgResourceRequest resReqlevel;
    CHK_RET(tempAlg->CalcRes(comm, param, topoInfo, resReqlevel));
    resourceRequest.slaveThreadNum += resReqlevel.slaveThreadNum + 1;
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.emplace_back(resReqlevel.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqlevel.notifyNumPerThread.begin(),
                                              resReqlevel.notifyNumPerThread.end());
    resourceRequest.channels.emplace_back(resReqlevel.channels[0]);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    PrepareResForTemplateLevel(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    u32 levelThreadNum = tempBase->GetThreadNum();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreads_[OMNIPIPE_LEVEL0].assign(threads_.begin() + 1, threads_.begin() + 1 + levelThreadNum);
        tempMainThreadsXY_.push_back(levelThreads_[OMNIPIPE_LEVEL0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreads_[OMNIPIPE_LEVEL1].assign(threads_.begin() + 1 + levelThreads_[OMNIPIPE_LEVEL0].size(),
                                              threads_.begin() + 1 + levelThreads_[0].size() + levelThreadNum);
        tempMainThreadsXY_.push_back(levelThreads_[OMNIPIPE_LEVEL1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreads_[OMNIPIPE_LEVEL2].assign(
            threads_.begin() + 1 + levelThreads_[OMNIPIPE_LEVEL0].size() + levelThreads_[OMNIPIPE_LEVEL1].size(),
            threads_.end());
        tempMainThreadsZ_.push_back(levelThreads_[OMNIPIPE_LEVEL2].at(0));
    }

    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    if (level < OMNIPIPE_LEVEL2) {
        ntfIdxCtrlToTempXY_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlXY_.push_back(tempMainThreadsXY_.size() + tempMainThreadsZ_.size() - 1);
    } else {
        ntfIdxCtrlToTempZ_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlZ_.push_back(tempMainThreadsXY_.size() + tempMainThreadsZ_.size() - 1);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    maxTmpMemSize_ = resCtx.cclMem.size;

    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;

    rankSizeLevel_.resize(OMNIPIPE_LEVEL_NUM);
    rankIdxLevel_.resize(OMNIPIPE_LEVEL_NUM);

    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo_,
            subCommRanks0, subCommRanks1, subCommRanks2ForZ_, tempMap, &resCtx.topoInfo));

    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] = myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]) /
                                      rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = myRank_ / (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]);

    threads_ = resCtx.threads;
    controlThread_ = threads_.at(0);
    levelThreads_.resize(OMNIPIPE_LEVEL_NUM);

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
        tempMap[OMNIPIPE_LEVEL1]->SetchannelsPerRank(remoteRankToChannelInfo_[1]);
    }
    for (auto& temp : tempMap) {
        CHK_RET(PrepareResForTemplateLevel(temp.first, temp.second));
    }

    HcclResult ret = OrchestrateLoop(param, resCtx, tempMap);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherSequenceOmniPipeExecutor][Orchestrate][rank:%u] errNo[0x%016llx] "
                   "AllGather executor kernel run failed", myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherSequenceOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                       InsAlgTemplate2>::GenTemplateAlgParamsByDimData(TemplateDataParams& tempAlgParams,
                                                                                       StepSliceInfo& stepSliceInfo) const
{
    CHK_RET(FillOmniPipeTemplateAlgParams(tempAlgParams, stepSliceInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherSequenceOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                       InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap)
{
    HCCL_INFO("[InsV2AllGatherSequenceOmniPipeExecutor][OrchestrateLoop] Start");

    u64 zRankSize = rankSizeLevel_[OMNIPIPE_LEVEL2];
    u64 xRankSize = rankSizeLevel_[OMNIPIPE_LEVEL0];
    u64 yRankSize = rankSizeLevel_[OMNIPIPE_LEVEL1];
    u64 xyRankSize = xRankSize * yRankSize;
    u64 xAxis = rankIdxLevel_[OMNIPIPE_LEVEL0];
    u64 yAxis = rankIdxLevel_[OMNIPIPE_LEVEL1];

    // ==================== Z轴AllGather模板参数准备 ====================
    TemplateDataParams zAlgParams;
    zAlgParams.buffInfo.inputPtr = param.inputPtr;
    zAlgParams.buffInfo.outputPtr = param.outputPtr;
    zAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    zAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    zAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    zAlgParams.buffInfo.hcclBuffBaseOff = 0;
    zAlgParams.repeatNum = 1;
    zAlgParams.inputRepeatStride = 0;
    zAlgParams.outputRepeatStride = 0;

    // ==================== 机内X+Y二维OmniPipe参数 ====================
    double bw_ag_l0 = BW_OMNI_DEFAULT;
    double bw_ag_l1 = BW_OMNI_UBX_AG_CLOS;
    double eqBw0 = bw_ag_l0;
    double eqBw1 = yRankSize > 1 ? bw_ag_l1 / (yRankSize - 1) : bw_ag_l1;
    std::vector<double> endpointAttrBwXY{eqBw0, eqBw1, 1.0};

    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;

    // Z轴loop参数：根据zRankSize计算（cclBuff总大小 / zRankSize / 数据类型大小）
    u64 zScratchBoundDataSize = maxTmpMemSize_ / (zRankSize > 1 ? zRankSize : 1) / HCCL_MIN_SLICE_ALIGN *
                                HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    u64 zMaxCountPerLoop = std::min(zScratchBoundDataSize, transportBoundDataSize);
    u64 zLoopTimes = dataCount_ / zMaxCountPerLoop +
                     static_cast<u64>(dataCount_ % zMaxCountPerLoop != 0);
    u64 zPerLoopSize = zMaxCountPerLoop * dataTypeSize_;

    // XY loop参数：根据总数据量计算（cclBuff总大小 / 总rank数 / 数据类型大小）
    u64 xyScratchBoundDataSize = maxTmpMemSize_ / rankSize_ / HCCL_MIN_SLICE_ALIGN *
                                 HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    u64 xyMaxCountPerLoop = std::min(xyScratchBoundDataSize, transportBoundDataSize);
    u64 xyLoopTimes = dataCount_ / xyMaxCountPerLoop +
                      static_cast<u64>(dataCount_ % xyMaxCountPerLoop != 0);
    u64 xyPerLoopSize = xyMaxCountPerLoop * dataTypeSize_;

    // 2D OmniPipe：zRankSize=1，使用XY的perLoopSize
    std::vector<u64> dataSizePerLoopXY(xyRankSize, xyPerLoopSize);
    std::vector<u64> dataWholeSizeXY(xyRankSize, xyPerLoopSize);

    OmniPipeSliceParam xySliceParam;
    xySliceParam.levelRankSize = {xRankSize, yRankSize, 1};
    xySliceParam.endpointAttrBw = endpointAttrBwXY;
    xySliceParam.dataSizePerLoop = dataSizePerLoopXY;
    xySliceParam.dataTypeSize = dataTypeSize_;
    xySliceParam.levelRankId = {xAxis, yAxis, 0};
    xySliceParam.opMode = opMode_;
    xySliceParam.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    xySliceParam.dataWholeSize = dataWholeSizeXY;

    OmniPipeSliceInfo xyAlignSliceInfo = CalcAGOmniPipeSliceInfo(xySliceParam);

    OmniPipeSliceInfo xyTailSliceInfo;
    if (dataCount_ % xyMaxCountPerLoop != 0) {
        u64 tailPerLoopSize = (dataCount_ % xyMaxCountPerLoop) * dataTypeSize_;
        std::vector<u64> tailDataSizePerLoopXY(xyRankSize, tailPerLoopSize);
        std::vector<u64> tailDataWholeSizeXY(xyRankSize, tailPerLoopSize);
        xySliceParam.dataSizePerLoop = tailDataSizePerLoopXY;
        xySliceParam.dataWholeSize = tailDataWholeSizeXY;
        xyTailSliceInfo = CalcAGOmniPipeSliceInfo(xySliceParam);
    }

    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;
    for (auto& temp : tempMap) {
        tempResMap[temp.first].channels = remoteRankToChannelInfo_[temp.first];
        tempResMap[temp.first].threads = levelThreads_[temp.first];
        tempAlgParamMap[temp.first].buffInfo.hcclBuff = resCtx.cclMem;
    }

    u64 zPieceCount = zRankSize > 1 ? zRankSize : 1;
    u64 xyLocalRankId = xAxis + yAxis * xRankSize;  // 本卡在同Z平面内的2D rank ID

    HCCL_DEBUG("zLoopTimes[%llu], xyLoopTimes[%llu], zRankSize[%llu], xRankSize[%llu], yRankSize[%llu]",
               zLoopTimes, xyLoopTimes, zRankSize, xRankSize, yRankSize);

    // ==================== Z轴Loop ====================
    // loop次数根据zRankSize计算，每个loop将本卡的一段数据从input拷到cclBuff，
    // 通过InsTempAllGatherNHRDPU发往同Z轴其他卡，接收数据写入output。
    if (zRankSize > 1) {
        u64 zProcessedDataCount = 0;
        for (u64 zLoop = 0; zLoop < zLoopTimes; zLoop++) {
            u64 zCurrDataCount = (zLoop == zLoopTimes - 1) ? dataCount_ - zProcessedDataCount : zMaxCountPerLoop;
            u64 zCurrDataSize = zCurrDataCount * dataTypeSize_;

            // 拷贝本卡数据从input到cclBuff
            DataSlice zSrc(param.inputPtr, zProcessedDataCount * dataTypeSize_,
                           zCurrDataSize, zCurrDataCount);
            DataSlice zDst(resCtx.cclMem.addr, myRank_ * zCurrDataSize,
                           zCurrDataSize, zCurrDataCount);
            CHK_RET(LocalCopy(controlThread_, zSrc, zDst));

            // Z轴AllGather：cclBuff中的本卡数据 → 同Z轴其他卡 → 接收数据写入output
            zAlgParams.count = zCurrDataCount;
            zAlgParams.buffInfo.inBuffBaseOff = zProcessedDataCount * dataTypeSize_;
            zAlgParams.buffInfo.outBuffBaseOff = xyLocalRankId * dataSize_ + zProcessedDataCount * dataTypeSize_;
            zAlgParams.sliceSize = zCurrDataSize;
            zAlgParams.tailSize = zAlgParams.sliceSize;
            zAlgParams.inputSliceStride = 0;
            zAlgParams.outputSliceStride = dataSize_ * xyRankSize;

            zAlgParams.allRankSliceSize.clear();
            zAlgParams.allRankDispls.clear();
            zAlgParams.allRankProcessedDataCount.clear();
            for (u32 i = 0; i < zRankSize; i++) {
                zAlgParams.allRankDispls.emplace_back(i * zCurrDataSize);
                zAlgParams.allRankSliceSize.emplace_back(zCurrDataSize);
                zAlgParams.allRankProcessedDataCount.emplace_back(zCurrDataCount);
            }

            CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxCtrlToTempZ_));
            CHK_RET(tempMap[OMNIPIPE_LEVEL2]->KernelRun(param, zAlgParams,
                                                         tempResMap[OMNIPIPE_LEVEL2]));
            CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxTempToCtrlZ_));

            zProcessedDataCount += zCurrDataCount;
        }
        HCCL_INFO("[InsV2AllGatherSequenceOmniPipeExecutor][OrchestrateLoop] Full Z AG done, "
                  "zLoopTimes[%llu]", zLoopTimes);
    }

    // ==================== XY Loop（机内2D OmniPipe + 循环多片Z数据） ====================
    // loop次数根据总数据量计算
    {
        u64 xyProcessedDataCount = 0;
        OmniPipeSliceInfo xyOmniPipeSliceInfo;

        for (u64 xyLoop = 0; xyLoop < xyLoopTimes; xyLoop++) {
            u64 xyCurrDataCount = (xyLoop == xyLoopTimes - 1) ? dataCount_ - xyProcessedDataCount : xyMaxCountPerLoop;
            u64 xyCurrDataSize = xyCurrDataCount * dataTypeSize_;

            if (xyLoop == xyLoopTimes - 1 && dataCount_ % xyMaxCountPerLoop != 0) {
                xyOmniPipeSliceInfo = xyTailSliceInfo;
            } else {
                xyOmniPipeSliceInfo = xyAlignSliceInfo;
            }

            u32 xyStepCount = xyOmniPipeSliceInfo.dataSliceLevel0.size();

            // 循环处理zRankSize片不连续数据
            for (u64 zPiece = 0; zPiece < zPieceCount; zPiece++) {
                u64 srcRankId = zPiece * xyRankSize + yAxis * xRankSize + xAxis;
                u64 pieceOutputOffset = srcRankId * dataSize_ + xyProcessedDataCount * dataTypeSize_;

                // 拷贝Z片数据从output到cclBuff
                DataSlice pieceSrc(param.outputPtr, pieceOutputOffset,
                                   xyCurrDataSize, xyCurrDataCount);
                DataSlice pieceDst(resCtx.cclMem.addr, myRank_ * xyCurrDataSize,
                                   xyCurrDataSize, xyCurrDataCount);
                CHK_RET(LocalCopy(controlThread_, pieceSrc, pieceDst));

                // 执行机内2D OmniPipe
                for (u32 j = 0; j < xyStepCount; j++) {
                    CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxCtrlToTempXY_));
                    if (xRankSize > 1) {
                        CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_LEVEL0],
                                                              xyOmniPipeSliceInfo.dataSliceLevel0[j]));
                        CHK_RET(tempMap[OMNIPIPE_LEVEL0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL0],
                                                                     tempResMap[OMNIPIPE_LEVEL0]));
                    }
                    if (yRankSize > 1) {
                        CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_LEVEL1],
                                                              xyOmniPipeSliceInfo.dataSliceLevel1[j]));
                        CHK_RET(tempMap[OMNIPIPE_LEVEL1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL1],
                                                                     tempResMap[OMNIPIPE_LEVEL1]));
                    }
                    CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxTempToCtrlXY_));
                }

                // 拷贝OmniPipe结果从cclBuff到output
                for (u32 rank = 0; rank < rankSize_; rank++) {
                    DataSlice dstSlice(param.outputPtr,
                                       (rank * dataCount_ + xyProcessedDataCount) * dataTypeSize_,
                                       xyCurrDataSize, xyCurrDataCount);
                    DataSlice srcSlice(resCtx.cclMem.addr, rank * xyCurrDataSize,
                                       xyCurrDataSize, xyCurrDataCount);
                    CHK_RET(LocalCopy(controlThread_, srcSlice, dstSlice));
                }
            }
            xyProcessedDataCount += xyCurrDataCount;
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherSequenceOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
    InsAlgTemplate2>::RestoreChannelMap(
    const AlgResourceCtxSerializable& resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const
{
    rankIdToChannelInfo.resize(OMNIPIPE_LEVEL_NUM);
    u32 level = 0;
    for (u32 i = 0; i < OMNIPIPE_LEVEL_NUM; i++) {
        if (rankSizeLevel_[i] > 1) {
            for (auto& channel : resCtx.channels[level]) {
                u32 remoteRank = channel.remoteRank;
                rankIdToChannelInfo[i][remoteRank].push_back(channel);
            }
            level++;
        }
    }
    return HCCL_SUCCESS;
}

// ==================== Registration ====================
// UBX拓扑: X=Mesh, Y=CLOS/NHR, Z=机间
// Z轴使用InsTempAllGatherNHRDPU（非OmniPipe），仅X和Y使用OmniPipe模板
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, InsV2AllGatherSequenceOmniPipe,
                       InsV2AllGatherSequenceOmniPipeExecutor, TopoMatchUBX,
                       InsTempAllGatherOmniPipeMesh1D, InsTempAllGatherOmniPipeNHR,
                       InsTempAllGatherNHRDPUNoLocalCopy);
}  // namespace ops_hccl
