/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_alg.h"
#include "gather_operator.h"
#include "send_receive_operator.h"
#include "alltoall_operator.h"
#include "all_reduce_operator.h"
#include "broadcast_operator_for_hetero.h"
#include "coll_alg_op_registry.h"
#include "topo_matcher.h"
#include "topo_info_extractor.h"
#include "alg_configurator.h"

namespace hccl
{
    constexpr u32 TINY_MEMORY_SIZE = 32; // sendBuff或recvBuff为空时, 使用的DeviceMem大小

    HcclAlg::HcclAlg(CCLBufferManager &cclBufferManager, const HcclDispatcher dispatcher, const HcclDispatcher vDispatcher) : cclBufferManager_(cclBufferManager), dispatcher_(dispatcher), vDispatcher_(vDispatcher)
    {
    }

    HcclAlg::~HcclAlg()
    {
    }

    HcclResult HcclAlg::Init(const void *transportResourceInfoAddr, size_t transportResourceInfoSize,
                             std::unique_ptr<WorkspaceResource> &workSpaceRes,
                             const std::unique_ptr<NotifyPool> &notifyPool, std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
                             const std::unique_ptr<QueueNotifyManager> &queueNotifyManager,
                             HcclAlgoAttr &algoAttr, HcclTopoAttr &topoAttr, bool isHeterogComm)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::Init(HcclAlgoAttr &algoAttr, HcclTopoAttr &topoAttr, bool isHeterogComm)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::GetAlltoAllStagedWorkSpaceMemSize(
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &memSize)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::GetTopoType(TopoType &topoType)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::SetAlgType(AlgType algType, HcclCMDType opType)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::GetAlgType(AlgType &algType, HcclCMDType opType)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::SupportDeterministicOptim(bool &isDeterministicOptim)
    {
        return HCCL_SUCCESS;
    }

    u8 HcclAlg::GetDeterministicConfig() const
    {
        return 0;
    }

    HcclResult HcclAlg::SetDeterministicConfig(const u8 deterministic)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::SetAivModeConfig(const bool aivMode)
    {
        return HCCL_SUCCESS;
    }

    bool HcclAlg::GetAicpuUnfoldConfig() const
    {
        return false;
    }

    HcclResult HcclAlg::SetAicpuUnfoldConfig(const bool aicpuUnfold)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::GetRankVecInfo(std::vector<std::vector<std::vector<u32>>> &serverAndsuperPodToRank)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::GetIsBridgeVector(std::vector<bool> &isBridgeVector)
    {
        return HCCL_SUCCESS;
    }
    HcclResult HcclAlg::GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>> &commPlaneRanks)
    {
        return HCCL_SUCCESS;
    }

    void HcclAlg::GetCommPlaneVector(std::vector<std::vector<std::vector<RankInfo>>> &commPlaneVector)
    {
    }

    HcclResult HcclAlg::GetCommPlaneSubGroupVector(std::vector<std::vector<std::vector<std::vector<u32>>>> &commPlaneSubGroupVector)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::GetAHCAlgOption(std::map<AHCConcOpType, TemplateType> &ahcAlgOption)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::GetIsUsedRdmaMap(std::unordered_map<u32, bool> &isUsedRdmaMap)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::GetTinyMem(DeviceMem &tinySendRecvMem)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::InitExternalEnable(HcclExternalEnable &externalEnable)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::InitTopoInfo(HcclTopoInfo &topoInfo, HcclTopoAttr &topoAttr)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::InitAlgoInfo(HcclAlgoInfo &algoInfo, HcclAlgoAttr &algoAttr)
    {
        return HCCL_SUCCESS;
    }

    // 上层保证，以下方法在初始化成功后才会调用，所以未对pimpl_进行保护判断
    HcclResult HcclAlg::ReleaseCommInfos()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::Broadcast(
        const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root, Stream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType, u32 destRank,
                             Stream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
                                     u32 destRank, Stream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
                                u32 srcRank, Stream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
                                        u32 srcRank, Stream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank, u64 inputCount,
                               HcclDataType dataType, Stream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::ClearOpResource(const std::string &tag)
    {
        return HCCL_SUCCESS;
    }

    bool HcclAlg::IsExistCommRes(const std::string &tag)
    {
        return false;
    }

    HcclResult HcclAlg::CreateMutiStreamRes(const std::string &tag, Stream &stream, level1StreamInfo_t &streamInfo,
                                            AlgType algType, bool isAicpuModeEn)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::CreateComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
                                   std::unique_ptr<CommInfo> &commInfo, u32 root, bool isP2p, bool isAicpuModeEn)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::CreateComm(
        const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType, u32 root, bool isP2p)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::CreateP2PCommQuerry(const std::string &tag, u32 &status)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclAlg::CreateP2PCommAsync(const std::string &tag, DeviceMem &mem, u32 peerRank, u32 &status)
    {
        return HCCL_SUCCESS;
    }

    void HcclAlg::CancelCommRes(const std::string &tag)
    {
    }

    void HcclAlg::Break()
    {
    }

    HcclResult HcclAlg::SetHDCModeInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &rankDevicePhyIdNicInfoMap,
        std::vector<u32> &ranksPort, bool isSetHDCModeInfo, bool isUseRankPort)
    {
        return HCCL_SUCCESS;
    }
}
