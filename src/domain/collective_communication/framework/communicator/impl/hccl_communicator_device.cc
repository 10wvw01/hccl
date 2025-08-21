/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_communicator.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <sys/time.h>
#include "externalinput_pub.h"
#include "env_config.h"
#include <memory>
#include "p2p_mgmt_pub.h"
#include "opexecounter_pub.h"
#include "config.h"
#include "stream_active_manager.h"
#include "device_capacity.h"
#include "profiling_manager_pub.h"
#include "task_exception_handler_pub.h"
#include "rank_consistentcy_checker.h"
#include "hccl_aiv.h"
#include "task_abort_handler_pub.h"
#include "adapter_rts_common.h"
#include "coll_alg_utils.h"
#include "../common/src/state_guard.h"
#include "detect_connect_anomalies.h"
#include "alg_profiling.h"
#include "preempt_port_manager.h"
#include "mmpa_api.h"
#include "config_log.h"
#include "../nslbdp/hccl_nslbdp.h"

using namespace std;

constexpr u32 MODULE_NUM_FOUR = 4;

namespace hccl
{
    HcclCommunicator::HcclCommunicator()
        : dispatcher_(nullptr), vDispatcher_(nullptr), notifyPool_(nullptr),
          initializedFlag_(ATOMIC_FLAG_INIT), userRank_(INVALID_VALUE_RANKID), realUserRank_(INVALID_VALUE_RANKID),
          userRankSize_(INVALID_VALUE_RANKSIZE), drvInit_(false), inlineReduceSwitchOn_(true),
          nicDeployment_(NICDeployment::NIC_DEPLOYMENT_DEVICE), devicePhyId_(INVALID_UINT),
          deviceLogicId_(-1), localRank_(INVALID_VALUE_RANKID), hostSocketHandle_(nullptr),
          isUsedRdmaLevel0_(false), nicInitialized_(0), hcomGroupNicInit_(false),
          profilingMode_(HcomProfilingMode::PROFILING_CLOSE), raResourceInit_(false),
          interServer_(false), isSingleMeshAggregation_(false), cclBufferManager_(CCLBufferManager()),
          isExecuteProfilingInit_(false), deviceType_(DevType::DEV_TYPE_COUNT),
          commHandle_(nullptr),
          commWorkMode_(WorkMode::HCCL_MODE_NORMAL), meshAggregationRankSize_(0), isHaveCpuRank_(false), ranktableCrc_(0),
          pMsgInfosMem_(nullptr), pReqInfosMem_(nullptr), memBlocksManager_(nullptr), pRecvWrInfosMem_(nullptr),
          transportResInfo_(mrManager_, pMsgInfosMem_, pReqInfosMem_, memBlocksManager_, pRecvWrInfosMem_),
          multiModuleDiffDeviceNumMode_(false), multiSuperPodDiffServerNumMode_(false),
          isStandardCard_(false), is310PDuoCard_(false), hccsPortNum_(-1),
          loopBackIp_(HcclIpAddress(COMM_LOOPBACK_IP)), profilingInitiated_(false), callbackThreadId_(INVALID_U64),
          role_(SERVER_ROLE_SOCKET), mrManagerInit_(false),
          isHostUseDevNic_(false),
          isAllRankSamePlane_(false), serverNum_(0), moduleNum_(0)
    {
    }

    HcclCommunicator::~HcclCommunicator()
    {
    }

    HcclResult HcclCommunicator::Init(HcclCommParams &params, const RankTable_t &rankTable)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Init(HcclCommParams &params, const std::vector<RankInfo> &rankList,
                                      WorldGroupInfo &groupCommonData)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitOneSidedService(const RankTable_t &rankTable)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitOneSidedServiceNetDevCtx(u32 remoteRankId)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DeInitOneSidedServiceNetDevCtx()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetOneSidedService(IHcclOneSidedService **service)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::OneSidedServiceStartListen(NicType nicType, HcclNetDevCtx netDevCtx)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetOneSidedServiceDevIpAndPort(NicType nicType, HcclIpAddress& ipAddress, u32& port)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DeinitOneSidedService()
    {
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::IsSupportZeroCopy(const OpParam &opParam)
    {
        return false;
    }

    HcclResult HcclCommunicator::PrepareZeroCopy(const std::string &algName, const AlgDesc &algDesc, OpParam &opParam)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::UpdateZeroCopy(const OpParam &opParam, const AlgResourceResponse &resp)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildZeroCopyParam()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitCommParams(HcclCommParams &params)
    {
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::Is310PDuoCard()
    {
        return false;
    }

    // 910B A+X 在RDMA未启用情况下，两模块间的device数目需要一致且两模块中使用的卡都在同一平面上
    HcclResult HcclCommunicator::CheckSingleServerComm(const std::vector<RankInfo_t> &rankList) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckDataType(const HcclDataType dataType, bool needReduce)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitZeroCopyMemoryAgent()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DeinitZeroCopyMemoryAgent(bool inDestructor)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ClearResMap(const std::string &tag, bool &findTag)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ClearOpResource(const std::string &tag)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
                                                        const HcomCollOpInfo &opInfo)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::CreateRemoteOpBasedResources(u64 memSize, const std::string &tag)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::DestroyRemoteOpBasedMem(const std::string &tag)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    bool HcclCommunicator::IsAtomicInit()
    {
        return false;
    }

    bool HcclCommunicator::IsNeedNicInit()
    {
        return false;
    }

    HcclResult HcclCommunicator::GetBandWidthPerNPU(u32 level, float &bandWidth)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::GetDeviceNumPerAggregation(u32 &deviceNumPerAggregation)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitHccp()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckReduceDataType(const HcclDataType dataType, const HcclReduceOp op)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetAlgType(AlgType &algType, HcclCMDType opType)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::GetCommParams(HcclCommParams &params)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::GetCommRankTable(RankTable_t &rankTable)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::InitPara()
    {
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::IsStandardCard()
    {
        return false;
    }

    HcclResult HcclCommunicator::InitOpRetry()
    {
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::CompareWithServerId(const ServerInfo_t &left, const ServerInfo_t &right)
    {
        return false;
    }

    bool HcclCommunicator::CompareWithNicName(const NetworkInfo_t &left, const NetworkInfo_t &right)
    {
        return false;
    }

    bool HcclCommunicator::CompareWithUserRank(const RankInfo &left, const RankInfo &right)
    {
        return false;
    }

    HcclResult HcclCommunicator::InitPreResource(const RankTable_t &rankTable)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitTcpMode(const RankTable_t &rankTable) const
    {
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::IsEnableBackupLink()
    {
        return false;
    }

    HcclResult HcclCommunicator::InitRaResource()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DisablePreResource()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetWorkspaceSubStreamNum(u64 &streamNum, u64 dataSize, HcclCMDType opType)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DestroyNetworkResources()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetWorkspaceResource(const std::string &tag, void *memPtr, u64 &maxSize,
                                                      std::vector<rtStream_t> &stream)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    void HcclCommunicator::DestroyWorkspaceResource(const std::string &tag)
    {
    }

    HcclResult HcclCommunicator::AtomicInitSet()
    {
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::AtomicInitClear()
    {
    }

    u32 HcclCommunicator::GetUserRank()
    {
        return 0;
    }

    u32 HcclCommunicator::GetGroupRank()
    {
        return 0;
    }

    u32 HcclCommunicator::GetRankSize()
    {
        return 0;
    }

    bool HcclCommunicator::GetNicInitialized()
    {
        return false;
    }

    /*
        1. 选择算法
        2. 计算resource，存到request内
        3. 创建和分配资源
    */
    HcclResult HcclCommunicator::HcclSelectAlg(HcclCMDType opType, u64 count, HcclDataType dataType,
                                               HcclReduceOp op, bool &ifAiv, std::string &algName, bool isSuperKernel)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::HcclCalcBlockDim(HcclCMDType opType, u64 count, HcclDataType dataType,
                                                  std::string &algName, u32 &blockDim)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::HcclGetAlgExecParam(const std::string &tag, HcclCMDType opType, u64 count, void *inputPtr, void *outputPtr,
                                                     bool clearEnable, HcclDataType dataType, HcclReduceOp op, void *&commContext, u64 &len, u32 aivCoreLimit)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckDeviceType(const DevType deviceType) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckReductionOp(const HcclReduceOp op) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckUserRank(const u32 userRank) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckCount(const u64 count) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetGroupRanksInfo(const std::vector<u32> &groupRanks, std::vector<RankInfo> &ranksInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetGroupCommonData(WorldGroupInfo &groupCommonData) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetWorkspaceMemSize(const std::string &opType, u64 count, HcclDataType dataType,
                                                     u32 &rankSize, u64 &memSize, DevType &deviceType) const
    {
        return HCCL_E_NOT_SUPPORT;
    }

    DeviceMem HcclCommunicator::GetWorkspaceScracthMem(const std::string &tag, u64 allocMemSize)
    {
        return DeviceMem();
    }

    std::vector<Stream> HcclCommunicator::GetWorkspaceSubStreams(const std::string &tag, u32 num)
    {
        return {};
    }

    HcclResult HcclCommunicator::InitProfiling()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DeinitProfiling()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::RegistTaskExceptionHandler() const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::UnRegistTaskExceptionHandler() const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetInCCLbuffer(void *&buffer, u64 &size)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::GetOutCCLbuffer(void *&buffer, u64 &size)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    void HcclCommunicator::ReleaseCommCCLbuffer()
    {
    }

    HcclResult HcclCommunicator::ReleaseCommInfos()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitProfiler()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateCommCCLbuffer()
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::CreateCommExpBuffer()
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    u32 HcclCommunicator::GetLocalNicPort(NicType nicType)
    {
        return 0;
    }

    HcclResult HcclCommunicator::InitNic(bool isMC2ReInit)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DeinitNic()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::RegisterRanksToDca()
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::RegisterToHeartBeat()
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::RegisterToHeartBeat(u32 peerRankId, string &tag)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    void HcclCommunicator::UnRegisterToHeartBeat()
    {
    }

    HcclResult HcclCommunicator::SetGlobalWorkSpace(std::vector<void *> &globalWorkSpaceAddr)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetandClearOverFlowTasks(std::vector<HcclDumpInfo> &hcclDumpInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetDeviceId(s32 &deviceId) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetQosCfg(const u32 qosCfg)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::ResetQosCfg()
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::GetQosCfg(u32 &qosCfg)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::GetCqeError(HcclResult &result)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::MrManagerInit()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::MrManagerDeInit()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SupportDeterministicOptim(bool &isDeterministicOptim)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetHccsLinkNum(u32 &numHccsLink)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
                                           HcclDataType dataType, HcclRtStream stream, HcomCollOpInfo *opInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllGatherV(const std::string &tag, const void *sendBuf, u64 sendCount, const void *recvBuf,
                                            const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                             HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcclCMDType cmdType)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
                                                   u64 inputCount, HcclDataType dataType, HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllGatherVOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
                                                    u64 inputCount, const void *outputCounts, const void *outputDispls, HcclDataType dataType, HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::GetAndSetSyncMode(SyncMode &preSyncMode, SyncMode newSyncMode)
    {
    }

    void HcclCommunicator::RestorePreSyncMode(SyncMode preSyncMode, SyncMode newSyncMode)
    {
    }

    HcclResult HcclCommunicator::AllReduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                           HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
                                           SyncMode syncMode, const HcomCollOpInfo *opInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllReduceAicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                                      HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                                   HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
                                                   SyncMode syncMode)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
                                           HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                                           rtStream_t stream, const std::string &tag)
    {

        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls,
                                                   HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                                                   rtStream_t stream, const std::string &tag)
    {

        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
                                            const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AlltoAllVCOutPlace(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
                                                    const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AlltoAll(const void *sendBuf, u64 sendCount, HcclDataType sendType,
                                          const void *recvBuf, u64 recvCount, HcclDataType recvType, rtStream_t stream, const std::string &tag)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
                                           HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BroadcastOutPlace(const std::string &tag, void *ptr, u64 count, HcclDataType dataType,
                                                   u32 root, HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Scatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
                                         HcclDataType dataType, u32 root, HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
                                                 HcclDataType dataType, u32 root, HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                        HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                                HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReduceScatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                               HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcomCollOpInfo *opInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
                                                       u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReduceScatterV(const std::string &tag, void *inputPtr,
                                                const void *inputCounts, const void *inputDispls, void *outputPtr, u64 outputCount,
                                                HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcomCollOpInfo *opInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReduceScatterVOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
                                                        const void *inputCounts, const void *inputDispls, u64 outputCount,
                                                        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BatchSendRecv(const std::string &tag, HcclSendRecvItem *sendRecvItemsPtr, u32 itemNum,
                                               rtStream_t stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
                                      u32 destRank, rtStream_t stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
                                              u32 destRank, rtStream_t stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
                                         u32 srcRank, rtStream_t stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count,
                                                 HcclDataType dataType, u32 srcRank, rtStream_t stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank,
                                        u64 inputCount, HcclDataType dataType, rtStream_t stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::RegressCalPreOp(AlltoAllOperator *&alltoAllOperator, const OpParam &opParam,
                                                 std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::RegressCalPreOp(AlltoAllOperator *&alltoAllOperator, const OpParam &opParam,
                                                 std::unique_ptr<PreProcessMetaInfo> &preMetaInfo, Stream &preProcessStream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ExecOp(HcclCMDType opType, OpParam &opParam, bool isCustom)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::FreeScratchMemOnOpBaseMode(DeviceMem &scratchMem, const OpParam &opParam,
                                                            const HcclCMDType &opType)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ExecOpAlltoAll(HcclCMDType opType, OpParam &opParam, bool isCustom)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::HandleAclGraphFirstOpAivBuff(rtStream_t mainStream)
    {
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::StreamIsCapture(rtStream_t mainStream)
    {
        bool isCapture = false;
        return isCapture;
    }

    HcclResult HcclCommunicator::CaptureSlaveStreams(rtStream_t mainStream, vector<Stream> &slaveStreams)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpLocalScratchMemResParam(
        const AlgResourceResponse &algResource, const std::string &newTag, LocalResInfoV2 *localResHostPtr)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckSetRetryStateToWaitResume()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpLocalResParam(const AlgResourceResponse &algResource, const std::string &newTag)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocAndGetStreamContextBuff(u32 streamId, u64 &addr, u64 &size)
    {
        return HCCL_SUCCESS;
    }

    u32 HcclCommunicator::UpdateOpIndex(const OpParam &opParam)
    {
        return 0;
    }

    HcclResult HcclCommunicator::BuildAicpuCustomParam()
    {
        return HCCL_SUCCESS;
    }

    template <typename T>
    HcclResult HcclCommunicator::CopyVectorToDeviceMem(const u64 len, DeviceMem &dstDeviceMem, const std::vector<T> &srcVec)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpTopoResTlvParam(const std::string &algName,
                                                        const std::vector<std::vector<std::vector<u32>>> &inputVectorInfo, DeviceMem &dstTlvDeviceMem, u64 &tlvLen)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpTopoResVectorTlvParam(const std::string &algName,
                                                              const std::vector<std::vector<std::vector<std::vector<u32>>>> &inputVectorInfo, DeviceMem &dstTlvDeviceMem, u64 &tlvLen)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildPairLinkCounter(const std::string &algName)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildIsUsedRdmaRank(const std::string &algName)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildNicList(const std::string &algName)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildBridgeRank(const std::string &algName)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildCommPlanRank(const std::string &algName)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildServerAndsuperPodRank(const std::string &algName)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpRetryParam(const AlgResourceResponse &algResource, const std::string &newTag)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildCommPlaneSubGroupRank(const std::string &algName)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildHierarchicalAlgOption(const std::string &algName)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpTopoResParam(const std::string &algName, const AlgResourceResponse &algResource)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpRemoteLinkP2pResParam(const LINK &link, HccltagRemoteResV3 &tagRemoteRes,
                                                              TransportLinkType linkType)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpRemoteLinkRoceResParam(const LINK &link, HccltagRemoteResV3 &tagRemoteRes,
                                                               bool isBackup, bool isRetry, bool IsSecondBuild)
    {
        return HCCL_SUCCESS;
    }

    template <typename T>
    HcclResult HcclCommunicator::CreateListNode(T **resHostPtr, T **resDevicePtr)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildRemoteResByTag(const std::string &newTag, const u32 &usrRankId,
                                                     HcclRankRelationResV2 *&rankRelationResHostPtr, HcclRankRelationResV2 *&rankRelationResDevicePtr, bool isBackup,
                                                     bool isRetry)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildRelationResByRemoteRankId(const TransportRequest &transportRequest, const LINK &link,
                                                                HcclRankRelationResV2 *&rankRelationResHostPtr, HcclRankRelationResV2 *&rankRelationResDevicePtr)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ParseRemoteDataToMem(const OpCommTransport &opTransportResponse, const std::string &newTag,
                                                      const HcclCMDType opType, bool isBackup, bool isRetry)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpRemoteResParam(const AlgResourceResponse &algResource, const std::string &newTag,
                                                       const HcclCMDType opType, bool isRetry)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CopyHostListResToDeviceParam(const std::string &newTag, const ListCommon *headHostList, const u64 size)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CopyHostOpResToDeviceParam(const std::string &newTag)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpResParam(
        const std::string &algName, const AlgResourceResponse &algResource, const std::string &newTag,
        const HcclCMDType opType)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildCustomOpResParam()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::RegisterDfxInfo(const OpParam &param, AlgType algType,
        std::vector<Stream> slaveStreams, bool isAiv)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetReportHcclMC2Info(const Stream &kfcStream, const std::vector<Stream> &aicpuStreams)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::OrchestrateAicpu(const HcclCMDType &opType, const std::string &algName,
                                                  const OpParam &param, const AlgResourceResponse &algResource, const std::string &newTag, AlgType algType,
                                                  bool isCustom)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CalcTinySendRecvMem(const OpParam &opParam, AlgResourceResponse &algResResponse,
                                                     DeviceMem &tinySendRecvMem)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocAlgNotifys(const std::string &tag, const NotifyLoadType notifyLoadType, const u32 notifyNum,
                                                 std::vector<std::shared_ptr<LocalNotify>> &notifiesMain, std::vector<std::shared_ptr<LocalNotify>> &notifiesAux)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocAlgResource(const std::string &newTag, HcclCMDType opType, const OpParam &opParam,
                                                  AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse)
    {
        SaveLinkRes(algResResponse.opTransportResponse);
        SaveLinkRes(algResResponse.opTransportResponseBackUp);

        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::IncreAllocLink(const std::string &newTag, const OpParam &opParam,
                                                AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse)
    {
        SaveLinkRes(algResResponse.opTransportResponse);
        SaveLinkRes(algResResponse.opTransportResponseBackUp);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitRecvMsgAndRequestBuffer()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitMemBlocksAndRecvWrMem()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetDevicePid(s32 devicePid)
    {
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::ReleaseWorkSpacebuffer()
    {
    }

    HcclResult HcclCommunicator::AllocAndClearDeviceMem(u64 size, std::shared_ptr<DeviceMem> &bufferPtr) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocAndClearHostMem(u64 size, std::shared_ptr<HostMem> &bufferPtr) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateWorkSpace(u64 size, DeviceMem &buffer) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetWorkSpace(u64 *workSpaceSize, u64 *workSpace) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitWorkSpace()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::FillOpParam(const HcclCMDType commType, OpParam &opParam,
                                             const uint64_t count, void *pCount, void *pDispls)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocComResource(const string &newTag, const string &algName,
                                                  const HcclCMDType commType, const OpParam &opParam, rtStream_t stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocComResourceByTiling(const std::string &algConfig, void *param)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateCommResource(const std::string &tag, rtStream_t aiCpuStream, bool isOpbaseMode,
                                                    void **commContext)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Mc2CreateAndLaunchContext(rtStream_t aiCpuStream, bool isOpbaseMode, void **commContext, const string &tag)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetAiCpuNotifyData(const std::shared_ptr<LocalNotify> &localNotify,
                                                    HcclSignalInfo &notifyInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateAndGetAiCpuNotify(std::shared_ptr<LocalNotify> &localNotify,
                                                         HcclSignalInfo &notifyInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Mc2AiCpuInitStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AiCpuKernelLaunch(const rtStream_t stm, u64 addr, const std::string &kernelName)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitKernelArgsPrepare(aclrtBinHandle binHandle, const std::string &kernelName,
        void* initTaskAddr, u32 initTaskSize, aclrtFuncHandle &funcHandle, aclrtArgsHandle &argsHandle)
    {
        return HCCL_SUCCESS;
    }
    
    HcclResult HcclCommunicator::TaskCommKernelArgsPrepare(aclrtBinHandle binHandle, const std::string &kernelName,
        void* contextAddr, u32 contextSize, void* tilingDataPtr, u32 tilingDataSize, aclrtFuncHandle &funcHandle,
        aclrtArgsHandle &argsHandle)
    {
        return HCCL_SUCCESS;
    }
    
    HcclResult HcclCommunicator::AicpuAclKernelLaunch(const rtStream_t stm, void* addr, u32 size, aclrtBinHandle binHandle,
        const std::string &kernelName, bool isInitTask, void* tilingDataPtr, u32 tilingDataSize)
    {
        return HCCL_SUCCESS;
    }
    

    HcclResult HcclCommunicator::AicpuKfcTilingDataLaunch(const OpParam &opParam, const HcclCMDType &opType,
                                                          const DeviceMem &deviceContext, const std::string &kernelName, const AicpuOpTiling opTilingInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuInitOpTilingDataBuf(const OpParam &opParam, const HcclCMDType &opType,
                                                          const std::string &kernelName, const AicpuOpTiling opTilingInfo, u64 dynamicDataSize)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuKfcTilingDataLaunchIn(const OpParam &opParam, const DeviceMem &deviceContext,
                                                            const std::string &kernelName, const AicpuOpTiling opTilingInfo, u64 opTilingDataSize, bool isCustom)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuKfcTilingDataLaunchExt(const OpParam &opParam, const HcclCMDType &opType,
                                                             const DeviceMem &deviceContext, const std::string &kernelName, const AicpuOpTiling opTilingInfo,
                                                             bool isCustom)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuUnfoldKernelLaunch(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
                                                         void *tilingDataPtr, u32 tilingDataSize, const std::string &kernelName, HcclWorkflowMode mode, const std::string &tag)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuUnfoldKernelLaunchV2(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
                                                           void *tilingDataPtr, u32 tilingDataSize, const std::string &kernelName, HcclWorkflowMode mode, const std::string &tag)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuUnfoldKernelLaunchIsCustom(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
                                                                 void *tilingDataPtr, u32 tilingDataSize, const std::string &kernelName, HcclWorkflowMode mode, const std::string &tag, bool isCustom)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitCombinOpara()
    {
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::GetCommResource(const std::string &tag, void **commContext)
    {
        return false;
    }

    bool HcclCommunicator::GetCommResource(void *&commContext)
    {
        return false;
    }

    HcclResult HcclCommunicator::GetAicpuOpStreamNotify(HcclRtStream *opStream, u8 aicpuNotifyNum, void **aicpuNotify)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetAicpuOpStreamAndNotify(HcclRtStream *opStream, u8 aicpuNotifyNum, void **aicpuNotify)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetAicpuNotifyInvaild()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReplaceCommInfoByTag(const std::string &tag, std::unique_ptr<CommInfo> &commInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateMutiStreamResFor310P(const std::string &tag, level1StreamInfo_t &streamInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateCommAndStreamRes(const std::string &tag, Stream &stream)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetComm(const std::string &tag, CommBase **comm)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetCommResource(u64 commBufferSize, void *commInPtr, void *commOutPtr, void *commExpPtr,
                                                 CommBase *comm, level1StreamInfo_t &streamInfo, Stream &stream)
    {
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::ReleaseCommContextbuffer()
    {
    }

    HcclResult HcclCommunicator::CreateDeviceCommContext(u64 size, DeviceMem &buffer) const
    {
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::Break()
    {
        return;
    }

    HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
                                                                   u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &memSize)
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::GetAllReduceScratchSize(
        const u32 count, const HcclDataType dataType, u64 &scratchSize) const
    {
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult HcclCommunicator::SetWorldGroupInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap,
        vector<RankInfo> worldRankInfoList, vector<u32> &nicRanksPort, vector<u32> &vnicRanksPort)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetTopoDesc(HcclTopoDescs *topoDescs, uint32_t topoSize)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetAivModeConfig(const bool aivMode)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetAicpuUnfoldConfig(const bool aicpuUnfold)
    {
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::SetQpQosAttr(u32 trafficClass, u32 serviceLevel)
    {
        transportManager_->SetQpQosAttr(trafficClass, serviceLevel);
    }

    HcclResult HcclCommunicator::CheckExitWaitResumeState(bool &isChangedLink)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetMemoryRange(void *baseVirPtr, size_t size, size_t alignment, uint64_t flags)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::UnsetMemoryRange(void *baseVirPtr)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ActivateCommMemory(void *virPtr, size_t size, size_t offset, void *handle, uint64_t flags)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DeactivateCommMemory(void *virPtr)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetSingleLinkInfo(std::unordered_map<u32, bool> &switchRanks, u32 remoteRankId,
                                                   ChangeLinkInfo &changeLinkInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetRemoteRankLinkInfo(std::unordered_map<u32, bool> &switchRanks,
                                                       ChangeLinkInfo &changeLinkInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ActiveStoppedLink(std::map<u32, bool> &remoteRankPortMap,
        OpCommTransport &opTransportResponse, bool isBackup)
    {
        return HCCL_SUCCESS;
    }
    
    HcclResult HcclCommunicator::PrepareLinkForSwitchNic(std::unordered_map<u32, bool> &switchRanks,
        ChangeLinkInfo &changeLinkInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ParseSwitchRanks(uint32_t nRanks, uint32_t *ranks, bool *useBackup,
                                                  std::unordered_map<u32, bool> &switchRanks)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SwitchNic(uint32_t nRanks, uint32_t *ranks, bool *useBackup,
                                           std::shared_ptr<HDCommunicate> &controlH2D, std::shared_ptr<HDCommunicate> &statusD2H)
    {
        HcclResult ret = HCCL_SUCCESS;
        return ret;
    }

    HcclResult HcclCommunicator::GetSwitchRanks(u32 *distSwitchRankList, bool *distSwitchUseBackup, u32 &distSwitchRankNum,
                                                u8 *distRemoteRankNicStatus, u32 &distNicStatusNum, bool &needCheckDefaultNic, bool &needCheckBackupNic)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::LoadCustomFile(const char *binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode,
        aclrtBinHandle& binHandle)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::LoadBinaryFromFile(const char *binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode,
                                                    aclrtBinHandle &binHandle)
    {
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::UnloadBinary(aclrtBinHandle &binHandle)
    {
        return;
    }
}
