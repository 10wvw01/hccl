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
#include "stream_utils.h"
#include "config_log.h"
#include "../nslbdp/hccl_nslbdp.h"
#include "hccl_one_sided_service.h"

using namespace std;
constexpr u32 MODULE_NUM_FOUR = 4;

namespace hccl
{
    static std::mutex g_hcomInitMutex;
    constexpr u32 MEMORY_CAPACITY = 256 * 1024;
    constexpr u32 WAIT_PREPARE_SLEEP_TIME = 5000;
    constexpr u32 SINGLE_SERVER_NUM = 1;
    constexpr u32 CONN_LIMIT = 4096;
    constexpr u32 COMM_DEV_TYPE_DIGIT_NUM = 8;
    constexpr u32 TILINGDATA_BUF_SIZE = 32 * 1024; // 单位：字节
    constexpr u32 ALLTOALL_INFO_MATRIX_SIZE = 4;
    constexpr u32 AICPU_RETRY_LINKROCE_DEFAULT = 0;
    constexpr u32 AICPU_RETRY_LINKROCE_BACKUP = 1;
    constexpr u32 SINGLE_PROCESS_MIN_PORT = 1024;
    constexpr u32 SINGLE_PROCESS_MAX_PORT = 65535;
    constexpr u32 NON_BATCH_WRITE_MAX_STREAM_NUM = 19U;
    enum TransferMemInfoIdx
    {
        TRANSFER_MEM_INFO_KEY_IDX = 0,
        TRANSFER_MEM_INFO_VALUE_IDX = 1,
        TRANSFER_MEM_INFO_RDMA_ENVELOPE_IDX = 2,
        TRANSFER_MEM_INFO_IDX_NUM = 3
    };

    enum class AicpuLocalNotifyIdx : u32 {
        // host-aicpu同步
        HOST_TO_AICPU_0 = 0,
        HOST_TO_AICPU_1 = 1,
        // 主流-kfc流同步
        MAIN_TO_KFC_0 = 2,
        MAIN_TO_KFC_1 = 3
    };

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
        mrManager_.reset(new (std::nothrow) MrManager());
        if (mrManager_ == nullptr) {
            HCCL_ERROR("new MrManager failed!");
        }
        zeroCopyAclGraph_.reset(new (std::nothrow) ZeroCopyAclGraph());
        if (zeroCopyAclGraph_ == nullptr)
        {
            HCCL_ERROR("new ZeroCopyAclGraph failed!");
        }
    }

    HcclCommunicator::~HcclCommunicator()
    {
        HCCL_DEBUG("Enter ~HcclCommunicator.");

        DeinitZeroCopyMemoryAgent(true);
        (void)DestroyAicpuComm();
        (void)UnRegisterBackGroundThread();

        UnRegisterToHeartBeat();
        AlgWrap::GetInstance().UnregisterAlgCallBack(identifier_);
        DetectConnectionAnomalies::GetInstance(deviceLogicId_).Deinit();

        if (zeroCopyAclGraph_ != nullptr) {
            zeroCopyAclGraph_ = nullptr;
        }

        if (implAlg_ != nullptr) {
            implAlg_ = nullptr;
        }

        for (auto &res : resMap_) {
            DestroyAlgResource(res.second);
        }

        if (opRetryManager_ != nullptr) {
            OpRetryManager::DeleteLinkInfoByIdentifier(deviceLogicId_, identifier_);
            opRetryManager_->UnRegisterOpRetryManager(identifier_);
            opRetryManager_ = nullptr;
        }

        resMap_.clear();
        deviceResOrigMem_.clear();
        hostResMap_.clear();
        tagCommInfo_.clear();
        tagWorkSpaceMem_.clear();
        tagStreamInfo_.clear();

        if (opRetryStreamPtr_ != nullptr) {
            opRetryStreamPtr_->clear();
            opRetryStreamPtr_ = nullptr;
        }

        (void)UnRegistTaskExceptionHandler();
        kfcControlTransferH2D_ = nullptr;
        kfcStatusTransferD2H_ = nullptr;
        customControlTransferH2D_ = nullptr;
        customStatusTransferD2H_ = nullptr;

        oneSideService_ = nullptr;
        if (isOneSidedServiceNetDevCtxInited) {
            DeInitOneSidedServiceNetDevCtx();
        }

        MrManagerDeInit();

        /* 网络资源销毁 */
        DestroyNetworkResources();
        notifyPool_ = nullptr;
        /* driver关联资源释放 */
        if (drvInit_){
            if (DisablePreResource() != HCCL_SUCCESS) {
                HCCL_WARNING("driver resource is not released successfully");
            }
        }

        if (isExecuteProfilingInit_) {
            (void)DeinitProfiling();
        }

        if (OpExeCounter::GetInstance(deviceLogicId_).DeInitCounter() != HCCL_SUCCESS) {
            HCCL_WARNING("op exec counter resource free failed");
        }

        /* 销毁当前trace句柄 */
        if (opBaseAtraceInfo_ != nullptr) {
            opBaseAtraceInfo_->DeInit();
            opBaseAtraceInfo_ = nullptr;
        }

        ReleaseWorkSpacebuffer();
        ReleaseCommContextbuffer();

        for (u32 i = 0; i < AICPU_LOCAL_NOTIFY_SIZE; i++) {
            if (localAiCpuOpNotify_[i]) {
                HcclResult ret = localAiCpuOpNotify_[i]->Destroy();
                localAiCpuOpNotify_[i] = nullptr;
                if (ret != RT_ERROR_NONE) {
                    HCCL_ERROR("[Destroy][AicpuNotify]errNo[0x%016llx] rt notify destroy fail, "
                               "aicpuOpNotify[%u] return[%d].",
                               HCCL_ERROR_CODE(HCCL_E_RUNTIME), i, ret);
                }
            }
        }

        while (!aiCpuNoIpcEvnet_.empty()) {
            rtEvent_t eventInfo = aiCpuNoIpcEvnet_.back();
            HcclResult ret = hrtEventDestroy(eventInfo);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[Destroy][AicpuNoIpcEvnet]errNo[0x%016llx] rt event destroy fail, "
                           "return[%d].",
                           HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
            }
            aiCpuNoIpcEvnet_.pop_back();
        }

        UnloadCustomKernel();
        if (dispatcher_ != nullptr) {
            HcclDispatcherDestroy(dispatcher_);
            dispatcher_ = nullptr;
        }
        if (vDispatcher_ != nullptr) {
            HcclDispatcherDestroy(vDispatcher_);
            vDispatcher_ = nullptr;
        }
        HCCL_DEBUG("~HcclCommunicator success.");
    }

    HcclResult HcclCommunicator::Init(HcclCommParams &params, const RankTable_t &rankTable)
    {
        CHK_RET(InitCommParams(params));
        CHK_RET(attrCollector_.Init(params, rankTable));
        CHK_RET(InitRankInfo(rankTable));
        CHK_RET(InitNetResource(rankTable));
        CHK_RET(InitDebug());
        CHK_RET(InitNotifyManager());
        CHK_RET(InitStreamManager());
        CHK_RET(InitTransportManager());
        CHK_RET(InitMemoryManager());
        CHK_RET(InitCombinOpara());
        CHK_RET(RegisterRanksToDca());
        /*--------------加锁区--------------*/
        std::unique_lock<std::mutex> lock(g_hcomInitMutex);
        CHK_RET(RegistTaskExceptionHandler());

        attrCollector_.GenCollectiveId(params, rankTable);
        collectiveId_ = attrCollector_.GetCollectiveId();

        // 初始化参数(需要放置在ranktable解析之后)
        HcclResult ret = InitPara();
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[HcclCommunicator][Init]errNo[0x%016llx] collectiveid[%s] parameter initialization failed",
                               HCCL_ERROR_CODE(ret), params.id.internal),
                    ret);
        lock.unlock();
        /*--------------加锁区--------------*/
        if (deviceType_ == DevType::DEV_TYPE_910B || deviceType_ == DevType::DEV_TYPE_910_93){
            CHK_RET(RegisterKernel(deviceType_));
        }
        CHK_RET(LoadCustomKernel());
        CHK_RET(InitHDCommunicate());
        CHK_RET(InitOpRetry());
        CHK_RET(InitOpResPara());

        CHK_RET(InitOneSidedService(rankTable));

        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Init(HcclCommParams &params, const std::vector<RankInfo> &rankList,
                                      WorldGroupInfo &groupCommonData)
    {
        CHK_RET(InitCommParams(params));
        CHK_RET(attrCollector_.Init(params, rankList, groupCommonData));
        CHK_RET(InitRankInfoSubGroup(rankList, groupCommonData));
        CHK_RET(InitDebugSubGroup());
        CHK_RET(InitNotifyManager());
        CHK_RET(InitDispatcher());
        CHK_RET(InitStreamManager());
        CHK_RET(InitRaResource());
        CHK_RET(InitTransportManager());
        CHK_RET(InitMemoryManagerSubGroup());
        CHK_RET(InitHcclAlg());
        CHK_RET(LoadCustomKernel());
        CHK_RET(InitHDCommunicate());
        CHK_RET(InitOpRetry());
        CHK_RET(InitOpResPara());
        CHK_RET(RegisterRanksToDca());
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitOneSidedService(const RankTable_t &rankTable)
    {
        EXECEPTION_CATCH((oneSideService_ = std::make_unique<HcclOneSidedService>(socketManager_, notifyPool_)),
                         return HCCL_E_INTERNAL);
        hcclRankLinkInfo_.userRank = userRank_;
        hcclRankLinkInfo_.devicePhyId = devicePhyId_;

        if (devIpAddr_.empty()) {
            HCCL_ERROR("[%s] device ip is invalid, please set device ip first.", __func__);
            return HCCL_E_NOT_FOUND;
        }
        hcclRankLinkInfo_.ip = devIpAddr_[0];
        if (nicRanksPort_.size() <= userRank_) {
            HCCL_ERROR("[%s] userRank_[%u] port is invalid, please set port first", __func__, userRank_);
            return HCCL_E_NOT_FOUND;
        }
        hcclRankLinkInfo_.port = nicRanksPort_[userRank_];
        hcclRankLinkInfo_.socketsPerLink = 1;
        HCCL_DEBUG("[%s]hcclRankLinkInfo_ userRank[%u], devicePhyId[%u], ip[%s], port[%u]", __func__,
                   hcclRankLinkInfo_.userRank, hcclRankLinkInfo_.devicePhyId, hcclRankLinkInfo_.ip.GetReadableIP(),
                   hcclRankLinkInfo_.port);
        CHK_RET(oneSideService_->Config(dispatcher_, hcclRankLinkInfo_, &rankTable));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitOneSidedServiceNetDevCtx(u32 remoteRankId)
    {
        if (nicDeployment_ != NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            // 单边操作当前只支持Device网卡，不支持host
            HCCL_ERROR("[%s]nicDeployment_[%d], userRankSize_[%u], do not support oneSidedService.",
                       __func__, nicDeployment_, userRankSize_);
            return HCCL_E_INTERNAL;
        }

        std::string localServerId = serverId_;
        std::string localSuperPodId = superPodId_;
        std::string remoteServerId = rankInfoList_.at(remoteRankId).serverId;
        std::string remoteSuperPodId = rankInfoList_.at(remoteRankId).superPodId;
        u32 intraRoceSwitch = GetExternalInputIntraRoceSwitch();
        bool useRdma = false;
        if (intraRoceSwitch ||
            (!useSuperPodMode_ && localServerId != remoteServerId) ||
            (localSuperPodId != remoteSuperPodId)) {
            // 1. 初始化网口
            CHK_RET(InitNic());
            isOneSidedServiceNicInited = true;

            // 2. 单边操作SetNetDevCtx, RDMA
            if (netDevCtxMap_.find(devIpAddr_[0]) == netDevCtxMap_.end()) {
                HCCL_ERROR("[%s] nicDeployment_[%d], device nic init fail, please check", __func__, nicDeployment_);
                return HCCL_E_NOT_FOUND;
            }
            useRdma = true;
            oneSideService_->SetNetDevCtx(netDevCtxMap_[devIpAddr_[0]], useRdma);
            HCCL_INFO("[%s]init device Nic for oneSidedService success.", __func__);
        }else {
            // 单边操作SetNetDevCtx, IPC
            oneSideService_->SetNetDevCtx(netDevCtxMap_[localVnicIp_], useRdma);
            HCCL_INFO("[%s]init vNic for oneSidedService success.", __func__);
        }
        isOneSidedServiceNetDevCtxInited = true;
        HCCL_DEBUG("[%s]nicDeployment_[%d], intraRoceSwitch[%u]", __func__, nicDeployment_, intraRoceSwitch);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DeInitOneSidedServiceNetDevCtx()
    {
        if (nicDeployment_ != NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            // 单边操作当前只支持Device网卡，不支持host
            HCCL_ERROR("[%s]nicDeployment_[%d], userRankSize_[%u], do not support oneSidedService.",
                       __func__, nicDeployment_, userRankSize_);
            return HCCL_E_INTERNAL;
        }
        if (isOneSidedServiceNicStartListen_) {
            socketManager_->DestroySockets();
            u32 port = GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
            CHK_RET(socketManager_->ServerDeInit(onesidedServiceNicIpAddr_, port));
            isOneSidedServiceNicStartListen_ = false;
            HCCL_INFO("[HcclCommunicator][%s] DeInit socket server success.tag[%s].", __func__, identifier_.c_str());
        }
        u32 intraRoceSwitch = GetExternalInputIntraRoceSwitch();
        if (isOneSidedServiceNicInited) {
            // 1. close sockets
            if (raResourceInit_) {
                socketManager_->DestroySockets();
            }
            // 2. 去初始化网口
            CHK_RET(DeinitNic());
            isOneSidedServiceNicInited = false;
            HCCL_INFO("[%s]Deinit device Nic for oneSidedService success.", __func__);
        }
        isOneSidedServiceNetDevCtxInited = false;
        HCCL_DEBUG("[%s]nicDeployment_[%d], intraRoceSwitch[%u]", __func__, nicDeployment_, intraRoceSwitch);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetOneSidedService(IHcclOneSidedService **service)
    {
        *service = oneSideService_.get();
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::OneSidedServiceStartListen(NicType nicType, HcclNetDevCtx netDevCtx)
    {
        HCCL_INFO("[HcclCommunicator][%s] Start prepare netDevCtx.", __func__);
        u32 port = GetLocalNicPort(nicType);
        CHK_RET(socketManager_->ServerInit(netDevCtx, port));
        if (nicType == NicType::DEVICE_NIC_TYPE) {
            CHK_RET(HcclNetDevGetLocalIp(netDevCtx, onesidedServiceNicIpAddr_));
            isOneSidedServiceNicStartListen_ = true;
        }
        isOneSidedServiceNetDevCtxInited = true;
        HCCL_INFO("[HcclCommunicator][%s] netDevCtx[%p] port[%u] server init success.", __func__, netDevCtx, port);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetOneSidedServiceDevIpAndPort(NicType nicType, HcclIpAddress &ipAddress, u32& port)
    {
        if (nicDeployment_ != NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            // 单边操作当前只支持Device网卡，不支持host
            HCCL_ERROR("[%s]nicDeployment_[%d], userRankSize_[%u], do not support oneSidedService.",
                       __func__, nicDeployment_, userRankSize_);
            return HCCL_E_INTERNAL;
        }
        port = GetLocalNicPort(nicType);
        if (nicType == NicType::VNIC_TYPE) {
            ipAddress = localVnicIp_;
            HCCL_INFO("[GetOneSidedServiceDevIpAddr] vnic ipAddress[%s] get success.", ipAddress.GetReadableAddress());
            return HCCL_SUCCESS;
        }else if (nicType == NicType::DEVICE_NIC_TYPE) {
            u32 nicNum = devIpAddr_.size();
            for (u32 i = 0; i < nicNum; i++) {
                if (devIpAddr_[i].IsInvalid()) {
                    HCCL_INFO("[GetOneSidedServiceDevIpAddr]nic num[%u] deviceip is invalid, total nicNum[%u]", i, nicNum);
                    continue;
                }
                ipAddress = devIpAddr_[i];
                HCCL_INFO("[GetOneSidedServiceDevIpAddr] nic ipAddress[%s] get success.", ipAddress.GetReadableAddress());
                return HCCL_SUCCESS;
            }
        }
        HCCL_ERROR("[HcclCommunicator][%s] ipAddress get fail. tag[%s]", __func__, identifier_.c_str());
        return HCCL_E_NOT_FOUND;
    }

    HcclResult HcclCommunicator::DeinitOneSidedService()
    {
        if (oneSideService_ != nullptr) {
            CHK_RET(oneSideService_->DeInit());
        }
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::IsSupportZeroCopy(const OpParam &opParam)
    {
        // 只支持aicpu展开、非重执行、单算子模式、910_93芯片
        CHK_PRT_RET(!opParam.aicpuUnfoldMode,
                    HCCL_INFO("[%s] aicpuUnfold:%d not support zero copy", __func__, opParam.aicpuUnfoldMode), false);
        CHK_PRT_RET(retryEnable_,
                    HCCL_INFO("[%s] retryEnable:%d not support zero copy", __func__, retryEnable_), false);
        CHK_PRT_RET(GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE,
                    HCCL_INFO("[%s] workflowMode:%d not support zero copy", __func__, GetWorkflowMode()), false);
        CHK_PRT_RET(deviceType_ != DevType::DEV_TYPE_910_93,
                    HCCL_INFO("[%s] deviceType:%d not support zero copy", __func__, deviceType_), false);

        // 判断拓扑逻辑是否支持zero copy
        // 每个节点只有一张卡或节点间非对称场景不支持零拷贝
        CHK_PRT_RET(deviceNumPerAggregation_ == 1 || multiModuleDiffDeviceNumMode_,
                    HCCL_INFO("[%s] deviceNumPerAggregation[%u], multiModuleDiffDeviceNumMode_[%d] not support zero copy",
                              __func__, deviceNumPerAggregation_, multiModuleDiffDeviceNumMode_),
                    false);

        // 判断输入输出地址是否都是支持零Copy特性的
        CHK_PRT_RET(!ZeroCopyMemoryAgent::IsActivateCommMemoryAddr(opParam.outputPtr, opParam.outputSize),
                    HCCL_INFO("[%s] input[%p] size[%llu] is not support zero copy", __func__, opParam.inputPtr, opParam.inputSize), false);
        CHK_PRT_RET(!ZeroCopyMemoryAgent::IsActivateCommMemoryAddr(opParam.inputPtr, opParam.inputSize),
                    HCCL_INFO("[%s] output[%p] size[%llu] is not support zero copy", __func__, opParam.outputPtr, opParam.outputSize), false);

        return true;
    }

    HcclResult HcclCommunicator::PrepareZeroCopy(const std::string &algName, const AlgDesc &algDesc, OpParam &opParam)
    {
        if (!algDesc.isZeroCopy) {
            HCCL_INFO("[HcclCommunicator][PrepareZeroCopy] algName[%s] not support zerocopy.", algName.c_str());
            return HCCL_SUCCESS;
        }

        // 如果自己侧的共享内存没有申请，那么进行申请，并设置给transportManager，后续p2p建链时进行交换
        if (zeroCopyLocalBuffer_.ptr() == nullptr) {
            zeroCopyLocalBuffer_ = DeviceMem::alloc(ZERO_COPY_IPC_BUFFER_LENGTH);
            CHK_PRT_RET(!zeroCopyLocalBuffer_, HCCL_ERROR("[HcclCommunicator][PrepareZeroCopy]Create buffer size[%llu] fail,", ZERO_COPY_IPC_BUFFER_LENGTH), HCCL_E_PTR);
            CHK_RET(hrtMemSet(zeroCopyLocalBuffer_.ptr(), zeroCopyLocalBuffer_.size(), zeroCopyLocalBuffer_.size()));
            zeroCopyIpcPtrs_[userRank_ % deviceNumPerAggregation_] = zeroCopyLocalBuffer_.ptr();

            HCCL_RUN_INFO("[HCCL_TRACE][PrepareZeroCopy]Create ZeroCopy buffer success. buffer ptr[%p] size[%llu]",
                          zeroCopyLocalBuffer_.ptr(), zeroCopyLocalBuffer_.size());
        }
        opParam.isZeroCopy = true;
        HCCL_INFO("[HcclCommunicator][PrepareZeroCopy] success to use zero copy feature");
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::UpdateZeroCopy(const OpParam &opParam, const AlgResourceResponse &resp)
    {
        if (!opParam.isZeroCopy) {
            return HCCL_SUCCESS;
        }

        // 遍历所有transport，找出里面的p2p链路对应的对端地址
        for (auto &singleSubCommTransport : resp.opTransportResponse[COMM_LEVEL0]) {
            for (u64 i = 0; i < singleSubCommTransport.links.size(); ++i) {
                LINK link = singleSubCommTransport.links[i];
                if (link == nullptr || !singleSubCommTransport.transportRequests[i].isValid) {
                    // 无效或者不支持的链路
                    continue;
                }

                // 在使能零拷贝场景，我们使用控制面内存做OpenIpc交换，因此这里取出input即可
                u32 remoteRank = link->GetRemoteRank();

                void *remotePtr = nullptr;
                CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &remotePtr));
                CHK_PRT_RET(remotePtr == nullptr,
                            HCCL_ERROR("[BuildZeroCopyParam] invalid remotePtr[%p]", remotePtr), HCCL_E_PARA);
                CHK_PRT_RET(zeroCopyIpcPtrs_[remoteRank % deviceNumPerAggregation_] != nullptr && zeroCopyIpcPtrs_[remoteRank % deviceNumPerAggregation_] != remotePtr,
                            HCCL_ERROR("[BuildZeroCopyParam] zeroCopyIpcPtrs_[%u] is [%p] not equal to %p", remoteRank, zeroCopyIpcPtrs_[remoteRank % deviceNumPerAggregation_],
                                       remotePtr),
                            HCCL_E_PARA);

                zeroCopyIpcPtrs_[remoteRank % deviceNumPerAggregation_] = remotePtr;
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildZeroCopyParam()
    {
        // 不支持ZeroCopy
        if (zeroCopyLocalBuffer_.ptr() == nullptr) {
            return HCCL_SUCCESS;
        }

        for (u32 i = 0; i < MAX_MODULE_DEVICE_NUM; ++i) {
            opResPara_.zeroCopyIpcPtrs[i] = reinterpret_cast<u64>(zeroCopyIpcPtrs_[i]);
        }

        for (u32 i = 0; i < rankInfoList_.size(); ++i) {
            opResPara_.zeroCopyDevicePhyId[i % deviceNumPerAggregation_] = rankInfoList_[i].devicePhyId;
        }

        CHK_RET(ZeroCopyMemoryAgent::GetRingBufferAddr(opResPara_.zeroCopyRingBuffer,
                                                       opResPara_.zeroCopyHeadPtr, opResPara_.zeroCopyTailPtr));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitCommParams(HcclCommParams &params)
    {
        commHandle_ = params.commHandle;
        userRank_ = params.rank;
        realUserRank_ = params.userRank;
        userRankSize_ = params.totalRanks;
        deviceLogicId_ = params.logicDevId;
        profilingOption_ = params.profilingOption;
        profilingInitiated_ = params.profilingInitiated;
        deviceType_ = params.deviceType;
        commWorkMode_ = params.commWorkMode;
        hcomGroupNicInit_ = params.hcomGroupNicInit;
        identifier_ = params.identifier;
        collectiveId_ = params.id.internal;
        ranktableCrc_ = params.ranktableCrc;
        commConnections_ = params.commConnections;
        commPortConfig_ = params.commPortConfig;

        HCCL_DEBUG(
            " userRank_: %u realUserRank_: %u userRankSize_: %u deviceLogicId_: %u deviceType_: %u commWorkMode_: %u.",
            userRank_,
            realUserRank_,
            userRankSize_,
            deviceLogicId_,
            deviceType_,
            commWorkMode_);

        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::Is310PDuoCard()
    {
        return (Is310P3Common(isHaveCpuRank_, deviceType_) &&
                (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == userRankSize_));
    }

    // 910B A+X 在RDMA未启用情况下，两模块间的device数目需要一致且两模块中使用的卡都在同一平面上
    HcclResult HcclCommunicator::CheckSingleServerComm(const std::vector<RankInfo_t> &rankList) const
    {
        if (serverNum_ == 1 && moduleNum_ == HCCL_MODULE_NUM_TWO && GetExternalInputIntraRoceSwitch() == 0) {
            std::vector<u32> devIdList0;
            std::vector<u32> devIdList1;
            for (RankInfo_t rankInfo : rankList){
                if (rankInfo.deviceInfo.devicePhyId == HOST_DEVICE_ID) {
                    HCCL_ERROR("[Check][SingleServerComm]not support cpu rank");
                    return HCCL_E_NOT_SUPPORT;
                }
                if (rankInfo.deviceInfo.devicePhyId < DEVICE_PER_MODULE) {
                    devIdList0.push_back(rankInfo.deviceInfo.devicePhyId);
                } else {
                    devIdList1.push_back(rankInfo.deviceInfo.devicePhyId);
                }
            }
            std::sort(devIdList0.begin(), devIdList0.end());
            std::sort(devIdList1.begin(), devIdList1.end());

            if (devIdList0.size() != devIdList1.size()) {
                char errorLogBuffer[LOG_TMPBUF_SIZE];
                s32 ret = snprintf_s(errorLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
                                     "errNo[0x%016llx]. In A+X serverNum_[%d], moduleNum_[%d] case: "
                                     "deviceNum in module0:[%d] not equal to deviceNum in module1:[%d], "
                                     "you can export HCCL_INTRA_ROCE_ENABLE=1 to enable this scenario.",
                                     HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), serverNum_, moduleNum_, devIdList0.size(), devIdList1.size());
                CHK_PRT_CONT(ret == -1, HCCL_ERROR("Failed to build log info"));
                HCCL_ERROR("[Check][SingleServerComm]%s", errorLogBuffer);
                RPT_INPUT_ERR(true, "EI0010", std::vector<std::string>({"reason"}),
                              std::vector<std::string>({std::string(errorLogBuffer)}));
                return HCCL_E_NOT_SUPPORT;
            }
            for (size_t i = 0; i < devIdList0.size(); i++) {
                if (devIdList0[i] % DEVICE_PER_MODULE != devIdList1[i] % DEVICE_PER_MODULE) {
                    char errorLogBuffer[LOG_TMPBUF_SIZE];
                    s32 ret = snprintf_s(errorLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
                                         "errNo[0x%016llx]. In A+X serverNum_[%d], moduleNum_[%d] case: "
                                         "deviceId[%d] in module0 and deviceId[%d] in module1 are not on the same plane, "
                                         "you can export HCCL_INTRA_ROCE_ENABLE=1 to enable this scenario.",
                                         HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), serverNum_, moduleNum_, devIdList0[i], devIdList1[i]);
                    CHK_PRT_CONT(ret == -1, HCCL_ERROR("Failed to build log info"));
                    HCCL_ERROR("[Check][SingleServerComm]%s", errorLogBuffer);
                    RPT_INPUT_ERR(true, "EI0010", std::vector<std::string>({"reason"}),
                                  std::vector<std::string>({std::string(errorLogBuffer)}));
                    return HCCL_E_NOT_SUPPORT;
                }
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckDataType(const HcclDataType dataType, bool needReduce)
    {
        const vector<string> infoTitle({"ccl_op", "parameter", "value", "tips"});
        if (needReduce) {
            if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
                if ((dataType == HCCL_DATA_TYPE_INT64) || (dataType == HCCL_DATA_TYPE_BFP16)) {
                    RPT_INPUT_ERR(true, "EI0003", infoTitle, vector<string>({"CheckDataType", "dataType", GetDataTypeEnumStr(dataType), "please check dataType"}));
                    HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                               HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                               GetSupportDataType(needReduce).c_str());
                    return HCCL_E_NOT_SUPPORT;
                }
            }

            if ((dataType == HCCL_DATA_TYPE_UINT64) ||
                (dataType == HCCL_DATA_TYPE_UINT8) || (dataType == HCCL_DATA_TYPE_UINT16) ||
                (dataType == HCCL_DATA_TYPE_UINT32) || (dataType == HCCL_DATA_TYPE_FP64) ||
                (dataType == HCCL_DATA_TYPE_RESERVED)) {
                RPT_INPUT_ERR(true, "EI0003", infoTitle, vector<string>({"CheckDataType", "dataType", GetDataTypeEnumStr(dataType), "please check dataType"}));
                HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                           HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                           GetSupportDataType(needReduce).c_str());
                return HCCL_E_NOT_SUPPORT;
            }
        } else {
            if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8) ||
                (Is310P3Common(isHaveCpuRank_, deviceType_) && dataType == HCCL_DATA_TYPE_BFP16)) {
                RPT_INPUT_ERR(true, "EI0003", infoTitle, vector<string>({"CheckDataType", "dataType", GetDataTypeEnumStr(dataType), "please check dataType"}));
                HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                           HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                           GetSupportDataType(needReduce).c_str());
                return HCCL_E_NOT_SUPPORT;
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitZeroCopyMemoryAgent()
    {
        CHK_PRT_RET(zeroCopyMemoryAgent_ != nullptr,
                    HCCL_ERROR("[HcclCommunicator][InitZeroCopyMemoryAgent] ipc memory agent has init"), HCCL_E_INTERNAL);

        // 获取节点内的ranktable
        std::vector<std::vector<std::vector<RankInfo>>> commPlaneVector;
        CHK_SMART_PTR_NULL(implAlg_);
        implAlg_->GetCommPlaneVector(commPlaneVector);
        rankInfoListIntraServer_ = commPlaneVector[COMM_LEVEL0][COMM_INDEX_0];
        zeroCopyMemoryAgent_.reset(static_cast<ZeroCopyMemoryAgent *>(new (std::nothrow) ZeroCopyMemoryAgent(socketManager_, devicePhyId_,
                                                                                                             deviceLogicId_, localVnicIp_, rankInfoListIntraServer_, userRank_, useSuperPodMode_, identifier_)));
        CHK_PTR_NULL(zeroCopyMemoryAgent_);
        CHK_RET(zeroCopyMemoryAgent_->Init());
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DeinitZeroCopyMemoryAgent(bool inDestructor)
    {
        if (zeroCopyMemoryAgent_ != nullptr) {
            if (!inDestructor) {
                // 析构函数释放场景不做barrier close
                CHK_RET(zeroCopyMemoryAgent_->BarrierClose());
            }
            CHK_RET(zeroCopyMemoryAgent_->DeInit());
            zeroCopyMemoryAgent_ = nullptr;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ClearResMap(const std::string &tag, bool &findTag)
    {
        auto resIter = resMap_.find(tag);
        if (resIter != resMap_.end()) {
            findTag = true;
            DestroyAlgResource(resIter->second);
            CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(resIter->second.slaveStreams));
            resMap_.erase(resIter);
            HCCL_INFO("[%s] clear resMap[%s]", __func__, tag.c_str());
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ClearOpResource(const std::string &tag)
    {
        bool findTag = false;
        CHK_RET(ClearResMap(tag, findTag));
        CHK_RET(ClearResMap(tag + "_host", findTag));
        CHK_RET(ClearResMap(tag + "_device", findTag));
        if (!findTag) {
            HCCL_WARNING("[%s] not find tag[%s] in resMap", __func__, tag.c_str());
        }

        tagCommInfo_.erase(tag);
        // stream解绑定
        auto iterStream = tagStreamInfo_.find(tag);
        if (iterStream != tagStreamInfo_.end()) {
            CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(iterStream->second.ringStreams));
        }
        tagStreamInfo_.erase(tag);
        if (opRetryStreamPtr_ != nullptr) {
            opRetryStreamPtr_->erase(tag);
        }
        if (implAlg_ != nullptr) {
            CHK_RET(implAlg_->ClearOpResource(tag));
        }
        DestroyWorkspaceResource(tag);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
                                                        const HcomCollOpInfo &opInfo)
    {
        return workSpaceRes_->CreateOpBasedResources(opType, tag, opInfo);
    }

    HcclResult HcclCommunicator::CreateRemoteOpBasedResources(u64 memSize, const std::string &tag)
    {
        return workSpaceRes_->CreateRemoteOpBasedResources(memSize, tag);
    }

    HcclResult HcclCommunicator::DestroyRemoteOpBasedMem(const std::string &tag)
    {
        return workSpaceRes_->DestroyRemoteOpBasedMem(tag);
    }

    bool HcclCommunicator::IsAtomicInit()
    {
        if (!initializedFlag_.test_and_set()) {
            initializedFlag_.clear();
            return false;
        }
        return true;
    }

    bool HcclCommunicator::IsNeedNicInit()
    {
        return ((nicInitialized_ == 0) && (!hcomGroupNicInit_) && (userRankSize_ > 1) && !isSingleMeshAggregation_ &&
                (superPodNum_ > 1 || !isUsedInterHccsMode_));
    }

    HcclResult HcclCommunicator::GetBandWidthPerNPU(u32 level, float &bandWidth)
    {
        return hccl::GetBandWidthPerNPU(level, userRankSize_, deviceNumPerAggregation_, bandWidth);
    }

    HcclResult HcclCommunicator::GetDeviceNumPerAggregation(u32 &deviceNumPerAggregation)
    {
        deviceNumPerAggregation = deviceNumPerAggregation_;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitHccp()
    {
        /* NSLB 填充  */
        if (hcclNslbDp::GetInstance().getHccpInitFlag() == true) {
            HCCL_INFO("hcclNslbDp getHccpInitFlag is init");
            return HCCL_SUCCESS;
        }

        u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
        nslb_inithccp_info nslb_hccp;
        nslb_hccp.version = NSLBDP_HCCP_VERSION;
        nslb_hccp.phy_id = devicePhyID;
        nslb_hccp.nic_posion = NSLBDP_HCCP_NICPOSION;

        u32 nslb_buffersize = 0;
        void *tlv_handle;
        HCCL_INFO("H2DTlvInit HCCL nslb InitHccp version:[%u]-phy_id:[%u]-nic_posion:[%u] .",
                  nslb_hccp.version, nslb_hccp.phy_id, nslb_hccp.nic_posion);

        HcclResult ret = H2DTlvInit(reinterpret_cast<tlv_init_info *>(&nslb_hccp), MODULE_TYPE_NSLB, &nslb_buffersize, &tlv_handle);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        if (nslb_buffersize == 0) {
            HCCL_INFO("HCCL InitHccp nslb_hccp .nslb_buffersize:[%u] failed.", nslb_buffersize);
            return HCCL_E_NOT_SUPPORT;
        }

        hcclNslbDp::GetInstance().setHccpInfo(nslb_buffersize, tlv_handle);
        if (tlv_handle == nullptr) {
            HCCL_INFO("HCCL ndlbdp  InitHccp is null.");
            return HCCL_E_NOT_SUPPORT;
        }

        HCCL_INFO("entry H2DTlvInit end");
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckReduceDataType(const HcclDataType dataType, const HcclReduceOp op)
    {
        if ((deviceType_ == DevType::DEV_TYPE_910B) || (deviceType_ == DevType::DEV_TYPE_910_93)) {
            if ((op == HCCL_REDUCE_PROD) &&
                ((dataType == HCCL_DATA_TYPE_INT16) || (dataType == HCCL_DATA_TYPE_BFP16))) {
                RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
                              std::vector<std::string>({"CheckReduceDataType",
                                                        "dataType",
                                                        GetDataTypeEnumStr(dataType),
                                                        "please check dataType when optype is prod"}));
                HCCL_ERROR(
                    "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s] and data "
                    "type[%s] for Op[%s]",
                    HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_,
                    GetDataTypeEnumStr(HCCL_DATA_TYPE_BFP16).c_str(),
                    GetDataTypeEnumStr(HCCL_DATA_TYPE_INT16).c_str(),
                    GetReduceOpEnumStr(op).c_str());
                return HCCL_E_NOT_SUPPORT;
            }
        } else if (deviceType_ == DevType::DEV_TYPE_910) {
            if (dataType == HCCL_DATA_TYPE_INT16) {
                RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
                              std::vector<std::string>({"CheckReduceDataType",
                                                        "dataType",
                                                        GetDataTypeEnumStr(dataType),
                                                        "please check the data type when the device type is 910."}));
                HCCL_ERROR(
                    "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s]",
                    HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_,
                    GetDataTypeEnumStr(dataType).c_str());
                return HCCL_E_NOT_SUPPORT;
            }
        } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
            if (dataType == HcclDataType::HCCL_DATA_TYPE_INT16 && op != HcclReduceOp::HCCL_REDUCE_SUM) {
                RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
                              std::vector<std::string>({"CheckReduceDataType",
                                                        "op",
                                                        GetReduceOpEnumStr(op),
                                                        "please check operation type when the data type is int16."}));
                HCCL_ERROR(
                    "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s] for Op[%s]",
                    HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_,
                    GetDataTypeEnumStr(HcclDataType::HCCL_DATA_TYPE_INT16).c_str(),
                    GetReduceOpEnumStr(op).c_str());
                return HCCL_E_NOT_SUPPORT;
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetAlgType(AlgType &algType, HcclCMDType opType)
    {
        CHK_SMART_PTR_NULL(implAlg_);
        return implAlg_->GetAlgType(algType, opType);
    }

    HcclResult HcclCommunicator::GetCommParams(HcclCommParams &params)
    {
        params.commHandle = commHandle_;
        params.rank = userRank_;
        params.userRank = realUserRank_;
        params.totalRanks = userRankSize_;
        params.logicDevId = deviceLogicId_;
        params.deviceType = deviceType_;
        params.hcomGroupNicInit = hcomGroupNicInit_;
        params.identifier = identifier_;
        params.ranktableCrc = ranktableCrc_;
        params.commConnections = commConnections_;
        params.commPortConfig.devPortSwitchOn = commPortConfig_.devPortSwitchOn;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetCommRankTable(RankTable_t &rankTable)
    {
        for (auto &server : servRankInfo_) {
            for (auto &rank : server.second) {
                rankTable.rankList.emplace_back(rank);
            }
        }
        rankTable.serverNum = serverNum_;
        rankTable.superPodNum = superPodNum_;
        rankTable.nicDeploy = nicDeployment_;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitPara()
    {
        // 检查当前user_rank 对应的devid和rt查到的一致
        CHK_RET(attrCollector_.CheckLocalRankInfo());
        CHK_RET(attrCollector_.CalAndSetMeshAggRankSize());
        meshAggregationRankSize_ = attrCollector_.GetMeshAggregationRankSize();

        CHK_RET(InitProfiler());

        CHK_RET(InitDispatcher());

        // 初始化计数任务
        CHK_RET(OpExeCounter::GetInstance(deviceLogicId_).InitCounter());

        notifyPool_.reset(new (std::nothrow) NotifyPool());
        CHK_SMART_PTR_NULL(notifyPool_);
        CHK_RET(notifyPool_->Init(devicePhyId_));

        callbackTask_.reset(new (std::nothrow) HcclCallbackTask(devicePhyId_, deviceLogicId_,
                                                                dispatcher_, nicDeployment_));
        CHK_SMART_PTR_NULL(callbackTask_);

        workSpaceRes_.reset(new (std::nothrow)
                                WorkspaceResource(devicePhyId_, deviceLogicId_, &cclBufferManager_));
        CHK_SMART_PTR_NULL(workSpaceRes_);

        HcclTopoAttr topoAttr{};
        attrCollector_.GetTopoAttr(topoAttr);

        HcclAlgoAttr algoAttr{};
        attrCollector_.GetAlgoAttr(algoAttr);

        implAlg_.reset(new (std::nothrow) HcclAlg(cclBufferManager_, dispatcher_, vDispatcher_));
        CHK_SMART_PTR_NULL(implAlg_);
        CHK_RET(implAlg_->Init(static_cast<const void *>(&transportResInfo_), sizeof(transportResInfo_),
                               workSpaceRes_, notifyPool_, netDevCtxMap_, queueNotifyManager_,
                               algoAttr, topoAttr, false));
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::IsStandardCard()
    {
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            HCCL_INFO("The current device just support this StandardCard case.");
            return true;
        }

        return ((pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == 0) &&
                (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)].size() == 0) &&
                (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::SIO_TYPE)].size() == 0));
    }

    HcclResult HcclCommunicator::InitOpRetry()
    {
        EXECEPTION_CATCH((opRetryStreamPtr_ = std::make_shared<HcclOpStreamRes>()), return HCCL_E_PTR);
        if (retryEnable_) {
            opRetryManager_.reset(new (std::nothrow) OpRetryManager());
            CHK_SMART_PTR_NULL(opRetryManager_);
            HcclIpAddress hostIp = !rankInfoList_.empty() ? rankInfoList_[0].hostIp : HcclIpAddress();
            u32 hostPort = !rankInfoList_.empty() ? rankInfoList_[0].hostPort : HCCL_INVALID_PORT;
            s32 hostDevId = !rankInfoList_.empty() ? rankInfoList_[0].devicePhyId : 0;
            HcclIpAddress localIp = rankInfoList_.size() > userRank_ ? rankInfoList_[userRank_].hostIp : HcclIpAddress();
            auto notifyResetCallback = [this](bool isSendRecv, s64 destRank) { 
                return isSendRecv ? this->ResetNotifyForDestRank(destRank) : this->ResetNotify(); 
            };

            auto setTransportStatusCallback = [this](const HcclOpIdentifier &opId, bool statusStop,
                const std::map<u32, bool> &remoteRankPortMap, const std::map<u32, bool> &isChangeLinkMap, bool isChangeLinkFlag) { 
                    return this->SetTransportStatus(opId, statusStop, remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag); 
            };
            auto getSwitchRanksCallback =
                [this](u32 *distSwitchRankList, bool *distSwitchUseBackup, u32 &distSwitchRankNum,
                       u8 *distRemoteRankNicStatus, u32 &distRankSize, bool &needCheckDefaultNic, bool &needCheckBackupNic) {
                return this->GetSwitchRanks(distSwitchRankList, distSwitchUseBackup, distSwitchRankNum,
                                            distRemoteRankNicStatus, distRankSize, needCheckDefaultNic, needCheckBackupNic);
            };
            auto setTransportResumeStatusCallback = [this](const std::map<u32, bool> &remoteRankPortMap, 
                const std::map<u32, bool> &isChangeLinkMap, bool isChangeLinkFlag, bool statusStop){
                    return this->SetTransportResumeStatus(remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag, statusStop); };
            HcclNetDevCtx netDevCtx = netDevCtxMap_[devIpAddr_[0]];
            HcclNetDevCtx backUpNetDevCtx;
            if (IsEnableBackupLink() && netDevCtxMap_.find(devBackupIpAddr_[0]) != netDevCtxMap_.end()) {
                backUpNetDevCtx = netDevCtxMap_[devBackupIpAddr_[0]];
            }
            OpRetryServerInfo serverInfo = {hostIp, hostPort, hostDevId};
            OpRetryAgentInfo agentInfo = {userRank_, deviceLogicId_, localIp, devIpAddr_[0], netDevCtx, backUpNetDevCtx};

            OpRetryAgentParam agentParam;
            agentParam.group = identifier_;
            agentParam.agentConnection = commConnections_.agentConnection;
            agentParam.h2dPtr = kfcControlTransferH2D_;
            agentParam.d2hPtr = kfcStatusTransferD2H_;
            agentParam.opStreamPtr = opRetryStreamPtr_;
            agentParam.notifyResetCallback = notifyResetCallback;
            agentParam.setTransportStatusCallback = setTransportStatusCallback;
            agentParam.setTransportResumeStatusCallback = setTransportResumeStatusCallback;
            agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
            agentParam.isEnableBackupLink = IsEnableBackupLink();
            agentParam.agentInfo = agentInfo;

            CHK_RET(opRetryManager_->RegisterOpRetryMachine(agentParam, userRankSize_, commConnections_.isRoot,
                                                            commConnections_.serverConnections, serverInfo));
        }
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::CompareWithServerId(const ServerInfo_t &left, const ServerInfo_t &right)
    {
        return (strcmp(left.serverId.c_str(), right.serverId.c_str()) < 0);
    }

    bool HcclCommunicator::CompareWithNicName(const NetworkInfo_t &left, const NetworkInfo_t &right)
    {
        return (strcmp(left.ethName.c_str(), right.ethName.c_str()) < 0);
    }

    bool HcclCommunicator::CompareWithUserRank(const RankInfo &left, const RankInfo &right)
    {
        return left.userRank < right.userRank;
    }

    HcclResult HcclCommunicator::InitPreResource(const RankTable_t &rankTable)
    {
        if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
            HCCL_ERROR("[Init][PreResource]not support cpu rank");
            return HCCL_E_NOT_SUPPORT;
        }
        (void)rankTable;
        // 查询本rank所在服务器
        auto iterServ = servRankInfo_.find(serverId_);

        bool check = (iterServ == servRankInfo_.end());
        CHK_PRT_RET(check, HCCL_ERROR("[Init][PreResource]can't find serverId[%s] in server map", serverId_.c_str()),
                    HCCL_E_NOT_FOUND);

        for (u32 i = 0; i < iterServ->second.size(); i++) {
            if (iterServ->second[i].deviceInfo.devicePhyId != HOST_DEVICE_ID) {
                enableP2PDevices_.push_back(iterServ->second[i].deviceInfo.devicePhyId);
            }
        }
        if (deviceType_ != DevType::DEV_TYPE_310P3) {
            HcclResult ret = P2PMgmtPub::EnableP2P(enableP2PDevices_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][PreResource]Enable P2P Failed, deviceLogicId[%d], ret[%u]", deviceLogicId_, ret), ret);
        }

        drvInit_ = true;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitTcpMode(const RankTable_t &rankTable) const
    {
        bool isTcpMode = false;
        HCCL_INFO("[TcpMode][%u] [1:TCP, 2:RDMA, 3:RESERVED]", GetExternalInputProtocolType());
        if (GetExternalInputProtocolType() == ProtocolType::TCP) {
            isTcpMode = true;
        }
        else if (GetExternalInputProtocolType() == ProtocolType::RDMA) {
            // 通信协议选择RDMA
        } else {
            isTcpMode = (rankTable.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST);
            HCCL_INFO("[Init][TcpMode]isTcpMode[%d] nicDeploy[%d]", isTcpMode, rankTable.nicDeploy);
        }
        SetTcpMode(isTcpMode);

        // 异构场景解析外部输入,放在SetTcpMode前防止Tcp用例走错分支，放在RecordProtocolType确保hdc模式下建链通信协议校验正确
        CHK_RET(InitExternalInputHeterog());
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::IsEnableBackupLink()
    {
        return deviceType_ == DevType::DEV_TYPE_910_93 && IsEnableRoce() && GetExternalInputHcclAicpuUnfold() &&
               GetExternalInputInterSuperPodRetryEnable() && !devBackupIpAddr_[0].IsInvalid() && rtsSupportChangeLink_ &&
               !isDiffDeviceType_;
    }

    HcclResult HcclCommunicator::InitRaResource()
    {
        /* 本通信域内只有1个device时，不需要初始化ra资源 */
        if (userRankSize_ <= 1) {
            HCCL_INFO("user rank size <= 1, ra is not needed for single device.");
            return HCCL_SUCCESS;
        }

        CHK_RET(IsHostUseDevNic(isHostUseDevNic_));

        if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID ||
            nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_, false));
            if (IsEnableBackupLink()) {
                // 超节点 && level2支持重执行 && Aicpu -> 初始化主备hccp资源(Pid粒度)
                CHK_RET(hrtGetPairDevicePhyId(devicePhyId_, deviceBackUpPhyId_));
                if (hrtGetDeviceIndexByPhyId(deviceBackUpPhyId_, deviceBackUpLogicId_) != HCCL_SUCCESS) {
                    rtsSupportChangeLink_ = false;
                    HCCL_ERROR("[%s]Runtime does not support changelink, deviceLogicId_[%d], devicePhyId_[%u], "
                               "deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], nicDeployment_[%d], IsEnableBackupLink[%d]"
                               "rtsSupportChangeLink_[%d]",
                               __func__, deviceLogicId_, devicePhyId_, deviceBackUpPhyId_,
                               deviceBackUpLogicId_, nicDeployment_, IsEnableBackupLink(), rtsSupportChangeLink_);
                    return HCCL_E_NOT_SUPPORT;
                } else {
                    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, deviceBackUpPhyId_, deviceBackUpLogicId_,
                                        false, true));
                    HCCL_DEBUG("[%s]Default & backup NetworkManager Init, deviceLogicId[%d], devicePhyId[%u], "
                               "deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], nicDeployment_[%d], IsEnableBackupLink[%d]",
                               __func__, deviceLogicId_, devicePhyId_, deviceBackUpPhyId_, deviceBackUpLogicId_,
                               nicDeployment_, IsEnableBackupLink());
                }
            }
        }

        if ((static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID && isHaveCpuRank_) ||
            (IsEnableRoce() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) ||
            (Is310PDevice() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST)) {
            u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
            CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhyID, deviceLogicId_, false));
        }

        CHK_RET(InitSocketManager());

        if (Is310PDevice()) {
            CHK_RET(InitNic());
        } else if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID) {
            std::shared_ptr<HcclSocket> &devVnicSocket = commPortConfig_.devVnicListen.first;
            if (devVnicSocket) {
                localVnicIp_ = devVnicSocket->GetLocalIp();
                localVnicListenPort_ = devVnicSocket->GetLocalPort();
                HcclNetDevCtx &devVnicCtx = commPortConfig_.devVnicListen.second;
                CHK_PTR_NULL(devVnicCtx);
                netDevCtxMap_.insert(std::make_pair(localVnicIp_, devVnicCtx));
                CHK_RET(socketManager_->ServerInit(devVnicCtx, localVnicListenPort_));
                commPortConfig_.devVnicListen.second = nullptr;
                HCCL_INFO("[HcclCommunicator][InitRaResource] init vnic with listened socket success, "
                          "listened ip[%s] port[%u]",
                          localVnicIp_.GetReadableAddress(), localVnicListenPort_);
            } else {
                localVnicListenPort_ = GetLocalNicPort(NicType::VNIC_TYPE);
                localVnicIp_ = HcclIpAddress(devicePhyId_);
                if (useSuperPodMode_) {
                    CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                        devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_SDID, superDeviceId_, localVnicIp_));
                } else {
                    CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                        devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, devicePhyId_, localVnicIp_));
                }

                HcclNetDevCtx vnicPortCtx;
                CHK_RET(HcclNetOpenDev(&vnicPortCtx, NicType::VNIC_TYPE, devicePhyId_, deviceLogicId_, localVnicIp_));
                CHK_PTR_NULL(vnicPortCtx);
                netDevCtxMap_.insert(std::make_pair(localVnicIp_, vnicPortCtx));
                CHK_RET(socketManager_->ServerInit(vnicPortCtx, localVnicListenPort_));
                HCCL_INFO("[HcclCommunicator][InitRaResource] init vnic with ip[%s] port[%u] success",
                          localVnicIp_.GetReadableAddress(), localVnicListenPort_);
            }

            if (isHaveCpuRank_) {
                HcclNetDevCtx hostPortCtx;
                CHK_RET(HcclNetOpenDev(&hostPortCtx, NicType::HOST_NIC_TYPE, devicePhyId_, deviceLogicId_, loopBackIp_));
                CHK_PTR_NULL(hostPortCtx);
                netDevCtxMap_.insert(std::make_pair(loopBackIp_, hostPortCtx));
                CHK_RET(socketManager_->ServerInit(hostPortCtx, hostPort_));
            }

            if (IsEnableRoce()) {
                CHK_RET(InitNic()); // isUsedRdmaLevel0_默认为false，若初始化网卡时，网卡IP有效才根据环境变量配置
            }
        }

        HCCL_INFO("isUsedRdmaLevel0_[%u] nicNum[%u] hostIP[%s], nicDeployment[%d].",
                  isUsedRdmaLevel0_, devIpAddr_.size(), hostIp_.GetReadableAddress(), nicDeployment_);

        raResourceInit_ = true; // 全局通信域会初始化，子通信域不会初始化，但是析构均会进入此逻辑，需要标记
        attrCollector_.GenSupportRdmaLite();
        isSupportRdmaLite_ = attrCollector_.GetSupportRdmaLite(); // 是否支持Rdma Lite
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DisablePreResource()
    {
        // 查询本rank所在服务器
        auto iterServ = servRankInfo_.find(serverId_);
        bool check = (iterServ == servRankInfo_.end());
        CHK_PRT_RET(check, HCCL_ERROR("[Disable][PreResource]can't find serverId[%s] in server map", serverId_.c_str()),
                    HCCL_E_NOT_FOUND);
        HcclResult ret = P2PMgmtPub::DisableP2P(enableP2PDevices_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Disable][PreResource]Disable all P2P Failed, deviceLogicId[%d], ret[%u]",
                               deviceLogicId_, ret),
                    ret);
        enableP2PDevices_.clear();
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetWorkspaceSubStreamNum(u64 &streamNum, u64 dataSize, HcclCMDType opType)
    {
        AlgType algType;

        CHK_RET(GetAlgType(algType, opType));

        std::map<HcclCMDType, u64> gapMap = {
            {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HCCL_SMALL_COUNT_2_MB + HCCL_SMALL_COUNT_512_KB},
            {HcclCMDType::HCCL_CMD_ALLGATHER, HCCL_SMALL_COUNT_2_MB + HCCL_SMALL_COUNT_512_KB},
            {HcclCMDType::HCCL_CMD_ALLREDUCE, (HCCL_SMALL_COUNT_1_MB + HCCL_SMALL_COUNT_512_KB) * userRankSize_}};

        if (serverNum_ == 1 && deviceType_ == DevType::DEV_TYPE_910_93 && gapMap.find(opType) != gapMap.end() &&
            dataSize <= gapMap[opType] &&
            deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) {
            streamNum = deviceNumPerAggregation_ - HCCL_SUB_STREAM_NP_MESH;
            HCCL_DEBUG("[GetWorkspaceSubStreamNum]DEV_TYPE_910_93 Single Server, the streamNum is %llu", streamNum);
            return HCCL_SUCCESS;
        }

        if (deviceType_ == DevType::DEV_TYPE_910_93) {
            streamNum = HCCL_SUB_STREAM_NUM_DOUBLE_RING + RDMA_PLANE_NUM_IN_NPRING_DOUBLE;
            if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING) {
                streamNum += 1U; // semi_ring算法server内增加一条从流，需要2条从流
            }
            if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
                opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
                streamNum = MAX_RANK_SIZE;
            }
            HCCL_DEBUG("[GetWorkspaceSubStreamNum]DEV_TYPE_910_93, the streamNum is %llu", streamNum);
            return HCCL_SUCCESS;
        }

        if (deviceType_ == DevType::DEV_TYPE_910B &&
            (opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) &&
            GetExternalInputHcclDeterministicV2() == DETERMINISTIC_STRICT) {
            // 图模式 A2规约保序场景，需要重新计算需要的streamNum
            streamNum = CalcStreamNumForReduceOrderPreservation();
            HCCL_DEBUG("[GetWorkspaceSubStreamNum]A2 reduce order preservation, the streamNum is %llu", streamNum);
            return HCCL_SUCCESS;
        }

        if (deviceType_ == DevType::DEV_TYPE_910B && opType == HcclCMDType::HCCL_CMD_ALLREDUCE && algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
            streamNum = userRankSize_ / moduleNum_ - 1;
            HCCL_DEBUG("[GetWorkspaceSubStreamNum]A2 pipeline allreduce, the streamNum is %llu", streamNum);
            return HCCL_SUCCESS;
        }

        // 设置AG和RS的图模式pipeline算法能够申请的streamNum
        if (deviceType_ == DevType::DEV_TYPE_910B &&                                                         // 910B
            algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_MESH &&                                       // fullmesh
            (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) && // AG或RS
            moduleNum_ > 1 && deviceNumPerAggregation_ > 1 &&                                                // 多机且每机器出多卡
            (moduleNum_ <= MODULE_NUM_FOUR ||                                                                // "机器数量小于等于4"
             dataSize > HCCL_SMALL_COUNT_1_MB ||                                                             // "大数据量"
             algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE)) { // "指定level1的算法为pipeline"
            streamNum = userRankSize_ / moduleNum_;
            HCCL_DEBUG("[GetWorkspaceSubStreamNum]DEV_TYPE_910B, the streamNum is %llu", streamNum);
            return HCCL_SUCCESS;
        }

        // 根据所用算法，选择所需的从stream数目
        switch (algType.algoLevel0) {
            case AlgTypeLevel0::ALG_LEVEL0_NP_MESH:
                streamNum = userRankSize_ / moduleNum_ - HCCL_SUB_STREAM_NP_MESH;
            break;
            case AlgTypeLevel0::ALG_LEVEL0_8P_RING:
                streamNum = HCCL_SUB_STREAM_NUM_8P_RING;
            break;
            case AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING:
                streamNum = (GetExternalInputEnableRdmaSdmaConcurrent() == false) ? HCCL_SUB_STREAM_NUM_DOUBLE_RING : HCCL_SUB_STREAM_NUM_DOUBLE_RING + RDMA_PLANE_NUM_IN_NPRING_DOUBLE;
            break;
            case AlgTypeLevel0::ALG_LEVEL0_4P_MESH:
                streamNum = HCCL_SUB_STREAM_NUM_4P_MESH;
            break;
            default:
                streamNum = (GetExternalInputEnableRdmaSdmaConcurrent() == false) ? HCCL_SUB_STREAM_NUM_ZERO : HCCL_SUB_STREAM_NUM_DOUBLE_RING;
            break;
        }

        if (SatisfyIntraSuperPod(deviceType_, userRankSize_, useSuperPodMode_, superPodNum_)) {
            streamNum = std::max(static_cast<u64>(userRankSize_ - 1u), streamNum);
        } else if (FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(deviceType_,
                                                                      meshAggregationRankSize_, useSuperPodMode_)) {
            streamNum = std::max(static_cast<u64>(meshAggregationRankSize_ - 1u), streamNum);
        }

        auto iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algType.algoLevel0);
        CHK_PRT_RET(iter == HCCL_ALGO_LEVEL0_NAME_MAP.end(),
                    HCCL_ERROR("[GetWorkspaceSubStreamNum]level0: algType[%u] is invalid.", algType.algoLevel0),
                    HCCL_E_INTERNAL);
        HCCL_DEBUG("[GetWorkspaceSubStreamNum]hccl algorithm: In level0, using %s algo, the streamNum is %llu",
                   iter->second.c_str(), streamNum);

        u64 sliceNum = CalculatePiplineSliceNum(opType, dataSize, algType, deviceType_, deviceNumPerServer_, serverNum_);
        // 图模式下数据量固定, 按照当前数据量判断是否支持pipline切分并申请从流
        if (implAlg_ != nullptr && sliceNum >= MIN_PIPLINE_SLICE_NUM) {
            streamNum++;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DestroyNetworkResources()
    {
        transportManager_ = nullptr;
        if (raResourceInit_) {
            socketManager_->DestroySockets();
        }

        /* 本通信域内只有1个device时，不需要卸载ra资源 */
        if (userRankSize_ <= 1) {
            HCCL_INFO("user rank size <= 1, ra is not needed for single device");
            return HCCL_SUCCESS;
        }

        // nic的初始化独立调用，在此单独判断是否需要解初始化
        if (nicInitialized_ > 0) {
            CHK_RET(DeinitNic());
        }

        if (raResourceInit_ && (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID) && !Is310PDevice()) {
            if (isHaveCpuRank_) {
                CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[loopBackIp_], hostPort_));
                HcclNetCloseDev(netDevCtxMap_[loopBackIp_]);
                netDevCtxMap_.erase(loopBackIp_);
            }
            CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[localVnicIp_], localVnicListenPort_));
            HcclNetCloseDev(netDevCtxMap_[localVnicIp_]);
            netDevCtxMap_.erase(localVnicIp_);
        }

        CHK_RET(ReleasePreemptSocket());

        if (raResourceInit_) {
            if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID ||
                nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
                if (IsEnableBackupLink()) {
                    // 超节点 && level2支持重执行 && Aicpu -> 释放主备hccp资源
                    CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_));
                    CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, deviceBackUpPhyId_,
                                          deviceBackUpLogicId_, true));
                    HCCL_DEBUG("[%s]Default & backup HcclNetDeInit, deviceLogicId[%d], devicePhyId[%u], "
                               "deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], nicDeployment_[%d], IsEnableBackupLink[%d]",
                               __func__, deviceLogicId_, devicePhyId_, deviceBackUpPhyId_, deviceBackUpLogicId_,
                               nicDeployment_, IsEnableBackupLink());
                }
                else {
                    CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_));
                }
            }

            if ((static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID && isHaveCpuRank_) ||
                (IsEnableRoce() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) ||
                (Is310PDevice() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST)) {
                u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
                CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhyID, deviceLogicId_));
            }

            socketManager_ = nullptr;
        }

        raResourceInit_ = false;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetWorkspaceResource(const std::string &tag, void *memPtr, u64 &maxSize,
                                                      std::vector<rtStream_t> &stream)
    {
        return workSpaceRes_->SetWorkspaceResource(tag, memPtr, maxSize, stream);
    }

    void HcclCommunicator::DestroyWorkspaceResource(const std::string &tag)
    {
        workSpaceRes_->DestroyWorkspaceResource(tag);
    }

    HcclResult HcclCommunicator::AtomicInitSet()
    {
        CHK_PRT_RET(initializedFlag_.test_and_set(),
                    HCCL_ERROR("[HcclCommunicator][AtomicInitSet]errNo[0x%016llx] instance "
                               "already been initialized",
                               HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
                    HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::AtomicInitClear()
    {
        initializedFlag_.clear();
    }

    u32 HcclCommunicator::GetUserRank()
    {
        return realUserRank_;
    }

    u32 HcclCommunicator::GetGroupRank()
    {
        return userRank_;
    }

    u32 HcclCommunicator::GetRankSize()
    {
        return userRankSize_;
    }

    bool HcclCommunicator::GetNicInitialized()
    {
        return nicInitialized_ > 0;
    }

    /*
        1. 选择算法
        2. 计算resource，存到request内
        3. 创建和分配资源
    */
    HcclResult HcclCommunicator::HcclSelectAlg(HcclCMDType opType, u64 count, HcclDataType dataType,
                                               HcclReduceOp op, bool &ifAiv, std::string &algName, bool isSuperKernel)
    {
        ifAiv = false;
        // super kernel仅支持A3单机
        if (isSuperKernel && (deviceType_ != DevType::DEV_TYPE_910_93 || serverNum_ > 1)) {
            HCCL_WARNING("[HcclCommunicator][HcclSelectAlg] super kernel not support deviceType_[%u] serverNum_[%u]",
                         deviceType_, serverNum_);
            return HCCL_SUCCESS;
        }

        /* 选择算法前，先更新成图模式 */
        auto originWorkflowMode = GetWorkflowMode();
        SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
        std::string newTag;
        std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
        CHK_SMART_PTR_NULL(algOperator);
        OpParam param;
        param.reduceType = op;
        param.opType = opType;
        if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
            param.All2AllDataDes.sendType = dataType;
            param.All2AllDataDes.recvType = dataType;
            param.All2AllDataDes.sendCount = count;
        } else {
            param.DataDes.count = count;
            param.DataDes.dataType = dataType;
        }

        ResourceLimit limit;
        AlgDesc algDesc;
        CHK_RET(algOperator->SelectAlg("", param, limit, algName, algDesc, newTag));
        /* 非AIV算法直接返回 */
        if (!algDesc.isAivMode) {
            HCCL_INFO("[HcclCommunicator][HcclSelectAlg] alg not select to AIV");
            return HCCL_SUCCESS;
        }
        ifAiv = true;
        /* 完成算法选择和记录后，恢复成原来的模式 */
        SetWorkflowMode(originWorkflowMode);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::HcclCalcBlockDim(HcclCMDType opType, u64 count, HcclDataType dataType,
                                                  std::string &algName, u32 &blockDim)
    {
        auto originWorkflowMode = GetWorkflowMode();
        SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
        std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
        CHK_SMART_PTR_NULL(algOperator);
        OpParam param;

        param.opType = opType;
        if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
            param.All2AllDataDes.sendType = dataType;
            param.All2AllDataDes.recvType = dataType;
            param.All2AllDataDes.sendCount = count;
        } else {
            param.DataDes.count = count;
            param.DataDes.dataType = dataType;
        }

        CHK_RET(algOperator->CalBlockDim(algName, param, blockDim));
        SetWorkflowMode(originWorkflowMode);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::HcclGetAlgExecParam(const std::string &tag, HcclCMDType opType, u64 count, void *inputPtr, void *outputPtr,
                                                     bool clearEnable, HcclDataType dataType, HcclReduceOp op, void *&commContext, u64 &len, u32 aivCoreLimit)
    {
        /* 将Host申请和注册好的资源，传给AICPU */
        // 1\ algName 从getstr里某一个名字里获取出来（要防止名字重复） commContext & len 从 response里拿
        // 2\ rtmemcopy 先获取一下algoperator对象，用这个调用getalgxxx
        AivSuperKernelArgs aivSuperKernelArgs;
        SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

        void *sendAlgParamMemPtr = nullptr;
        // alloc device 地址
        hrtMalloc(&sendAlgParamMemPtr, sizeof(AivSuperKernelArgs));
        HCCL_INFO("SPK sendalgparam %p", sendAlgParamMemPtr);
        OpParam param;
        param.DataDes.count = count;
        param.DataDes.dataType = dataType;
        param.reduceType = op;
        param.tag = tag;
        param.inputPtr = inputPtr;
        param.outputPtr = outputPtr;
        param.opType = opType;
        u64 totalSize;
        if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
            param.All2AllDataDes.sendType = dataType;
            param.All2AllDataDes.recvType = dataType;
            param.All2AllDataDes.sendCount = count;
        }

        if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
            totalSize = count * SIZE_TABLE[dataType] * userRankSize_;
        } else {
            totalSize = count * SIZE_TABLE[dataType]; // allreduce就是输入
        }
        param.inputSize = totalSize;
        std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
        CHK_SMART_PTR_NULL(algOperator);
        std::string algName;
        AlgResourceResponse algResResponse;
        std::string newTag;
        ResourceLimit limit;
        AlgDesc algDesc;
        CHK_RET(algOperator->SelectAlg(param.tag, param, limit, algName, algDesc, newTag));

        // 资源创建
        InsertNewTagToTagMap(newTag, param.tag);
        if (resMap_.find(newTag) == resMap_.end()) {
            HCCL_INFO("[HcclCoommunicator][HcclAllocRes] algName[%s], alloc new res", algName.c_str());
            AlgResourceRequest resRequest;
            CHK_RET(algOperator->CalcResRequest(algName, param, resRequest)); // [重构建议] 计算和alloc可以拆开
            CHK_RET(AllocAlgResource(newTag, opType, param, resRequest, resMap_[newTag]));
            // 暂不作心跳注册
        }

        CHK_RET(algOperator->GetAivExecParam(algName, param, resMap_[newTag], aivSuperKernelArgs));

        // gettag
        HCCL_INFO("SPK, rank %llu", userRank_);
        u32 maxCorePerRank = aivCoreLimit / userRankSize_;
        aivSuperKernelArgs.tag = aivTag_;
        aivSuperKernelArgs.clearEnable = (clearEnable ? 1 : 0);
        aivTag_ = GetNextAivTag(aivTag_, algDesc.aivTagNum);

        if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
            aivSuperKernelArgs.blockdim = userRankSize_; // allreduce现在不支持多核，固定ranksize核
        } else {
            aivSuperKernelArgs.blockdim = userRankSize_ *
                                          (maxCorePerRank > BLOCK_DIM_FOUR_PER_RANK_A3 ? BLOCK_DIM_FOUR_PER_RANK_A3 : maxCorePerRank);
        }

        HCCL_INFO("SPK, Tag %llu clearEnable %llu, aivCoreLimit %u, blockdim %llu", aivSuperKernelArgs.tag,
                  aivSuperKernelArgs.clearEnable, aivCoreLimit, aivSuperKernelArgs.blockdim);
        // clearenable
        //  拷贝到Device
        SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_RET(hrtMemSyncCopy(
            sendAlgParamMemPtr, sizeof(AivSuperKernelArgs),
            &aivSuperKernelArgs, sizeof(AivSuperKernelArgs),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        commContext = sendAlgParamMemPtr;
        len = sizeof(AivSuperKernelArgs);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckDeviceType(const DevType deviceType) const
    {
        if ((deviceType >= DevType::DEV_TYPE_COUNT) || (deviceType < DevType::DEV_TYPE_910)) {
            HCCL_ERROR("[Check][DeviceType]errNo[0x%016llx] deivce Type[%d] out of range[%d, %d]",
                       HCCL_ERROR_CODE(HCCL_E_PARA), deviceType, DevType::DEV_TYPE_910, DevType::DEV_TYPE_NOSOC);
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckReductionOp(const HcclReduceOp op) const
    {
        if ((op >= HCCL_REDUCE_RESERVED) || (op < HCCL_REDUCE_SUM)) {
            HCCL_ERROR("[Check][ReductionOp]errNo[0x%016llx] op:[%d] not supported", HCCL_ERROR_CODE(HCCL_E_PARA), op);
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckUserRank(const u32 userRank) const
    {
        if (userRankSize_ <= userRank) {
            HCCL_ERROR("[Check][UserRank]errNo[0x%016llx] userRank:[%u] is out of range[0 ~ %u]",
                       HCCL_ERROR_CODE(HCCL_E_PARA), userRank, userRankSize_);
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckCount(const u64 count) const
    {
        if (count > SYS_MAX_COUNT) {
            HCCL_ERROR("[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
                       HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetGroupRanksInfo(const std::vector<u32> &groupRanks, std::vector<RankInfo> &ranksInfo)
    {
        ranksInfo.clear();
        std::vector<RankInfo> tmpRankInfoList;
        tmpRankInfoList.assign(rankInfoList_.begin(), rankInfoList_.end());

        for (u32 index = 0; index < groupRanks.size(); index++) {
            if (tmpRankInfoList.size() <= groupRanks[index]) {
                HCCL_ERROR("[Get][GroupRanksInfo]errNo[0x%016llx] groupRanks[%u]=[%u], >= rankinfolist size[%zu]",
                           HCCL_ERROR_CODE(HCCL_E_PARA), index, groupRanks[index], tmpRankInfoList.size());
                return HCCL_E_PARA;
            }
            tmpRankInfoList[groupRanks[index]].userRank = index;
            ranksInfo.push_back(tmpRankInfoList[groupRanks[index]]);
            HCCL_DEBUG("index: %d userRank: %dhost ip: %s host port: %u dev phy id: %d serverIdx:%d",
                       index,
                       tmpRankInfoList[groupRanks[index]].userRank,
                       tmpRankInfoList[groupRanks[index]].hostIp.GetReadableAddress(),
                       tmpRankInfoList[groupRanks[index]].hostPort,
                       tmpRankInfoList[groupRanks[index]].devicePhyId,
                       tmpRankInfoList[groupRanks[index]].serverIdx);
        }

        // 按rank id从小到大的顺序返回
        std::sort(ranksInfo.begin(), ranksInfo.end(), CompareWithUserRank);

        for (u32 index = 0; index < ranksInfo.size(); ++index) {
            if (index != ranksInfo[index].userRank) {
                HCCL_ERROR("[Get][GroupRanksInfo]errNo[0x%016llx] index[%u] !=  user rank[%u]",
                           HCCL_ERROR_CODE(HCCL_E_PARA), index, ranksInfo[index].userRank);
                return HCCL_E_PARA;
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetGroupCommonData(WorldGroupInfo &groupCommonData) const
    {
        groupCommonData.inlineReduceSwitchOn = inlineReduceSwitchOn_;
        groupCommonData.deviceType = deviceType_;
        groupCommonData.deviceLogicId = deviceLogicId_;
        groupCommonData.profilingInitiated = profilingInitiated_;
        groupCommonData.serverId = serverId_;
        groupCommonData.phyIdNicInfoMap = rankDevicePhyIdNicInfoMap_;
        groupCommonData.worldRankInfoList = rankInfoList_;
        groupCommonData.ranksPort = nicRanksPort_;
        groupCommonData.vnicRanksPort = vnicRanksPort_;
        groupCommonData.useSuperPodMode = useSuperPodMode_;
        groupCommonData.devPortSwitchOn = commPortConfig_.devPortSwitchOn;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetWorkspaceMemSize(const std::string &opType, u64 count, HcclDataType dataType,
                                                     u32 &rankSize, u64 &memSize, DevType &deviceType) const
    {
        return workSpaceRes_->GetWorkspaceMemSize(opType, count, dataType, rankSize, memSize, deviceType);
    }

    DeviceMem HcclCommunicator::GetWorkspaceScracthMem(const std::string &tag, u64 allocMemSize)
    {
        return workSpaceRes_->AllocDeviceMem(tag, allocMemSize);
    }

    std::vector<Stream> HcclCommunicator::GetWorkspaceSubStreams(const std::string &tag, u32 num)
    {
        return workSpaceRes_->AllocSlaveStreams(tag, num);
    }

    HcclResult HcclCommunicator::InitProfiling()
    {
        if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
            HCCL_ERROR("[Init][Profiling]not support cpu rank");
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_PRT_RET(profilingInitiated_, HCCL_DEBUG("Profiling plugin has already been Initiated."), HCCL_SUCCESS);

        if (profilingMode_ != HcomProfilingMode::PROFILING_OPEN && GetExternalInputProfilingMode()) {
            profilingMode_ = HcomProfilingMode::PROFILING_OPEN;
            profilingOption_ = GetExternalInputProfilingOption();
        }
        HCCL_INFO("profiling config information:options[%s], mode[%d]", profilingOption_.c_str(), profilingMode_);

        // profilingInitiated_会广播给所有子通信域，用于避免taskInfoSaver的重复初始化
        profilingInitiated_ = true;
        // isExecuteProfilingInit_用于记录本impl是否执行了taskInfoSaver的初始化，用于进行对应的释放
        isExecuteProfilingInit_ = true;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DeinitProfiling()
    {
        CHK_PRT_RET(!profilingInitiated_, HCCL_DEBUG("Profiling plugin has not been Initiated"), HCCL_SUCCESS);
        profilingInitiated_ = false;
        HCCL_INFO("Profiling is deinitiated.");
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::RegistTaskExceptionHandler() const
    {
        CHK_RET(TaskExceptionHandler::Init());
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::UnRegistTaskExceptionHandler() const
    {
        CHK_RET(TaskExceptionHandler::DeInit());
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetInCCLbuffer(void *&buffer, u64 &size)
    {
        return cclBufferManager_.GetInCCLbuffer(buffer, size);
    }

    HcclResult HcclCommunicator::GetOutCCLbuffer(void *&buffer, u64 &size)
    {
        return cclBufferManager_.GetOutCCLbuffer(buffer, size);
    }

    void HcclCommunicator::ReleaseCommCCLbuffer()
    {
        cclBufferManager_.ReleaseCommCCLbuffer();
    }

    HcclResult HcclCommunicator::ReleaseCommInfos()
    {
        if (implAlg_ != nullptr) {
            return implAlg_->ReleaseCommInfos();
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitProfiler()
    {
        profilerManager_.reset(new (std::nothrow) ProfilerManager(devicePhyId_, deviceLogicId_, realUserRank_));
        CHK_SMART_PTR_NULL(profilerManager_);
        HcclResult ret = profilerManager_->InitProfiler();
        CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[BASE][InitProfiler]profilerManager_ InitProfiler failed."),
                    HCCL_E_PARA);

        HCCL_INFO("[BASE][InitProfiler]Register CtrlCallBack success.");
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateCommCCLbuffer()
    {
        return cclBufferManager_.CreateCommCCLbuffer();
    }

    HcclResult HcclCommunicator::CreateCommExpBuffer()
    {
        return cclBufferManager_.CreateCommExpBuffer();
    }

    HcclResult HcclCommunicator::InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize)
    {
        return cclBufferManager_.InitCCLbuffer(inCCLbufferSize, outCCLbufferSize);
    }

    u32 HcclCommunicator::GetLocalNicPort(NicType nicType)
    {
        u32 port = HCCL_INVALID_PORT;
        if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
            return GetHostPort(devicePhyId_);
        }
        // isUseRankPort_在ranksPort初始化时一同配置：1. 异构场景 2. 开启device侧端口配置
        // groupRanksPort_为空说明此时处于全局通信域，要从ranksPort_取监听端口；否则取groupRanksPort_
        bool devicePortSwitchOn = commPortConfig_.devPortSwitchOn;
        if (nicType == NicType::HOST_NIC_TYPE) {
            port = GetHostPort(devicePhyId_);
        } else if (devicePortSwitchOn && nicType == NicType::VNIC_TYPE) {
            // vnic ports仅在开启device侧端口配置时单独配置
            std::vector<u32> &ranksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
            port = GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);
        } else {
            // 1. 开启device侧端口配置时的nic port时使用ranksPorts
            // 2. 异构场景使用ranksPorts
            // 3. 其余场景场景isUseRankPort_应当为false，使用默认port
            std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
            port = GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);
        }
        HCCL_INFO("[HcclCommunicator][GetLocalNicPort] nicType[%u], devicePortSwitchOn[%u], isUseRankPort[%u], "
                  "get port[%u], devId[%u]",
                  nicType, devicePortSwitchOn, isUseRankPort_, port, devicePhyId_);
        return port;
    }

    HcclResult HcclCommunicator::InitNic(bool isMC2ReInit)
    {
        if (!GetExternalInputIntraRoceSwitch() && servRankInfo_.size() == 1 && isDiffDeviceModule_ && !isMC2ReInit) {
            return HCCL_SUCCESS;
        }

        if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            std::shared_ptr<HcclSocket> &devNicSocket = commPortConfig_.devNicListen.first;
            if (devNicSocket) {
                HcclNetDevCtx &devNicCtx = commPortConfig_.devNicListen.second;
                CHK_PTR_NULL(devNicCtx);
                netDevCtxMap_.insert(std::make_pair(devNicSocket->GetLocalIp(), devNicCtx));
                CHK_RET(socketManager_->ServerInit(devNicCtx, devNicSocket->GetLocalPort()));
                commPortConfig_.devNicListen.second = nullptr;
                HCCL_INFO("[HcclCommunicator][InitNic] init nic with listened socket success, "
                          "listened ip[%s] port[%u]",
                          devNicSocket->GetLocalIp().GetReadableAddress(), devNicSocket->GetLocalPort());
            } else {
                u32 port = GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
                u32 nicNum = devIpAddr_.size();
                for (u32 i = 0; i < nicNum; i++) {
                    if (devIpAddr_[i].IsInvalid()) {
                        HCCL_INFO("[Init][Nic]nic num[%u] deviceip is invalid, total nicNum[%u]", i, nicNum);
                        continue;
                    }
                    HcclNetDevCtx nicPortCtx;
                    CHK_RET(HcclNetOpenDev(&nicPortCtx, NicType::DEVICE_NIC_TYPE, devicePhyId_, deviceLogicId_, devIpAddr_[i]));
                    CHK_PTR_NULL(nicPortCtx);
                    netDevCtxMap_.insert(std::make_pair(devIpAddr_[i], nicPortCtx));
                    CHK_RET(socketManager_->ServerInit(nicPortCtx, port));
                    HCCL_INFO("[HcclCommunicator][InitNic] init nic with ip[%s] port[%u] success",
                              devIpAddr_[i].GetReadableAddress(), port);
                }
            }
            attrCollector_.GenUsedRdmaLevel0();
            isUsedRdmaLevel0_ = attrCollector_.GetUsedRdmaLevel0();

            if (IsEnableBackupLink()) {
                std::shared_ptr<HcclSocket> &backupNicSocket = commPortConfig_.backupDevNicListen.first;
                if (backupNicSocket) {
                    HcclNetDevCtx &backupNicCtx = commPortConfig_.backupDevNicListen.second;
                    CHK_PTR_NULL(backupNicCtx);
                    netDevCtxMap_.insert(std::make_pair(backupNicSocket->GetLocalIp(), backupNicCtx));
                    CHK_RET(socketManager_->ServerInit(backupNicCtx, backupNicSocket->GetLocalPort()));
                    commPortConfig_.backupDevNicListen.second = nullptr;
                    HCCL_INFO("[HcclCommunicator][InitNic] init backup nic with listened socket success, "
                              "listened ip[%s] port[%u]",
                              backupNicSocket->GetLocalIp().GetReadableAddress(), backupNicSocket->GetLocalPort());
                } else {
                    // 超节点 && level2支持重执行 && Aicpu -> 备用网卡 initRdma
                    HcclNetDevCtx nicPortBackUpCtx;
                    CHK_RET(HcclNetOpenDev(&nicPortBackUpCtx, NicType::DEVICE_NIC_TYPE, deviceBackUpPhyId_,
                                           deviceBackUpLogicId_, devBackupIpAddr_[0], devIpAddr_[0]));
                    CHK_PTR_NULL(nicPortBackUpCtx);
                    netDevCtxMap_.insert(std::make_pair(devBackupIpAddr_[0], nicPortBackUpCtx));
                    CHK_RET(socketManager_->ServerInit(nicPortBackUpCtx, devBackupPort_));
                    HCCL_DEBUG("[%s]finish backup ServerInit, deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], "
                               "devBackupIpAddr_[%s], devBackupPort_[%u], nicDeployment_[%d], IsEnableBackupLink[%d], "
                               "netDevCtxMap_.size[%d]",
                               __func__, deviceBackUpPhyId_, deviceBackUpLogicId_, devBackupIpAddr_[0].GetReadableAddress(),
                               devBackupPort_, nicDeployment_, IsEnableBackupLink(), netDevCtxMap_.size());
                    HCCL_INFO("[HcclCommunicator][InitNic] init backup nic with ip[%s] port[%u] success",
                              devBackupIpAddr_[0].GetReadableAddress(), devBackupPort_);
                }
            }
        } else if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
            u32 port = GetLocalNicPort(NicType::HOST_NIC_TYPE);
            CHK_PRT_RET((hostIp_.IsInvalid()), HCCL_ERROR("[Init][Nic] host ip is invalid when NIC "
                                                          "deployment is host. "),
                        HCCL_E_PARA);
            attrCollector_.GenUsedRdmaLevel0();
            isUsedRdmaLevel0_ = attrCollector_.GetUsedRdmaLevel0();
            u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
            HCCL_INFO("[Init][Nic], hostPort[%u], devicePhyID[%u]", port, devicePhyID);
            HcclNetDevCtx hostnicPortCtx;
            CHK_RET(HcclNetOpenDev(&hostnicPortCtx, NicType::HOST_NIC_TYPE, devicePhyId_, deviceLogicId_, hostIp_));
            CHK_PTR_NULL(hostnicPortCtx);
            netDevCtxMap_.insert(std::make_pair(hostIp_, hostnicPortCtx));
            CHK_RET(socketManager_->ServerInit(hostnicPortCtx, port));
        } else {
            HCCL_ERROR("[Init][Nic]nic deployment[%d] is not supported", nicDeployment_);
            return HCCL_E_PARA;
        }
        isNeedInitNic_ = true;
        attrCollector_.SetNeedInitNicFlag(isNeedInitNic_);
        nicInitialized_++;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DeinitNic()
    {
        if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            u32 port = GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
            u32 nicNum = devIpAddr_.size();
            for (u32 i = 0; i < nicNum; i++) {
                if (devIpAddr_[i].IsInvalid()) {
                    HCCL_INFO("continue invalid devIp %s", devIpAddr_[i].GetReadableAddress());
                    continue;
                }
                if (netDevCtxMap_.find(devIpAddr_[i]) == netDevCtxMap_.end()) {
                    HCCL_INFO("devIp[%s] not found in netDevCtxMap_", devIpAddr_[i].GetReadableAddress());
                    continue;
                }
                CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[devIpAddr_[i]], port));
                // 最后一次调用才删除netCtx
                if (nicInitialized_ - 1 <= 0) {
                    HcclNetCloseDev(netDevCtxMap_[devIpAddr_[i]]);
                    netDevCtxMap_.erase(devIpAddr_[i]);
                }
            }
            if (IsEnableBackupLink() && netDevCtxMap_.find(devBackupIpAddr_[0]) != netDevCtxMap_.end()) {
                // 超节点 && level2支持重执行 && Aicpu -> 备用网卡 deinit
                CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[devBackupIpAddr_[0]], devBackupPort_));
                if (nicInitialized_ - 1 <= 0) {
                    HcclNetCloseDev(netDevCtxMap_[devBackupIpAddr_[0]]);
                    netDevCtxMap_.erase(devBackupIpAddr_[0]);
                    HCCL_DEBUG("[%s]finish backup ServerDeInit devBackupIpAddr_[%s], port[%u], IsEnableBackupLink[%d]",
                               __func__, devBackupIpAddr_[0].GetReadableAddress(), devBackupPort_, IsEnableBackupLink());
                }
            }
        } else if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
            u32 port = GetLocalNicPort(NicType::HOST_NIC_TYPE);
            CHK_PRT_RET((hostIp_.IsInvalid()), HCCL_ERROR("[DeInit][Nic] host ip is invalid when NIC "
                                                          "deployment is host. "),
                        HCCL_E_PARA);
            CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[hostIp_], port));
            HcclNetCloseDev(netDevCtxMap_[hostIp_]);
            netDevCtxMap_.erase(hostIp_);
        } else {
            HCCL_ERROR("[Deinit][Nic]nic deployment[%d] is not supported", nicDeployment_);
            return HCCL_E_PARA;
        }
        nicInitialized_--;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::RegisterRanksToDca()
    {
        if (deviceType_ != DevType::DEV_TYPE_910_93 && deviceType_ != DevType::DEV_TYPE_910B) {
            HCCL_WARNING("[RegisterRanksToDca] not support deviceType[%d]", deviceType_);
            return HCCL_SUCCESS;
        }
        CHK_RET(setVnicIpToRankInfoList());
        DetectConnectionAnomalies::GetInstance(deviceLogicId_).Init(rankInfoList_, isNeedInitNic_);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::RegisterToHeartBeat()
    {
        if (Is310PDevice() || deviceType_ == DevType::DEV_TYPE_310P3) {
            return HCCL_SUCCESS;
        }
        u32 localPort = commPortConfig_.devPortSwitchOn ? HCCL_INVALID_PORT : GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
        return Heartbeat::GetInstance(deviceLogicId_).RegisterToHeartBeat(userRank_, deviceType_, rankInfoList_, localPort, isNeedInitNic_, identifier_, useSuperPodMode_, isUsedRdmaLevel0_, retryEnable_, IsEnableBackupLink());
    }

    HcclResult HcclCommunicator::RegisterToHeartBeat(u32 peerRankId, string &tag)
    {
        u32 localPort = commPortConfig_.devPortSwitchOn ? HCCL_INVALID_PORT : GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
        return Heartbeat::GetInstance(deviceLogicId_).RegisterToHeartBeat(userRank_, deviceType_, rankInfoList_, localPort, isNeedInitNic_, peerRankId, identifier_, tag, useSuperPodMode_, isUsedRdmaLevel0_, retryEnable_, IsEnableBackupLink());
    }

    void HcclCommunicator::UnRegisterToHeartBeat()
    {
        for (auto tag : hbSendRecvTags_) {
            Heartbeat::GetInstance(deviceLogicId_).UnRegisterToHeartBeat(deviceType_, identifier_, tag);
        }
        Heartbeat::GetInstance(deviceLogicId_).UnRegisterToHeartBeat(deviceType_, identifier_);
    }

    HcclResult HcclCommunicator::SetGlobalWorkSpace(std::vector<void *> &globalWorkSpaceAddr)
    {
        CHK_RET(HcclSetGlobalWorkSpace(dispatcher_, globalWorkSpaceAddr));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetandClearOverFlowTasks(std::vector<HcclDumpInfo> &hcclDumpInfo)
    {
        if (profilerManager_ != nullptr) {
            CHK_RET(profilerManager_->GetandClearOverFlowTasks(hcclDumpInfo));
        } else {
            HCCL_WARNING("[impl][GetDumpTask] profilerManager_ not set");
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetDeviceId(s32 &deviceId) const
    {
        deviceId = deviceLogicId_;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetQosCfg(const u32 qosCfg)
    {
        CHK_PTR_NULL(dispatcher_);
        return HcclSetQosCfg(dispatcher_, qosCfg);
    }

    HcclResult HcclCommunicator::ResetQosCfg()
    {
        CHK_PTR_NULL(dispatcher_);
        return HcclResetQosCfg(dispatcher_);
    }

    HcclResult HcclCommunicator::GetQosCfg(u32 &qosCfg)
    {
        CHK_PTR_NULL(dispatcher_);
        return HcclGetQosCfg(dispatcher_, &qosCfg);
    }

    HcclResult HcclCommunicator::GetCqeError(HcclResult &result)
    {
        CHK_RET(Heartbeat::GetInstance(deviceLogicId_).CheckErrorCqe(identifier_, result));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::MrManagerInit()
    {
        // 拉远、下沉、推理场景(ps、worker)支持使用mrManager
        if (!GetExternalInputHcclIsTcpMode() && (Is310PDevice())) {
            mrManager_.reset(new (std::nothrow) MrManager(netDevCtxMap_[devIpAddr_[0]]));
            CHK_SMART_PTR_NULL(mrManager_);

            CHK_RET(mrManager_->Init());
            mrManagerInit_ = true;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::MrManagerDeInit()
    {
        if (mrManagerInit_) {
            CHK_SMART_PTR_NULL(mrManager_);
            CHK_RET(mrManager_->DeInit());
            mrManager_ = nullptr;
            mrManagerInit_ = false;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SupportDeterministicOptim(bool &isDeterministicOptim)
    {
        CHK_SMART_PTR_NULL(implAlg_);
        CHK_RET(implAlg_->SupportDeterministicOptim(isDeterministicOptim));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetHccsLinkNum(u32 &numHccsLink)
    {
        auto iter = pairLinkInfo_.find(static_cast<u32>(LinkTypeInServer::HCCS_TYPE));
        if (iter == pairLinkInfo_.end()) {
            HCCL_ERROR("[HcclCommunicator][GetHccsLinkNum]HCCS_TYPE is not found");
            return HCCL_E_PARA;
        }
        numHccsLink = iter->second.size();
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
                                           HcclDataType dataType, HcclRtStream stream, HcomCollOpInfo *opInfo)
    {
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true &&
            (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][AllGather]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = inputCount * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize * userRankSize_;
        opParam.DataDes.count = inputCount;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
        opParam.stream = streamObj;
        opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLGATHER,
                                                                    tag, inputCount, dataType, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(),
                                                                    identifier_.c_str(), ranktableCrc_));

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllGatherV(const std::string &tag, const void *sendBuf, u64 sendCount, const void *recvBuf,
                                            const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclRtStream stream)
    {
        bool aicpuUnfoldMode = false;

        if (GetExternalInputHcclAicpuUnfold() && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][AllGatherV]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = sendCount * perDataSize;

        u64 outputSize = 0;
        const u64 *counts = static_cast<const u64 *>(recvCounts);
        for (u32 i = 0; i < userRankSize_; i++) {
            outputSize += counts[i] * perDataSize;
        }

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = const_cast<void *>(sendBuf);
        opParam.inputSize = totalSize;
        opParam.outputPtr = const_cast<void *>(recvBuf);
        opParam.outputSize = outputSize;
        opParam.VDataDes.dataType = dataType;
        opParam.VDataDes.counts = const_cast<void *>(recvCounts);
        opParam.VDataDes.displs = const_cast<void *>(rdispls);
        opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
        opParam.stream = streamObj;
        opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER_V;

        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLGATHER_V,
                                                                    tag, recvCounts, rdispls, userRankSize_, dataType, cclBufferManager_.GetInCCLbufferSize(),
                                                                    cclBufferManager_.GetInCCLbufferSize(), identifier_.c_str(), ranktableCrc_));
        if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
            for (u32 i = 0; i < userRankSize_; i++) {
                HCCL_CONFIG_DEBUG(HCCL_ALG,
                                  "[HcclCommunicator][AllGatherV]userRank_[%u], rankIdx[%u], recvCounts[%llu], rdispls[%llu]",
                                  userRank_, i, counts[i], static_cast<const u64 *>(rdispls)[i]);
            }
        }

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                             HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcclCMDType cmdType)
    {
        Stream streamObj(stream);
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;
        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.stream = streamObj;
        opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
        AlgType algType;
        algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
        algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        CHK_RET(RegisterDfxInfo(opParam, algType));
        HcclResult ret = HCCL_SUCCESS;
        if (!IsExistCommRes(identifier_)) {
            HCCL_INFO("[AicpuUnfold] tag[%s] count[%llu] dataType[%s] op[%s].", identifier_.c_str(),
                      count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
            uint64_t streamMode = 0;
            CHK_RET(hrtStreamGetMode(stream, &streamMode));

            rtStream_t aicpuStream;
            ret = Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream);
            void *commContext = nullptr;
            ret = CreateCommResource(identifier_, stream, true, &commContext);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[hcclImpl][CreateComm]create aicpu unfold comminfo by tag[%s] failed. return[%d]",
                           identifier_.c_str(), ret);
                return ret;
            }
        }

        std::string kernelName = "RunAicpuRpcSrvLaunch";
        AicpuOpTiling opTilingInfo;
        ret = AicpuKfcTilingDataLaunch(opParam, cmdType, commContext_, kernelName, opTilingInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[hcclImpl][TilingData]aicpu unfold tiling data launch failed. return[%d] inputPtr[%p]"
                       "outputPtr[%p] count[%llu] dataType[%s] op[%s]",
                       ret, inputPtr, outputPtr, count,
                       GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
            return ret;
        }
        CHK_RET(UnRegisterDfxInfo(opParam));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
                                                   u64 inputCount, HcclDataType dataType, HcclRtStream stream)
    {
        CHK_RET(CheckSuspendingStatus());
        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][AllGatherOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        bool isCapture = StreamIsCapture(stream);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = inputCount * perDataSize * userRankSize_;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = inputCount * perDataSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = inputCount;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
        opParam.stream = streamObj;
        opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.isCapture = isCapture;
        opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLGATHER,
                                                                    tag, inputCount, dataType, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
                                                                    identifier_.c_str(), ranktableCrc_));

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllGatherVOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
                                                    u64 inputCount, const void *outputCounts, const void *outputDispls, HcclDataType dataType, HcclRtStream stream)
    {
        CHK_RET(CheckSuspendingStatus());
        if (userRankSize_ == 1) {
            // rankSize为1时，退化为AllGather
            return AllGather(tag, inputPtr, outputPtr, inputCount, dataType, stream);
        }

        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][AllGatherVOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        bool isCapture = StreamIsCapture(stream);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 outputSize = 0;
        const u64 *counts = static_cast<const u64 *>(outputCounts);
        for (u32 i = 0; i < userRankSize_; i++) {
            outputSize += counts[i] * perDataSize;
        }

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = inputCount * perDataSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = outputSize;
        opParam.VDataDes.counts = const_cast<void *>(outputCounts);
        opParam.VDataDes.displs = const_cast<void *>(outputDispls);
        opParam.VDataDes.dataType = dataType;
        opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
        opParam.stream = streamObj;
        opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.isCapture = isCapture;

        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLGATHER_V,
                                                                    tag, outputCounts, outputDispls, userRankSize_, dataType, cclBufferManager_.GetInCCLbufferSize(),
                                                                    cclBufferManager_.GetInCCLbufferSize(), identifier_.c_str(), ranktableCrc_));

        if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
            for (u32 i = 0; i < userRankSize_; i++) {
                HCCL_CONFIG_DEBUG(HCCL_ALG,
                                  "[HcclCommunicator][AllGatherVOutPlace]userRank_[%u], rankIdx[%u],"
                                  "outputCounts[%llu], outputDispls[%llu]",
                                  userRank_, i, counts[i], static_cast<const u64 *>(outputDispls)[i]);
            }
        }

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::GetAndSetSyncMode(SyncMode &preSyncMode, SyncMode newSyncMode)
    {

        if (newSyncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE) {
            if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
                HCCL_WARNING("310P don't support unlimited notify wait mode");
            } else {
                HcclGetNotifyWaitMode(dispatcher_, &preSyncMode);
                HcclSetNotifyWaitMode(dispatcher_, newSyncMode);
            }
        }
    }

    void HcclCommunicator::RestorePreSyncMode(SyncMode preSyncMode, SyncMode newSyncMode)
    {
        if (newSyncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE && !Is310P3Common(isHaveCpuRank_, deviceType_)) {
            HcclSetNotifyWaitMode(dispatcher_, preSyncMode);
        }
    }

    HcclResult HcclCommunicator::AllReduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                           HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
                                           SyncMode syncMode, const HcomCollOpInfo *opInfo)
    {
        CHK_RET(CheckSuspendingStatus());
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true &&
            IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) &&
            deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][AllReduce]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        // 设置notify wait模式
        SyncMode preSyncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
        GetAndSetSyncMode(preSyncMode, syncMode);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.stream = streamObj;
        opParam.syncMode = syncMode;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
        // 用于inplace支持重执行场景的图模式归一至单算子模式
        retryOrigWorkflowMode_ = GetWorkflowMode();
        bool isHcclOpInplace = IsHcclOpInplace(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, userRank_, userRankSize_,
                                               isInplaceStatus_);
        if (aicpuUnfoldMode && retryEnable_ && isHcclOpInplace) {
            HCCL_DEBUG("The retry with inplace case is expected to be supported, "
                       "aicpuUnfoldMode[%d], retryEnable_[%d], isHcclOpInplace[%d], "
                       "therefore HcclWorkflowMode is converted from [%d] to HCCL_WORKFLOW_MODE_OP_BASE",
                       aicpuUnfoldMode, retryEnable_, isHcclOpInplace, static_cast<u8>(retryOrigWorkflowMode_));
            CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
        }

        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLREDUCE,
                                                                    tag, count, dataType, op, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(),
                                                                    identifier_.c_str(), ranktableCrc_));

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        RestorePreSyncMode(preSyncMode, syncMode);
        CHK_RET(SetWorkflowMode(retryOrigWorkflowMode_));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllReduceAicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                                      HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    {
        Stream streamObj(stream);
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;
        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.stream = streamObj;
        opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
        AlgType algType;
        algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
        algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        CHK_RET(RegisterDfxInfo(opParam, algType));
        HcclResult ret;
        if (!IsExistCommRes(tag)) {
            uint64_t streamMode = 0;
            CHK_RET(hrtStreamGetMode(stream, &streamMode));

            rtStream_t aicpuStream;
            ret = Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream);
            void *commContext = nullptr;
            ret = CreateCommResource(tag, stream, true, &commContext);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[hcclImpl][CreateComm]create aicpu unfold comminfo by tag[%s] failed. return[%d]",
                           tag.c_str(), ret);
                return ret;
            }
        }

        AicpuOpTiling opTilingInfo;
        std::string kernelName = "RunAicpuRpcSrvLaunch";
        ret = AicpuKfcTilingDataLaunch(opParam, HcclCMDType::HCCL_CMD_ALLREDUCE, commContext_, kernelName, opTilingInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[hcclImpl][TilingData]aicpu unfold tiling data launch failed. return[%d] inputPtr[%p]"
                       "outputPtr[%p] count[%llu] dataType[%s] op[%s]",
                       ret, inputPtr, outputPtr, count,
                       GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
            return ret;
        }
        CHK_RET(UnRegisterDfxInfo(opParam));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                                   HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
                                                   SyncMode syncMode)
    {
        CHK_RET(CheckSuspendingStatus());
        const u32 RANK_SIZE_TWO = 2;
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true &&
            IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op)) {
            if (userRankSize_ >= RANK_SIZE_TWO && Is310P3Common(isHaveCpuRank_, deviceType_)) {
                HcclResult ret = AllReduceAicpuUnfold(tag, inputPtr, outputPtr, count, dataType, op, stream);
                CHK_PRT_RET((ret != HCCL_SUCCESS),
                            HCCL_ERROR("[HcclCommunicator][AllReduce]errNo[0x%016llx]  tag[%s],all reduce aicpu unfold failed",
                                       HCCL_ERROR_CODE(ret), tag.c_str()),
                            ret);

                return HCCL_SUCCESS;
            }
            if ((deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
                aicpuUnfoldMode = true;
            }
        }

        bool isCapture = StreamIsCapture(stream);

        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][AllReduceOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        // 设置notify wait模式
        SyncMode preSyncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
        GetAndSetSyncMode(preSyncMode, syncMode);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.stream = streamObj;
        opParam.syncMode = syncMode;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.isCapture = isCapture;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLREDUCE,
                                                                    tag, count, dataType, op, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
                                                                    identifier_.c_str(), ranktableCrc_));
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        RestorePreSyncMode(preSyncMode, syncMode);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
                                           HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                                           rtStream_t stream, const std::string &tag)
    {
        CHK_RET(CheckSuspendingStatus());

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][AlltoAllV]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        if (IsNeedNicInit()) {
            HCCL_INFO("InitNic.");
            CHK_RET(InitNic());
        }

        bool isCapture = StreamIsCapture(stream);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = const_cast<void *>(sendBuf);
        opParam.outputPtr = const_cast<void *>(recvBuf);
        opParam.All2AllDataDes.sendType = sendType;
        opParam.All2AllDataDes.recvType = recvType;
        opParam.All2AllDataDes.sendCounts = const_cast<void *>(sendCounts);
        opParam.All2AllDataDes.recvCounts = const_cast<void *>(recvCounts);
        opParam.All2AllDataDes.sdispls = const_cast<void *>(sdispls);
        opParam.All2AllDataDes.rdispls = const_cast<void *>(rdispls);
        opParam.stream = streamObj;
        opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
        opParam.aicpuUnfoldMode = deviceType_ == DevType::DEV_TYPE_910_93 && GetExternalInputHcclAicpuUnfold();
        opParam.isCapture = isCapture;

        if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
            for (u32 i = 0; i < userRankSize_; i++) {
                HCCL_CONFIG_INFO(HCCL_ALG, "[HcclCommunicator][AlltoAllV] rank[%u], sendCounts[%llu], sendDispls[%llu] "
                                           "recvCounts[%llu], recvDispls[%llu]",
                                 userRank_,
                                 *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCounts) + i),
                                 *(static_cast<const u64 *>(opParam.All2AllDataDes.sdispls) + i),
                                 *(static_cast<const u64 *>(opParam.All2AllDataDes.recvCounts) + i),
                                 *(static_cast<const u64 *>(opParam.All2AllDataDes.rdispls) + i));
            }
        }

        CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLV, opParam));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls,
                                                   HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                                                   rtStream_t stream, const std::string &tag)
    {
        CHK_RET(CheckSuspendingStatus());
        CHK_PRT_RET(Is310P3Common(isHaveCpuRank_, deviceType_),
                    HCCL_RUN_INFO("[AlltoAllVOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);
        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][AlltoAllVOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        if (IsNeedNicInit()) {
            HCCL_INFO("InitNic.");
            CHK_RET(InitNic());
        }

        bool isCapture = StreamIsCapture(stream);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = const_cast<void *>(sendBuf);
        opParam.outputPtr = const_cast<void *>(recvBuf);
        opParam.All2AllDataDes.sendType = sendType;
        opParam.All2AllDataDes.recvType = recvType;
        opParam.All2AllDataDes.sendCounts = const_cast<void *>(sendCounts);
        opParam.All2AllDataDes.recvCounts = const_cast<void *>(recvCounts);
        opParam.All2AllDataDes.sdispls = const_cast<void *>(sdispls);
        opParam.All2AllDataDes.rdispls = const_cast<void *>(rdispls);
        opParam.stream = streamObj;
        opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
        opParam.aicpuUnfoldMode = deviceType_ == DevType::DEV_TYPE_910_93 && GetExternalInputHcclAicpuUnfold();
        opParam.isCapture = isCapture;

        if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
            for (u32 i = 0; i < userRankSize_; i++) {
                HCCL_CONFIG_INFO(HCCL_ALG, "[HcclCommunicator][AlltoAllVOutPlace] rank[%u], sendCounts[%llu],"
                                           "sendDispls[%llu], recvCounts[%llu], recvDispls[%llu]",
                                 userRank_,
                                 *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCounts) + i),
                                 *(static_cast<const u64 *>(opParam.All2AllDataDes.sdispls) + i),
                                 *(static_cast<const u64 *>(opParam.All2AllDataDes.recvCounts) + i),
                                 *(static_cast<const u64 *>(opParam.All2AllDataDes.rdispls) + i));
            }
        }

        CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLV, opParam));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
                                            const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
    {
        CHK_RET(CheckSuspendingStatus());
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
                        vector<string>({"310P", std::string(__func__) + " is not supported"}));
            HCCL_ERROR("[HcclCommunicator][AlltoAllVC]AlltoAllVC is not supported");
            return HCCL_E_NOT_SUPPORT;
        }
        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][AlltoAllVC]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        if (IsNeedNicInit()) {
            HCCL_INFO("InitNic.");
            CHK_RET(InitNic());
        }

        bool isCapture = StreamIsCapture(stream);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = const_cast<void *>(sendBuf);
        opParam.outputPtr = const_cast<void *>(recvBuf);
        opParam.All2AllDataDes.sendType = sendType;
        opParam.All2AllDataDes.recvType = recvType;
        opParam.All2AllDataDes.sendCountMatrix = const_cast<void *>(sendCountMatrix);
        opParam.stream = streamObj;
        opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
        opParam.aicpuUnfoldMode = deviceType_ == DevType::DEV_TYPE_910_93 && GetExternalInputHcclAicpuUnfold();
        opParam.isCapture = isCapture;

        if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
            for (u32 i = 0; i < userRankSize_; i++) {
                for (u32 j = 0; j < userRankSize_; j++) {
                    HCCL_CONFIG_DEBUG(HCCL_ALG, "[HcclCommunicator][AlltoAllVC] usrRank[%u] rank[%u] to remoteRank[%u], "
                                                "sendCounts[%llu]",
                                      userRank_, i, j,
                                      *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCountMatrix) + i * userRankSize_ + j));
                }
            }
        }

        CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLVC, opParam));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AlltoAllVCOutPlace(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
                                                    const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
    {
        CHK_RET(CheckSuspendingStatus());
        CHK_PRT_RET(Is310P3Common(isHaveCpuRank_, deviceType_),
                    HCCL_RUN_INFO("[AlltoAllVCOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][AlltoAllVCOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        if (IsNeedNicInit()) {
            HCCL_INFO("InitNic");
            CHK_RET(InitNic());
        }

        bool isCapture = StreamIsCapture(stream);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = const_cast<void *>(sendBuf);
        opParam.outputPtr = const_cast<void *>(recvBuf);
        opParam.All2AllDataDes.sendType = sendType;
        opParam.All2AllDataDes.recvType = recvType;
        opParam.All2AllDataDes.sendCountMatrix = const_cast<void *>(sendCountMatrix);
        opParam.stream = streamObj;
        opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
        opParam.aicpuUnfoldMode = deviceType_ == DevType::DEV_TYPE_910_93 && GetExternalInputHcclAicpuUnfold();
        opParam.isCapture = isCapture;

        if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
            for (u32 i = 0; i < userRankSize_; i++) {
                for (u32 j = 0; j < userRankSize_; j++) {
                    HCCL_CONFIG_DEBUG(HCCL_ALG, "[HcclCommunicator][AlltoAllVCOutPlace] usrRank[%u] rank[%u]"
                                                "to remoteRank[%u], sendCounts[%llu]",
                                      userRank_, i, j,
                                      *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCountMatrix) + i * userRankSize_ + j));
                }
            }
        }

        CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLVC, opParam));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AlltoAll(const void *sendBuf, u64 sendCount, HcclDataType sendType,
                                          const void *recvBuf, u64 recvCount, HcclDataType recvType, rtStream_t stream, const std::string &tag)
    {
        CHK_RET(CheckSuspendingStatus());
        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][AlltoAll]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        if (IsNeedNicInit()) {
            HCCL_INFO("InitNic.");
            CHK_RET(InitNic());
        }

        bool isCapture = StreamIsCapture(stream);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
        std::vector<u64> sendCountMatrix(userRankSize_ * userRankSize_, sendCount);

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = const_cast<void *>(sendBuf);
        opParam.outputPtr = const_cast<void *>(recvBuf);
        opParam.All2AllDataDes.sendType = sendType;
        opParam.All2AllDataDes.recvType = recvType;
        opParam.All2AllDataDes.sendCount = sendCount;
        opParam.All2AllDataDes.sendCountMatrix = static_cast<void *>(sendCountMatrix.data());
        opParam.stream = streamObj;
        opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALL;
        opParam.aicpuUnfoldMode = false;
        opParam.isCapture = isCapture;
        if (deviceType_ == DevType::DEV_TYPE_910_93) {
            opParam.aicpuUnfoldMode = GetExternalInputHcclAicpuUnfold();
        }

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);
        CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALL, opParam));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
                                           HcclRtStream stream)
    {
        CHK_RET(CheckSuspendingStatus());
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true && deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][Broadcast]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        if (isHaveCpuRank_ && !isSetHDCModeInfo_ && isServerInter_) {
            isSetHDCModeInfo_ = true;
        }
        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);
        if (isHaveCpuRank_) {
            CHK_RET(implAlg_->Broadcast(tag, ptr, count, dataType, root, streamObj));
        } else {
            u32 perDataSize = SIZE_TABLE[dataType];
            u64 totalSize = count * perDataSize;

            OpParam opParam;
            opParam.tag = tag;
            opParam.inputPtr = ptr;
            opParam.outputPtr = ptr;
            opParam.inputSize = totalSize;
            opParam.outputSize = totalSize;
            opParam.DataDes.count = count;
            opParam.DataDes.dataType = dataType;
            opParam.root = root;
            opParam.stream = streamObj;
            opParam.aicpuUnfoldMode = aicpuUnfoldMode;
            opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
            opParam.opType = HcclCMDType::HCCL_CMD_BROADCAST;

            CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_BROADCAST, opParam));
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BroadcastOutPlace(const std::string &tag, void *ptr, u64 count, HcclDataType dataType,
                                                   u32 root, HcclRtStream stream)
    {
        CHK_RET(CheckSuspendingStatus());
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true && deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        CHK_PRT_RET(Is310P3Common(isHaveCpuRank_, deviceType_),
                    HCCL_RUN_INFO("[BroadcastOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][BroadcastOutPlace]errNo[0x%016llx] hccl init must be called before"
                       " call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        bool isCapture = StreamIsCapture(stream);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        if (isHaveCpuRank_) {
            CHK_RET(implAlg_->Broadcast(tag, ptr, count, dataType, root, streamObj));
        } else {
            u32 perDataSize = SIZE_TABLE[dataType];
            u64 totalSize = count * perDataSize;

            OpParam opParam;
            opParam.tag = tag;
            opParam.inputPtr = ptr;
            opParam.outputPtr = ptr;
            opParam.inputSize = totalSize;
            opParam.outputSize = totalSize;
            opParam.DataDes.count = count;
            opParam.DataDes.dataType = dataType;
            opParam.root = root;
            opParam.stream = streamObj;
            opParam.aicpuUnfoldMode = aicpuUnfoldMode;
            opParam.isCapture = isCapture;
            opParam.opType = HcclCMDType::HCCL_CMD_BROADCAST;

            // 记录指令信息用于一致性校验
            CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_BROADCAST, tag, count,
                                                                        dataType, root, cclBufferManager_.GetInCCLbufferSize(), 0, identifier_.c_str(), ranktableCrc_));

            CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_BROADCAST, opParam));

            // 移除tag对应的指令信息
            CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Scatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
                                         HcclDataType dataType, u32 root, HcclRtStream stream)
    {
        CHK_RET(CheckSuspendingStatus());
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
                        vector<string>({"310P", std::string(__func__) + " is not supported"}));
            HCCL_ERROR("[HcclCommunicator][Scatter]Scatter Not Supported Yet");
            return HCCL_E_NOT_SUPPORT;
        }
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true && deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][Scatter]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 outputSize = recvCount * perDataSize;
        u64 totalSize = outputSize * userRankSize_;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = recvCount;
        opParam.DataDes.dataType = dataType;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.root = root;
        opParam.opType = HcclCMDType::HCCL_CMD_SCATTER;
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SCATTER, opParam));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
                                                 HcclDataType dataType, u32 root, HcclRtStream stream)
    {
        CHK_RET(CheckSuspendingStatus());
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
                        vector<string>({"310P", std::string(__func__) + " is not supported"}));
            HCCL_ERROR("[HcclCommunicator][ScatterOutPlace]ScatterOutPlace Not Supported Yet");
            return HCCL_E_NOT_SUPPORT;
        }

        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        bool isCapture = StreamIsCapture(stream);

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][ScatterOutPlace]errNo[0x%016llx] hccl init must be called before"
                       " call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 outputSize = recvCount * perDataSize;
        u64 totalSize = outputSize * userRankSize_;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = recvCount;
        opParam.DataDes.dataType = dataType;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.isCapture = isCapture;
        opParam.root = root;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.opType = HcclCMDType::HCCL_CMD_SCATTER;
        /* 记录指令信息用于一致性校验 */
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_SCATTER, tag,
                                                                    recvCount, dataType, root, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
                                                                    identifier_.c_str(), ranktableCrc_));

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SCATTER, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                        HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
    {
        CHK_RET(CheckSuspendingStatus());
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
                        vector<string>({"310P", std::string(__func__) + " is not supported"}));
            HCCL_ERROR("[HcclCommunicator][Reduce]Reduce Not Supported Yet");
            return HCCL_E_NOT_SUPPORT;
        }
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true &&
            IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][Reduce]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;
        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.root = root;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opType = HcclCMDType::HCCL_CMD_REDUCE;

        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE,
                                                                    tag, count, dataType, op, root, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(),
                                                                    identifier_.c_str(), ranktableCrc_));

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                                HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
    {
        CHK_RET(CheckSuspendingStatus());
        CHK_PRT_RET(Is310P3Common(isHaveCpuRank_, deviceType_),
                    HCCL_RUN_INFO("[ReduceOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true &&
            IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        bool isCapture = StreamIsCapture(stream);

        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][ReduceOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;
        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.root = root;
        opParam.stream = streamObj;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.isCapture = isCapture;
        opParam.opType = HcclCMDType::HCCL_CMD_REDUCE;
        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE,
                                                                    tag, count, dataType, op, root, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
                                                                    identifier_.c_str(), ranktableCrc_));

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReduceScatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                               HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcomCollOpInfo *opInfo)
    {
        CHK_RET(CheckSuspendingStatus());
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true &&
            IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][ReduceScatter]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        u32 perDataSize = SIZE_TABLE[dataType];

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = userRankSize_ * count * perDataSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = count * perDataSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.stream = streamObj;
        opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        // 用于inplace支持重执行场景的图模式归一至单算子模式
        retryOrigWorkflowMode_ = GetWorkflowMode();
        bool isHcclOpInplace = IsHcclOpInplace(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, userRank_, userRankSize_,
                                               isInplaceStatus_);
        if (aicpuUnfoldMode && retryEnable_ && isHcclOpInplace) {
            HCCL_DEBUG("The retry with inplace case is expected to be supported, "
                       "aicpuUnfoldMode[%d], retryEnable_[%d], isHcclOpInplace[%d], "
                       "therefore HcclWorkflowMode is converted from [%d] to HCCL_WORKFLOW_MODE_OP_BASE",
                       aicpuUnfoldMode, retryEnable_, isHcclOpInplace, static_cast<u8>(retryOrigWorkflowMode_));
            CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
        }
        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, tag,
                                                                    count, dataType, op, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(),
                                                                    identifier_.c_str(), ranktableCrc_));

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        CHK_RET(SetWorkflowMode(retryOrigWorkflowMode_));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
                                                       u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    {
        CHK_RET(CheckSuspendingStatus());
        if (userRankSize_ > 1) {
            CHK_RET(CreateCommCCLbuffer());
            CHK_RET(CreateCommExpBuffer());
        }

        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true &&
            IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(),
                                dataType, op) &&
            (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        bool isCapture = StreamIsCapture(stream);

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][ReduceScatterOutPlace]errNo[0x%016llx] hccl init must be called before"
                       " call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        u32 perDataSize = SIZE_TABLE[dataType];

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = userRankSize_ * count * perDataSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = count * perDataSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.stream = streamObj;
        opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.isCapture = isCapture;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();

        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, tag,
                                                                    count, dataType, op, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
                                                                    identifier_.c_str(), ranktableCrc_));

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReduceScatterV(const std::string &tag, void *inputPtr,
                                                const void *inputCounts, const void *inputDispls, void *outputPtr, u64 outputCount,
                                                HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcomCollOpInfo *opInfo)
    {
        CHK_RET(CheckSuspendingStatus());
        if (userRankSize_ == 1) {
            // rankSize为1时，退化为ReduceScatter
            return ReduceScatter(tag, inputPtr, outputPtr, outputCount, dataType, op, stream);
        }

        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][ReduceScatterV]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        const bool aicpuUnfoldMode = GetExternalInputHcclAicpuUnfold() &&
                                     IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_93);

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 inputSize = 0;
        const u64 *counts = static_cast<const u64 *>(inputCounts);
        for (u32 i = 0; i < userRankSize_; i++) {
            inputSize += counts[i] * perDataSize;
        }

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = inputSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = outputCount * perDataSize;
        opParam.srcRank = userRank_; // rankId for access counts
        opParam.VDataDes.counts = const_cast<void *>(inputCounts);
        opParam.VDataDes.displs = const_cast<void *>(inputDispls);
        opParam.VDataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.stream = streamObj;
        opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;

        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, tag,
                                                                    inputCounts, inputDispls, userRankSize_, dataType, op, cclBufferManager_.GetInCCLbufferSize(),
                                                                    cclBufferManager_.GetInCCLbufferSize(), identifier_.c_str(), ranktableCrc_));

        if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
            for (u32 i = 0; i < userRankSize_; i++) {
                HCCL_CONFIG_DEBUG(HCCL_ALG,
                                  "[HcclCommunicator][ReduceScatterV]userRank_[%u], rankIdx[%u], inputCounts[%llu], inputDispls[%llu]",
                                  userRank_, i, counts[i], static_cast<const u64 *>(inputDispls)[i]);
            }
        }

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReduceScatterVOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
                                                        const void *inputCounts, const void *inputDispls, u64 outputCount,
                                                        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    {
        CHK_RET(CheckSuspendingStatus());
        if (userRankSize_ == 1) {
            // rankSize为1时，退化为ReduceScatter
            return ReduceScatterOutPlace(tag, inputPtr, outputPtr, outputCount, dataType, op, stream);
        }

        CHK_RET(CreateCommCCLbuffer());
        CHK_RET(CreateCommExpBuffer());
        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][ReduceScatterVOutPlace]errNo[0x%016llx] hccl init must be called before"
                       " call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        bool isCapture = StreamIsCapture(stream);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        const bool aicpuUnfoldMode = GetExternalInputHcclAicpuUnfold() &&
                                     IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_93);

        u32 perDataSize = SIZE_TABLE[dataType];
        u64 inputSize = 0;
        const u64 *counts = static_cast<const u64 *>(inputCounts);
        for (u32 i = 0; i < userRankSize_; i++) {
            inputSize += counts[i] * perDataSize;
        }

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = inputSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = outputCount * perDataSize;
        opParam.srcRank = userRank_; // rankId for access counts
        opParam.VDataDes.counts = const_cast<void *>(inputCounts);
        opParam.VDataDes.displs = const_cast<void *>(inputDispls);
        opParam.VDataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.stream = streamObj;
        opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.isCapture = isCapture;

        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, tag,
                                                                    inputCounts, inputDispls, userRankSize_, dataType, op, cclBufferManager_.GetInCCLbufferSize(),
                                                                    cclBufferManager_.GetInCCLbufferSize(), identifier_.c_str(), ranktableCrc_));

        if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
            for (u32 i = 0; i < userRankSize_; i++) {
                HCCL_CONFIG_DEBUG(HCCL_ALG,
                                  "[HcclCommunicator][ReduceScatterVOutPlace]userRank_[%u],"
                                  "rankIdx[%u], inputCounts[%llu], inputDispls[%llu]",
                                  userRank_, i, counts[i], static_cast<const u64 *>(inputDispls)[i]);
            }
        }

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BatchSendRecv(const std::string &tag, HcclSendRecvItem *sendRecvItemsPtr, u32 itemNum,
                                               rtStream_t stream)
    {
        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][BatchSendRecv]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        bool isCapture = StreamIsCapture(stream);

        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][BatchSendRecv]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }
        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);
        OpParam opParam;
        opParam.tag = tag;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.isCapture = isCapture;
        opParam.BatchSendRecvDataDes.sendRecvItemsPtr = sendRecvItemsPtr;
        opParam.BatchSendRecvDataDes.itemNum = itemNum;
        opParam.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV,
                                                                    tag, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
                                                                    identifier_.c_str(), ranktableCrc_));

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
                                      u32 destRank, rtStream_t stream)
    {
        CHK_RET(CheckSuspendingStatus());
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][Send]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        if (isHaveCpuRank_) {
            CHK_RET(implAlg_->Send(tag, inputPtr, count, dataType, destRank, streamObj));
        } else {
            u32 perDataSize = SIZE_TABLE[dataType];
            u64 totalSize = count * perDataSize;

            OpParam opParam;
            opParam.tag = tag;
            opParam.inputPtr = inputPtr;
            opParam.inputSize = totalSize;
            opParam.outputPtr = inputPtr;
            opParam.outputSize = totalSize;
            opParam.DataDes.count = count;
            opParam.DataDes.dataType = dataType;
            opParam.stream = streamObj;
            opParam.aicpuUnfoldMode = aicpuUnfoldMode;
            opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
            opParam.dstRank = destRank;
            opParam.opType = HcclCMDType::HCCL_CMD_SEND;
            CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SEND, opParam));
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
                                              u32 destRank, rtStream_t stream)
    {
        CHK_RET(CheckSuspendingStatus());
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
                        vector<string>({"310P", std::string(__func__) + " is not supported"}));
            HCCL_ERROR("[HcclCommunicator][SendOutPlace]SendOutPlace is not supported");
            return HCCL_E_NOT_SUPPORT;
        }
        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][SendOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        bool isCapture = StreamIsCapture(stream);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        // 记录指令信息用于一致性校验
        HcclResult ret = RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_SEND, tag, count,
                                                                             dataType, cclBufferManager_.GetInCCLbufferSize(), 0, identifier_.c_str(), ranktableCrc_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("errNo[0x%016llx] record CMD with parameter error", HCCL_ERROR_CODE(ret)), ret);

        if (isHaveCpuRank_) {
            CHK_RET(implAlg_->SendOutPlace(tag, inputPtr, count, dataType, destRank, streamObj));
        } else {
            u32 perDataSize = SIZE_TABLE[dataType];
            u64 totalSize = count * perDataSize;

            OpParam opParam;
            opParam.tag = tag;
            opParam.inputPtr = inputPtr;
            opParam.inputSize = totalSize;
            opParam.outputPtr = inputPtr;
            opParam.outputSize = totalSize;
            opParam.DataDes.count = count;
            opParam.DataDes.dataType = dataType;
            opParam.stream = streamObj;
            opParam.aicpuUnfoldMode = aicpuUnfoldMode;
            opParam.isCapture = isCapture;
            opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
            opParam.dstRank = destRank;
            opParam.opType = HcclCMDType::HCCL_CMD_SEND;
            CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SEND, opParam));
        }

        // 移除tag对应的指令信息
        ret = RankConsistentcyChecker::GetInstance().DelOpPara(tag);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("errNo[0x%016llx] delete CMD with parameters error. tag[%s]", HCCL_ERROR_CODE(ret),
                               tag.c_str()),
                    ret);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
                                         u32 srcRank, rtStream_t stream)
    {
        CHK_RET(CheckSuspendingStatus());
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][Receive]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        if (isHaveCpuRank_) {
            CHK_RET(implAlg_->Receive(tag, outputPtr, count, dataType, srcRank, streamObj));
        } else {
            u32 perDataSize = SIZE_TABLE[dataType];
            u64 totalSize = count * perDataSize;

            OpParam opParam;
            opParam.tag = tag;
            opParam.inputPtr = outputPtr;
            opParam.inputSize = totalSize;
            opParam.outputPtr = outputPtr;
            opParam.outputSize = totalSize;
            opParam.DataDes.count = count;
            opParam.DataDes.dataType = dataType;
            opParam.stream = streamObj;
            opParam.aicpuUnfoldMode = aicpuUnfoldMode;
            opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
            opParam.srcRank = srcRank;
            opParam.opType = HcclCMDType::HCCL_CMD_RECEIVE;
            CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_RECEIVE, opParam));
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count,
                                                 HcclDataType dataType, u32 srcRank, rtStream_t stream)
    {
        CHK_RET(CheckSuspendingStatus());
        bool aicpuUnfoldMode = false;
        if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }

        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
                        vector<string>({"310P", std::string(__func__) + " is not supported"}));
            HCCL_ERROR("[HcclCommunicator][ReceiveOutPlace]ReceiveOutPlace is not supported");
            return HCCL_E_NOT_SUPPORT;
        }
        if (!IsAtomicInit()) {
            HCCL_ERROR(
                "[HcclCommunicator][ReceiveOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
                HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        bool isCapture = StreamIsCapture(stream);

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

        // 记录指令信息用于一致性校验
        HcclResult ret = RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_RECEIVE, tag, count,
                                                                             dataType, cclBufferManager_.GetInCCLbufferSize(), 0, identifier_.c_str(), ranktableCrc_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("errNo[0x%016llx] record CMD with parameter error", HCCL_ERROR_CODE(ret)), ret);

        if (isHaveCpuRank_) {
            CHK_RET(implAlg_->ReceiveOutPlace(tag, outputPtr, count, dataType, srcRank, streamObj));
        } else {
            u32 perDataSize = SIZE_TABLE[dataType];
            u64 totalSize = count * perDataSize;

            OpParam opParam;
            opParam.tag = tag;
            opParam.inputPtr = outputPtr;
            opParam.inputSize = totalSize;
            opParam.outputPtr = outputPtr;
            opParam.outputSize = totalSize;
            opParam.DataDes.count = count;
            opParam.DataDes.dataType = dataType;
            opParam.stream = streamObj;
            opParam.aicpuUnfoldMode = aicpuUnfoldMode;
            opParam.isCapture = isCapture;
            opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
            opParam.srcRank = srcRank;
            opParam.opType = HcclCMDType::HCCL_CMD_RECEIVE;
            CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_RECEIVE, opParam));
        }

        // 移除tag对应的指令信息
        ret = RankConsistentcyChecker::GetInstance().DelOpPara(tag);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("errNo[0x%016llx] delete CMD with parameters error. tag[%s]", HCCL_ERROR_CODE(ret),
                               tag.c_str()),
                    ret);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank,
                                        u64 inputCount, HcclDataType dataType, rtStream_t stream)
    {
        CHK_RET(CheckSuspendingStatus());
        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][Gather]errNo[0x%016llx] hccl init must be called before call this function",
                       HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }

        Stream streamObj(stream);
        CHK_RET(callbackTask_->CallbackRegStream(stream));

        if (isHaveCpuRank_ && !isSetHDCModeInfo_ && isServerInter_) {
            isSetHDCModeInfo_ = true;
        }
        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);
        CHK_RET(implAlg_->Gather(tag, inputPtr, outputPtr, rootRank, inputCount, dataType, streamObj));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::RegressCalPreOp(AlltoAllOperator *&alltoAllOperator, const OpParam &opParam,
                                                 std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
    {
        HCCL_INFO("Run with Graph, alloc new stream");
        Stream stream(StreamType::STREAM_TYPE_ONLINE);
        return RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo, stream);
    }

    HcclResult HcclCommunicator::RegressCalPreOp(AlltoAllOperator *&alltoAllOperator, const OpParam &opParam,
                                                 std::unique_ptr<PreProcessMetaInfo> &preMetaInfo, Stream &preProcessStream)
    {
        OpParam preProcessOpParam;
        HcclWorkflowMode mode = GetWorkflowMode();
        CHK_PRT_RET(mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED, HCCL_ERROR("Invalid Workflow Mode[%d]", mode), HCCL_E_INTERNAL);

        // h to d
        CHK_RET(SetInfoToDevice(opParam, preMetaInfo, mode, preProcessStream));
        // opParam准备
        CHK_RET(alltoAllOperator->PreparePreOpParam(preProcessOpParam, preMetaInfo, preProcessStream));

        // 回归调用其它算子
        HCCL_INFO("[HcclCommunicator][RegressCalPreOp] Regression calls other operators and opType[%u]",
                  preMetaInfo->opType);
        CHK_RET(ExecOp(preMetaInfo->opType, preProcessOpParam));
        CHK_RET(hcclStreamSynchronize(preProcessStream.ptr()));
        HCCL_DEBUG("[HcclCommunicator][RegressCalPreOp] preProcess tag[%s].", preProcessOpParam.tag.c_str());
        SetWorkflowMode(mode);

        // d to h
        HostMem hostCollectBuffer = HostMem::alloc(preMetaInfo->outputSize);
        CHK_PTR_NULL(hostCollectBuffer.ptr());
        CHK_RET(GetInfoFromDevice(opParam, preMetaInfo, mode, preProcessStream, hostCollectBuffer));

        hostCollectBuffer_ = hostCollectBuffer;
        alltoAllOperator->SetPreProcessResult(std::move(hostCollectBuffer));
        HCCL_INFO("[HcclCommunicator][RegressCalPreOp] run success!");
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ExecOp(HcclCMDType opType, OpParam &opParam, bool isCustom)
    {
        bool isInGraphCaptureZeroCopy = false;
        zeroCopyAclGraph_->SetRetryEnable(retryEnable_);
        isInGraphCaptureZeroCopy = zeroCopyAclGraph_->SetAclGraphZeroCopyMode(
            deviceType_, opType, opParam, implAlg_.get(), cclBufferManager_.GetOutCCLbufferSize());
        if (isInGraphCaptureZeroCopy && userRankSize_ > 1) {
            CHK_RET(CreateCommCCLbuffer());
        }

        std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
        CHK_SMART_PTR_NULL(algOperator);
        // 算法选择
        std::string algName;
        std::string newTag;
        if (opParam.aicpuUnfoldMode) {
            // 用于inplace支持重执行判断
            CHK_RET(algOperator->SetRetryEnable(retryEnable_));
        }
        if (GetExternalInputHcclAivMode()) {
            // 用于判断图模式是否清零
            CHK_RET(algOperator->SetAivClearEnable(aivClearEnable_));
        }

        ResourceLimit limit;
        AlgDesc algDesc;
        opParam.supportZeroCopy = IsSupportZeroCopy(opParam);
        CHK_RET(algOperator->SelectAlg(opParam.tag, opParam, limit, algName, algDesc, newTag));
        CHK_RET(PrepareZeroCopy(algName, algDesc, opParam));

        newTag += !opParam.isCapture ? "" : "_Capture"; // aclgraph使用新的Tag，避免影响其他操作

        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && userRankSize_ > 1) {
            CHK_RET(CreateCommCCLbuffer());
        }
        CHK_RET(CreateCommExpBuffer());
        if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0) {
            NslbDp_CollectOperTable(opType, opParam, algOperator->GetAlgType(), algName);
        }
        // 资源创建
        if ((resMap_.find(newTag) != resMap_.end()) && opParam.isCapture)
        {
            auto resTmp = resMap_[newTag];
            ++captureCnt_;
            newTag += std::to_string(captureCnt_);
            resMap_[newTag] = resTmp;
            AlgResourceRequest resRequest;
            CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
            resRequest.isInGraphCaptureZeroCopy = isInGraphCaptureZeroCopy;
            CHK_RET(CleanTransportLinks(resRequest.opTransport, resMap_[newTag].opTransportResponse));
            if (IsEnableBackupLink()) {
                CHK_RET(CleanTransportLinks(resRequest.opTransport, resMap_[newTag].opTransportResponseBackUp));
            }
            CHK_RET(IncreAllocLink(newTag, opParam, resRequest, resMap_[newTag]));
        }
        InsertNewTagToTagMap(newTag, opParam.tag);
        if (resMap_.find(newTag) == resMap_.end()) {
            AlgResourceRequest resRequest;
            CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
            resRequest.isInGraphCaptureZeroCopy = isInGraphCaptureZeroCopy;
            HcclResult ret = AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag]);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommunicator][ExecOp] AllocAlgResource failed, algName=[%s]", algName.c_str()), ret);

            // 对于91093超节点内aiv跨机通信算子，将不同机的CCLbuffer地址存在约定好的aiv将读取的HBM位置
            CHK_RET(algOperator->PrepareCommInfoToDevice(algName, resMap_[newTag]));

            if (!isHaveCpuRank_) {
                if (isUseRankPort_) {
                    std::vector<u32> &nicPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
                    std::vector<u32> &vnicPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
                    Heartbeat::GetInstance(deviceLogicId_).SetRankPortInfo(isUseRankPort_, nicPorts, vnicPorts, commPortConfig_.devPortSwitchOn);
                }
                // 开始注册心跳
                if (opType == HcclCMDType::HCCL_CMD_SEND) {
                    CHK_RET(RegisterToHeartBeat(opParam.dstRank, newTag));
                    hbSendRecvTags_.emplace(newTag);
                } else if (opType == HcclCMDType::HCCL_CMD_RECEIVE) {
                    CHK_RET(RegisterToHeartBeat(opParam.srcRank, newTag));
                    hbSendRecvTags_.emplace(newTag);
                } else {
                    CHK_RET(RegisterToHeartBeat());
                }
            }
            CHK_RET(UpdateZeroCopy(opParam, resMap_[newTag]));
        } else if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            // batchsendrecv需要根据任务来确定和哪些卡建链，因此复用tag，并在此基础上实现增量建链
            AlgResourceRequest resRequest;
            CHK_RET(algOperator->CalcIncreLinkRequest(algName, opParam, resRequest));
            CHK_RET(IncreAllocLink(newTag, opParam, resRequest, resMap_[newTag]));
        }

        if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0)
        {
            AdjInfo nslbAdjInfo = {};
            CHK_RET(algOperator->GetAdjInfo(algName, opParam, resMap_[newTag], nslbAdjInfo));
            HCCL_RUN_INFO("[NSLBDP-SWK]-nslbAdjInfosize[%u]-algName[%s]-rankSize[%u]-commDesc[%s]..",
                          nslbAdjInfo.dstRankNum, algName.c_str(), userRankSize_, identifier_.c_str());
            NslbDp_CollectSendAdjTable(opType, opParam, algOperator->GetAlgType(), nslbAdjInfo);
        }
        // 算法执行
        bool selectAivAlg = algDesc.isAivMode;
        if (selectAivAlg) {
            HCCL_INFO("[HcclCommunicator][ExecOp] userRank[%u] cur aiv tag [%d]", userRank_, aivTag_);
            opParam.aivTag = aivTag_;
            opParam.aicpuUnfoldMode = false;
            aivTag_ = GetNextAivTag(aivTag_, algDesc.aivTagNum);
            CHK_RET(HandleAclGraphFirstOpAivBuff(opParam.stream.ptr()));
            u32 aivCoreLimit;
            aclError acl_ret = aclrtGetDeviceResLimit(deviceLogicId_, ACL_RT_DEV_RES_VECTOR_CORE, &aivCoreLimit);
            CHK_PRT_RET(acl_ret != ACL_SUCCESS,
                HCCL_ERROR("[HcclCommunicator][ExecOp] aclrtGetDeviceResLimit failed, ret=[%d]", acl_ret),
                HCCL_E_PARA);
            CHK_RET(algOperator->SetBlockDim(aivCoreLimit));
            if (aivClearEnable_) {
                // 用于判断图模式是否清零
                CHK_RET(algOperator->SetAivClearEnable(aivClearEnable_));
            }
        }
        auto algType = algOperator->GetAlgType();
        CHK_RET(RegisterDfxInfo(opParam, algType, resMap_[newTag].slaveStreams, selectAivAlg));
        // 头计数
        CHK_RET(StarsCounter(dispatcher_, opParam.stream, HEAD, opParam.aicpuUnfoldMode, retryEnable_, selectAivAlg));
        if (opParam.aicpuUnfoldMode) {
            isInplaceStatus_ = 0;
            inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::INPLACE_STATUS_END;
            // algOperator->SupportRetryWithInplaceCheck 依赖 algOperator->SetRetryEnable 才能正确返回是否支持inplace

            inplaceSupportRetry_ = algOperator->SupportRetryWithInplaceCheck(
                opType, opParam, algName, isInplaceStatus_, inPlaceSupportRetryStatus_);
            auto algType = algOperator->GetAlgType();
            HCCL_INFO("[HcclCommunicator][ExecOp] aicpu Unfold mode algType[%s], inplaceSupportRetry_[%d], opType[%d], "
                      "isInplaceStatus_[%d], inPlaceSupportRetryStatus_[%d]",
                      AlgTypeToStr(algType).c_str(), inplaceSupportRetry_, opType, isInplaceStatus_, inPlaceSupportRetryStatus_);
            CHK_RET(OrchestrateAicpu(opType, algName, opParam, resMap_[newTag], newTag, algType, isCustom));
        } else {
            // HOST展开aclgraph场景，capture从流
            if (!selectAivAlg) {
                CHK_RET(CaptureSlaveStreams(opParam.stream.ptr(), resMap_[newTag].slaveStreams));
            }
            OpCounterInfo opCounter;
            CHK_RET(GetOpCountInfo(opCounter));
            CHK_RET(algOperator->SetOpCounter(opCounter));
            CHK_RET(algOperator->Orchestrate(algName, opParam, resMap_[newTag]));
            if (hostResMap_.find(newTag) == hostResMap_.end()) {
                hostResMap_.insert(newTag);
            }
            CHK_RET(algOperator->GetBlockDim(blockDim_));
        }
        // 尾计数
        CHK_RET(StarsCounter(dispatcher_, opParam.stream, TAIL, opParam.aicpuUnfoldMode, retryEnable_, selectAivAlg));
        if (selectAivAlg) {
            CHK_RET(algOperator->SetAivClearEnable(false));
            aivClearEnable_ = false;
        }

        if (isInGraphCaptureZeroCopy) {
            SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::FreeScratchMemOnOpBaseMode(DeviceMem &scratchMem, const OpParam &opParam,
                                                            const HcclCMDType &opType)
    {
        // 当前单算子模式下scratch内存为手动申请，需要手动进行释放
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || IsForceAicpuOpBaseMode(opParam, opType)) {
            scratchMem.free();
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ExecOpAlltoAll(HcclCMDType opType, OpParam &opParam, bool isCustom)
    {
        bool isInGraphCaptureZeroCopy = false;
        zeroCopyAclGraph_->SetRetryEnable(retryEnable_);
        isInGraphCaptureZeroCopy = zeroCopyAclGraph_->SetAclGraphZeroCopyMode(
            deviceType_, opType, opParam, implAlg_.get(), cclBufferManager_.GetOutCCLbufferSize());
        if (isInGraphCaptureZeroCopy && userRankSize_ > 1) {
            CHK_RET(CreateCommCCLbuffer());
        }

        std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
        AlltoAllOperator *alltoAllOperator = dynamic_cast<AlltoAllOperator *>(algOperator.get());
        CHK_PTR_NULL(alltoAllOperator);

        // 算法选择
        std::string algName;
        std::string newTag;
        if (opParam.aicpuUnfoldMode) {
            // 用于inplace支持重执行判断
            CHK_RET(algOperator->SetRetryEnable(retryEnable_));
        }
        std::unique_ptr<PreProcessMetaInfo> preMetaInfo = std::make_unique<PreProcessMetaInfo>();
        CHK_SMART_PTR_NULL(preMetaInfo);

        bool preProcessFlag = alltoAllOperator->JudgeIfNeedPreProcessAndGetParam(opParam, preMetaInfo);
        if (preProcessFlag) {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo, const_cast<Stream &>(opParam.stream)));
            } else {
                CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo));
            }
        }

        ResourceLimit limit;
        AlgDesc algDesc;
        CHK_RET(algOperator->SelectAlg(opParam.tag, opParam, limit, algName, algDesc, newTag));

        newTag += !opParam.isCapture ? "" : "_Capture";
        bool supportAicpuAlg = algName == "RunAlltoAllVFullMesh" || algName == "RunAlltoAllDirectFullmesh" || algName == "RunAlltoAllVTwoLevelPipeline";
        bool supportAlg = ((algName == "RunAlltoAllVFullMesh" || algName == "RunAlltoAllVTwoLevelPipeline") && opParam.aicpuUnfoldMode) || algName == "RunAlltoAllDirectFullmesh";
        bool isOpbaseMode = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
        if ((supportAlg) || (isOpbaseMode && userRankSize_ > 1)) {
            CHK_RET(CreateCommCCLbuffer());
        }
        CHK_RET(CreateCommExpBuffer());

        u32 srcLocalRankId = userRank_;
        u32 rootRank = opParam.root;
        if (opParam.root == INVALID_VALUE_RANKID) {
            rootRank = 0;
        }
        std::string nslb_identifier = identifier_;
        HCCL_INFO("NSLBDP-SWK NslbDp_CollectOperTable nslb_identifier[%s] .", nslb_identifier.c_str());
        u32 rankSize = userRankSize_;
        u64 count = opParam.All2AllDataDes.sendCount * SIZE_TABLE[opParam.All2AllDataDes.sendType];

        // 资源创建
        if ((resMap_.find(newTag) != resMap_.end()) && opParam.isCapture)
        {
            auto resTmp = resMap_[newTag];
            ++captureCnt_;
            newTag += std::to_string(captureCnt_);
            resMap_[newTag] = resTmp;
            AlgResourceRequest resRequest;
            CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
            resRequest.isInGraphCaptureZeroCopy = isInGraphCaptureZeroCopy;
            CHK_RET(CleanTransportLinks(resRequest.opTransport, resMap_[newTag].opTransportResponse));
            if (IsEnableBackupLink()) {
                CHK_RET(CleanTransportLinks(resRequest.opTransport, resMap_[newTag].opTransportResponseBackUp));
            }
            CHK_RET(IncreAllocLink(newTag, opParam, resRequest, resMap_[newTag]));
        }
        InsertNewTagToTagMap(newTag, opParam.tag);
        if (resMap_.find(newTag) == resMap_.end()) {
            AlgResourceRequest resRequest;
            CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
            resRequest.isInGraphCaptureZeroCopy = isInGraphCaptureZeroCopy;
            CHK_RET(AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag]));

            // 对于91093超节点内aiv跨机通信算子，将不同机的CCLbuffer地址存在约定好的aiv将读取的HBM位置
            CHK_RET(algOperator->PrepareCommInfoToDevice(algName, resMap_[newTag]));

            if (!isHaveCpuRank_) {
                if (isUseRankPort_) {
                    std::vector<u32> &nicPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
                    std::vector<u32> &vnicPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
                    Heartbeat::GetInstance(deviceLogicId_).SetRankPortInfo(isUseRankPort_, nicPorts, vnicPorts, commPortConfig_.devPortSwitchOn);
                }
                CHK_RET(RegisterToHeartBeat());
            }
        }
        else
        {
            bool needRecreateAlltoallComm = false;
            CHK_RET(alltoAllOperator->CheckNeedRecreateComm(algName, opParam, resMap_[newTag].scratchMem.size(),
                                                            needRecreateAlltoallComm));
            HCCL_INFO("resMap_ find this newTag[%s], and need to judge whether recreate comm [%d]", newTag.c_str(),
                      needRecreateAlltoallComm);
            if (needRecreateAlltoallComm) {
                CHK_RET(hcclStreamSynchronize(opParam.stream.ptr()));
                AlgResourceRequest resRequest;
                CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
                // alltoall算子重分配内存前需清除scratchMMem，防止内存泄漏
                CHK_RET(FreeScratchMemOnOpBaseMode(resMap_[newTag].scratchMem, opParam, opType));
                CHK_RET(AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag]));
                if (!isHaveCpuRank_) {
                    if (isUseRankPort_) {
                        std::vector<u32> &nicPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
                        std::vector<u32> &vnicPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
                        Heartbeat::GetInstance(deviceLogicId_).SetRankPortInfo(isUseRankPort_, nicPorts, vnicPorts, commPortConfig_.devPortSwitchOn);
                    }
                    CHK_RET(RegisterToHeartBeat());
                }
            } else {
                DeviceMem tinySendRecvMem;
                CHK_RET(implAlg_->GetTinyMem(tinySendRecvMem));
                CHK_RET(CalcTinySendRecvMem(opParam, resMap_[newTag], tinySendRecvMem));
            }
        }
        /* NSLB 填充 表  */
        AlgType nslbAlgType = algOperator->GetAlgType();
        AlgTypeLevel1 algValue = nslbAlgType.algoLevel1;
        uint8_t nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel1AlgType(algValue);

        if (algName == "RunAlltoAllVFullMesh" || algName == "RunAlltoAllDirectFullmesh") {
            nslbAlg = NSLBDP_PAIRWISE;
            if (deviceType_ == DevType::DEV_TYPE_910_93) {
                nslbAlg = NSLB_ALGO_TYPE_FULLMESH;
            }
        }

        if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0) {
            // 填充表2
            hcclNslbDp::GetInstance().GenerateOpAndAdjTable(opType, rootRank, srcLocalRankId, nslbAlg, nslb_identifier, count, rankSize);
            AdjInfo nslbAdjInfo = {};
            CHK_RET(algOperator->GetAdjInfo(algName, opParam, resMap_[newTag], nslbAdjInfo));
            HCCL_RUN_INFO("[NSLBDP-WEN]-nslbAdjInfosize[%u]-algName[%s]-rankSize[%u]-commDesc[%s]..",
                          nslbAdjInfo.dstRankNum, algName.c_str(), userRankSize_, identifier_.c_str());
            HCCL_INFO("NSLBDP-HCCL opType:[%u],srcLocalRankId[%u],rootRank[%u],algValue[%u]",
                      opType, srcLocalRankId, rootRank, algValue);
            // 填充表3
            hcclNslbDp::GetInstance().GetAlgAdjacencyTable(opType, srcLocalRankId, rootRank, nslbAlg, nslb_identifier, nslbAdjInfo);
            /*发送流程*/
            hcclNslbDp::GetInstance().SendAlgorithmInfoTable();
        }
        // 算法执行
        bool selectAivAlg = algDesc.isAivMode;
        if (selectAivAlg) {
            HCCL_INFO("[HcclCommunicator][ExecOpAlltoAll] userRank[%u] cur aiv tag [%d]", userRank_, aivTag_);
            opParam.aivTag = aivTag_;
            opParam.aicpuUnfoldMode = false;
            aivTag_ = GetNextAivTag(aivTag_, algDesc.aivTagNum);
            CHK_RET(HandleAclGraphFirstOpAivBuff(opParam.stream.ptr()));
            u32 aivCoreLimit;
            aclError acl_ret = aclrtGetDeviceResLimit(deviceLogicId_, ACL_RT_DEV_RES_VECTOR_CORE, &aivCoreLimit);
            CHK_PRT_RET(acl_ret != ACL_SUCCESS,
                HCCL_ERROR("[HcclCommunicator][ExecOpAlltoAll] aclrtGetDeviceResLimit failed, ret=[%d]", acl_ret),
                HCCL_E_PARA);
            CHK_RET(algOperator->SetBlockDim(aivCoreLimit));
            if (aivClearEnable_) {
                // 用于判断图模式是否清零
                CHK_RET(algOperator->SetAivClearEnable(aivClearEnable_));
            }
        }
        // 头计数
        CHK_RET(StarsCounter(dispatcher_, opParam.stream, HEAD, opParam.aicpuUnfoldMode, retryEnable_, selectAivAlg));
        // 算法执行
        if (opParam.aicpuUnfoldMode && supportAicpuAlg) {
            isInplaceStatus_ = 0;
            inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::INPLACE_STATUS_END;
            // algOperator->SupportRetryWithInplaceCheck 依赖 algOperator->SetRetryEnable 才能正确返回是否支持inplace

            inplaceSupportRetry_ = algOperator->SupportRetryWithInplaceCheck(
                opType, opParam, algName, isInplaceStatus_, inPlaceSupportRetryStatus_);
            auto algType = algOperator->GetAlgType();
            HCCL_INFO("[HcclCommunicator][ExecOp] aicpu Unfold mode algType[%s], inplaceSupportRetry_[%d], opType[%d], "
                      "isInplaceStatus_[%d], inPlaceSupportRetryStatus_[%d]",
                      AlgTypeToStr(algType).c_str(), inplaceSupportRetry_, opType, isInplaceStatus_, inPlaceSupportRetryStatus_);
            CHK_RET(OrchestrateAicpu(opType, algName, opParam, resMap_[newTag], newTag, algType, isCustom));
        } else {
            // HOST展开aclgraph场景，capture从流
            if (!selectAivAlg) {
                CHK_RET(CaptureSlaveStreams(opParam.stream.ptr(), resMap_[newTag].slaveStreams));
            }
            OpCounterInfo opCounter;
            CHK_RET(GetOpCountInfo(opCounter));
            CHK_RET(algOperator->SetOpCounter(opCounter));
            CHK_RET(algOperator->Orchestrate(algName, opParam, resMap_[newTag]));
            // for profiling, blockDim upload
            CHK_RET(algOperator->GetBlockDim(blockDim_));
        }
        // 尾计数
        CHK_RET(StarsCounter(dispatcher_, opParam.stream, TAIL, opParam.aicpuUnfoldMode, retryEnable_, selectAivAlg));
        if (selectAivAlg) {
            CHK_RET(algOperator->SetAivClearEnable(false));
            aivClearEnable_ = false;
        }

        if (isInGraphCaptureZeroCopy) {
            SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::HandleAclGraphFirstOpAivBuff(rtStream_t mainStream)
    {
        rtModel_t rtModel = nullptr;
        bool isCapture = false;
        u32 modelId = 0;
        CHK_RET(GetStreamCaptureInfo(mainStream, rtModel, isCapture));
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            if (captureModelIds_.find(modelId) == captureModelIds_.end()) {
                // aclgraph场景，首算子清理AIV buff
                aivClearEnable_ = true;
                captureModelIds_.insert(modelId);
                HCCL_INFO("[HcclCommunicator][%s] modelId[%u] is inserted to captureModelIds_", __func__, modelId);
            }
        }
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::StreamIsCapture(rtStream_t mainStream)
    {
        bool isCapture = false;
        rtModel_t rtModel = nullptr;
        u32 modelId = 0;
        CHK_RET(GetStreamCaptureInfo(mainStream, rtModel, isCapture));
        return isCapture;
    }

    HcclResult HcclCommunicator::CaptureSlaveStreams(rtStream_t mainStream, vector<Stream> &slaveStreams)
    {
        if (deviceType_ != DevType::DEV_TYPE_910_93) {
            HCCL_INFO("[HcclCommunicator][%s]Only A3 device in host expan mode need to capture slave streams.", __func__);
            return HCCL_SUCCESS;
        }
        rtModel_t rtModel = nullptr;
        bool isCapture = false;
        u32 modelId = 0;
        CHK_RET(GetStreamCaptureInfo(mainStream, rtModel, isCapture));
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            CHK_RET(GetModelId(rtModel, modelId));
            for (auto slaveStream : slaveStreams) {
                CHK_RET(AddStreamToModel(slaveStream.ptr(), rtModel));
                HCCL_DEBUG("[HcclCommunicator][%s]Add stream[%d] to model[%u] success.", __func__, slaveStream.id(),
                           modelId);
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpLocalScratchMemResParam(
        const AlgResourceResponse &algResource, const std::string &newTag, LocalResInfoV2 *localResHostPtr)
    {
        if (algResource.scratchMem.size() > 0) {
            hostMemVec_.resize(hostMemVec_.size() + 1);
            CHK_RET(AllocAndClearHostMem(sizeof(HccltagLocalResV2), hostMemVec_.back()));
            HccltagLocalResV2 *tagLocalResHostPtr = static_cast<HccltagLocalResV2 *>(hostMemVec_.back().get()->ptr());

            deviceMemVec_.resize(deviceMemVec_.size() + 1);
            CHK_RET(AllocAndClearDeviceMem(sizeof(HccltagLocalResV2), deviceMemVec_.back()));
            HccltagLocalResV2 *tagLocalResDevicePtr = static_cast<HccltagLocalResV2 *>(deviceMemVec_.back().get()->ptr());

            // 初始化HcclRankRelationResV2中的tagRes链表
            ListCommonInit(&tagLocalResDevicePtr->nextTagRes, &tagLocalResHostPtr->nextTagRes);
            // 刷新host空间内容
            CHK_SAFETY_FUNC_RET(
                memcpy_s(tagLocalResHostPtr->tag, sizeof(tagLocalResHostPtr->tag), newTag.c_str(), newTag.length() + 1));
            tagLocalResHostPtr->ScratchmemSize = algResource.scratchMem.size();
            tagLocalResHostPtr->Scratchmem = reinterpret_cast<u64>(algResource.scratchMem.ptr());

            // 3、将节点插入链表头
            ListCommonAddHead(&tagLocalResDevicePtr->nextTagRes,
                              &tagLocalResHostPtr->nextTagRes,
                              &localResHostPtr->nextTagRes,
                              &opResDeviceParaPtr_->localRes.nextTagRes);
            HCCL_DEBUG("[HcclCommunicator][BuildOpLocalScratchMemResParam] LocalResHostPtr head addr[%p], nextHost[%p], "
                       "preHost[%p]",
                       &localResHostPtr->nextTagRes,
                       localResHostPtr->nextTagRes.nextHost,
                       localResHostPtr->nextTagRes.preHost);

            HCCL_DEBUG("[HcclCommunicator][BuildOpLocalScratchMemResParam] tag LocalResHostPtr head addr[%p], nextHost[%p],"
                       "preHost[%p]",
                       &tagLocalResHostPtr->nextTagRes,
                       tagLocalResHostPtr->nextTagRes.nextHost,
                       tagLocalResHostPtr->nextTagRes.preHost);
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CheckSetRetryStateToWaitResume()
    {
        if (retryEnable_ && opRetryManager_ != nullptr) {
            HcclResult ret = opRetryManager_->SetRetryStateToWaitResume(identifier_, commConnections_.isRoot);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[NsRecovery]set opretry state to wait resume timeout."), HCCL_E_INTERNAL);
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpLocalResParam(const AlgResourceResponse &algResource, const std::string &newTag)
    {
        LocalResInfoV2 *localResHostPtr = &opResPara_.localRes;
        ListCommonInit(&opResDeviceParaPtr_->localRes.nextTagRes, &opResPara_.localRes.nextTagRes);
        if (algResource.slaveDevStreams.size() > LOCAL_STREAM_MAX_NUM) {
            HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]Fail to assign stream for tag[%s]", newTag.c_str());
            return HCCL_E_PARA;
        }
        auto signalM2SNum = algResource.notifiesDevMain.size();
        auto signalS2MNum = algResource.notifiesDevAux.size();
        auto signalNum = signalM2SNum + signalS2MNum;
        if (signalNum > LOCAL_NOTIFY_MAX_NUM) {
            HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]Fail to assign local notify for tag[%s]", newTag.c_str());
            return HCCL_E_PARA;
        }

        localResHostPtr->streamNum = algResource.slaveDevStreams.size();
        for (u32 i = 0; i < algResource.slaveDevStreams.size(); i++) {
            localResHostPtr->streamParam[i].streamInfo.streamIds = algResource.slaveDevStreams[i].id();
            localResHostPtr->streamParam[i].streamInfo.sqIds = algResource.slaveDevStreams[i].sqId();
            localResHostPtr->streamParam[i].streamInfo.cqIds = algResource.slaveDevStreams[i].cqId();
            localResHostPtr->streamParam[i].streamInfo.logicCqids = algResource.slaveDevStreams[i].logicCqId();
            CHK_RET(AllocAndGetStreamContextBuff(algResource.slaveDevStreams[i].id(),
                                                 localResHostPtr->streamParam[i].sqCqContextAddr, localResHostPtr->streamParam[i].sqCqContextSize));
        }

        localResHostPtr->signalNum = signalNum;

        for (u32 i = 0; i < signalM2SNum; i++) {
            algResource.notifiesDevMain[i]->GetNotifyData(localResHostPtr->localSignals[i << 1]);
            algResource.notifiesDevAux[i]->GetNotifyData(localResHostPtr->localSignals[(i << 1) + 1]);
        }
        HcclResult ret = HCCL_SUCCESS;
        ret = CreateAndGetAiCpuNotify(localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)],
            localResHostPtr->aicpuOpNotify[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)]);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]get aicpu notify 0 error,"
                               "errNo[0x%016llx]",
                               HCCL_ERROR_CODE(ret)),
                    ret);
        ret = CreateAndGetAiCpuNotify(localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)],
            localResHostPtr->aicpuOpNotify[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)]);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[HcclCommunicator][BuildOpLocalResParam]get aicpu notify 1 error,errNo[0x%016llx]", HCCL_ERROR_CODE(ret)),
                    ret);

        HcclSignalInfo mainToKfcNotifyInfo0;
        ret = CreateAndGetAiCpuNotify(localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::MAIN_TO_KFC_0)],
            mainToKfcNotifyInfo0);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]get aicpu notify MAIN_TO_KFC_0 error",
            HCCL_ERROR_CODE(ret)), ret);
        
        HcclSignalInfo mainToKfcNotifyInfo1;
        ret = CreateAndGetAiCpuNotify(localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::MAIN_TO_KFC_1)],
            mainToKfcNotifyInfo1);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]get aicpu notify MAIN_TO_KFC_1 error",
            HCCL_ERROR_CODE(ret)), ret);
        HCCL_INFO("[HcclCommunicator][BuildOpLocalResParam] MAIN_TO_KFC_0: resId[%u], MAIN_TO_KFC_1: resId[%u]",
            mainToKfcNotifyInfo0.resId, mainToKfcNotifyInfo1.resId);

        if (opMainStream_.ptr() == nullptr) {
            opMainStream_ = Stream(StreamType::STREAM_TYPE_DEVICE);
        }
        localResHostPtr->mainStreamParam.streamInfo.streamIds = opMainStream_.id();
        localResHostPtr->mainStreamParam.streamInfo.sqIds = opMainStream_.sqId();
        localResHostPtr->mainStreamParam.streamInfo.cqIds = opMainStream_.cqId();
        localResHostPtr->mainStreamParam.streamInfo.logicCqids = opMainStream_.logicCqId();
        CHK_RET(AllocAndGetStreamContextBuff(opMainStream_.id(),
                                             localResHostPtr->mainStreamParam.sqCqContextAddr, localResHostPtr->mainStreamParam.sqCqContextSize));

        CHK_RET(BuildOpLocalScratchMemResParam(algResource, newTag, localResHostPtr));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocAndGetStreamContextBuff(u32 streamId, u64 &addr, u64 &size)
    {
        if (streamIdToStreamContext_.find(streamId) == streamIdToStreamContext_.end()) {
            DeviceMem streamContext;
            CHK_RET(CreateWorkSpace(sizeof(SqCqeContext), streamContext));
            streamIdToStreamContext_.insert({streamId, std::move(streamContext)});
        }
        addr = reinterpret_cast<u64>(streamIdToStreamContext_.at(streamId).ptr());
        size = streamIdToStreamContext_.at(streamId).size();
        HCCL_INFO("%s success, streamId:%u, addr:0x%llx, size:%llu", __func__, streamId, addr, size);
        return HCCL_SUCCESS;
    }

    u32 HcclCommunicator::UpdateOpIndex(const OpParam &opParam)
    {
        u32 opIndex = 0;
        u32 commIndex = 0;
        // 用于重执行和taskException打印的算子计数，bsr/sendrecv/其他算子分别计数
        if (opParam.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            constexpr s32 batSendRecvIndex = -1; // batchSendRecv使用 key = -1
            commIndex = batSendRecvIndex;
        } else if (opParam.opType == HcclCMDType::HCCL_CMD_SEND) {
            commIndex = opParam.dstRank;
        } else if (opParam.opType == HcclCMDType::HCCL_CMD_RECEIVE) {
            commIndex = opParam.srcRank;
        } else {
            commIndex = userRank_;
        }

        auto it = opIndexMap_.find(commIndex);
        if (it != opIndexMap_.end()) {
            opIndex = ++(it->second);
        } else {
            opIndexMap_.insert({commIndex, 1});
            opIndex = 1;
        }

        HCCL_DEBUG("%s tag:%s opType:%u commIndex:%u opIndex:%u",
                   __func__, opParam.tag.c_str(), opParam.opType, commIndex, opIndex);
        return opIndex;
    }

    HcclResult HcclCommunicator::BuildAicpuCustomParam()
    {
        if (aicpuCustomDev_.ptr() == nullptr) {
            CHK_RET(CreateWorkSpace(sizeof(AicpuCustomParam), aicpuCustomDev_));
        }

        opResPara_.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev_.ptr());
        opResPara_.aicpuCustomParamSize = aicpuCustomDev_.size();
        HCCL_INFO("%s success, aicpuCustomParamAddr:0x%llx, aicpuCustomParamSize:%llu",
                  __func__, opResPara_.aicpuCustomParamAddr, opResPara_.aicpuCustomParamSize);
        return HCCL_SUCCESS;
    }

    template <typename T>
    HcclResult HcclCommunicator::CopyVectorToDeviceMem(const u64 len, DeviceMem &dstDeviceMem, const std::vector<T> &srcVec)
    {
        CHK_PRT_RET(!len,
                    HCCL_INFO("[HcclCommunicator][CopyVectorToDeviceMem] space size is zero. not need to malloc memory"),
                    HCCL_SUCCESS);

        CHK_PRT_RET((len > ULONG_MAX),
                    HCCL_ERROR("[HcclCommunicator][CopyVectorToDeviceMem] space size is greater than %llu", ULONG_MAX),
                    HCCL_E_PARA);

        CHK_RET(CreateWorkSpace(len, dstDeviceMem));
        std::shared_ptr<HostMem> srcHostMem;
        CHK_RET(AllocAndClearHostMem(len, srcHostMem));
        std::copy(srcVec.begin(), srcVec.end(), static_cast<T *>(srcHostMem.get()->ptr()));
        CHK_RET(hrtMemSyncCopy(
            dstDeviceMem.ptr(), len, srcHostMem.get()->ptr(), len, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpTopoResTlvParam(const std::string &algName,
                                                        const std::vector<std::vector<std::vector<u32>>> &inputVectorInfo, DeviceMem &dstTlvDeviceMem, u64 &tlvLen)
    {
        vector<u32> tlv;
        CommonTlv commonTlv;
        HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResTlvParam] input vector size[%lu], group[%s]",
                   inputVectorInfo.size(), identifier_.c_str());
        for (u16 level0Idx = 0; level0Idx < inputVectorInfo.size(); level0Idx++) {
            for (u16 level1Idx = 0; level1Idx < inputVectorInfo[level0Idx].size(); level1Idx++) {
                commonTlv.type = ((level0Idx << TOP_COMM_LEVEL0_SHIFT) | level1Idx);
                commonTlv.length = (sizeof(LENGTH_TYPE) + sizeof(TAG_TYPE)) +
                                   inputVectorInfo[level0Idx][level1Idx].size() * sizeof(RANK_TYPE);
                tlv.push_back(commonTlv.type);
                tlv.push_back(commonTlv.length);
                tlv.insert(tlv.end(), inputVectorInfo[level0Idx][level1Idx].begin(),
                           inputVectorInfo[level0Idx][level1Idx].end());
            }
        }
        for (u64 idx = 0; idx < tlv.size(); idx++) {
            HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResTlvParam] idx[%lu] tlv[%lu]", idx, tlv[idx]);
        }
        tlvLen = tlv.size() * sizeof(u32);
        CHK_RET(CopyVectorToDeviceMem(tlvLen, dstTlvDeviceMem, tlv));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpTopoResVectorTlvParam(const std::string &algName,
                                                              const std::vector<std::vector<std::vector<std::vector<u32>>>> &inputVectorInfo, DeviceMem &dstTlvDeviceMem, u64 &tlvLen)
    {
        vector<u32> tlv;
        CommonTlv commonTlv;
        HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResVectorTlvParam] input vector size[%lu], group[%s]",
                   inputVectorInfo.size(), identifier_.c_str());
        for (u16 level0Idx = 0; level0Idx < inputVectorInfo.size(); level0Idx++) {
            for (u16 level1Idx = 0; level1Idx < inputVectorInfo[level0Idx].size(); level1Idx++) {
                for (u16 level2Idx = 0; level2Idx < inputVectorInfo[level0Idx][level1Idx].size(); level2Idx++) {
                    commonTlv.type = (((level0Idx << TOP_HIERARCHICAL_COMM_LEVEL0_SHIFT) | level1Idx) << TOP_HIERARCHICAL_COMM_LEVEL1_SHIFT) | level2Idx;
                    commonTlv.length = (sizeof(LENGTH_TYPE) + sizeof(TAG_TYPE)) +
                                       inputVectorInfo[level0Idx][level1Idx][level2Idx].size() * sizeof(RANK_TYPE);
                    tlv.push_back(commonTlv.type);
                    tlv.push_back(commonTlv.length);
                    tlv.insert(tlv.end(), inputVectorInfo[level0Idx][level1Idx][level2Idx].begin(),
                               inputVectorInfo[level0Idx][level1Idx][level2Idx].end());
                }
            }
        }
        for (u64 idx = 0; idx < tlv.size(); idx++) {
            HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResVectorTlvParam] idx[%lu] tlv[%lu]", idx, tlv[idx]);
        }
        tlvLen = tlv.size() * sizeof(u32);
        CHK_RET(CopyVectorToDeviceMem(tlvLen, dstTlvDeviceMem, tlv));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildPairLinkCounter(const std::string &algName)
    {
        constexpr u32 KEY_VALUE_TO_VECTOR_MODULUS = 2;
        if (pairLinkCounterDevice_.ptr() == nullptr) {
            u64 pairLinkCounterSize = pairLinkCounter_.size();
            HCCL_DEBUG("[HcclCommunicator][BuildPairLinkCounter] pairLinkCounter size[%lu], group[%s]",
                       pairLinkCounterSize, identifier_.c_str());
            std::vector<u32> pairLinkCounterVec(pairLinkCounterSize * KEY_VALUE_TO_VECTOR_MODULUS);
            u64 index = 0;
            for (auto &kt : pairLinkCounter_) {
                pairLinkCounterVec[index] = kt.first;
                pairLinkCounterVec[index + 1] = kt.second;
                index += KEY_VALUE_TO_VECTOR_MODULUS; // 每次根据
            }
            u64 len = pairLinkCounterSize * sizeof(u32) * KEY_VALUE_TO_VECTOR_MODULUS; // key-value，都为u32
            CHK_RET(CopyVectorToDeviceMem(len, pairLinkCounterDevice_, pairLinkCounterVec));
            opResPara_.topoInfo.pairLinkCounter = reinterpret_cast<u64>(pairLinkCounterDevice_.ptr());
            opResPara_.topoInfo.pairLinkCounterNum = pairLinkCounterSize * KEY_VALUE_TO_VECTOR_MODULUS;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildIsUsedRdmaRank(const std::string &algName)
    {
        constexpr u32 KEY_VALUE_TO_VECTOR_MODULUS = 2;
        if (isUsedRdmaRankPairDevice_.ptr() == nullptr) {
            std::unordered_map<u32, bool> isUsedRdmaMap;
            CHK_RET(implAlg_->GetIsUsedRdmaMap(isUsedRdmaMap));
            u64 isUsedRdmaMapSize = isUsedRdmaMap.size();
            HCCL_DEBUG("[HcclCommunicator][BuildIsUsedRdmaRank] is used Rdma rank size[%lu], group[%s]",
                       isUsedRdmaMapSize, identifier_.c_str());
            std::vector<u32> isUsedRdmaPairVec(isUsedRdmaMapSize * KEY_VALUE_TO_VECTOR_MODULUS);
            u64 index = 0;
            for (auto &kt : isUsedRdmaMap) {
                isUsedRdmaPairVec[index] = kt.first;
                isUsedRdmaPairVec[index + 1] = static_cast<u32>(kt.second);
                index += KEY_VALUE_TO_VECTOR_MODULUS;
            }
            u64 len = isUsedRdmaMapSize * sizeof(u32) * KEY_VALUE_TO_VECTOR_MODULUS; // key-value，都为u32
            CHK_RET(CopyVectorToDeviceMem(len, isUsedRdmaRankPairDevice_, isUsedRdmaPairVec));
            opResPara_.topoInfo.isUsedRdmaRankPair = reinterpret_cast<u64>(isUsedRdmaRankPairDevice_.ptr());
            opResPara_.topoInfo.isUsedRdmaRankPairNum = isUsedRdmaMapSize * KEY_VALUE_TO_VECTOR_MODULUS;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildNicList(const std::string &algName)
    {
        if (nicListDevice_.ptr() == nullptr) {
            u64 len = nicList_.size() * sizeof(u32);
            HCCL_DEBUG("[HcclCommunicator][BuildNicList] niclist size[%lu], group[%s]",
                       nicList_.size(), identifier_.c_str());
            CHK_RET(CopyVectorToDeviceMem(len, nicListDevice_, nicList_));
            opResPara_.topoInfo.nicList = reinterpret_cast<u64>(nicListDevice_.ptr());
            opResPara_.topoInfo.nicNum = nicList_.size();
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildBridgeRank(const std::string &algName)
    {
        if (bridgeRankDevice_.ptr() == nullptr) {
            std::vector<bool> isBridgeVector;
            CHK_RET(implAlg_->GetIsBridgeVector(isBridgeVector));
            u64 len = isBridgeVector.size() * sizeof(bool);
            HCCL_DEBUG("[HcclCommunicator][BuildBridgeRank] Bridge size[%lu], group[%s]",
                       isBridgeVector.size(), identifier_.c_str());
            CHK_RET(CopyVectorToDeviceMem(len, bridgeRankDevice_, isBridgeVector));
            opResPara_.topoInfo.bridgeRank = reinterpret_cast<u64>(bridgeRankDevice_.ptr());
            opResPara_.topoInfo.bridgeRankNum = isBridgeVector.size();
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildCommPlanRank(const std::string &algName)
    {
        opResPara_.topoInfo.complanRank = 0;
        opResPara_.topoInfo.complanRankLength = 0;
        if (complanRankDevice_.ptr() == nullptr) {
            std::vector<std::vector<std::vector<u32>>> commPlaneRanks;
            CHK_RET(implAlg_->GetCommPlaneRanks(commPlaneRanks));
            u64 tlvLen = 0;
            CHK_RET(BuildOpTopoResTlvParam(algName, commPlaneRanks, complanRankDevice_, tlvLen));
            opResPara_.topoInfo.complanRank = reinterpret_cast<u64>(complanRankDevice_.ptr());
            opResPara_.topoInfo.complanRankLength = tlvLen;
            HCCL_DEBUG("[HcclCommunicator][BuildCommPlanRank] comm plane ranks tlv length[%lu], ptr[%p], group[%s], "
                       "local user rankId[%u] ",
                       tlvLen, complanRankDevice_.ptr(), identifier_.c_str(), userRank_);
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildServerAndsuperPodRank(const std::string &algName)
    {
        opResPara_.topoInfo.serverAndsuperPodRank = 0;
        opResPara_.topoInfo.serverAndsuperPodRankLength = 0;
        if (serverAndsuperPodToRankDevice_.ptr() == nullptr) {
            std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank;
            CHK_RET(implAlg_->GetRankVecInfo(serverAndsuperPodToRank));
            u64 tlvLen = 0;
            CHK_RET(BuildOpTopoResTlvParam(algName, serverAndsuperPodToRank, serverAndsuperPodToRankDevice_, tlvLen));
            opResPara_.topoInfo.serverAndsuperPodRank = reinterpret_cast<u64>(serverAndsuperPodToRankDevice_.ptr());
            opResPara_.topoInfo.serverAndsuperPodRankLength = tlvLen;
            HCCL_DEBUG("[HcclCommunicator][BuildServerAndsuperPodRank] server and super pod ranks tlv length[%lu], ptr[%p], "
                       "group[%s],  local user rankId[%u] ",
                       tlvLen, serverAndsuperPodToRankDevice_.ptr(),
                       identifier_.c_str(), userRank_);
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpRetryParam(const AlgResourceResponse &algResource, const std::string &newTag)
    {
        opResPara_.config.retryEnable = static_cast<u8>(retryEnable_);
        opResPara_.config.retryHoldTime = GetExternalInputRetryHoldTime();
        opResPara_.config.retryIntervalTime = GetExternalInputRetryIntervalTime();
        // aicpu和custom共用同一个opResPara_，aicpu初始化完成后，会修改h2d/d2h的指针，然后重新传给custom
        opResPara_.kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
        opResPara_.kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();
        opResPara_.debugConfig = GetDebugConfig();

        CHK_SMART_PTR_NULL(opRetryStreamPtr_);
        if (opRetryStreamPtr_->find(newTag) == opRetryStreamPtr_->end()) {
            std::vector<Stream> retryStreams(algResource.slaveDevStreams.begin(), algResource.slaveDevStreams.end());
            retryStreams.push_back(opMainStream_);
            opRetryStreamPtr_->insert(std::make_pair(newTag, retryStreams));
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildCommPlaneSubGroupRank(const std::string &algName)
    {
        opResPara_.hierarchicalAlgInfo.commplaneSubGroupRank = 0;
        opResPara_.hierarchicalAlgInfo.commplaneSubGroupRankLength = 0;
        if (commplaneSubGroupRankDevice_.ptr() == nullptr) {
            std::vector<std::vector<std::vector<std::vector<u32>>>> commplaneSubGroupVector;
            CHK_RET(implAlg_->GetCommPlaneSubGroupVector(commplaneSubGroupVector));
            u64 tlvLen = 0;
            CHK_RET(BuildOpTopoResVectorTlvParam(algName, commplaneSubGroupVector, commplaneSubGroupRankDevice_, tlvLen));
            opResPara_.hierarchicalAlgInfo.commplaneSubGroupRank = reinterpret_cast<u64>(commplaneSubGroupRankDevice_.ptr());
            opResPara_.hierarchicalAlgInfo.commplaneSubGroupRankLength = tlvLen;
            HCCL_DEBUG("[HcclCommunicator][BuildCommPlaneSubGroupRank] comm plane subGroups ranks tlv length[%lu], ptr[%p], "
                       "group[%s],  local user rankId[%u] ",
                       tlvLen, commplaneSubGroupRankDevice_.ptr(),
                       identifier_.c_str(), userRank_);
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildHierarchicalAlgOption(const std::string &algName)
    {
        std::map<AHCConcOpType, TemplateType> hierarchicalAlgOption;
        CHK_RET(implAlg_->GetAHCAlgOption(hierarchicalAlgOption));
        opResPara_.hierarchicalAlgInfo.hierarchicalAlgOptionVec = 0;
        opResPara_.hierarchicalAlgInfo.hierarchicalAlgOptionNum = 0;
        if (hierarchicalAlgOptionDevice_.ptr() == nullptr) {
            std::vector<u32> hierarchicalAlgOptionVec;
            for (auto it = hierarchicalAlgOption.begin(); it != hierarchicalAlgOption.end(); ++it) {
                HCCL_DEBUG("[HcclCommunicator][BuildHierarchicalAlgOption] Level [%n], ConcType[%n] AHCOpType[%n], TemplateType [%n]",
                           it->first.ahcLevel, it->first.concType, it->first.ahcOpType, it->second);
                hierarchicalAlgOptionVec.push_back(static_cast<u32>(it->first.ahcLevel));
                hierarchicalAlgOptionVec.push_back(static_cast<u32>(it->first.concType));
                hierarchicalAlgOptionVec.push_back(static_cast<u32>(it->first.ahcOpType));
                hierarchicalAlgOptionVec.push_back(static_cast<u32>(it->second));
            }
            u64 len = hierarchicalAlgOptionVec.size() * sizeof(u32);
            HCCL_DEBUG("[HcclCommunicator][BuildHierarchicalAlgOption] hierarchicalAlgOptionVec size[%lu], group[%s], local user rankId[%u]",
                       hierarchicalAlgOptionVec.size(), identifier_.c_str(), userRank_);
            CHK_RET(CopyVectorToDeviceMem(len, hierarchicalAlgOptionDevice_, hierarchicalAlgOptionVec));
            opResPara_.hierarchicalAlgInfo.hierarchicalAlgOptionVec = reinterpret_cast<u64>(hierarchicalAlgOptionDevice_.ptr());
            opResPara_.hierarchicalAlgInfo.hierarchicalAlgOptionNum = hierarchicalAlgOptionVec.size();
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpTopoResParam(const std::string &algName, const AlgResourceResponse &algResource)
    {
        opResPara_.topoInfo.userRank = userRank_;
        opResPara_.topoInfo.userRankSize = userRankSize_;
        opResPara_.topoInfo.deviceLogicId = deviceLogicId_;
        opResPara_.topoInfo.isSingleMeshAggregation = isSingleMeshAggregation_;
        opResPara_.topoInfo.deviceNumPerAggregation = deviceNumPerAggregation_;
        opResPara_.topoInfo.superPodNum = superPodNum_;
        opResPara_.topoInfo.devicePhyId = devicePhyId_;
        opResPara_.topoInfo.deviceType = static_cast<u32>(deviceType_);
        TopoType topoType;
        CHK_RET(implAlg_->GetTopoType(topoType));
        opResPara_.topoInfo.topoType = static_cast<u32>(topoType);
        opResPara_.topoInfo.serverNum = serverNum_;
        opResPara_.topoInfo.meshAggregationRankSize = meshAggregationRankSize_;
        opResPara_.topoInfo.multiModuleDiffDeviceNumMode = multiModuleDiffDeviceNumMode_;
        opResPara_.topoInfo.multiSuperPodDiffServerNumMode = multiSuperPodDiffServerNumMode_;
        opResPara_.topoInfo.realUserRank = realUserRank_;
        opResPara_.topoInfo.isDiffDeviceModule = isDiffDeviceModule_;
        opResPara_.topoInfo.isDiffDeviceType = isDiffDeviceType_;
        opResPara_.topoInfo.gcdDeviceNumPerAggregation = gcdDeviceNumPerAggregation_;
        opResPara_.topoInfo.moduleNum = moduleNum_;
        CHK_RET(BuildPairLinkCounter(algName));
        CHK_RET(BuildIsUsedRdmaRank(algName));
        CHK_RET(BuildNicList(algName));
        CHK_RET(BuildBridgeRank(algName));
        CHK_RET(BuildCommPlanRank(algName));
        CHK_RET(BuildServerAndsuperPodRank(algName));
        CHK_RET(BuildCommPlaneSubGroupRank(algName));
        CHK_RET(BuildHierarchicalAlgOption(algName));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpRemoteLinkP2pResParam(const LINK &link, HccltagRemoteResV3 &tagRemoteRes,
                                                              TransportLinkType linkType)
    {
        // hccs sio并发场景，sio链路（linkTyp为SIO）打包到linkP2pSio, hccs链路（linkTyp为HCCS）打包到linkP2p；
        // 其他场景打包到linkP2p
        HcclLinkP2pV2 *linkp2p = &(tagRemoteRes.tagRemoteResPtr->linkP2p);
        if (linkType == TransportLinkType::SIO) {
            linkp2p = &(tagRemoteRes.tagRemoteResPtr->linkP2pSio);
        }
        if (linkp2p->localIpcSignal[0].resId != INVALID_U64) {
            HCCL_INFO("[%s]the linkP2p is existed, no need to refresh transport resource, resId[%llu]",
                      __func__, linkp2p->localIpcSignal[0].resId);
            return HCCL_SUCCESS;
        }
        // localMem & remoteMem
        void *inbufferPtr = nullptr;
        void *outbufferPtr = nullptr;
        CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &inbufferPtr));
        CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &outbufferPtr));
        (linkp2p->remoteMem)[INPUT].addr = reinterpret_cast<u64>(inbufferPtr);
        (linkp2p->remoteMem)[OUTPUT].addr = reinterpret_cast<u64>(outbufferPtr);
        CHK_RET(link->GetRemoteMemSize(UserMemType::INPUT_MEM, (linkp2p->remoteMem)[INPUT].size));
        CHK_RET(link->GetRemoteMemSize(UserMemType::OUTPUT_MEM, (linkp2p->remoteMem)[OUTPUT].size));
        MemDetails localMem; // 暂时预留，赋值为空
        (linkp2p->localMem)[0] = localMem;
        (linkp2p->localMem)[1] = localMem;
        HCCL_DEBUG("[%s] finish set localMem & remoteMem info", __func__);
        // localnotify & remotenotify
        u64 notifyNum = 0;
        std::vector<HcclSignalInfo> locIpcSignals;
        std::vector<HcclSignalInfo> rmtIpcSignals;
        CHK_RET(link->GetLocalNotify(locIpcSignals));
        CHK_RET(link->GetRemoteNotify(rmtIpcSignals));

        for (size_t i = 0; i < locIpcSignals.size(); i++) {
            CHK_RET(CheckNotifyOrQPMaxNum(notifyNum, LINK_P2P_MAX_NUM, true));
            linkp2p->localIpcSignal[notifyNum] = locIpcSignals[i];
            linkp2p->remoteIpcSignal[notifyNum] = rmtIpcSignals[i];
            notifyNum++;
        }
        tagRemoteRes.p2pNotifyNum = notifyNum;
        HCCL_DEBUG("[%s] finish set localnotify & remotenotify info, notifyNum[%llu]", __func__, notifyNum);
        // transportAttr
        CHK_RET(link->GetTransportAttr(linkp2p->transportAttr));
        HCCL_DEBUG("[%s] finish set RemoteLinkP2pResParam info", __func__);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpRemoteLinkRoceResParam(const LINK &link, HccltagRemoteResV3 &tagRemoteRes,
                                                               bool isBackup, bool isRetry, bool IsSecondBuild)
    {
        u32 iter = IsSecondBuild ? 2 : 0;
        HcclLinkRoceV2 *linkRoce = isBackup ? &(tagRemoteRes.tagRemoteResPtr->linkRoce[AICPU_RETRY_LINKROCE_BACKUP + iter])
                                            : &(tagRemoteRes.tagRemoteResPtr->linkRoce[AICPU_RETRY_LINKROCE_DEFAULT + iter]);
        if (!isRetry && linkRoce->localNotifyList != 0) {
            HCCL_INFO("[%s]the linkRoce is existed, no need to refresh transport resource, localNotifyListPtr[%p], iter[%u]",
                      __func__, reinterpret_cast<void *>(linkRoce->localNotifyList), iter);
            return HCCL_SUCCESS;
        }
        // localMem & remoteMem
        CHK_RET(link->GetLocalMemDetails(UserMemType::INPUT_MEM, (linkRoce->localMem)[INPUT]));
        CHK_RET(link->GetLocalMemDetails(UserMemType::OUTPUT_MEM, (linkRoce->localMem)[OUTPUT]));
        void *inbufferPtr = nullptr;
        void *outbufferPtr = nullptr;
        CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &inbufferPtr));
        CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &outbufferPtr));
        HCCL_DEBUG("[%s]inbufferPtr[%p], outbufferPtr[%p]", __func__, inbufferPtr, outbufferPtr);
        if (inbufferPtr == nullptr || outbufferPtr == nullptr) {
            HCCL_ERROR("[%s]inbufferPtr[%p], outbufferPtr[%p]", __func__, inbufferPtr, outbufferPtr);
            return HCCL_E_INTERNAL;
        }
        (linkRoce->remoteMem)[INPUT].addr = reinterpret_cast<u64>(inbufferPtr);
        (linkRoce->remoteMem)[OUTPUT].addr = reinterpret_cast<u64>(outbufferPtr);
        CHK_RET(link->GetRemoteMemKey(UserMemType::INPUT_MEM, &((linkRoce->remoteMem)[INPUT].key)));
        CHK_RET(link->GetRemoteMemKey(UserMemType::OUTPUT_MEM, &((linkRoce->remoteMem)[OUTPUT].key)));
        CHK_RET(link->GetRemoteMemSize(UserMemType::INPUT_MEM, (linkRoce->remoteMem)[INPUT].size));
        CHK_RET(link->GetRemoteMemSize(UserMemType::OUTPUT_MEM, (linkRoce->remoteMem)[OUTPUT].size));
        HCCL_DEBUG("[%s] finish set localMem & remoteMem info", __func__);
        // notifyValue & Key
        std::vector<AddrKey> notifyValueAddrKey;
        CHK_RET(link->GetLocalNotifyValueAddrKey(notifyValueAddrKey));
        linkRoce->notifyValue = notifyValueAddrKey[0].addr;
        linkRoce->notifyValueKey = notifyValueAddrKey[0].key;
        // QPInfo
        std::vector<HcclQpInfoV2> aiQpInfos;
        CHK_RET(link->GetAiQpInfo(aiQpInfos));
        u32 qpNum = aiQpInfos.size();
        if (qpNum > RDMA_QP_MAX_NUM || qpNum < 1) {
            return HCCL_E_INTERNAL;
        }
        std::copy_n(aiQpInfos.begin(), qpNum, linkRoce->QpInfo);
        linkRoce->qpsPerConnection = qpNum - static_cast<u32>(qpNum > 1); // 多QP数量或单QP模式

        // localnotify & remotenotify
        std::vector<AddrKey> notifyAddrKey;
        std::vector<HcclSignalInfo> signalInfos;
        CHK_RET(link->GetLocalRdmaNotify(signalInfos));
        CHK_RET(link->GetRemoteRdmaNotifyAddrKey(notifyAddrKey));
        constexpr u32 RDMA_NOTIFY_MIN_NUM = 3;
        constexpr u32 RDMA_NOTIFY_MAX_NUM = 8192;
        if ((signalInfos.size() != notifyAddrKey.size()) || (signalInfos.size() < RDMA_NOTIFY_MIN_NUM) ||
            (signalInfos.size() > RDMA_NOTIFY_MAX_NUM) || (notifyAddrKey.size() < RDMA_NOTIFY_MIN_NUM) ||
            (notifyAddrKey.size() > RDMA_NOTIFY_MAX_NUM) ||
            ((signalInfos.size() - RDMA_NOTIFY_MIN_NUM) % linkRoce->qpsPerConnection) ||
            ((notifyAddrKey.size() - RDMA_NOTIFY_MIN_NUM) % linkRoce->qpsPerConnection)) {
            return HCCL_E_INTERNAL;
        }
        u64 notifyNum = (notifyAddrKey.size() - RDMA_NOTIFY_MIN_NUM) / linkRoce->qpsPerConnection - static_cast<u32>(linkRoce->qpsPerConnection > 1);
        linkRoce->singleQPNotifyNum = notifyNum;

        u64 len = signalInfos.size() * sizeof(HcclSignalInfo);
        DeviceMem localNotifyListMem;
        CHK_RET(CopyVectorToDeviceMem(len, localNotifyListMem, signalInfos));
        linkRoce->localNotifyList = reinterpret_cast<u64>(localNotifyListMem.ptr());
        ibverbsLocalNotify_.emplace_back(std::move(localNotifyListMem));

        len = notifyAddrKey.size() * sizeof(AddrKey);
        DeviceMem remoteNotifyListMem;
        CHK_RET(CopyVectorToDeviceMem(len, remoteNotifyListMem, notifyAddrKey));
        linkRoce->remoteNotifyList = reinterpret_cast<u64>(remoteNotifyListMem.ptr());
        ibverbsRemoteNotify_.emplace_back(std::move(remoteNotifyListMem));

        HCCL_DEBUG("[%s] finish set localnotify & remotenotify info, notifyNum[%llu], linkNotifyNum[%llu]",
                   __func__, notifyNum, signalInfos.size());

        if (isBackup) {
            tagRemoteRes.roceNotifyNumBackup = linkRoce->singleQPNotifyNum;
            tagRemoteRes.qpNumBackup = linkRoce->qpsPerConnection;
        } else {
            tagRemoteRes.roceNotifyNum = linkRoce->singleQPNotifyNum;
            tagRemoteRes.qpNum = linkRoce->qpsPerConnection;
        }
        HCCL_DEBUG("[%s] finish set Qp info qpNum[%u], linkRoce->localNotifyList[0].resId[%llu], "
                   "notifyNum[%u], isBackup[%d], isSecond[%d], qpPtr[%llu]",
                   __func__, linkRoce->qpsPerConnection,
                   signalInfos[0].resId, linkRoce->singleQPNotifyNum, isBackup, IsSecondBuild,
                   linkRoce->QpInfo[0].qpPtr);
        return HCCL_SUCCESS;
    }

    template <typename T>
    HcclResult HcclCommunicator::CreateListNode(T **resHostPtr, T **resDevicePtr)
    {
        hostMemVec_.resize(hostMemVec_.size() + 1);
        CHK_RET(AllocAndClearHostMem(sizeof(T), hostMemVec_.back()));
        *resHostPtr = static_cast<T *>(hostMemVec_.back().get()->ptr());

        deviceMemVec_.resize(deviceMemVec_.size() + 1);
        CHK_RET(AllocAndClearDeviceMem(sizeof(T), deviceMemVec_.back()));

        *resDevicePtr = static_cast<T *>(deviceMemVec_.back().get()->ptr());
        // 初始化HcclRankRelationResV2中的tagRes链表
        ListCommonInit(&((*resDevicePtr)->nextTagRes), &((*resHostPtr)->nextTagRes));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildRemoteResByTag(const std::string &newTag, const u32 &usrRankId,
                                                     HcclRankRelationResV2 *&rankRelationResHostPtr, HcclRankRelationResV2 *&rankRelationResDevicePtr, bool isBackup,
                                                     bool isRetry)
    {
        HCCL_DEBUG("[%s]start to add RemoteRes with newtag[%s] and remoteRankId[%u] to list",
                   __func__, newTag.c_str(), usrRankId);
        if (rankTagRemoteRes_.find(usrRankId) == rankTagRemoteRes_.end() ||
            rankTagRemoteRes_[usrRankId].find(newTag) == rankTagRemoteRes_[usrRankId].end()) {
            HccltagRemoteResV2 *tagRemoteResHostPtr = nullptr;
            HccltagRemoteResV2 *tagRemoteResDevicePtr = nullptr;
            CHK_RET(CreateListNode(&tagRemoteResHostPtr, &tagRemoteResDevicePtr));
            CHK_SAFETY_FUNC_RET(memcpy_s(tagRemoteResHostPtr->tag, sizeof(tagRemoteResHostPtr->tag),
                                         newTag.c_str(), newTag.length() + 1));
            tagRemoteResHostPtr->linkP2p.localIpcSignal[0].resId = INVALID_U64;
            tagRemoteResHostPtr->linkP2pSio.localIpcSignal[0].resId = INVALID_U64;
            tagRemoteResHostPtr->linkRoce[0].localNotifyList = 0;
            tagRemoteResHostPtr->linkRoce[1].localNotifyList = 0;
            tagRemoteResHostPtr->linkRoce[2].localNotifyList = 0;
            tagRemoteResHostPtr->linkRoce[3].localNotifyList = 0;
            ListCommonAddHead(&tagRemoteResDevicePtr->nextTagRes, &tagRemoteResHostPtr->nextTagRes,
                              &rankRelationResHostPtr->nextTagRes, &rankRelationResDevicePtr->nextTagRes);
            HccltagRemoteResV3 tempTagRemoteRes;
            tempTagRemoteRes.tagRemoteResPtr = tagRemoteResHostPtr;
            rankTagRemoteRes_[usrRankId][newTag] = tempTagRemoteRes;
            HCCL_DEBUG("[%s] successfully add RemoteRes to list with newtag[%s], remoteRankId[%u]"
                       "rankRelationResHostPtr head addr[%p], nextHost[%p], preHost[%p], nextDevice[%p], preDevice[%p], "
                       "tagRemoteResDevicePtr head addr[%p]",
                       __func__, newTag.c_str(), usrRankId,
                       &rankRelationResHostPtr->nextTagRes, rankRelationResHostPtr->nextTagRes.nextHost,
                       rankRelationResHostPtr->nextTagRes.preHost, rankRelationResHostPtr->nextTagRes.nextDevice,
                       rankRelationResHostPtr->nextTagRes.preDevice, &tagRemoteResDevicePtr->nextTagRes);
        } else {
            HCCL_DEBUG("[%s] the RemoteRes with usr rankid[%u] tag[%s] has been added list",
                       __func__, usrRankId, newTag.c_str());
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildRelationResByRemoteRankId(const TransportRequest &transportRequest, const LINK &link,
                                                                HcclRankRelationResV2 *&rankRelationResHostPtr, HcclRankRelationResV2 *&rankRelationResDevicePtr)
    {
        const u32 usrRankId = transportRequest.remoteUserRank;
        HCCL_INFO("[%s]start to add RelationRes with remote usr rankid[%u] to list", __func__, usrRankId);
        if (opResPara_.remoteRes[usrRankId].nextHostPtr != 0 && opResPara_.remoteRes[usrRankId].nextDevicePtr != 0) {
            rankRelationResHostPtr =
                reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[usrRankId].nextHostPtr);
            rankRelationResDevicePtr =
                reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[usrRankId].nextDevicePtr);
            HCCL_DEBUG("[%s] RelationRes with remote usr rankid[%u] has been added to list, "
                       "rankRelationResHostPtr[%p], rankRelationResDevicePtr[%p]",
                       __func__, usrRankId, rankRelationResHostPtr, rankRelationResDevicePtr);
        } else {
            CHK_RET(CreateListNode(&rankRelationResHostPtr, &rankRelationResDevicePtr));
            opResPara_.remoteRes[usrRankId].nextHostPtr = reinterpret_cast<u64>(rankRelationResHostPtr);
            opResPara_.remoteRes[usrRankId].nextDevicePtr = reinterpret_cast<u64>(rankRelationResDevicePtr);
            rankRelationResHostPtr->remoteUsrRankId = usrRankId;
            rankRelationResHostPtr->remoteWorldRank = rankInfoList_[usrRankId].worldRank;
            HCCL_DEBUG("[%s]successfully add RelationRes with remote usr rankid[%u] to list, rankRelationResHostPtr[%p],"
                       "rankRelationResDevicePtr[%p]",
                       __func__, usrRankId, rankRelationResHostPtr, rankRelationResDevicePtr);
        }
        // 刷新远端对应的cclbuffer
        std::vector<void *> extraMemVector;
        if (transportRequest.inputMemType == TransportMemType::CCL_INPUT && rankRelationResHostPtr->windowsIn == 0) {
            void *inbufferPtr = nullptr;
            CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &inbufferPtr));
            rankRelationResHostPtr->windowsIn = reinterpret_cast<u64>(inbufferPtr);
        }
        if (transportRequest.outputMemType == TransportMemType::CCL_OUTPUT && rankRelationResHostPtr->windowsOut == 0) {
            void *outbufferPtr = nullptr;
            CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &outbufferPtr));
            rankRelationResHostPtr->windowsOut = reinterpret_cast<u64>(outbufferPtr);
        }
        if (rankRelationResHostPtr->windowsExp == 0) {
            std::vector<void *> memPtrVec = {};
            CHK_RET(link->GetRemoteMem(&memPtrVec));
            if (memPtrVec.size() != 0) {
                rankRelationResHostPtr->windowsExp = reinterpret_cast<u64>(memPtrVec[0]);
            }
        }
        HCCL_INFO("group[%s] successfully set windowsIn & windowsOut & windowsExp info: userRank[%u], groupRank[%u], "
                  "remoteRank[%u], windowsIn[0x%llx], InSize[0x%llx], windowOut[0x%llx], OutSize[0x%llx], "
                  "windowExp[0x%llx], ExpSize[0x%llx]",
                  identifier_.c_str(), GetUserRank(), GetGroupRank(), transportRequest.remoteUserRank,
                  rankRelationResHostPtr->windowsIn, cclBufferManager_.GetInCCLbufferSize(),
                  rankRelationResHostPtr->windowsOut, cclBufferManager_.GetOutCCLbufferSize(),
                  rankRelationResHostPtr->windowsExp, cclBufferManager_.GetExpBufferSize());
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ParseRemoteDataToMem(const OpCommTransport &opTransportResponse, const std::string &newTag,
                                                      const HcclCMDType opType, bool isBackup, bool isRetry)
    {
        HCCL_INFO("[%s] entry process newtag[%s], isBackup[%d]", __func__, newTag.c_str(), isBackup);
        std::set<u32> bsrTansportRank;
        for (auto &levelNSubCommTransport : opTransportResponse) {
            for (auto &singleSubCommTransport : levelNSubCommTransport) {
                u32 linkIdx = 0;
                for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                    if (transportRequest.isValid) {
                        auto tempLink = singleSubCommTransport.links[linkIdx];
                        HCCL_INFO("[%s]transportRequest.isUsedRdma[%d], isBackup[%d]", __func__,
                                  transportRequest.isUsedRdma, isBackup);
                        if ((!transportRequest.isUsedRdma || tempLink->GetLinkType() == LinkType::LINK_SIO) &&
                            (isBackup || isRetry)) {
                            HCCL_INFO("[%s]no need to add p2p backup Link resource, transportRequest.isUsedRdma[%d], "
                                      "isBackup[%d]",
                                      __func__, transportRequest.isUsedRdma, isBackup);
                            linkIdx++;
                            continue;
                        }
                        HcclRankRelationResV2 *rankRelationResHostPtr = nullptr;
                        HcclRankRelationResV2 *rankRelationResDevicePtr = nullptr;
                        CHK_RET(BuildRelationResByRemoteRankId(transportRequest, tempLink, rankRelationResHostPtr,
                                                               rankRelationResDevicePtr));
                        const u32 usrRankId = transportRequest.remoteUserRank;
                        HCCL_INFO("[%s]successfully BuildRelationResByRemoteRankId with remote usr rankid[%u], "
                                  "rankRelationResHostPtr[%p], rankRelationResDevicePtr[%p], newTage[%s]",
                                  __func__, usrRankId, rankRelationResHostPtr, rankRelationResDevicePtr, newTag.c_str());
                        CHK_RET(BuildRemoteResByTag(newTag, usrRankId, rankRelationResHostPtr,
                                                    rankRelationResDevicePtr, isBackup, isRetry));
                        // transport信息保存（notify、qp）
                        if (!transportRequest.isUsedRdma || tempLink->GetLinkType() == LinkType::LINK_SIO) {
                            // sdma -> P2P
                            CHK_RET(BuildOpRemoteLinkP2pResParam(tempLink, rankTagRemoteRes_[usrRankId][newTag],
                                                                 transportRequest.linkType));
                        } else {
                            // rdma -> roce
                            bool isSecondBuild = false;
                            if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV &&
                                bsrTansportRank.find(transportRequest.remoteUserRank) != bsrTansportRank.end()) {
                                isSecondBuild = true;
                            }
                            bsrTansportRank.insert(transportRequest.remoteUserRank);
                            CHK_RET(BuildOpRemoteLinkRoceResParam(tempLink, rankTagRemoteRes_[usrRankId][newTag],
                                                                  isBackup, isRetry, isSecondBuild));
                        }
                        HCCL_INFO("[%s] successfully add RemoteRes to list with newtag[%s] rankRelationResHostPtr "
                                  "head addr[%p], nextHost[%p], preHost[%p], nextDevice[%p], preDevice[%p], "
                                  "rankRelationResDevicePtr head addr[%p]",
                                  __func__, newTag.c_str(),
                                  &rankRelationResHostPtr->nextTagRes, rankRelationResHostPtr->nextTagRes.nextHost,
                                  rankRelationResHostPtr->nextTagRes.preHost, rankRelationResHostPtr->nextTagRes.nextDevice,
                                  rankRelationResHostPtr->nextTagRes.preDevice, &rankRelationResDevicePtr->nextTagRes);
                        HCCL_INFO("[%s] create link success with newtag[%s], linkIdx[%u], isBackup[%d], usrRankId[%u]",
                                  __func__, newTag.c_str(), linkIdx, isBackup, usrRankId);
                    }
                    linkIdx++;
                }
            }
        }
        HCCL_DEBUG("[%s] process success newtag[%s]", __func__, newTag.c_str());
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpRemoteResParam(const AlgResourceResponse &algResource, const std::string &newTag,
                                                       const HcclCMDType opType, bool isRetry)
    {
        HCCL_DEBUG("[%s]start ParseRemoteDataToMem, IsEnableBackupLink[%d]", __func__, IsEnableBackupLink());
        CHK_RET(ParseRemoteDataToMem(algResource.opTransportResponse, newTag, opType, false, isRetry));
        if (IsEnableBackupLink()) {
            HCCL_DEBUG("[%s]start Parse backupRemoteDataToMem, IsEnableBackupLink[%d]", __func__, IsEnableBackupLink());
            CHK_RET(ParseRemoteDataToMem(algResource.opTransportResponseBackUp, newTag, opType, true, isRetry));
        }
        if (deviceType_ == DevType::DEV_TYPE_910_93 || deviceType_ == DevType::DEV_TYPE_910B) {
            opResPara_.notifysize = 4; // 910B & 910_93 每个notify占4个字节
        } else {
            opResPara_.notifysize = 8; // 其他芯片类型每个notify占8个字节
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CopyHostListResToDeviceParam(const std::string &newTag, const ListCommon *headHostList, const u64 size)
    {
        ListCommon *nextHostList = reinterpret_cast<ListCommon *>(headHostList->nextHost);
        ListCommon *nextDeviceList = reinterpret_cast<ListCommon *>(headHostList->nextDevice);

        while (nextHostList != headHostList) {
            HCCL_INFO(
                "[HcclCommunicator][CopyHostListResToDeviceParam] remote resource, tag[%s], head Host List[%p], next "
                "Host List[%p],next Device List[%p]",
                newTag.c_str(), headHostList, nextHostList, nextDeviceList);
            CHK_RET(hrtMemSyncCopy(reinterpret_cast<void *>(nextDeviceList), size, reinterpret_cast<void *>(nextHostList),
                                   size, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
            nextDeviceList = reinterpret_cast<ListCommon *>(nextHostList->nextDevice);
            nextHostList = reinterpret_cast<ListCommon *>(nextHostList->nextHost);
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CopyHostOpResToDeviceParam(const std::string &newTag)
    {
        // 1、将opResPara_，H2D到device
        CHK_RET(hrtMemSyncCopy(opResDevicePara_.ptr(), sizeof(HcclOpResParam), reinterpret_cast<void *>(&opResPara_),
                               sizeof(HcclOpResParam), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        HCCL_DEBUG("[HcclCommunicator][CopyHostOpResToDeviceParam] tag[%s] local rankId[%u] workspace[%p] "
                   "workspacesize[%lu] ranksize[%u], cclbuffersize[%lu], cclinbuffer[%p], ccloutbuffer[%p], "
                   "remote winStart[%u], remote rWinOffset[%u], hostStateInfo[%p], aicpuStateInfo[%p], notifysize[%u]",
                   newTag.c_str(), userRank_, opResPara_.mc2WorkSpace.workSpace, opResPara_.mc2WorkSpace.workSpaceSize,
                   opResPara_.rankSize, opResPara_.winSize, opResPara_.localWindowsIn, opResPara_.localWindowsOut,
                   opResPara_.rWinStart, opResPara_.rWinOffset, opResPara_.hostStateInfo, opResPara_.aicpuStateInfo,
                   opResPara_.notifysize);
        // 2、将opResPara_中localres的tagRes，H2D到device
        HCCL_DEBUG("[HcclCommunicator][CopyHostOpResToDeviceParam] local resource, tag[%s] streamNum[%u] signalNum[%u]",
                   newTag.c_str(), opResPara_.localRes.streamNum, opResPara_.localRes.signalNum);
        CHK_RET(CopyHostListResToDeviceParam(
            newTag, reinterpret_cast<ListCommon *>(&opResPara_.localRes.nextTagRes), sizeof(HccltagLocalResV2)));
        // 3、遍历rank中tag资源，H2D到device
        CHK_RET(CopyHostOpRemoteResToDeviceParam(newTag));
        HCCL_DEBUG("[HcclCommunicator][CopyHostOpResToDeviceParam] copy host resource success!, tag[%s]", newTag.c_str());
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildOpResParam(
        const std::string &algName, const AlgResourceResponse &algResource, const std::string &newTag,
        const HcclCMDType opType)
    {
        opResPara_.localUsrRankId = userRank_;
        opResPara_.rankSize = userRankSize_;

        opResPara_.winSize = algResource.cclInputMem.size();
        opResPara_.localWindowsIn = reinterpret_cast<u64>(algResource.cclInputMem.ptr());
        opResPara_.localWindowsOut = reinterpret_cast<u64>(algResource.cclOutputMem.ptr());
        // 填充Exp相关信息 当前该块内存大小恒为1M
        opResPara_.winExpSize = EXP_BUFFER_SIZE;
        opResPara_.localWindowsExp = reinterpret_cast<u64>(cclBufferManager_.GetCommExpBuffer().ptr());

        CHK_SAFETY_FUNC_RET(
            memcpy_s(opResPara_.hcomId, sizeof(opResPara_.hcomId), identifier_.c_str(), identifier_.length() + 1));

        opResPara_.config.deterministic = GetDeterministicConfig();
        opResPara_.config.highPerfEnable = 0;
        rtFloatOverflowMode_t floatOverflowMode = RT_OVERFLOW_MODE_UNDEF;
        CHK_RET(hrtGetDeviceSatMode(&floatOverflowMode));
        opResPara_.config.floatOverflowMode = floatOverflowMode;
        opResPara_.config.notifyWaitTime =
            (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET)
                ? GetExternalInputHcclExecTimeOut()
                : NOTIFY_DEFAULT_WAIT_TIME;
        opResPara_.config.linkTimeOut = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
        opResPara_.config.retryEnable = static_cast<u8>(retryEnable_);
        opResPara_.config.interHccsDisable = GetExternalInputInterHccsDisable();
        opResPara_.config.multiQpThreshold = GetExternalInputMultiQpThreshold();
        opResPara_.rWinStart = offsetof(HcclOpResParam, remoteRes);
        opResPara_.rWinOffset = sizeof(RemoteResPtr);
        opResPara_.notifysize = 0;
        opResPara_.lockAddr = hostDeviceLock_->GetDevMemAddr();
        opResPara_.utraceStatusFlag = GetExternalInputHcclEnableEntryLog();
        DeviceMem tinySendRecvMem;
        CHK_RET(implAlg_->GetTinyMem(tinySendRecvMem));
        opResPara_.tinyMem = reinterpret_cast<u64>(tinySendRecvMem.ptr());
        opResPara_.tinyMemSize = reinterpret_cast<u64>(tinySendRecvMem.size());

        CHK_RET(BuildOpLocalResParam(algResource, newTag));
        CHK_RET(BuildOpRemoteResParam(algResource, newTag, opType));
        CHK_RET(BuildOpTopoResParam(algName, algResource));
        CHK_RET(BuildOpRetryParam(algResource, newTag));
        CHK_RET(BuildZeroCopyParam());
        CHK_RET(BuildAicpuCustomParam());
        CHK_RET(CopyHostOpResToDeviceParam(newTag));
        HCCL_RUN_INFO("[%s]build aicpu unfold resource success!, tag[%s] rWinStart[%u] rWinOffset[%u]",
                      __func__, newTag.c_str(), opResPara_.rWinStart, opResPara_.rWinOffset);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::BuildCustomOpResParam()
    {
        // custom进程需要刷新h2d/d2h内存
        opResPara_.kfcControlTransferH2DParams = customControlTransferH2D_->GetCommunicateParams();
        opResPara_.kfcStatusTransferD2HParams = customStatusTransferD2H_->GetCommunicateParams();
        CHK_RET(hrtMemSyncCopy(opResDevicePara_.ptr(), sizeof(HcclOpResParam), reinterpret_cast<void *>(&opResPara_),
                               sizeof(HcclOpResParam), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::RegisterDfxInfo(const OpParam &param, AlgType algType,
                                                 std::vector<Stream> slaveStreams, bool isAiv)
    {
        u64 count = 0;
        HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
        switch (param.opType) {
        case HcclCMDType::HCCL_CMD_SEND:
        case HcclCMDType::HCCL_CMD_RECEIVE:
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV:
            count = param.GetDataCount(userRank_);
            dataType = param.GetDataType();
            HCCL_PROFILER_ADD_TAG_SENDRECV(param.tag, identifier_, GetWorkflowMode());
            HCCL_PROFILER_ADD_GROUPRANK_SENDRECV(identifier_, userRankSize_, userRank_, param.dstRank);
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALL:
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
            CHK_RET(AddGroupTagInfo(param.tag, isAiv));
            count = param.All2AllDataDes.sendCount;
            dataType = param.All2AllDataDes.sendType;
            break;
        default:
            CHK_RET(AddGroupTagInfo(param.tag, isAiv));
            count = param.GetDataCount(userRank_);
            dataType = param.GetDataType();
        }
        // task exception使用: 算子计数，算子入参信息(src/dst/datatype/reducetype)
        HCCL_PROFILER_ADD_OPDATA_OP(param.tag, count, param.inputPtr, param.outputPtr, dataType, param.root, identifier_,
                                    param.reduceType);
        // 记录主流相关信息, 给profiling和task exception使用
        HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType);
        if (((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
             hccl::ProfilingManagerPub::GetAddtionInfoState() &&
             hccl::ProfilingManagerPub::GetTaskApiState())) {
            return HCCL_SUCCESS;
        }
        // 从流信息profiling开关打开的话再注册
        for (u32 streamIndex = 0; streamIndex < slaveStreams.size(); streamIndex++) {
            HCCL_PROFILER_ADD_STREAM_BY_STREAMID(slaveStreams[streamIndex].id(), param.tag, streamIndex + 1, algType);
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetReportHcclMC2Info(const Stream &kfcStream, const std::vector<Stream> &aicpuStreams)
    {
        hcclMc2Info_.groupName = hrtMsprofGetHashId(identifier_.c_str(), identifier_.length());
        hcclMc2Info_.rankSize = userRankSize_;
        hcclMc2Info_.rankId = userRank_;
        hcclMc2Info_.usrRankId = realUserRank_;
        hcclMc2Info_.aicpuKfcStreamId = static_cast<uint32_t>(kfcStream.id());
        hcclMc2Info_.reserve = 0;
        const uint32_t ONCE_REPORT_STREAM_NUM_MAX = 8;
        for (uint32_t streamIndex = 0, reportId = 0; streamIndex < aicpuStreams.size(); streamIndex++) {
            HCCL_INFO("streamIndex:%u, reportId:%u, streamId:%d", streamIndex, reportId, aicpuStreams[streamIndex].id());
            hcclMc2Info_.commStreamIds[reportId++] = aicpuStreams[streamIndex].id();
            if (reportId == ONCE_REPORT_STREAM_NUM_MAX) {
                hcclMc2Info_.commStreamSize = reportId;
                CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(hrtMsprofSysCycleTime(), &hcclMc2Info_,
                                                                         sizeof(hcclMc2Info_)));
                reportId = 0;
            }
            if (streamIndex == (aicpuStreams.size() - 1)) {
                HCCL_INFO("streamIndex:%u, reportId:%u, streamId:%d", streamIndex, reportId, opMainStream_.id());
                hcclMc2Info_.commStreamIds[reportId++] = opMainStream_.id();
                hcclMc2Info_.commStreamSize = reportId;
                CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(hrtMsprofSysCycleTime(), &hcclMc2Info_,
                                                                         sizeof(hcclMc2Info_)));
                reportId = 0;
            }
        }
        if (aicpuStreams.empty()) {
            HCCL_INFO("only exist main stream, streamId:%d", opMainStream_.id());
            hcclMc2Info_.commStreamIds[0] = opMainStream_.id();
            hcclMc2Info_.commStreamSize = 1; // 只有主流1条
            CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(hrtMsprofSysCycleTime(), &hcclMc2Info_,
                                                                     sizeof(hcclMc2Info_)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::OrchestrateAicpu(const HcclCMDType &opType, const std::string &algName,
                                                  const OpParam &param, const AlgResourceResponse &algResource, const std::string &newTag, AlgType algType,
                                                  bool isCustom)
    {
        uint64_t streamMode = 0;
        CHK_RET(hrtStreamGetMode(param.stream.ptr(), &streamMode));
        rtStream_t aicpuStream;
        Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream); // aicpuStream需要在首次下发时申请
        if (!isContextLaunched_) {
            // 1、通信域内首次下发，从algResource中获取资源，H2D刷新资源，launch init
            rtStream_t aicpuInitStream;
            Mc2AiCpuInitStreamAllocAndGet(streamMode, aicpuInitStream); // 使用aicpuInitStream_下初始化kernel
            Stream tmpStream(aicpuInitStream);
            HCCL_DEBUG("%s ContextLaunched, aicpuInitStream:%p, aicpuStream:%p", __func__, aicpuInitStream, aicpuStream);
            CHK_RET(AicpuResourceInit(algName, algResource, newTag, aicpuInitStream, opType, isCustom));
            CHK_RET(GetReportHcclMC2Info(tmpStream, algResource.slaveDevStreams));
        } else if (newTagResAlloced_.find(newTag) == newTagResAlloced_.end() ||
                 opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            // 2、通信域内非首次，但是有新的newTag，查看是否需要补充资源。
            PetersonLockGuard guard(hostDeviceLock_.get());
            CHK_PRT_RET(guard.IsLockFailed(),
                        HCCL_ERROR("[HcclCommunicator][OrchestrateAicp] hostDeviceLock lock failed"), HCCL_E_INTERNAL);
            CHK_RET(AicpuResourceRefresh(algResource, newTag, opType));
        }
        bool isUsedMainStream = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
        // inplace支持重执行的stream资源处理逻辑
        bool isHcclOpInplace = IsHcclOpInplace(opType, param, userRank_, userRankSize_, isInplaceStatus_);
        if ((retryOrigWorkflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) &&
            retryEnable_ && isHcclOpInplace &&
            (opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER)) {
            isUsedMainStream = true;
        }
        AicpuOpTiling opTilingInfo;
        opTilingInfo.algName = algName;
        opTilingInfo.newTag = newTag;
        opTilingInfo.algType = algType;
        opTilingInfo.isUsedMainStream = isUsedMainStream;
        opTilingInfo.dumpDebug = GetExternalInputHcclDumpDebug();
        rtFloatOverflowMode_t floatOverflowMode = RT_OVERFLOW_MODE_UNDEF;
        CHK_RET(hrtGetDeviceSatMode(&floatOverflowMode));
        opTilingInfo.floatOverflowMode = floatOverflowMode;
        HcclResult ret = HCCL_SUCCESS;
        std::string kernelName = "RunAicpuRpcSrvLaunchV2";
        ret = AicpuKfcTilingDataLaunchExt(param, opType, opResDevicePara_, kernelName, opTilingInfo, isCustom);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclCommunicator][OrchestrateAicpu]aicpu unfold launch kernel failed. return[%d] inputPtr[%p]"
                       "outputPtr[%p] count[%llu] dataType[%s] op[%s]",
                       ret, param.inputPtr, param.outputPtr,
                       param.DataDes.count, GetDataTypeEnumStr(param.DataDes.dataType).c_str(),
                       GetReduceOpEnumStr(param.reduceType).c_str());
            return ret;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CalcTinySendRecvMem(const OpParam &opParam, AlgResourceResponse &algResResponse,
                                                     DeviceMem &tinySendRecvMem)
    {
        u64 sendCount = 0;
        u64 recvCount = 0;
        if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            for (u32 i = 0; i < userRankSize_; i++) {
                u64 curSendCount = *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCounts) + i) +
                                   *(static_cast<const u64 *>(opParam.All2AllDataDes.sdispls) + i);
                sendCount = std::max(sendCount, curSendCount);
                u64 curRecvCount = *(static_cast<const u64 *>(opParam.All2AllDataDes.recvCounts) + i) +
                                   *(static_cast<const u64 *>(opParam.All2AllDataDes.rdispls) + i);
                recvCount = std::max(recvCount, curRecvCount);
            }
        } else {
            for (u32 i = 0; i < userRankSize_; i++) {
                sendCount += *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCountMatrix) +
                               userRank_ * userRankSize_ + i);
                recvCount += *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCountMatrix) +
                               userRank_ + userRankSize_ * i);
            }
        }

        u32 sendTypeSize = 0, recvTypeSize = 0;
        CHK_RET(SalGetDataTypeSize(opParam.All2AllDataDes.sendType, sendTypeSize));
        CHK_RET(SalGetDataTypeSize(opParam.All2AllDataDes.recvType, recvTypeSize));

        // 在sendCount/recvCount全0时, 使用tinySendRecvMem, 避免使用空deviceMem
        algResResponse.paramInputMem = sendCount == 0 ? DeviceMem::create(tinySendRecvMem.ptr(), tinySendRecvMem.size()) : DeviceMem::create(opParam.inputPtr, sendCount * sendTypeSize);
        algResResponse.paramOutputMem = recvCount == 0 ? DeviceMem::create(tinySendRecvMem.ptr(), tinySendRecvMem.size()) : DeviceMem::create(opParam.outputPtr, recvCount * recvTypeSize);

        HCCL_INFO("[HcclCommunicator][CalcTinySendRecvMem] senMem addr[%p], sendSize[%llu], "
                  "RecvMem addr[%p], RecvSize[%llu],",
                  algResResponse.paramInputMem.ptr(),
                  algResResponse.paramInputMem.size(), algResResponse.paramOutputMem.ptr(),
                  algResResponse.paramOutputMem.size());
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CleanTransportLinks(OpCommTransport &opTransportReq, OpCommTransport &opTransportResponse)
    {
        for (u32 levelIndex = 0; levelIndex < opTransportReq.size(); levelIndex++) {
            for (u32 ringIndex = 0; ringIndex < opTransportReq[levelIndex].size(); ringIndex++) {
                SingleSubCommTransport &reqSingleSubComm = opTransportReq[levelIndex][ringIndex];
                SingleSubCommTransport &respSingleSubComm = opTransportResponse[levelIndex][ringIndex];
                for (u32 rankIndex = 0; rankIndex < reqSingleSubComm.transportRequests.size();  rankIndex++) {
                    TransportRequest &transportRequest = reqSingleSubComm.transportRequests[rankIndex];
                    CHK_PRT_RET(rankIndex >= respSingleSubComm.links.size(),
                        HCCL_ERROR("[CleanTransportLinks] The remote rank_id[%u] is larger than the existent respSingleSubComm map "\
                        "size[%u]", rankIndex, respSingleSubComm.links.size()), HCCL_E_PARA);
                    if (respSingleSubComm.links[rankIndex] != nullptr &&
                        respSingleSubComm.links[rankIndex]->GetLinkType() != hccl::LinkType::LINK_RESERVED && !transportRequest.isUsedRdma) {
                        HCCL_INFO("[CleanTransportLinks] The link to remote userRank[%u] has existed", transportRequest.remoteUserRank);
                        continue;
                    }
                    respSingleSubComm.links[rankIndex] = nullptr;
                }
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocAlgNotifys(const std::string &tag, const NotifyLoadType notifyLoadType, const u32 notifyNum,
                                                 std::vector<std::shared_ptr<LocalNotify>> &notifiesMain, std::vector<std::shared_ptr<LocalNotify>> &notifiesAux)
    {
        std::vector<std::shared_ptr<LocalNotify>> notifys(notifyNum, nullptr);
        CHK_RET(queueNotifyManagerRefac_->Alloc(tag, notifyNum, notifys, notifyLoadType));

        u32 signalNum = notifyNum >> 1;
        notifiesMain.resize(signalNum);
        notifiesAux.resize(signalNum);
        for (u32 i = 0; i < signalNum; i++) {
            notifiesMain[i] = notifys[i << 1];
            notifiesAux[i] = notifys[(i << 1) + 1];
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocAlgResource(const std::string &newTag, HcclCMDType opType, const OpParam &opParam,
                                                  AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse)
    {
        HcclResult ret = HCCL_SUCCESS;
        bool isGraphZeroCopyAlgAlloc = false;
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
            !IsForceAicpuOpBaseMode(opParam, opType)) {
            isGraphZeroCopyAlgAlloc = resRequest.isInGraphCaptureZeroCopy;
            if (isGraphZeroCopyAlgAlloc) {
                if (resRequest.scratchMemSize > 0) {
                    algResResponse.scratchMem =
                        DeviceMem::create(cclBufferManager_.GetOutCCLbuffer().ptr(), resRequest.scratchMemSize);
                }
            } else if (resRequest.scratchMemSize > 0) {
                algResResponse.scratchMem = GetWorkspaceScracthMem(opParam.tag, resRequest.scratchMemSize);
                if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
                    // cce reduce地址32字节对齐，截取32字节对齐后的内存地址
                    u32 addOffset = (reinterpret_cast<uintptr_t>(algResResponse.scratchMem.ptr())) % CCE_REDUCE_ALIGN_SIZE;
                    u64 totalSize = userRankSize_ * opParam.DataDes.count * SIZE_TABLE[opParam.DataDes.dataType];
                    algResResponse.scratchMem = algResResponse.scratchMem.range(addOffset, totalSize);
                }
            }

            if (resRequest.streamNum > 0) {
                if (isGraphZeroCopyAlgAlloc) {
                    CHK_RET(opStreamManager_->RegisterMaster(opParam.stream));
                    algResResponse.slaveStreams =
                        opStreamManager_->AllocSlaves(StreamType::STREAM_TYPE_ONLINE, resRequest.streamNum);
                    CHK_PRT_RET(algResResponse.slaveStreams.empty(),
                                HCCL_ERROR("[AllocAlgResource]tag[%s] get slave stream failed, "
                                           "expect to get size [%u], but only alloc 0.",
                                           newTag.c_str(), resRequest.streamNum),
                                HCCL_E_INTERNAL);
                } else {
                    algResResponse.slaveStreams = GetWorkspaceSubStreams(opParam.tag, resRequest.streamNum);
                }
            }
        } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ||
                 IsForceAicpuOpBaseMode(opParam, opType)) {
            CHK_RET(AllocOpBaseModeScratchMem(opType, opParam, resRequest, algResResponse));
            if (resRequest.streamNum > 0) {
                CHK_RET(opStreamManager_->RegisterMaster(opParam.stream));
                algResResponse.slaveStreams =
                    opStreamManager_->AllocSlaves(StreamType::STREAM_TYPE_ONLINE, resRequest.streamNum);
                CHK_PRT_RET(algResResponse.slaveStreams.empty(),
                            HCCL_ERROR("[AllocAlgResource]tag[%s] get slave stream failed, "
                                       "expect to get size [%u], but only alloc 0.",
                                       newTag.c_str(), resRequest.streamNum),
                            HCCL_E_INTERNAL);
            }
        } else {
            HCCL_ERROR("[AllocAlgResource]WorkflowMode is not set.");
            return HCCL_E_PARA;
        }

        if (opParam.aicpuUnfoldMode && ((userRankSize_ != 1) || IsForceAicpuOpBaseMode(opParam, opType))) {
            CHK_RET(opStreamManager_->RegisterMaster(opParam.stream));
            // 非batchwrite通信使用的最大从流个数为19
            const u32 streamNum =
                (opType == HcclCMDType::HCCL_CMD_BATCH_WRITE ? LOCAL_STREAM_MAX_NUM : NON_BATCH_WRITE_MAX_STREAM_NUM);
            HCCL_DEBUG("%u stream(s) on device will be created for op %u tag %s.",
                       streamNum, static_cast<u_int32_t>(opType), newTag.c_str());
            algResResponse.slaveDevStreams = opStreamManager_->AllocSlaves(StreamType::STREAM_TYPE_DEVICE, streamNum);
            CHK_PRT_RET(algResResponse.slaveDevStreams.empty(),
                        HCCL_ERROR("[AllocAlgResource]tag[%s] get slave device stream failed, "
                                   "expect to get size [%u], but only alloc 0.",
                                   newTag.c_str(), LOCAL_STREAM_MAX_NUM),
                        HCCL_E_INTERNAL);
            CHK_RET(AllocAlgNotifys(opParam.tag, NotifyLoadType::DEVICE_NOTIFY, LOCAL_NOTIFY_MAX_NUM,
                                    algResResponse.notifiesDevMain, algResResponse.notifiesDevAux));
        }
        CHK_RET(AllocAlgNotifys(opParam.tag, NotifyLoadType::HOST_NOTIFY, resRequest.notifyNum, algResResponse.notifiesMain,
                                algResResponse.notifiesAux));

        algResResponse.cclInputMem = cclBufferManager_.GetInCCLbuffer();
        algResResponse.cclOutputMem = cclBufferManager_.GetOutCCLbuffer();
        DeviceMem expMem = cclBufferManager_.GetCommExpBuffer(); // 获取拓展内存
        if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
            DeviceMem tinySendRecvMem;
            CHK_RET(implAlg_->GetTinyMem(tinySendRecvMem));
            CHK_RET(CalcTinySendRecvMem(opParam, algResResponse, tinySendRecvMem));
        } else {
            algResResponse.paramInputMem = DeviceMem::create(opParam.inputPtr, opParam.inputSize);
            algResResponse.paramOutputMem = DeviceMem::create(opParam.outputPtr, opParam.outputSize);
        }

        if (AIV_COMM_BUFFER_BITMASK & resRequest.aivBufferRequest) {
            bool useOpbaseFlag = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !opParam.isCapture);
            ret = cclBufferManager_.CreateCommAIVbuffer(useOpbaseFlag);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Alloc][AlgResource]Create CommAIVbuffer failed"), ret);
            if (useOpbaseFlag) { // 单算子非Capture模式
                algResResponse.aivInputMem = cclBufferManager_.GetInAivOpbaseBuffer();
                algResResponse.aivOutputMem = cclBufferManager_.GetOutAivOpbaseBuffer();
            } else { // 静态图或者Capture模式
                algResResponse.aivInputMem = cclBufferManager_.GetInAivOffloadbuffer();
                algResResponse.aivOutputMem = cclBufferManager_.GetOutAivOffloadbuffer();
            }
            HCCL_INFO("[AllocAlgResource] tag[%s] alloc aiv buffer", newTag.c_str());
        }
        if (AIV_COMM_INFO_BUFFER_BITMASK & resRequest.aivBufferRequest) {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                DeviceMem aivCommInfoMem = DeviceMem::alloc(AIV_COMM_INFO_SIZE); // 图模式每个算子单独一块内存
                CHK_PTR_NULL(aivCommInfoMem.ptr());
                algResResponse.aivCommInfoMem = aivCommInfoMem;
                aivOffloadCommInfoMem_.emplace_back(std::move(aivCommInfoMem));
            } else {
                ret = cclBufferManager_.CreateCommInfoAIVbuffer(); // Capture模式用CCL
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Alloc][AlgResource]Create CommInfoAIVbuffer failed"), ret);
                algResResponse.aivCommInfoMem = cclBufferManager_.GetAivCommInfoBuffer(); // 单算子每个通信域只用一块内存
            }
            HCCL_INFO("[AllocAlgResource] tag[%s] alloc aiv comm info buffer", newTag.c_str());
        }

        TransportIOMem transMem{algResResponse.cclInputMem, algResResponse.cclOutputMem,
                                algResResponse.paramInputMem, algResResponse.paramOutputMem, algResResponse.scratchMem,
                                algResResponse.aivInputMem, algResResponse.aivOutputMem, expMem};
        HCCL_DEBUG("algResResponse.cclInputMem[%p], size[%llu]; algResResponse.cclOutputMem[%p], "
                   "size[%llu]; algResResponse.paramInputMem[%p], size[%llu]; algResResponse.paramOutputMem[%p], size[%llu].",
                   algResResponse.cclInputMem.ptr(), algResResponse.cclInputMem.size(),
                   algResResponse.cclOutputMem.ptr(), algResResponse.cclOutputMem.size(),
                   algResResponse.paramInputMem.ptr(), algResResponse.paramInputMem.size(),
                   algResResponse.paramOutputMem.ptr(), algResResponse.paramOutputMem.size());
        algResResponse.opTransportResponse = resRequest.opTransport;

        // 零拷贝场景这里只借助P2p的openIpc能力交换控制面zeroCopyLocalBuffer_，不交换实际用户的输出输出
        if (opParam.isZeroCopy) {
            HCCL_INFO("[AllocAlgResource] zero copy change paramInput[%p] paramOutput[%p] scratchMem[%p] to localBuffer[%p]",
                      transMem.paramInputMem.ptr(), transMem.paramOutputMem.ptr(), transMem.scratchMem.ptr(), zeroCopyLocalBuffer_.ptr());
            transMem.scratchMem = zeroCopyLocalBuffer_;
            transMem.paramInputMem = zeroCopyLocalBuffer_;
            transMem.paramOutputMem = zeroCopyLocalBuffer_;
        } else {
            if (isGraphZeroCopyAlgAlloc) {
                transMem.scratchMem =
                    DeviceMem::create(cclBufferManager_.GetOutCCLbuffer().ptr(), resRequest.scratchMemSize);
                HCCL_INFO("[AllocAlgResource] acl graph set transMem.scratchMem =%ul", transMem.scratchMem.size());
            }
        }

        ClearOpTransportResponseLinks(algResResponse.opTransportResponse);
        if (IsEnableBackupLink()) {
            algResResponse.opTransportResponseBackUp = resRequest.opTransport;
            ClearOpTransportResponseLinks(algResResponse.opTransportResponseBackUp);
            HCCL_DEBUG("[%s]IsEnableBackupLink[%d] init backup & default opTransportResponse", __func__,
                       IsEnableBackupLink());
        }

        if (!GetExternalInputHcclEnableFfts() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            u32 slaveNum = algResResponse.slaveStreams.size();
            algResResponse.threadManage.resize(slaveNum);
            for (u32 ringIndex = 0; ringIndex < slaveNum; ringIndex++) {
                algResResponse.threadManage[ringIndex].reset(new (std::nothrow) ThreadManage(deviceLogicId_,
                                                                                             userRank_,
                                                                                             dispatcher_));
                CHK_SMART_PTR_NULL(algResResponse.threadManage[ringIndex]);
                HcclResult ret = algResResponse.threadManage[ringIndex]->Init();
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                            HCCL_ERROR("[Init][MultiRingResource]ringIndex[%u] ThreadManage failed,return[%d]",
                                       ringIndex, ret),
                            ret);
                HCCL_INFO("ringThreadsManage Init success[%u]", ringIndex);
            }
        }
        {
            StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
            ret = transportManager_->Alloc(opParam.tag, transMem, algResResponse.opTransportResponse,
                                           opParam.aicpuUnfoldMode, false, opParam.isZeroCopy, opParam.opType, opParam.isCapture);
        }
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[%s]Alloc transports failed, tag[%s]", __func__, newTag.c_str()), ret);

        if (retryEnable_) {
            // 获取当前rdma相连的所有对端rankList
            std::vector<u32> rankList;
            CHK_RET(transportManager_->GetRemoteRankList(algResResponse.opTransportResponse, rankList,
                                                         TransportType::TRANS_TYPE_IBV_EXP));
            std::string rankListStr = "";
            for (auto remoteRank : rankList) {
                rankListStr += (std::to_string(remoteRank) + ";");
            }
            HCCL_DEBUG("identifier[%s] newTag[%s] rankList[%s]", identifier_.c_str(), newTag.c_str(), rankListStr.c_str());
            CHK_RET(OpRetryManager::AddLinkInfoByIdentifier(deviceLogicId_, identifier_, newTag, rankList));
        }

        if (IsEnableBackupLink()) {
            // 超节点 && level2支持重执行 && Aicpu：创建备用Transport资源
            StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
            ret = transportManager_->Alloc(opParam.tag, transMem, algResResponse.opTransportResponseBackUp,
                                           opParam.aicpuUnfoldMode, true, opParam.isCapture);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[%s]Alloc backup transports failed, tag[%s]", __func__, newTag.c_str()), ret);
        }
        SaveLinkRes(algResResponse.opTransportResponse);
        SaveLinkRes(algResResponse.opTransportResponseBackUp);
        HCCL_DEBUG("[%s] process success newtag[%s]", __func__, newTag.c_str());
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::IncreAllocLink(const std::string &newTag, const OpParam &opParam,
                                                AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse)
    {
        algResResponse.cclInputMem = cclBufferManager_.GetInCCLbuffer();
        algResResponse.cclOutputMem = cclBufferManager_.GetOutCCLbuffer();
        DeviceMem expMem = cclBufferManager_.GetCommExpBuffer();

        TransportIOMem transMem{algResResponse.cclInputMem, algResResponse.cclOutputMem,
                                algResResponse.paramInputMem, algResResponse.paramOutputMem, algResResponse.scratchMem,
                                algResResponse.aivInputMem, algResResponse.aivOutputMem, expMem};
        {
            StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
            CHK_RET(transportManager_->IncreAlloc(opParam.tag, transMem, resRequest.opTransport,
                                                  algResResponse.opTransportResponse, opParam.aicpuUnfoldMode, false, opParam.isCapture));
        }
        if (retryEnable_) {
            // 获取当前rdma相连的所有对端rankList
            std::vector<u32> rankList;
            CHK_RET(transportManager_->GetIncreRemoteRankList(resRequest.opTransport,
                                                              algResResponse.opTransportResponse, rankList, TransportType::TRANS_TYPE_IBV_EXP));
            std::string rankListStr = "";
            for (auto remoteRank : rankList)
            {
                rankListStr += (std::to_string(remoteRank) + ";");
            }
            HCCL_DEBUG("identifier[%s] newTag[%s] rankList[%s]", identifier_.c_str(), newTag.c_str(), rankListStr.c_str());
            CHK_RET(OpRetryManager::AddLinkInfoByIdentifier(deviceLogicId_, identifier_, newTag, rankList, true));
        }
        if (IsEnableBackupLink()) {
            StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
            CHK_RET(transportManager_->IncreAlloc(opParam.tag, transMem, resRequest.opTransport,
                                                  algResResponse.opTransportResponseBackUp, opParam.aicpuUnfoldMode, true, opParam.isCapture));
        }
        SaveLinkRes(algResResponse.opTransportResponse);
        SaveLinkRes(algResResponse.opTransportResponseBackUp);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitRecvMsgAndRequestBuffer()
    {
        CHK_RET(CheckSuspendingStatus());
        // 拉远、下沉、推理场景(ps、worker)支持使用msg/request内存池
        if (pMsgInfosMem_ == nullptr) {
            pMsgInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclMessageInfo>(MEMORY_CAPACITY));
            CHK_SMART_PTR_NULL(pMsgInfosMem_);
            CHK_RET(pMsgInfosMem_->Init());
            HCCL_INFO("InitRecvMsgBuffer Success!");
        }

        if (pReqInfosMem_ == nullptr) {
            pReqInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclRequestInfo>(MEMORY_CAPACITY));
            CHK_SMART_PTR_NULL(pReqInfosMem_);
            CHK_RET(pReqInfosMem_->Init());
            HCCL_INFO("InitRequestBuffer Success!");
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitMemBlocksAndRecvWrMem()
    {
        u32 memBlockNum = MEM_BLOCK_NUM;
        CHK_PRT(GetMemBlockNum(devicePhyId_, memBlockNum));

        if (!GetExternalInputHcclIsTcpMode() && (Is310PDevice() || isHostUseDevNic_)) {
            // 注册mr,hdc模式下在通信类内进行
            if (!isHostUseDevNic_) {
                // 初始化信封内存
                memBlocksManager_.reset(new (std::nothrow) HeterogMemBlocksManager());
                CHK_SMART_PTR_NULL(memBlocksManager_);
                CHK_RET(memBlocksManager_->Init(memBlockNum));

                // 信封内存注册
                CHK_RET(mrManager_->GetKey(memBlocksManager_->GetMemAddr(), memBlocksManager_->GetMemSize(),
                                           transportResInfo_.lkey));
            }

            // 初始化wr内存
            pRecvWrInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<RecvWrInfo>(MEMORY_CAPACITY));
            CHK_SMART_PTR_NULL(pRecvWrInfosMem_);
            CHK_RET(pRecvWrInfosMem_->Init());
            HCCL_INFO("InitMemBlocksAndRecvWrMem Success!");
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetDevicePid(s32 devicePid)
    {
        devicePid_ = devicePid;
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::ReleaseWorkSpacebuffer()
    {
        workSpace_.free();
    }

    HcclResult HcclCommunicator::AllocAndClearDeviceMem(u64 size, std::shared_ptr<DeviceMem> &bufferPtr) const
    {
        CHK_PRT_RET(!size,
                    HCCL_INFO("[HcclCommunicator][AllocAndClearDeviceMem]device memory size is zero. not need to malloc memory"),
                    HCCL_SUCCESS);

        CHK_PRT_RET((size > ULONG_MAX),
                    HCCL_ERROR("[HcclCommunicator][AllocAndClearDeviceMem]device memory size is greater than %llu", ULONG_MAX),
                    HCCL_E_PARA);

        DeviceMem tmpBuffer = DeviceMem::alloc(size);
        EXECEPTION_CATCH((bufferPtr = std::make_shared<DeviceMem>(std::move(tmpBuffer))), return HCCL_E_PTR);

        CHK_PRT_RET(size && !bufferPtr.get()->ptr(),
                    HCCL_ERROR("[HcclCommunicator][AllocAndClearDeviceMem]Create DeviceMem size[%llu] fail,"
                               "please check workspace size.",
                               size),
                    HCCL_E_PTR);
        CHK_RET(hrtMemSet(bufferPtr.get()->ptr(), size, size));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocAndClearHostMem(u64 size, std::shared_ptr<HostMem> &bufferPtr) const
    {
        CHK_PRT_RET(!size,
                    HCCL_INFO("[HcclCommunicator][AllocAndClearHostMem] host memory size is zero. not need to malloc memory"),
                    HCCL_SUCCESS);

        CHK_PRT_RET((size > ULONG_MAX),
                    HCCL_ERROR("[HcclCommunicator][AllocAndClearHostMem] host memory size is greater than %llu", ULONG_MAX),
                    HCCL_E_PARA);

        HostMem tmpBuffer = HostMem::alloc(size);
        EXECEPTION_CATCH((bufferPtr = std::make_shared<HostMem>(std::move(tmpBuffer))), return HCCL_E_PTR);

        CHK_PRT_RET(size && !bufferPtr.get()->ptr(),
                    HCCL_ERROR("[HcclCommunicator][AllocAndClearHostMem]host memory space size[%llu] fail,"
                               "please check workspace size.",
                               size),
                    HCCL_E_PTR);
        CHK_SAFETY_FUNC_RET(memset_s(bufferPtr.get()->ptr(), size, 0, size));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateWorkSpace(u64 size, DeviceMem &buffer) const
    {
        CHK_PRT_RET(!size, HCCL_INFO("[Create][WorkSpace]work space size is zero. not need to malloc memory"),
                    HCCL_SUCCESS);

        CHK_PRT_RET((size > ULONG_MAX),
                    HCCL_ERROR("[Create][WorkSpace]work space size is greater than %llu",
                               ULONG_MAX),
                    HCCL_E_PARA);

        u64 memSize = size;
        buffer = DeviceMem::alloc(memSize);
        CHK_PRT_RET(size && !buffer, HCCL_ERROR("[Create][WorkSpace]Create work space size[%llu] fail,"
                                                "please check workspace size.",
                                                size),
                    HCCL_E_PTR);
        CHK_RET(hrtMemSet(buffer.ptr(), size, size));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetWorkSpace(u64 *workSpaceSize, u64 *workSpace) const
    {
        *workSpaceSize = workSpaceSize_;
        *workSpace = reinterpret_cast<u64>(workSpace_.ptr());
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitWorkSpace()
    {
        if (workSpace_.ptr() == nullptr) {
            workSpaceSize_ = COMM_MAX_WORK_SPACE_SIZE;
            CHK_RET(CreateWorkSpace(workSpaceSize_, workSpace_));
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::FillOpParam(const HcclCMDType commType, OpParam &opParam,
                                             const uint64_t count, void *pCount, void *pDispls)
    {
        if (commType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
            commType == HcclCMDType::HCCL_CMD_ALLGATHER ||
            commType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
            opParam.DataDes.count = count;
            opParam.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 按照fp16配置
        } else if (commType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
                 commType == HcclCMDType::HCCL_CMD_ALLTOALL ||
                 commType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
            opParam.All2AllDataDes.sendType = HcclDataType::HCCL_DATA_TYPE_FP16;
            opParam.All2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_FP16;
            opParam.All2AllDataDes.sendCounts = pCount;
            opParam.All2AllDataDes.recvCounts = pCount;
            opParam.All2AllDataDes.sdispls = pDispls;
            opParam.All2AllDataDes.rdispls = pDispls;
            opParam.All2AllDataDes.sendCountMatrix = pCount;
        } else if (commType == HcclCMDType::HCCL_CMD_BATCH_WRITE) {
        } else {
            HCCL_ERROR("[%s] invalid commType=[%u]",
                       __func__, static_cast<uint32_t>(commType));
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocComResource(const string &newTag, const string &algName,
                                                  const HcclCMDType commType, const OpParam &opParam, rtStream_t stream)
    {
        if (resMap_.find(newTag) == resMap_.end()) { // 计算&申请通信资源
            unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(commType);
            CHK_PRT_RET(algOperator == nullptr,
                        HCCL_ERROR("[%s] algOperator is nullptr", __func__), HCCL_E_INTERNAL);
            AlgResourceRequest resRequest;
            CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
            CHK_RET(AllocAlgResource(newTag, commType, opParam, resRequest, resMap_[newTag]));
            CHK_RET(RegisterToHeartBeat());
        }

        CHK_RET(InitWorkSpace());
        HcclResult ret = GetWorkSpace(&(opResPara_.mc2WorkSpace.workSpaceSize), &(opResPara_.mc2WorkSpace.workSpace));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("%s GetWorkSpace fail, size[%llu] space[%llu]", __func__,
                               opResPara_.mc2WorkSpace.workSpaceSize, opResPara_.mc2WorkSpace.workSpace),
                    ret);

        if (!isContextLaunched_) { // 通信域内首次下发
            uint64_t streamMode = 0;
            CHK_RET(hrtStreamGetMode(opParam.stream.ptr(), &streamMode));
            rtStream_t aicpuStream;
            Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream); // aicpuStream需要在首次下发时申请

            rtStream_t aicpuInitStream;
            Mc2AiCpuInitStreamAllocAndGet(streamMode, aicpuInitStream);
            Stream tmpStream(aicpuInitStream);
            HCCL_DEBUG("%s ContextLaunched, aicpuInitStream:%p, aicpuStream:%p", __func__, aicpuInitStream, aicpuStream);
            CHK_RET(AicpuResourceInit(algName, resMap_[newTag], newTag, stream, commType));
            CHK_RET(GetReportHcclMC2Info(tmpStream, resMap_[newTag].slaveDevStreams));
        } else if (newTagResAlloced_.find(newTag) == newTagResAlloced_.end()) {
            // 通信域内非首次，但是有新的newTag
            PetersonLockGuard guard(hostDeviceLock_.get());
            CHK_PRT_RET(guard.IsLockFailed(),
                        HCCL_ERROR("[%s] hostDeviceLock lock failed", __func__), HCCL_E_INTERNAL);
            CHK_RET(AicpuResourceRefresh(resMap_[newTag], newTag, commType));
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AllocComResourceByTiling(const string &algConfig, void *param)
    {
        string algName, newTag;
        OpParam &opParam = *static_cast<OpParam *>(param);
        CHK_RET(GetAlgInfo(algConfig, opParam.tag, opParam.opType, algName, newTag));
        CHK_RET(CreateAndGetAiCpuNotifyWithNotifyRes(combinOpara_.signalInfo.aicpuNotify));
        HCCL_INFO("Create aicpu notify %p.", localAiCpuNotifyRes_[0]->ptr());

        // 只有第一次创建，此处通过CCL Buffer地址有效来防止通信域内非首次重新申请内存
        CHK_RET(CreateCommCCLbuffer());
        CHK_RET(CreateCommExpBuffer());

        CHK_RET(cclBufferManager_.GetInCCLbuffer(opParam.inputPtr, opParam.inputSize));
        CHK_RET(cclBufferManager_.GetOutCCLbuffer(opParam.outputPtr, opParam.outputSize));

        // 按照 ccl buffer size 折算，不同算子折算方式不同, allreduce和cclbuffer size相同
        // allgather、reducescatter、alltoall需除以rank size
        uint64_t count = opParam.outputSize / SIZE_TABLE[HcclDataType::HCCL_DATA_TYPE_FP16];
        if (opParam.opType != HcclCMDType::HCCL_CMD_ALLREDUCE) {
            count = (count + userRankSize_ - 1) / userRankSize_;
        }
        HCCL_INFO("[%s] userRankSize=[%u], count=[%u]", __func__, userRankSize_, count);
        vector<uint64_t> countList(userRankSize_ * userRankSize_, count);
        vector<uint64_t> displsList(userRankSize_, 0);
        void *pCount = reinterpret_cast<void *>(&countList[0]);
        void *pDispls = reinterpret_cast<void *>(&displsList[0]);
        CHK_RET(FillOpParam(opParam.opType, opParam, count, pCount, pDispls));
        CHK_RET(AllocComResource(newTag, algName, opParam.opType, opParam, opParam.stream.ptr()));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateCommResource(const std::string &tag, rtStream_t aiCpuStream, bool isOpbaseMode,
                                                    void **commContext)
    {
        const std::string &suffix = HCCL_MC2_MULTISERVER_SUFFIX;
        if (tag.size() > suffix.size() && tag.compare(tag.size() - suffix.size(), suffix.size(), suffix) == 0) {
            HCCL_INFO("[HcclCommunicator][CreateCommResource] Set isA2MC2MultiServer_ to [true]");
            isA2MC2MultiServer_ = true;
        }
        if (isA2MC2MultiServer_ && !isNeedInitNic_) {
            InitNic(true);
        }

        if ((deviceType_ != DevType::DEV_TYPE_910_93 && moduleNum_ > 1 && !isA2MC2MultiServer_) ||
            (deviceType_ == DevType::DEV_TYPE_910_93 && superPodNum_ > 1)) {
            HCCL_ERROR("[HcclCommunicator][CommResource]MC2 does not support in the current scenario, "
                       "device type[%d] moduleNum[%d] serverNum[%d] superPodNum[%d], isMC2MultiServer[%d].",
                       deviceType_, moduleNum_, serverNum_, superPodNum_, isA2MC2MultiServer_);
            return HCCL_E_NOT_SUPPORT;
        }

        HCCL_INFO("[HcclCommunicator][CommResource]tag[%s] aicpu stream[%p] isOpbaseMode[%u]", tag.c_str(), aiCpuStream,
                  isOpbaseMode);

        Stream stream(aiCpuStream);
        CHK_RET(CreateCommAndStreamRes(tag, stream));

        CHK_RET(Mc2CreateAndLaunchContext(aiCpuStream, isOpbaseMode, commContext, tag));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Mc2CreateAndLaunchContext(rtStream_t aiCpuStream, bool isOpbaseMode, void **commContext, const string &tag)
    {
        u32 qosCfg = INVALID_QOSCFG;
        CHK_RET(GetQosCfg(qosCfg));
        CHK_RET(InitWorkSpace());
        HcclResult ret = GetWorkSpace(&(combinOpara_.mc2WorkSpace.workSpaceSize), &(combinOpara_.mc2WorkSpace.workSpace));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[HcclCommunicator][CommResource]errNo[0x%016llx] size[%llu] space[%llu]",
                               HCCL_ERROR_CODE(ret), combinOpara_.mc2WorkSpace.workSpaceSize, combinOpara_.mc2WorkSpace.workSpace),
                    ret);

        CHK_SAFETY_FUNC_RET(memcpy_s(combinOpara_.hcomId, sizeof(combinOpara_.hcomId),
                                     identifier_.c_str(), identifier_.length() + 1));

        Stream tmpStream(aiCpuStream);
        CHK_RET(CreateAndGetAiCpuNotifyWithNotifyRes(combinOpara_.signalInfo.aicpuNotify));
        CHK_RET(CreateAndGetAiCpuNotify(localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)],
            combinOpara_.signalInfo.aicpuOpNotify[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)]));
        CHK_RET(CreateAndGetAiCpuNotify(localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)],
            combinOpara_.signalInfo.aicpuOpNotify[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)]));
        // 申请集合通信域存储context的device空间
        CHK_RET(CreateDeviceCommContext(sizeof(HcclCombinOpParam), commContext_));
        combinOpara_.config.deterministic = GetDeterministicConfig();
        // retryEnable 写入aicpu_ctx
        combinOpara_.config.retryEnable = static_cast<u8>(retryEnable_);
        combinOpara_.config.retryHoldTime = GetExternalInputRetryHoldTime();
        combinOpara_.config.retryIntervalTime = GetExternalInputRetryIntervalTime();
        combinOpara_.config.notifyWaitTime =
            (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET) ? GetExternalInputHcclExecTimeOut() : NOTIFY_DEFAULT_WAIT_TIME;
        combinOpara_.config.linkTimeOut = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

        combinOpara_.kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
        combinOpara_.kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();

        void *overflowAddr = nullptr;
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            CHK_RET(hrtCtxGetOverflowAddr(&overflowAddr));
            combinOpara_.overFlowAddr = reinterpret_cast<u64>(overflowAddr);
            HCCL_INFO("[HcclImplBase][Mc2CreateAndLaunchContext]get combinOpara_.overFlowAddr %llx",
                      combinOpara_.overFlowAddr);
            // 非整卡 (2DUO卡各取1芯的场景) 因为受到PCIE限制，不可以使用读操作进行数据拷贝
            if (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() != userRankSize_) {
                combinOpara_.onlyRead = 1;
            }
        }
        HCCL_INFO("read only is set to %u", combinOpara_.onlyRead);
        char *mmSysGetEnvValue = nullptr;
        MM_SYS_GET_ENV(MM_ENV_HCCL_INTRA_PCIE_ENABLE, mmSysGetEnvValue);
        std::string intraPcieEnableEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";

        if (isA2MC2MultiServer_) {
            // 拷贝normal transport信息到device侧
            const u64 ibverbsDataSize = transDevIbverbsData_.size() * sizeof(TransportDeviceNormalData);
            ibverbsDataBuffer_ = DeviceMem::alloc(ibverbsDataSize);
            CHK_PTR_NULL(ibverbsDataBuffer_.ptr());
            CHK_RET(hrtMemAsyncCopyByQos(ibverbsDataBuffer_.ptr(),
                                         ibverbsDataBuffer_.size(),
                                         transDevIbverbsData_.data(),
                                         ibverbsDataSize,
                                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE,
                                         aiCpuStream,
                                         qosCfg));
            combinOpara_.ibverbsData = reinterpret_cast<u64>(ibverbsDataBuffer_.ptr());
            combinOpara_.ibverbsDataSize = ibverbsDataSize;
            combinOpara_.multiServerFlag = static_cast<u8>(true);

            const u64 capabilitySize = sizeof(combinedCapability_);
            combinedCapabilityBuffer_ = DeviceMem::alloc(capabilitySize);
            CHK_PTR_NULL(combinedCapabilityBuffer_.ptr());
            CHK_RET(hrtMemAsyncCopyByQos(combinedCapabilityBuffer_.ptr(),
                                         combinedCapabilityBuffer_.size(),
                                         &combinedCapability_,
                                         capabilitySize,
                                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE,
                                         aiCpuStream,
                                         qosCfg));
            combinOpara_.capabilityPtr = reinterpret_cast<u64>(combinedCapabilityBuffer_.ptr());
            combinOpara_.capabilitySize = capabilitySize;

            HCCL_INFO("[HcclImplBase][Mc2CreateAndLaunchContext] set combinOpara_.ibverbsData to [%llu], "
                      "combinOpara_.multiServerFlag to [%u]",
                      combinOpara_.ibverbsData, combinOpara_.multiServerFlag);

            bool isMC2Hie = (intraPcieEnableEnv == "1") && (GetExternalInputIntraRoceSwitch() == 0);
            bool isSupportAIVNormalQP = false;
            CHK_RET(IsSupportAIVNormalQP(devicePhyId_, isSupportAIVNormalQP));
              
            if (isSupportAIVNormalQP && isMC2Hie)
            {
                CHK_RET(H2DAiRMAInfo(tag, aiCpuStream));
            }
        }

        HostMem src = HostMem::create(&combinOpara_, sizeof(HcclCombinOpParam));
        // 将通信数据拷贝到device侧，供AICPU算法编排使用
        CHK_RET(hrtMemAsyncCopyByQos(commContext_.ptr(), commContext_.size(), src.ptr(), src.size(),
                                     HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream, qosCfg));

        std::string kernelName = "RunAicpuKfcResInit";
        CHK_RET(AiCpuKernelLaunch(tmpStream.ptr(), reinterpret_cast<u64>(commContext_.ptr()), kernelName));
        SetMC2EnvFlag();
        if (isOpbaseMode == true) {
            CHK_RET(hcclStreamSynchronize(tmpStream.ptr()));
        }

        *commContext = commContext_.ptr();
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetAiCpuNotifyData(const std::shared_ptr<LocalNotify> &localNotify,
                                                    HcclSignalInfo &notifyInfo)
    {
        if (localNotify == nullptr) {
            HCCL_INFO("[HcclCommunicator][GetAiCpuNotifyData]notifyHandle is null");
            notifyInfo.resId = INVALID_U64;
            return HCCL_SUCCESS;
        }

        CHK_RET(localNotify->GetNotifyData(notifyInfo));
        HCCL_INFO("[HcclCommunicator][GetAiCpuNotifyData]esId[%lld], addr[%lld], devId[%u], tsId[%u].",
                  notifyInfo.resId, notifyInfo.addr, notifyInfo.devId, notifyInfo.tsId);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateAndGetAiCpuNotify(std::shared_ptr<LocalNotify> &localNotify,
                                                         HcclSignalInfo &notifyInfo)
    {
        if (localNotify != nullptr) {
            CHK_RET(GetAiCpuNotifyData(localNotify, notifyInfo));
            HCCL_INFO("[HcclCommunicator][CreateAndGetAiCpuNotify]aicpu notify allready create ptr[%p]",
                      localNotify->ptr());
            return HCCL_SUCCESS;
        }

        EXECEPTION_CATCH((localNotify = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
        CHK_RET(localNotify->Init(NotifyLoadType::DEVICE_NOTIFY));
        CHK_RET(localNotify->SetIpc());

        CHK_RET(GetAiCpuNotifyData(localNotify, notifyInfo));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream)
    {
        if (opStream_.ptr() != nullptr) {
            HCCL_INFO("%s allready alloc, group:%s, stream id:%u", __func__, identifier_.c_str(), opStream_.id());
            aiCpuStream = opStream_.ptr();
            return HCCL_SUCCESS;
        }

        constexpr u32 aicpuStreamMode = 1; // 单独申请的kernel流，使能遇错即停，避免出错后流卡住不退
        opStream_ = Stream(StreamType::STREAM_TYPE_ONLINE);
        CHK_RET(hrtStreamSetMode(opStream_.ptr(), aicpuStreamMode));
        aiCpuStream = opStream_.ptr();
        HCCL_RUN_INFO("%s alloc success, group:%s, stream id:%u, mainStreamMode:%u, aicpuStreamMode:%u",
                      __func__, identifier_.c_str(), opStream_.id(), streamMode, aicpuStreamMode);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::Mc2AiCpuInitStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream)
    {
        if (aicpuInitStream_.ptr() != nullptr) {
            HCCL_INFO("%s allready alloc, group:%s, stream id:%u", __func__, identifier_.c_str(), aicpuInitStream_.id());
            aiCpuStream = aicpuInitStream_.ptr();
            return HCCL_SUCCESS;
        }

        constexpr u32 aicpuStreamMode = 1; // 单独申请的kernel流，使能遇错即停，避免出错后流卡住不退
        aicpuInitStream_ = Stream(StreamType::STREAM_TYPE_ONLINE);
        CHK_RET(hrtStreamSetMode(aicpuInitStream_.ptr(), aicpuStreamMode));
        aiCpuStream = aicpuInitStream_.ptr();
        HCCL_RUN_INFO("%s alloc success, group:%s, stream id:%u, mainStreamMode:%u, aicpuStreamMode:%u",
                      __func__, identifier_.c_str(), aicpuInitStream_.id(), streamMode, aicpuStreamMode);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AiCpuKernelLaunch(const rtStream_t stm, u64 addr, const std::string &kernelName)
    {
        uint64_t beginTime = hrtMsprofSysCycleTime();
        const std::string profName = "hcomAicpuInit";
        rtAicpuArgsEx_t argsInfo;
        struct ApiParamDef
        {
            u64 commContext;
            bool isCustom;
            char kernelName[64] = "";
            char soName[64] = "libccl_kernel.so";
            char opName[64] = "HcclAicpuOp";
        };
        struct ApiParamDef apiParam;
        CHK_SAFETY_FUNC_RET(
            memcpy_s(apiParam.kernelName, sizeof(apiParam.kernelName), kernelName.c_str(), kernelName.length() + 1));
        apiParam.commContext = static_cast<uint64_t>(addr);
        apiParam.isCustom = false;

        if (mc2DeviceMem_.ptr() == nullptr) {
            mc2DeviceMem_ = DeviceMem::alloc(sizeof(apiParam));
        }
        CHK_SMART_PTR_NULL(mc2DeviceMem_);
        CHK_RET(hrtMemSyncCopy(mc2DeviceMem_.ptr(), sizeof(apiParam), &apiParam, sizeof(apiParam),
                               HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        argsInfo.args = mc2DeviceMem_.ptr();
        argsInfo.hostInputInfoPtr = nullptr;
        argsInfo.kernelOffsetInfoPtr = nullptr;
        argsInfo.argsSize = sizeof(apiParam);
        argsInfo.hostInputInfoNum = 0;
        argsInfo.kernelOffsetInfoNum = 0;
        argsInfo.soNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.soName) -
                                                          reinterpret_cast<const char *>(&apiParam));

        argsInfo.kernelNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.kernelName) -
                                                              reinterpret_cast<const char *>(&apiParam));
        argsInfo.isNoNeedH2DCopy = true;
        CHK_RET(hrtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU, apiParam.opName, 1, &argsInfo, nullptr, stm, 0));
        uint64_t endTime = hrtMsprofSysCycleTime();
        s32 threadId = SalGetTid();
        CHK_RET(ProfilingManagerPub::CallMsprofReportNodeInfo(beginTime, endTime, profName, threadId));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitKernelArgsPrepare(aclrtBinHandle binHandle, const std::string &kernelName,
                                                       void *initTaskAddr, u32 initTaskSize, aclrtFuncHandle &funcHandle, aclrtArgsHandle &argsHandle)
    {
        aclError ret = aclrtBinaryGetFunction(binHandle, kernelName.c_str(), &funcHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, kernelName:%s",
                               ret, kernelName.c_str()),
                    HCCL_E_RUNTIME);

        ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName:%s", ret, kernelName.c_str()),
                    HCCL_E_RUNTIME);

        aclrtParamHandle paraHandle;
        ret = aclrtKernelArgsAppend(argsHandle, initTaskAddr, initTaskSize, &paraHandle); // 假设opResPara_为结构体参数
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append size %u, kernelName:%s", ret,
                               initTaskSize, kernelName.c_str()),
                    HCCL_E_RUNTIME);

        ret = aclrtKernelArgsFinalize(argsHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName:%s", ret,
                               kernelName.c_str()),
                    HCCL_E_RUNTIME);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::TaskCommKernelArgsPrepare(aclrtBinHandle binHandle, const std::string &kernelName,
                                                           void *contextAddr, u32 contextSize, void *tilingDataPtr, u32 tilingDataSize, aclrtFuncHandle &funcHandle,
                                                           aclrtArgsHandle &argsHandle)
    {
        aclError ret = aclrtBinaryGetFunction(binHandle, kernelName.c_str(), &funcHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, kernelName:%s",
                               ret, kernelName.c_str()),
                    HCCL_E_RUNTIME);

        ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName:%s", ret, kernelName.c_str()),
                    HCCL_E_RUNTIME);
        // 拼凑aicpu侧KFCTaskComm结构体
        // 1、先存放HcclOpResParam的context指针
        aclrtParamHandle paraHandle;
        ret = aclrtKernelArgsAppend(argsHandle, contextAddr, contextSize, &paraHandle); // 假设opResPara_为结构体参数
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append size %u, kernelName:%s", ret,
                               contextSize, kernelName.c_str()),
                    HCCL_E_RUNTIME);
        // 2、再将OpTilingData的tilingData指针拼凑，并将tilingData的hostMem给rts进行H2D
        void *dataAddr;
        ret = aclrtKernelArgsAppendPlaceHolder(argsHandle, &paraHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtKernelArgsAppendPlaceHolder]errNo[0x%016llx] args append place holder failed",
                               ret),
                    HCCL_E_RUNTIME);

        ret = aclrtKernelArgsGetPlaceHolderBuffer(argsHandle, paraHandle, tilingDataSize, &dataAddr);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtKernelArgsGetPlaceHolderBuffer]errNo[0x%016llx] args get place holder buffer failed,"
                               "tilingDataSize %u",
                               ret, tilingDataSize),
                    HCCL_E_RUNTIME);
        CHK_SAFETY_FUNC_RET(memcpy_s(dataAddr, tilingDataSize, tilingDataPtr, tilingDataSize));

        ret = aclrtKernelArgsFinalize(argsHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName:%s", ret,
                               kernelName.c_str()),
                    HCCL_E_RUNTIME);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuAclKernelLaunch(const rtStream_t stm, void *addr, u32 size, aclrtBinHandle binHandle,
                                                      const std::string &kernelName, bool isInitTask, void *tilingDataPtr, u32 tilingDataSize)
    {
        if (binHandle == nullptr) {
            HCCL_ERROR("binHandle is nullptr, no need to launch aicpu kernel, binHandle[%p], customHandle[%p]", binHandle_,
                       binCustomHandle_);
            return HCCL_E_PTR;
        }

        aclrtFuncHandle funcHandle;
        aclrtArgsHandle argsHandle;
        HcclResult ret;
        if (isInitTask) {
            ret = InitKernelArgsPrepare(binHandle, kernelName, addr, size, funcHandle, argsHandle);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[InitKernelArgsPrepare]errNo[0x%016llx]init args prepare failed, kernelName:%s", ret,
                                   kernelName.c_str()),
                        HCCL_E_RUNTIME);
        } else {
            ret = TaskCommKernelArgsPrepare(binHandle, kernelName, addr, size, tilingDataPtr, tilingDataSize, funcHandle,
                                            argsHandle);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[TaskCommKernelArgsPrepare]errNo[0x%016llx]taskCOmm args prepare failed, kernelName:%s", ret,
                                   kernelName.c_str()),
                        HCCL_E_RUNTIME);
        }

        aclrtLaunchKernelCfg cfg;
        aclrtLaunchKernelAttr attr;
        attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
        attr.value.timeout = static_cast<u16>((opResPara_.config.notifyWaitTime == 0) ? opResPara_.config.notifyWaitTime : (opResPara_.config.notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC));
        cfg.numAttrs = 1;
        cfg.attrs = &attr;
        constexpr u32 blockDim = 1;
        aclError aclRet = aclrtLaunchKernelWithConfig(funcHandle, blockDim, stm, &cfg, argsHandle, nullptr);
        CHK_PRT_RET(aclRet != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed", ret), HCCL_E_OPEN_FILE_FAILURE);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuKfcTilingDataLaunch(const OpParam &opParam, const HcclCMDType &opType,
                                                          const DeviceMem &deviceContext, const std::string &kernelName, const AicpuOpTiling opTilingInfo)
    {
        HCCL_DEBUG("AicpuKfcTilingDataLaunch count %llu dataType %s op %s opType %u", opParam.GetDataCount(userRank_),
                   GetDataTypeEnumStr(opParam.GetDataType()).c_str(), GetReduceOpEnumStr(opParam.reduceType).c_str(), opType);
        struct HcclKFCTilingData tilingDate = {0};
        tilingDate.sendCnt = opParam.DataDes.count;
        tilingDate.dataType = opParam.DataDes.dataType;
        tilingDate.commType = static_cast<uint8_t>(opType);
        tilingDate.reduceOp = opParam.reduceType;
        tilingDate.taskType = HCCL_KFC_TASK_HCCL_ONLY_EXE;
        tilingDate.totalCnt = 1;
        tilingDate.turnNum = 1;
        tilingDate.hasCommOut = 1;
        tilingDate.debugMode = 0;
        CHK_RET(SetNormalMode(dispatcher_));
        HcclWorkflowMode mode = GetWorkflowMode();
        Stream mainStream(opParam.stream.ptr());
        CHK_RET(LocalNotify::Post(mainStream, dispatcher_,
            localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)], INVALID_VALUE_STAGE));
        rtStream_t kfcOpStream = opStream_.ptr();
        if (opTilingInfo.isUsedMainStream) {
            kfcOpStream = opParam.stream.ptr();
        }
        CHK_RET(AicpuUnfoldKernelLaunch(opParam.inputPtr, opParam.outputPtr, kfcOpStream,
                                        reinterpret_cast<u64>(deviceContext.ptr()), &tilingDate, sizeof(HcclKFCTilingData),
                                        kernelName, mode, opParam.tag));
        CHK_RET(LocalNotify::Wait(mainStream, dispatcher_,
            localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)], INVALID_VALUE_STAGE));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuInitOpTilingDataBuf(const OpParam &opParam, const HcclCMDType &opType,
                                                          const std::string &kernelName, const AicpuOpTiling opTilingInfo, u64 dynamicDataSize)
    {
        u32 opTilingDataSize = sizeof(struct OpTilingData) + dynamicDataSize;

        if (opTilingDataBuf_.ptr() == nullptr) {
            opTilingDataBuf_ = HostMem::alloc(TILINGDATA_BUF_SIZE);
            CHK_PRT_RET(opTilingDataBuf_.ptr() == nullptr,
                        HCCL_ERROR("[HcclCommunicator][AicpuInitOpTilingDataBuf] Alloc opTilingDataBuf failed!"),
                        HCCL_E_INTERNAL);
        }

        if (opTilingDataBuf_.ptr() != nullptr && opTilingDataSize > opTilingDataBuf_.size()) {
            opTilingDataBuf_.free();
            opTilingDataBuf_ = HostMem::alloc(opTilingDataSize);
            CHK_PRT_RET(opTilingDataBuf_.ptr() == nullptr,
                        HCCL_ERROR("[HcclCommunicator][AicpuInitOpTilingDataBuf] increate opTilingDataBuf len[%llu] failed!",
                                   opTilingDataSize),
                        HCCL_E_INTERNAL);
        }

        // 填充固定内容
        HostMem opTilingDataMem = opTilingDataBuf_.range(0, opTilingDataSize);
        struct OpTilingData *opTilingData = static_cast<struct OpTilingData *>(opTilingDataMem.ptr());
        u32 algTypeTranfer = (static_cast<u32>(opTilingInfo.algType.algoLevel2) << (HCCL_LEVEL_ALGO_WIDTH + HCCL_LEVEL_ALGO_WIDTH)) +
                             (static_cast<u32>(opTilingInfo.algType.algoLevel1) << HCCL_LEVEL_ALGO_WIDTH) +
                             static_cast<u32>(opTilingInfo.algType.algoLevel0);
        opTilingData->algType = static_cast<u64>(algTypeTranfer);
        opTilingData->floatOverflowMode = opTilingInfo.floatOverflowMode;
        opTilingData->dumpDebug = opTilingInfo.dumpDebug;
        CHK_RET(AicpuInitOpTilingDataFromOpParam(opParam, opType, opTilingData));
        opTilingData->length = dynamicDataSize;
        opTilingData->customDataLength = 0;
        opTilingData->index = UpdateOpIndex(opParam);
        opTilingData->debugMode = 0;
        opTilingData->isZeroCopy = opParam.isZeroCopy;

        // 填充动态内容
        HostMem dynamicDataMem = opTilingDataBuf_.range(sizeof(struct OpTilingData), dynamicDataSize);
        CHK_PTR_NULL(dynamicDataMem.ptr());
        if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            struct OpTilingBatchSendRecvDataDes *batchSendRecvDataPtr =
                reinterpret_cast<struct OpTilingBatchSendRecvDataDes *>(dynamicDataMem.ptr());
            batchSendRecvDataPtr->itemNum = opParam.BatchSendRecvDataDes.itemNum;
            for (u32 i = 0; i < opParam.BatchSendRecvDataDes.itemNum; i++)
            {
                CHK_PTR_NULL(opParam.BatchSendRecvDataDes.sendRecvItemsPtr + i);
                batchSendRecvDataPtr->batchSendRecvItem[i] = *(opParam.BatchSendRecvDataDes.sendRecvItemsPtr + i);
            }
        } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
            CHK_RET(SetDynamicTilingDataAlltoall(opParam, dynamicDataMem));
        } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            CHK_RET(SetDynamicTilingDataAlltoallv(opParam, dynamicDataMem, opTilingInfo.algName));
        } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
            CHK_RET(SetDynamicTilingDataAlltoallvc(opParam, dynamicDataMem));
        } else if (opType == HcclCMDType::HCCL_CMD_ALLGATHER_V || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
            CHK_RET(SetDynamicTilingDataV(opParam, dynamicDataMem));
        } else {
            struct OpTilingDataDes *opDataDesPtr = reinterpret_cast<struct OpTilingDataDes *>(dynamicDataMem.ptr());
            opDataDesPtr->count = opParam.DataDes.count;
            opDataDesPtr->dataType = static_cast<u8>(opParam.DataDes.dataType);
        }
        HCCL_INFO("[HcclCommunicator][AicpuInitOpTilingDataBuf]algType[%lu]", opTilingData->algType);
        CHK_SAFETY_FUNC_RET(memcpy_s(opTilingData->algName, sizeof(opTilingData->algName), opTilingInfo.algName.c_str(),
                                     opTilingInfo.algName.length() + 1));
        CHK_SAFETY_FUNC_RET(memcpy_s(opTilingData->newTag, sizeof(opTilingData->newTag),
                                     opTilingInfo.newTag.c_str(), opTilingInfo.newTag.length() + 1));
        CHK_SAFETY_FUNC_RET(memcpy_s(opTilingData->tag, sizeof(opTilingData->tag), opParam.tag.c_str(),
                                     opParam.tag.length() + 1));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuKfcTilingDataLaunchIn(const OpParam &opParam, const DeviceMem &deviceContext,
                                                            const std::string &kernelName, const AicpuOpTiling opTilingInfo, u64 opTilingDataSize, bool isCustom)
    {
        HostMem opTilingDataMem = opTilingDataBuf_.range(0, opTilingDataSize);
        CHK_RET(SetNormalMode(dispatcher_));
        Stream &mainStream = const_cast<Stream &>(opParam.stream);
        CHK_RET(LocalNotify::Post(mainStream, dispatcher_,
            localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)], INVALID_VALUE_STAGE));

        // 使能重执行时，在主流下kernel，避免提前展开占核
        // aclgraph：实际执行model的流是另一条流，无法保序；使用从流下发，从流AddToModel后通过RTS插入event保序
        bool isKfcUseMainStream = (opTilingInfo.isUsedMainStream || retryEnable_) && !opParam.isCapture;
        rtStream_t kfcOpStream = isKfcUseMainStream ? opParam.stream.ptr() : opStream_.ptr();
        HcclWorkflowMode mode = GetWorkflowMode();
        // 如果是图模式，则尝试从附属从流中获取一下stream，如果能拿到则使用，否则用原有的
        if (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
            !attachedStreams_.empty() && attachedStreams_[0].ptr() != nullptr) {
            kfcOpStream = attachedStreams_[0].ptr();
            HCCL_INFO("[HcclCommunicator][AicpuKfcTilingDataLaunchExt] Use attached stream [%p]", kfcOpStream);
        }
        uint64_t beginTime = hrtMsprofSysCycleTime();
        std::string profName = GetCMDTypeEnumStr(opParam.opType);
        if (profName == "Invalid HcclCMDType" || profName == "invalid") {
            profName = "HcclOpAicpuKernel";
        } else {
            profName += "AicpuKernel";
        }
        s32 streamId = 0;
        CHK_RET(hrtGetStreamId(kfcOpStream, streamId));
        auto getAicpuTaskExceptionCallBack = [this]() { 
            return this->GetAicpuTaskException(); 
        };
        RegisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId_, getAicpuTaskExceptionCallBack);
        if (streamId != opParam.stream.id()) {
            RegisterGetAicpuTaskExceptionCallBack(opParam.stream.id(), deviceLogicId_, getAicpuTaskExceptionCallBack);
        }
        HCCL_INFO("%s profName[%s] tag[%s] kfcOpStreamId[%d] mainStreamId[%u] kfcStreamId[%d]",
            __func__, profName.c_str(), opParam.tag.c_str(), streamId, opParam.stream.id(), opStream_.id());

        if (opParam.isCapture && !isKfcUseMainStream) { // 非主流下发时，acl graph场景，capture从流
            rtModel_t rtModel = nullptr;
            bool isCapture = false;
            CHK_RET(GetStreamCaptureInfo(opParam.stream.ptr(), rtModel, isCapture));
            CHK_PTR_NULL(rtModel);
            CHK_RET(AddStreamToModel(kfcOpStream, rtModel));
            HCCL_INFO("[HcclCommunicator][%s]Tag[%s], Add stream[%d] to model success.", __func__, opParam.tag.c_str(),
                      streamId);
        }

        u32 timeOut = (opResPara_.config.notifyWaitTime == 0) ? opResPara_.config.notifyWaitTime :
                                                               (opResPara_.config.notifyWaitTime + AICPU_H2D_TIMEOUT_INC);
        // 单算子在主流下kernel的场景，需要添加和kfc流之间的同步，避免和mc2算子并发
        bool needSyncMainToKfc = (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && streamId == mainStream.id());
        if (needSyncMainToKfc) {
            // 主流 wait kfc流
            CHK_RET(LocalNotify::Post(opStream_, dispatcher_,
                localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::MAIN_TO_KFC_0)], INVALID_VALUE_STAGE));
            CHK_RET(LocalNotify::Wait(mainStream, dispatcher_,
                localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::MAIN_TO_KFC_0)], INVALID_VALUE_STAGE, timeOut));
        }

        CHK_RET(KernelLaunchChooseAicpuOrCustom(opParam.inputPtr, opParam.outputPtr, kfcOpStream,
                                                reinterpret_cast<u64>(deviceContext.ptr()), opTilingDataMem.ptr(), opTilingDataSize,
                                                kernelName, mode, opParam.tag, isCustom));
        
        if (needSyncMainToKfc) {
            // 主流 record kfc流
            CHK_RET(LocalNotify::Post(mainStream, dispatcher_,
                localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::MAIN_TO_KFC_1)], INVALID_VALUE_STAGE));
            CHK_RET(LocalNotify::Wait(opStream_, dispatcher_,
                localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::MAIN_TO_KFC_1)], INVALID_VALUE_STAGE, timeOut));
        }

        uint64_t endTime = hrtMsprofSysCycleTime();
        s32 threadId = SalGetTid();
        CHK_RET(ProfilingManagerPub::CallMsprofReportNodeInfo(beginTime, endTime, profName, threadId));
        CHK_RET(LocalNotify::Wait(mainStream, dispatcher_,
            localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)], INVALID_VALUE_STAGE, timeOut));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuKfcTilingDataLaunchExt(const OpParam &opParam, const HcclCMDType &opType,
                                                             const DeviceMem &deviceContext, const std::string &kernelName, const AicpuOpTiling opTilingInfo,
                                                             bool isCustom)
    {
        const u64 dataCount = opParam.GetDataCount(userRank_);
        const HcclDataType dataType = opParam.GetDataType();
        HCCL_DEBUG("AicpuKfcTilingDataLaunchExt count %llu dataType %s op %s opType %u retryEnable_ %d, "
                   "inPlaceSupportRetryStatus_ %d",
                   dataCount, GetDataTypeEnumStr(dataType).c_str(),
                   GetReduceOpEnumStr(opParam.reduceType).c_str(), opType, retryEnable_, inPlaceSupportRetryStatus_);

        u32 severNum4PostSync = 4;
        bool needPostSync = superPodNum_ > 1 || serverNum_ >= severNum4PostSync; // reduce/reduce scatter算子是否需要PostSync
        if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE &&
            retryEnable_ && (inPlaceSupportRetryStatus_ == InplaceSupportRetryStatus::USER_LARGER_THAN_CCL)) {
            u32 itemNum = 2;
            for (u32 i = 0; i < itemNum; i++) {
                if (i == 0) {
                    isInplacePreSync_ = true;
                } else {
                    isInplacePreSync_ = false;
                }
                HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with isInplacePreSync_[%d].",
                           isInplacePreSync_);
                u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
                CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
                CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
                                                   sizeof(struct OpTilingData) + dynamicDataSize, isCustom));
                isInplacePreSync_ = false;
            }
        } else if (opType == HcclCMDType::HCCL_CMD_REDUCE && retryEnable_ && needPostSync) {
            isPostSync_ = true;
            HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with isPostSync_[%d].",
                       isPostSync_);
            u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
            CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
            CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
                                               sizeof(struct OpTilingData) + dynamicDataSize, isCustom));
            isPostSync_ = false;
        } else if (retryEnable_ && opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
            if (inPlaceSupportRetryStatus_ == InplaceSupportRetryStatus::USER_LARGER_THAN_CCL) {
                isInplacePreSync_ = true;
                HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with isInplacePreSync_[%d].",
                           isInplacePreSync_);
                u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
                CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
                CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
                                                   sizeof(struct OpTilingData) + dynamicDataSize, isCustom));
                isInplacePreSync_ = false;
            }
            isInplacePreSync_ = false;
            if (needPostSync) {
                isPostSync_ = true;
            }
            HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with "
                       "isInplacePreSync_[%d], isPostSync_[%d].",
                       isInplacePreSync_, isPostSync_);
            u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
            CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
            CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
                                               sizeof(struct OpTilingData) + dynamicDataSize, isCustom));
            isPostSync_ = false;
        } else if (retryEnable_ &&
                 (opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
                  opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
                  opType == HcclCMDType::HCCL_CMD_ALLTOALLVC)) {
            isPostSync_ = true;
            HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with "
                       "isInplacePreSync_[%d], isPostSync_[%d].",
                       isInplacePreSync_, isPostSync_);
            u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
            CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
            CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
                                               sizeof(struct OpTilingData) + dynamicDataSize, isCustom));
            isPostSync_ = false;
        } else {
            u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
            HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt]dynamicDataSize[%u]", dynamicDataSize);
            CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
            CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
                                               sizeof(struct OpTilingData) + dynamicDataSize, isCustom));
        }

        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuUnfoldKernelLaunch(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
                                                         void *tilingDataPtr, u32 tilingDataSize, const std::string &kernelName, HcclWorkflowMode mode, const std::string &tag)
    {
        struct ApiParamDef
        {
            uint64_t x1; // 算子sendbuffer地址
            uint64_t y = 0;
            uint64_t gatherOut;     // 算子recvbuffer地址
            uint64_t context;       // 通信资源准备的地址
            uint64_t workspace;     // 消息区地址
            uint64_t tilingDataPtr; // tilingData地址
            uint8_t tilingData[2048];
            char soName[32] = "libccl_kernel.so";
            char kernelName[32] = "";
            char opName[32] = "HcclAicpuOp";
            char hostInputInfo[16];
        };

        struct ApiParamDef apiParam;
        CHK_SAFETY_FUNC_RET(
            memcpy_s(apiParam.kernelName, sizeof(apiParam.kernelName), kernelName.c_str(), kernelName.length() + 1));
        apiParam.x1 = reinterpret_cast<uint64_t>(inputPtr);
        apiParam.gatherOut = reinterpret_cast<uint64_t>(outputPtr);
        apiParam.context = addr;
        apiParam.workspace = (u64)workSpace_.ptr();
        CHK_SAFETY_FUNC_RET(memcpy_s(apiParam.tilingData, sizeof(apiParam.tilingData), tilingDataPtr,
                                     tilingDataSize));

        rtAicpuArgsEx_t argsInfo;

        argsInfo.args = (void *)&apiParam;
        apiParam.tilingDataPtr = reinterpret_cast<uint64_t>(apiParam.tilingData);

        rtHostInputInfo_t *hostInfo = (rtHostInputInfo_t *)apiParam.hostInputInfo;
        hostInfo->addrOffset = 5 * sizeof(void *); // aclnn与aicore协定，addr地址偏移时5*(void*）
        hostInfo->dataOffset = 6 * sizeof(void *); // aclnn与aicore协定，data偏移6*(void*)
        argsInfo.hostInputInfoPtr = hostInfo;
        argsInfo.kernelOffsetInfoPtr = nullptr;
        argsInfo.argsSize = sizeof(apiParam);
        argsInfo.hostInputInfoNum = 1;
        argsInfo.kernelOffsetInfoNum = 0;
        argsInfo.soNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.soName) -
                                                          reinterpret_cast<const char *>(&apiParam));

        argsInfo.kernelNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.kernelName) -
                                                              reinterpret_cast<const char *>(&apiParam));
        argsInfo.isNoNeedH2DCopy = false;

        CHK_RET(hrtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, apiParam.opName, 1, &argsInfo, nullptr, stm, 0));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuUnfoldKernelLaunchV2(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
                                                           void *tilingDataPtr, u32 tilingDataSize, const std::string &kernelName, HcclWorkflowMode mode, const std::string &tag)
    {
        struct ApiParamDef
        {
            uint64_t context;
            uint64_t tilingDataPtr; // tilingData 地址
            char soName[32];
            char kernelName[32];
            char opName[32];
            char hostInputInfo[16];
            uint8_t tilingData[];
        };

        u32 apiTilingDataSize = sizeof(struct ApiParamDef) + tilingDataSize;
        if (apiTilingDataMem_.ptr() == nullptr) {
            apiTilingDataMem_ = HostMem::alloc(TILINGDATA_BUF_SIZE);
            CHK_PRT_RET(apiTilingDataMem_.ptr() == nullptr,
                        HCCL_ERROR("[HcclCommunicator][AicpuUnfoldKernelLaunch] Alloc apiTilingDataMem_ failed!"),
                        HCCL_E_INTERNAL);
            std::string opName = "HcclAicpuOp";
            std::string soName = "libccl_kernel.so";
            struct ApiParamDef *apiParamTmp = static_cast<struct ApiParamDef *>(apiTilingDataMem_.ptr());
            CHK_SAFETY_FUNC_RET(
                memcpy_s(apiParamTmp->soName, sizeof(apiParamTmp->soName), soName.c_str(), soName.length() + 1));
            CHK_SAFETY_FUNC_RET(
                memcpy_s(apiParamTmp->opName, sizeof(apiParamTmp->opName), opName.c_str(), opName.length() + 1));
        }
        void *pytr;
        HostMem apiTilingDataMemTmp_;
        bool isNeedReAlloc = apiTilingDataSize > TILINGDATA_BUF_SIZE;
        if (isNeedReAlloc) {
            apiTilingDataMemTmp_ = HostMem::alloc(apiTilingDataSize);
            CHK_PRT_RET(apiTilingDataMemTmp_.ptr() == nullptr,
                        HCCL_ERROR("[HcclCommunicator][AicpuUnfoldKernelLaunch] Alloc apiTilingDataMemTmp_ failed!"),
                        HCCL_E_INTERNAL);
            HCCL_INFO("[HcclCommunicator][AicpuUnfoldKernelLaunch] apiTilingDataSize is larger than "
                      "the size of TILINGDATA_BUF_SIZE.");
            std::string opName = "HcclAicpuOp";
            std::string soName = "libccl_kernel.so";
            struct ApiParamDef *apiParamTmp = static_cast<struct ApiParamDef *>(apiTilingDataMemTmp_.ptr());
            CHK_SAFETY_FUNC_RET(
                memcpy_s(apiParamTmp->soName, sizeof(apiParamTmp->soName), soName.c_str(), soName.length() + 1));
            CHK_SAFETY_FUNC_RET(
                memcpy_s(apiParamTmp->opName, sizeof(apiParamTmp->opName), opName.c_str(), opName.length() + 1));
            pytr = apiTilingDataMemTmp_.range(0, apiTilingDataSize).ptr();
        } else {
            pytr = apiTilingDataMem_.range(0, apiTilingDataSize).ptr();
        }
        struct ApiParamDef *apiParam = static_cast<struct ApiParamDef *>(pytr);
        CHK_SAFETY_FUNC_RET(
            memcpy_s(apiParam->kernelName, sizeof(apiParam->kernelName), kernelName.c_str(), kernelName.length() + 1));
        CHK_SAFETY_FUNC_RET(
            memcpy_s(apiParam->tilingData, tilingDataSize, tilingDataPtr, tilingDataSize));
        apiParam->context = addr;

        rtAicpuArgsEx_t argsInfo;
        argsInfo.args = (void *)apiParam;
        apiParam->tilingDataPtr = reinterpret_cast<uint64_t>(apiParam->tilingData);
        rtHostInputInfo_t *hostInfo = (rtHostInputInfo_t *)(apiParam->hostInputInfo);
        hostInfo->addrOffset = 1 * sizeof(void *);  // tiliingDataPtr, addr地址偏移时1*(void*）
        hostInfo->dataOffset = 16 * sizeof(void *); // tilingData，data偏移16*(void*)
        argsInfo.hostInputInfoPtr = hostInfo;
        argsInfo.kernelOffsetInfoPtr = nullptr;
        argsInfo.argsSize = apiTilingDataSize;
        argsInfo.hostInputInfoNum = 1;
        argsInfo.kernelOffsetInfoNum = 0;
        argsInfo.soNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&(apiParam->soName)) -
                                                          reinterpret_cast<const char *>(apiParam));
        argsInfo.kernelNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&(apiParam->kernelName)) -
                                                              reinterpret_cast<const char *>(apiParam));
        argsInfo.timeout = static_cast<u16>((opResPara_.config.notifyWaitTime == 0) ? opResPara_.config.notifyWaitTime : (opResPara_.config.notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC));
        argsInfo.isNoNeedH2DCopy = false;
        CHK_RET(hrtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, apiParam->opName, 1, &argsInfo, nullptr, stm,
                                               RT_KERNEL_USE_SPECIAL_TIMEOUT));
        HCCL_INFO("[HcclCommunicator][AicpuUnfoldKernelLaunchV2] exec succ.");
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::AicpuUnfoldKernelLaunchIsCustom(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
                                                                 void *tilingDataPtr, u32 tilingDataSize, const std::string &kernelName, HcclWorkflowMode mode, const std::string &tag, bool isCustom)
    {
        u64 context = addr;
        HCCL_INFO("[HcclCommunicator]context[%p] tilingDataPtr[%p] tilingData[%p]", context,
                  tilingDataPtr, tilingDataSize);

        aclrtBinHandle binHandle = isCustom ? binCustomHandle_ : binHandle_;
        if (isCustom == true && binHandle == nullptr) {
            HCCL_ERROR("[AicpuUnfoldKernelLaunchIsCustom]custom task but custom switch is not open, please check.");
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_PRT(AicpuAclKernelLaunch(stm, reinterpret_cast<void *>(&context), sizeof(context), binHandle, kernelName,
                                     false, tilingDataPtr, tilingDataSize));
        HCCL_INFO("[HcclCommunicator][AicpuUnfoldKernelLaunchIsCustom] exec succ.");
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::InitCombinOpara()
    {
        CHK_SAFETY_FUNC_RET(memset_s(&combinOpara_, sizeof(combinOpara_), 0, sizeof(combinOpara_)));

        combinOpara_.rankId = INVALID_UINT;
        combinOpara_.signalInfo.aicpuNotify.rankId = INVALID_UINT;

        for (u32 i = 0; i < sizeof(combinOpara_.signalInfo.noIpcNotifys) / sizeof(combinOpara_.signalInfo.noIpcNotifys[0]);
             i++) {
            combinOpara_.signalInfo.noIpcNotifys[i].rankId = INVALID_UINT;
        }

        for (u32 i = 0; i < sizeof(combinOpara_.signalInfo.ipcNotifys) / sizeof(combinOpara_.signalInfo.ipcNotifys[0]);
             i++) {
            combinOpara_.signalInfo.ipcNotifys[i].rankId = INVALID_UINT;
        }

        for (u32 i = 0; i < sizeof(combinOpara_.signalInfo.noIpcEvents) / sizeof(combinOpara_.signalInfo.noIpcEvents[0]);
             i++) {
            combinOpara_.signalInfo.noIpcEvents[i].rankId = INVALID_UINT;
        }
        return HCCL_SUCCESS;
    }

    bool HcclCommunicator::GetCommResource(const std::string &tag, void **commContext)
    {
        if (LIKELY(IsExistCommRes(tag))) {
            *commContext = commContext_.ptr();
            return true;
        }
        return false;
    }

    bool HcclCommunicator::GetCommResource(void *&commContext)
    {
        commContext = opResDevicePara_.ptr();
        return true;
    }

    HcclResult HcclCommunicator::GetAicpuOpStreamNotify(HcclRtStream *opStream, u8 aicpuNotifyNum, void **aicpuNotify)
    {
        CHK_RET(GetAicpuOpStreamAndNotify(opStream, aicpuNotifyNum, aicpuNotify));
        HCCL_INFO("[HcclCommunicator][GetAicpuOpStreamNotify]opStream %p aicpuNotify %p.", *opStream, *aicpuNotify);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetAicpuOpStreamAndNotify(HcclRtStream *opStream, u8 aicpuNotifyNum, void **aicpuNotify)
    {
        *opStream = opStream_.ptr();
        if (localAiCpuNotifyRes_.size() < aicpuNotifyNum) {
            for (u16 i = localAiCpuNotifyRes_.size(); i < aicpuNotifyNum; i++) {
                std::shared_ptr<LocalNotify> localNotify = {nullptr};
                HcclSignalInfo aicpuNotify;
                CHK_RET(CreateAndGetAiCpuNotify(localNotify, aicpuNotify));
                localAiCpuNotifyRes_.push_back(localNotify);
            }
        }

        for (u16 i = 0; i < aicpuNotifyNum; i++) {
            *(aicpuNotify + i) = localAiCpuNotifyRes_[i]->ptr();
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetAicpuNotifyInvaild()
    {
        combinOpara_.signalInfo.aicpuNotify.resId = INVALID_U64;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ReplaceCommInfoByTag(const std::string &tag, std::unique_ptr<CommInfo> &commInfo)
    {
        std::unique_lock<std::mutex> replLock(commLock_);
        tagCommInfo_.erase(tag);
        tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(*commInfo)));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateMutiStreamResFor310P(const std::string &tag, level1StreamInfo_t &streamInfo)
    {
        u32 rankSize = GetRankSize();
        u32 pid;
        if (SalGetBareTgid(&pid) != HCCL_SUCCESS) {
            HCCL_DEBUG("get pid fail");
        }
        HCCL_INFO("[HcclCommunicator][CreateMutiStreamRes]tag[%s] ranksize[%u] comminfo ranksize[%u] "
                  "auxRingCommStreamsDev_ size[%u] ringDeviceSignalAux size[%u] ringDeviceSignal size[%u] "
                  "ringDeviceStreams size[%u]",
                  tag.c_str(), rankSize, tagCommInfo_[tag].commIntraServer->RankSize(),
                  auxRingCommStreamsDev_.size(), streamInfo.ringDeviceSignalAux.size(),
                  streamInfo.ringDeviceSignal.size(), streamInfo.ringDeviceStreams.size());
        if (auxRingCommStreamsDev_.empty() || auxRingCommStreamsDev_.size() < rankSize) {
            auxRingCommStreamsDev_.resize(rankSize);
            u32 resNum = rankSize - 1;
            streamInfo.ringDeviceSignalAux.resize(resNum);
            streamInfo.ringDeviceSignal.resize(resNum);
            for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
                auxRingCommStreamsDev_[ringIndex] = Stream(StreamType::STREAM_TYPE_DEVICE);
                // 给device侧申请的流不需要setmode，否则rts会捕获流成员Flags为1024的异常
            }
            for (auto &signal : streamInfo.ringDeviceSignal) {
                signal = nullptr;
            }
            for (auto &signal : streamInfo.ringDeviceSignalAux) {
                signal = nullptr;
            }

            u32 notifyNum = resNum * 2; // 2:Signal + SignalAux
            std::vector<std::shared_ptr<LocalNotify>> notifys(notifyNum, nullptr);
            CHK_RET(queueNotifyManager_->Alloc(tag, notifyNum, notifys, NotifyLoadType::DEVICE_NOTIFY));
            for (u32 i = 0; i < resNum; i++) {
                streamInfo.ringDeviceSignal[i] = notifys[2 * i];
                streamInfo.ringDeviceSignalAux[i] = notifys[2 * i + 1];
            }
        }

        if (streamInfo.ringDeviceStreams.empty() || streamInfo.ringDeviceStreams.size() < rankSize) {
            streamInfo.ringDeviceStreams.resize(rankSize);
            for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
                streamInfo.ringDeviceStreams[ringIndex] = auxRingCommStreamsDev_[ringIndex];
                CHK_SMART_PTR_NULL(streamInfo.ringDeviceStreams[ringIndex]);
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::CreateCommAndStreamRes(const std::string &tag, Stream &stream)
    {
        CHK_SMART_PTR_NULL(implAlg_);
        void *commInputPtr = nullptr;
        void *commOutputPtr = nullptr;
        u64 commInputSize, commOutputSize;

        HcclResult ret = CreateCommCCLbuffer();
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[HcclImplBase][CreateCommAndStreamRes]errNo[0x%016llx],create cclbuff failed",
                               HCCL_ERROR_CODE(ret)),
                    ret);

        ret = CreateCommExpBuffer();
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[HcclImplBase][CreateCommAndStreamRes]errNo[0x%016llx],create expbuff failed",
                               HCCL_ERROR_CODE(ret)),
                    ret);

        if (isA2MC2MultiServer_) {
            // 该场景下ccl buffer有一块区域在上层会被用作flag区，因此需要先清理一下
            CHK_RET(cclBufferManager_.CleanCCLbuffer());
        }

        CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
        CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
        DeviceMem expMem = cclBufferManager_.GetCommExpBuffer();
        DeviceMem inputMem = DeviceMem::create(commInputPtr, commInputSize);
        DeviceMem outputMem = DeviceMem::create(commOutputPtr, commOutputSize);
        AlgType algType;
        AlgType algTypeTmp;

        CHK_RET(GetAlgType(algType, HcclCMDType::HCCL_CMD_ALL));
        algTypeTmp = algType;

        CHK_RET(notifyPool_->RegisterOp(tag));

        // 根据tag创建comm和流资源
        if (!(IsExistCommRes(tag))) {
            std::unique_ptr<CommInfo> commInfo = nullptr;
            HcclResult ret = implAlg_->CreateComm(tag, inputMem, outputMem, algType, commInfo,
                                                  INVALID_VALUE_RANKID, false, true);

            CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR(
                            "[HcclCommunicator][CreateCommAndStreamRes]errNo[0x%016llx]tag[%s],comm resource create comm failed",
                            HCCL_ERROR_CODE(ret),
                            tag.c_str()),
                        ret);

            CHK_RET(ReplaceCommInfoByTag(tag, commInfo));
        }

        if (!(IsExistMutiStreamRes(tag))) {
            level1StreamInfo_t streamInfo;
            std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
            // 2p场景下，mc2当前algType为518，streamInfo.ringNum走默认流程值为1导致资源申请不足，910_93 mc2固定在节点内默认用mesh
            constexpr u32 RANK_SIZE_TWO = 2;
            if ((GetRankSize() == RANK_SIZE_TWO && !isA2MC2MultiServer_) || (deviceType_ == DevType::DEV_TYPE_910_93)) {
                algTypeTmp.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
                algTypeTmp.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
            }
            HcclResult ret = HCCL_SUCCESS;
            if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
                ret = CreateMutiStreamResFor310P(tag, streamInfo);
            } else {
                ret = implAlg_->CreateMutiStreamRes(tag, stream, streamInfo, algTypeTmp, true);
            }
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[HcclCommunicator][CreateCommAndStreamRes]errNo[0x%016llx]tag[%s],comm resource create stream "
                                   "resource",
                                   HCCL_ERROR_CODE(ret),
                                   tag.c_str()),
                        ret);
            tagStreamInfo_.insert(std::pair<std::string, Level1StreamInfo>(tag, std::move(streamInfo)));
            opRetryStreamPtr_->insert(std::make_pair(tag, tagStreamInfo_[tag].ringDeviceStreams));
            mutiStreamLock.unlock();
        }

        HCCL_INFO("resource creation (allreduce) success, tag[%s]", tag.c_str());
        CHK_RET(notifyPool_->UnregisterOp(tag));
        CHK_RET(RegisterToHeartBeat());

        CommBase *comm = nullptr;
        CHK_RET(GetComm(tag, &comm));
        if (comm == nullptr) {
            HCCL_ERROR("comm get err, comm %p", comm);
            return HCCL_E_PTR;
        }
        CHK_RET(SetCommResource(commInputSize, commInputPtr, commOutputPtr, expMem.ptr(),
                                comm, tagStreamInfo_[tag], stream));

        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetComm(const std::string &tag, CommBase **comm)
    {
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            *comm = tagCommInfo_[tag].commIntraServer.get();
        } else if (isA2MC2MultiServer_) {
            // 使用打平RDMA Mesh子通信域
            *comm = tagCommInfo_[tag].commLevel1Rdma[0].get();
        } else {
            *comm = tagCommInfo_[tag].commLevel0[0].get();
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetCommResource(u64 commBufferSize, void *commInPtr, void *commOutPtr, void *commExpPtr,
                                                 CommBase *comm, level1StreamInfo_t &streamInfo, Stream &stream)
    {
        u32 rankSize = comm->RankSize();
        u32 curRankId = comm->Rank();
        u32 usrRankId = comm->UserRank();
        combinOpara_.rankId = curRankId;
        combinOpara_.signalInfo.aicpuNotify.rankId = curRankId;
        combinOpara_.rankNum = rankSize;
        combinOpara_.winSize = commBufferSize;
        combinOpara_.winExpSize = EXP_BUFFER_SIZE;
        combinOpara_.config.deterministic = GetDeterministicConfig();
        combinOpara_.config.notifyWaitTime =
            (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET) ? GetExternalInputHcclExecTimeOut() : NOTIFY_DEFAULT_WAIT_TIME;
        hcclMc2Info_.groupName = hrtMsprofGetHashId(identifier_.c_str(), identifier_.length());
        combinOpara_.config.linkTimeOut = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
        hcclMc2Info_.rankSize = rankSize;
        hcclMc2Info_.rankId = curRankId;
        hcclMc2Info_.usrRankId = usrRankId;
        hcclMc2Info_.aicpuKfcStreamId = static_cast<uint32_t>(stream.id());
        hcclMc2Info_.commStreamSize = rankSize;
        hcclMc2Info_.reserve = 0;
        rtEvent_t event = nullptr;
        u32 eventId = 0;
        u32 idx = 0;
        u32 txSigleBase = 2;
        u32 rxSigleBase = 3;

        if (isA2MC2MultiServer_) {
            // MoE融合算子优化，MC2多机场景
            // 判断是否支持NormalQP创建，若不支持，需要额外下发敲Doorbell任务
            bool isSupportNormalQP = false;
            CHK_RET(IsSupportAicpuNormalQP(devicePhyId_, isSupportNormalQP));
            CHK_RET(SetDevIbverbsData(comm, isSupportNormalQP, commBufferSize, commInPtr, commOutPtr));
            char *mmSysGetEnvValue = nullptr;
            MM_SYS_GET_ENV(MM_ENV_HCCL_INTRA_PCIE_ENABLE, mmSysGetEnvValue);
            std::string intraPcieEnableEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
            bool isMC2Hie = (intraPcieEnableEnv == "1") && (GetExternalInputIntraRoceSwitch() == 0);
            bool isSupportAIVNormalQP = false;
            CHK_RET(IsSupportAIVNormalQP(devicePhyId_, isSupportAIVNormalQP));
            if (isSupportAIVNormalQP && isMC2Hie) {
                CHK_RET(GenAiRMAInfo(comm));
            } else {
                HCCL_WARNING("[%s] db transfer normal qp not support. tag[%s] curRankId[%u] rankNum[%u] isSupportAIVNormalQP[%u]",
                             __func__, comm->Tag().c_str(), aiRMAInfoHost_.curRankId, aiRMAInfoHost_.rankNum, isSupportAIVNormalQP);
            }

            SalSetBitOne(combinedCapability_.dataplaneModeBitmap, POS_DATA_PLANE_MODE_HOST);
            if (isSupportAIVNormalQP) {
                SalSetBitOne(combinedCapability_.dataplaneModeBitmap, POS_DATA_PLANE_MODE_AIV);
            }
            SalSetBitOne(combinedCapability_.dataplaneModeBitmap, POS_DATA_PLANE_MODE_AICPU);
            
            HCCL_INFO("[SetCommResource] Set dataplaneModeBitmap to [%llu]", combinedCapability_.dataplaneModeBitmap);

            // 非NormalQP场景需要传一条流，用于敲Doorbell
            combinOpara_.streamInfo[0].streamIds = streamInfo.ringDeviceStreams[0].id();
            combinOpara_.streamInfo[0].sqIds = streamInfo.ringDeviceStreams[0].sqId();
            combinOpara_.streamInfo[0].cqIds = streamInfo.ringDeviceStreams[0].cqId();
            combinOpara_.streamInfo[0].logicCqids = streamInfo.ringDeviceStreams[0].logicCqId();
            HCCL_DEBUG("[SetCommResource] Set streamInfo[0].streamIds[%u].sqIds[%u].cqIds[%u].logicCqids[%u]",
                       combinOpara_.streamInfo[0].streamIds,
                       combinOpara_.streamInfo[0].sqIds,
                       combinOpara_.streamInfo[0].cqIds,
                       combinOpara_.streamInfo[0].logicCqids);
        } else {
            for (u32 i = 0; i < rankSize; i++) {
                if (i != curRankId) {
                    void *bufferIn;
                    void *bufferOut;
                    std::vector<void *> remotePtrVec;
                    CHK_RET(comm->GetTransportByRank(i)->GetRemoteMem(UserMemType::INPUT_MEM, &bufferIn));
                    combinOpara_.windowsIn[i] = reinterpret_cast<u64>(bufferIn);

                    CHK_RET(comm->GetTransportByRank(i)->GetRemoteMem(UserMemType::OUTPUT_MEM, &bufferOut));
                    combinOpara_.windowsOut[i] = reinterpret_cast<u64>(bufferOut);

                    CHK_RET(comm->GetTransportByRank(i)->GetRemoteMem(&remotePtrVec));
                    if (remotePtrVec.size() != 0) {
                        combinOpara_.windowsExp[i] = reinterpret_cast<u64>(remotePtrVec[0]);
                    }
                    CHK_RET(comm->GetTransportByRank(i)->GetTxAckDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i]));
                    CHK_RET(comm->GetTransportByRank(i)->GetRxAckDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i + rankSize]));
                    CHK_RET(comm->GetTransportByRank(i)->GetTxDataSigleDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase]));
                    CHK_RET(comm->GetTransportByRank(i)->GetRxDataSigleDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase]));
                    CHK_RET(GetAiCpuNotifyData(streamInfo.ringDeviceSignalAux[idx],
                                               combinOpara_.signalInfo.noIpcNotifys[i]));

                    CHK_RET(GetAiCpuNotifyData(streamInfo.ringDeviceSignal[idx],
                                               combinOpara_.signalInfo.noIpcNotifys[i + rankSize]));
                    idx++;
                } else {
                    combinOpara_.windowsIn[i] = reinterpret_cast<u64>(commInPtr);
                    combinOpara_.windowsOut[i] = reinterpret_cast<u64>(commOutPtr);
                    combinOpara_.windowsExp[i] = reinterpret_cast<u64>(commExpPtr);
                    // 在与aicpu商议后，本卡不再防止无效值。后续代码要删掉
                    combinOpara_.signalInfo.ipcNotifys[i].resId = INVALID_U64;
                    combinOpara_.signalInfo.ipcNotifys[i + rankSize].resId = INVALID_U64;
                    combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].resId = INVALID_U64;
                    combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].resId = INVALID_U64;
                }
                HCCL_INFO("group[%s] successfully set windowsIn & windowsOut & windowsExp info: userRank[%u], groupRank[%u], "
                          "windowsIn[0x%llx], InSize[0x%llx], windowOut[0x%llx], OutSize[0x%llx], windowExp[0x%llx], ExpSize[0x%llu]",
                          identifier_.c_str(), GetUserRank(), GetGroupRank(),
                          combinOpara_.windowsIn[i], cclBufferManager_.GetInCCLbufferSize(),
                          combinOpara_.windowsOut[i], cclBufferManager_.GetOutCCLbufferSize(),
                          combinOpara_.windowsExp[i], cclBufferManager_.GetExpBufferSize());

                combinOpara_.signalInfo.ipcNotifys[i].rankId = i;
                combinOpara_.signalInfo.ipcNotifys[i + rankSize].rankId = i;
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].rankId = i;
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].rankId = i;
                combinOpara_.signalInfo.noIpcNotifys[i].rankId = i;

                hcclMc2Info_.commStreamIds[i] = streamInfo.ringDeviceStreams[i].id();
                combinOpara_.streamInfo[i].streamIds = streamInfo.ringDeviceStreams[i].id();
                combinOpara_.streamInfo[i].sqIds = streamInfo.ringDeviceStreams[i].sqId();
                combinOpara_.streamInfo[i].cqIds = streamInfo.ringDeviceStreams[i].cqId();
                combinOpara_.streamInfo[i].logicCqids = streamInfo.ringDeviceStreams[i].logicCqId();
                HCCL_DEBUG("[hccl_Mc2_Info] commStreamIds[%u]:[%u]", i, streamInfo.ringDeviceStreams[i].id());

                CHK_RET(hrtEventCreateWithFlag(&event));

                CHK_RET(hrtGetEventID(event, &eventId));
                aiCpuNoIpcEvnet_.push_back(event);
                combinOpara_.signalInfo.noIpcEvents[i].resId = eventId;
                HCCL_DEBUG("SetCommResource ipc notify info pre record local rankid: %u: remote rankid:%u, resId:%llu, "
                           "devId:%u, tsId:%u, addr:%llu.",
                           curRankId, combinOpara_.signalInfo.ipcNotifys[i].rankId, combinOpara_.signalInfo.ipcNotifys[i].resId,
                           combinOpara_.signalInfo.ipcNotifys[i].devId, combinOpara_.signalInfo.ipcNotifys[i].tsId,
                           combinOpara_.signalInfo.ipcNotifys[i].addr);
                HCCL_DEBUG("SetCommResource ipc notify info pre wait local rankid: %u: remote rankid:%u, resId:%llu, "
                           "devId:%u, tsId:%u, addr:%llu.",
                           curRankId, combinOpara_.signalInfo.ipcNotifys[i + rankSize].rankId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize].resId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize].devId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize].tsId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize].addr);
                HCCL_DEBUG("SetCommResource ipc notify info post record local rankid: %u: remote rankid:%u, resId:%llu, "
                           "devId:%u, tsId:%u, addr:%llu.",
                           curRankId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].rankId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].resId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].devId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].tsId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].addr);
                HCCL_DEBUG("SetCommResource ipc notify info post wait local rankid: %u: remote rankid:%u, resId:%llu, "
                           "devId:%u, tsId:%u, addr:%llu.",
                           curRankId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].rankId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].resId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].devId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].tsId,
                           combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].addr);
            }
        }
        HCCL_DEBUG("[hccl_Mc2_Info] groupname:[%s][%llu], rankSize[%u], rankId[%u], usrRankId[%u], aicpuKfcStreamId[%u], "
                   "commStreamSize[%u]",
                   identifier_.c_str(), hcclMc2Info_.groupName, rankSize, curRankId, usrRankId,
                   static_cast<uint32_t>(stream.id()), rankSize);
        CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(hrtMsprofSysCycleTime(), &hcclMc2Info_,
                                                                 sizeof(hcclMc2Info_)));
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::ReleaseCommContextbuffer()
    {
        commContext_.free();
    }

    HcclResult HcclCommunicator::CreateDeviceCommContext(u64 size, DeviceMem &buffer) const
    {
        CHK_PRT_RET(!size, HCCL_INFO("[Create][DeviceCommContext]device commContext size is zero. "
                                     "not need to malloc memory"),
                    HCCL_SUCCESS);

        CHK_PRT_RET((size > ULONG_MAX),
                    HCCL_ERROR("[Create][DeviceCommContext]device commContext size %llu is large than ULONG_MAX",
                               size),
                    HCCL_E_PARA);

        if (!buffer.ptr()) {
            u64 memSize = size;
            buffer = DeviceMem::alloc(memSize);
            CHK_PRT_RET(size && !buffer, HCCL_ERROR("[Create][DeviceCommContext]Create device commContext size[%llu] fail,"
                                                    "please check deviceCommContext size.",
                                                    size),
                        HCCL_E_PTR);
        }
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::Break()
    {
        if (implAlg_ != nullptr) {
            implAlg_->Break();
        }
        return;
    }

    HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
                                                                   u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize)
    {
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
                        vector<string>({"310P", std::string(__func__) + " is not supported"}));
            HCCL_ERROR("[HcclCommunicator][GetAlltoAllStagedWorkSpaceMemSize]Not Supported!");
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(implAlg_);
        std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(HcclCMDType::HCCL_CMD_ALLTOALLV);
        AlltoAllOperator *alltoAllOperator = dynamic_cast<AlltoAllOperator *>(algOperator.get());
        CHK_PTR_NULL(alltoAllOperator);

        OpParam opParam;
        opParam.All2AllDataDes.sendType = sendType;
        opParam.All2AllDataDes.recvType = recvType;
        opParam.All2AllDataDes.sendCounts = static_cast<void *>(sendCounts);
        opParam.All2AllDataDes.recvCounts = static_cast<void *>(recvCounts);
        opParam.All2AllDataDes.sdispls = static_cast<void *>(sdispls);
        opParam.All2AllDataDes.rdispls = static_cast<void *>(rdispls);
        opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
        opParam.aicpuUnfoldMode = false;

        if (alltoAllOperator->IsSatisfyAlltoAllAivCondition(opParam) ||
            alltoAllOperator->IsSatisfy91093OffloadCondition()) {
            memSize = 0;
            HCCL_INFO("Calculate workSpace MemSize for aiv alltoall done, memSize[%llu]", memSize);
            return HCCL_SUCCESS;
        }

        std::unique_ptr<PreProcessMetaInfo> preMetaInfo = std::make_unique<PreProcessMetaInfo>();
        CHK_SMART_PTR_NULL(preMetaInfo);

        CHK_RET(alltoAllOperator->PrepareAlltoAllAddrInfo(opParam.All2AllDataDes.sendCounts, opParam.All2AllDataDes.sdispls,
                                                          opParam.All2AllDataDes.sendType, opParam.All2AllDataDes.recvCounts, opParam.All2AllDataDes.rdispls,
                                                          opParam.All2AllDataDes.recvType, preMetaInfo));

        preMetaInfo->opType = HcclCMDType::HCCL_CMD_ALLGATHER;

        CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo));

        return alltoAllOperator->GetAlltoAllStagedWorkSpaceMemSize(opParam, memSize);
    }

    HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &memSize)
    {
        CHK_PRT_RET(Is310P3Common(isHaveCpuRank_, deviceType_),
                    HCCL_ERROR("[HcclCommunicator][GetAlltoAllStagedWorkSpaceMemSize]Not Supported!"), HCCL_E_NOT_SUPPORT);

        CHK_SMART_PTR_NULL(implAlg_);
        return implAlg_->GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize);
    }

    HcclResult HcclCommunicator::GetAllReduceScratchSize(
        const u32 count, const HcclDataType dataType, u64 &scratchSize) const
    {
        CHK_SMART_PTR_NULL(implAlg_);
        return implAlg_->GetAllReduceScratchSize(count, dataType, scratchSize);
    }

    HcclResult HcclCommunicator::SetWorldGroupInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap,
        vector<RankInfo> worldRankInfoList, vector<u32> &nicRanksPort, vector<u32> &vnicRanksPort)
    {
        for (auto &ipInfo : phyIdNicInfoMap) {
            for (auto &devInfo : ipInfo.second) {
                rankDevicePhyIdNicInfoMap_[ipInfo.first][devInfo.first] = devInfo.second;
                HCCL_DEBUG("phyIdNicInfoMap print hostIp[%s] devId[%u] devIp[%s]",
                           ipInfo.first.c_str(), devInfo.first, devInfo.second.GetReadableAddress());
            }
        }

        for (auto &rankInfo : worldRankInfoList) {
            worldRankInfoList_.push_back(rankInfo);
        }

        for (auto &port : nicRanksPort) {
            nicRanksPort_.push_back(port);
            HCCL_DEBUG("nicRanksPort port[%u]", port);
        }
        for (auto &port : vnicRanksPort) {
            vnicRanksPort_.push_back(port);
            HCCL_DEBUG("vnicRanksPort port[%u]", port);
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::GetTopoDesc(HcclTopoDescs *topoDescs, uint32_t topoSize)
    {
        if (topoSize < static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_MAX)) {
            HCCL_ERROR("topoDescs size is not enough, please check topoSize[%u]", topoSize);
            return HCCL_E_PARA;
        }

        if (deviceType_ == DevType::DEV_TYPE_910_93) {
            topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_SWITCH | HCCL_ALG_RING;
            topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = HCCL_ALG_RING;
        } else if (deviceType_ == DevType::DEV_TYPE_910B) {
            topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_MESH;
            topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = 0;
        } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
            topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_RING;
            topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = 0;
        }

        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].rankSize = userRankSize_;
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].rankSize = 0;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetAivModeConfig(const bool aivMode)
    {
        CHK_SMART_PTR_NULL(implAlg_);
        CHK_RET(implAlg_->SetAivModeConfig(aivMode));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetAicpuUnfoldConfig(const bool aicpuUnfold)
    {
        CHK_SMART_PTR_NULL(implAlg_);
        CHK_RET(implAlg_->SetAicpuUnfoldConfig(aicpuUnfold));
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::SetQpQosAttr(u32 trafficClass, u32 serviceLevel)
    {
        if (oneSideService_) {
            oneSideService_->SetTCAndSL(trafficClass, serviceLevel);
            HCCL_INFO("[%s]Set TC[%u] and SL[%u] for oneSidedService success.", __func__, trafficClass, serviceLevel);
        }
        transportManager_->SetQpQosAttr(trafficClass, serviceLevel);
    }

    HcclResult HcclCommunicator::CheckExitWaitResumeState(bool &isChangedLink)
    {
        if (retryEnable_ && opRetryManager_ != nullptr) {
            HcclResult ret = opRetryManager_->ExitWaitResumeState(identifier_, commConnections_.isRoot, isChangedLink);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[HcclCommunicator][Resume]opretry exit wait resume state failed."), ret);
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetMemoryRange(void *baseVirPtr, size_t size, size_t alignment, uint64_t flags)
    {
        CHK_PRT_RET(deviceType_ != DevType::DEV_TYPE_910_93,
                    HCCL_ERROR("[HcclCommunicator][SetMemoryRange] deviceType[%d] not support zero copy", deviceType_), HCCL_E_NOT_SUPPORT);
        if (zeroCopyMemoryAgent_ == nullptr) {
            CHK_RET(InitZeroCopyMemoryAgent());
        }
        CHK_RET(zeroCopyMemoryAgent_->SetMemoryRange(baseVirPtr, size, alignment, flags));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::UnsetMemoryRange(void *baseVirPtr)
    {
        CHK_PRT_RET(zeroCopyMemoryAgent_ == nullptr,
                    HCCL_ERROR("[HcclCommunicator][UnsetMemoryRange] not call HcclCommSetMemoryRange()"), HCCL_E_PARA);
        CHK_RET(zeroCopyMemoryAgent_->UnsetMemoryRange(baseVirPtr));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ActivateCommMemory(void *virPtr, size_t size, size_t offset, void *handle, uint64_t flags)
    {
        CHK_PRT_RET(zeroCopyMemoryAgent_ == nullptr,
                    HCCL_ERROR("[HcclCommunicator][ActivateCommMemory] not call HcclCommSetMemoryRange()"), HCCL_E_PARA);
        CHK_RET(zeroCopyMemoryAgent_->ActivateCommMemory(virPtr, size, offset, handle, flags));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::DeactivateCommMemory(void *virPtr)
    {
        CHK_PRT_RET(zeroCopyMemoryAgent_ == nullptr,
                    HCCL_ERROR("[HcclCommunicator][DeactivateCommMemory] not call HcclCommSetMemoryRange()"), HCCL_E_PARA);
        CHK_RET(zeroCopyMemoryAgent_->DeactivateCommMemory(virPtr));
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetSingleLinkInfo(std::unordered_map<u32, bool> &switchRanks, u32 remoteRankId,
                                                   ChangeLinkInfo &changeLinkInfo)
    {
        auto iterLocal = switchRanks.find(userRank_);
        auto iterRemote = switchRanks.find(remoteRankId);

        bool useBackupLink = false;
        if (iterLocal != switchRanks.end() && iterRemote != switchRanks.end()) {
            // 本端卡和对端卡都切，如果两者的目标网卡冲突，则切换失败；否则使用一致的目标网卡的的对应链路
            CHK_PRT_RET(iterLocal->second ^ iterRemote->second,
                        HCCL_ERROR("[HcclCommunicator][SetSingleLinkInfo] local rank[%u] plan to switch to nic[%u], "
                                   "which is conflict with remote rank[%u] planning to switch to nic[%u].",
                                   userRank_, iterLocal->second, remoteRankId, iterRemote->second),
                        HCCL_E_PARA);
            useBackupLink = iterLocal->second;
        } else if (iterLocal != switchRanks.end()) {
            // 仅切换本端卡，根据本端卡的目标网卡，刷新对应链路
            useBackupLink = iterLocal->second;
        } else if (iterRemote != switchRanks.end()) {
            // 仅切换对端卡，根据对端卡的目标网卡，刷新对应链路
            useBackupLink = iterRemote->second;
        } else {
            HCCL_INFO("[HcclCommunicator][SetSingleLinkInfo] comm identifier[%s], local rank[%u], "
                      "remote rank[%u], neither the rank need switch, link will not be refreshed.",
                      identifier_.c_str(), userRank_, remoteRankId);
            return HCCL_SUCCESS;
        }

        changeLinkInfo.remoteRankList[changeLinkInfo.remoteRankNum] = remoteRankId;
        changeLinkInfo.isUseDefaultPort[changeLinkInfo.remoteRankNum] = !(useBackupLink);
        changeLinkInfo.remoteRankNum++;
        remoteRankNicStatus_[remoteRankId] = useBackupLink ? CONNECT_REMOTE_BACKUP : CONNECT_REMOTE_DEFAULT;
        needCheckBackupNic_ |= useBackupLink;
        needCheckDefaultNic_ |= !useBackupLink;

        HCCL_RUN_INFO("[HcclCommunicator][SetSingleLinkInfo] comm identifier[%s], local rank[%u], "
                      "remote rank[%u], useBackupLink[%u], link info refreshed.",
                      identifier_.c_str(), userRank_, remoteRankId, useBackupLink);
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SetRemoteRankLinkInfo(std::unordered_map<u32, bool> &switchRanks,
                                                       ChangeLinkInfo &changeLinkInfo)
    {
        // 初始化重置changeLinkInfo
        changeLinkInfo.remoteRankNum = 0;
        needCheckBackupNic_ = false;
        needCheckDefaultNic_ = false;
        // 初始化重置remoteRankNicStatus_
        (void)memset_s(remoteRankNicStatus_, sizeof(remoteRankNicStatus_), 0, sizeof(remoteRankNicStatus_));

        for (auto resIt : resMap_) {
            for (auto &levelNSubCommTransport : resIt.second.opTransportResponse) {
                for (auto &singleSubCommTransport : levelNSubCommTransport) {
                    for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                        if (transportRequest.isValid && transportRequest.isUsedRdma) { // 仅RDMA链路需要刷新
                            CHK_RET(SetSingleLinkInfo(switchRanks, transportRequest.remoteUserRank, changeLinkInfo));
                        }
                    }
                }
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ActiveStoppedLink(std::map<u32, bool> &remoteRankPortMap,
                                                   OpCommTransport &opTransportResponse, bool isBackup)
    {
        for (auto &levelNSubCommTransport : opTransportResponse) {
            for (auto &singleSubCommTransport : levelNSubCommTransport) {
                if (singleSubCommTransport.status.size() == 0) {
                    continue;
                }
                if (singleSubCommTransport.status.size() != singleSubCommTransport.transportRequests.size() 
                    || singleSubCommTransport.links.size() != singleSubCommTransport.transportRequests.size()) {
                    HCCL_ERROR("[HcclCommunicator][ActiveStoppedLink] comm identifier[%s], local rank[%u], "
                               "status num[%u] or links num[%u] is inconsistent with transport request num[%u]. "
                               "Please check whether the resources are allocated correctly.",
                               identifier_.c_str(), userRank_, singleSubCommTransport.status.size(),
                               singleSubCommTransport.links.size(), singleSubCommTransport.transportRequests.size());
                    return HCCL_E_INTERNAL;
                }

                for (size_t i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
                    auto &transportRequest = singleSubCommTransport.transportRequests[i];
                    auto remoteRankIter = remoteRankPortMap.find(transportRequest.remoteUserRank);
                    bool needLink = transportRequest.isValid && transportRequest.isUsedRdma && remoteRankIter != remoteRankPortMap.end() 
                        && (remoteRankIter->second ^ isBackup);
                    // STOP状态的Transport需要唤醒，重置位到READY
                    if (needLink && singleSubCommTransport.status[i] == TransportStatus::STOP) {
                        HCCL_INFO("[HcclCommunicator][ActiveStoppedLink] comm identifier[%s], local rank[%u], "
                                  "resuming link of remote rank[%u]",
                                  identifier_.c_str(), userRank_,
                                  transportRequest.remoteUserRank);
                        CHK_RET(singleSubCommTransport.links[i]->Resume());
                        singleSubCommTransport.status[i] = TransportStatus::READY;
                    }
                }
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::PrepareLinkForSwitchNic(std::unordered_map<u32, bool> &switchRanks,
                                                         ChangeLinkInfo &changeLinkInfo)
    {
        CHK_RET(SetRemoteRankLinkInfo(switchRanks, changeLinkInfo));

        std::map<u32, bool> remoteRankPortMap;
        for (u32 i = 0; i < changeLinkInfo.remoteRankNum; i++) {
            remoteRankPortMap.emplace(changeLinkInfo.remoteRankList[i], changeLinkInfo.isUseDefaultPort[i]);
        }
        for (auto resIt : resMap_) {
            CHK_RET(ActiveStoppedLink(remoteRankPortMap, resIt.second.opTransportResponse, false));
            CHK_RET(ActiveStoppedLink(remoteRankPortMap, resIt.second.opTransportResponseBackUp, true));
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::ParseSwitchRanks(uint32_t nRanks, uint32_t *ranks, bool *useBackup,
                                                  std::unordered_map<u32, bool> &switchRanks)
    {
        CHK_PTR_NULL(ranks);
        CHK_PTR_NULL(useBackup);
        switchRanksNum_ = nRanks;
        (void)memset_s(switchRankList_, sizeof(switchRankList_), 0, sizeof(switchRankList_));
        (void)memset_s(switchUseBackup_, sizeof(switchUseBackup_), 0, sizeof(switchUseBackup_));
        s32 ret = memcpy_s(switchRankList_, sizeof(switchRankList_), ranks, sizeof(u32) * nRanks);
        CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HcclCommunicator][ParseSwitchRanks] mem copy switch ranks fail."),
                    HCCL_E_INTERNAL);
        ret = memcpy_s(switchUseBackup_, sizeof(switchUseBackup_), useBackup, sizeof(bool) * nRanks);
        CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HcclCommunicator][ParseSwitchRanks] mem copy switch use backup fail."),
                    HCCL_E_INTERNAL);

        std::string switchRankStr{};
        for (uint32_t i = 0; i < nRanks; i++) {
            CHK_PTR_NULL(ranks + i);
            CHK_PTR_NULL(useBackup + i);
            uint32_t switchRankId = ranks[i];
            bool backup = useBackup[i];
            CHK_PRT_RET(switchRankId >= userRankSize_,
                        HCCL_ERROR("[HcclCommunicator][ParseSwitchRanks] invalid switchRankId[%u], "
                                   "which should not be greater than rankSize[%u]",
                                   switchRankId, userRankSize_),
                        HCCL_E_PARA);
            CHK_PRT_RET(switchRanks.find(switchRankId) != switchRanks.end(),
                        HCCL_ERROR("[HcclCommunicator][ParseSwitchRanks] duplicated switchRankId[%u]", switchRankId), HCCL_E_PARA);
            switchRanks.emplace(switchRankId, backup);
            switchRankStr += std::to_string(switchRankId) + ":" + std::to_string(backup) + ";";
        }
        HCCL_RUN_INFO("[HcclCommunicator][ParseSwitchRanks] comm identifier[%s], userRank[%u], load switchRanks:%s.",
                      identifier_.c_str(), userRank_, switchRankStr.c_str());
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::SwitchNic(uint32_t nRanks, uint32_t *ranks, bool *useBackup,
                                           std::shared_ptr<HDCommunicate> &controlH2D, std::shared_ptr<HDCommunicate> &statusD2H)
    {
        HcclResult ret = HCCL_SUCCESS;
        CHK_PRT_RET(!IsEnableBackupLink(), HCCL_RUN_WARNING("[HcclCommunicator][%s]Backup link is not enabled, "
                                                            "switch nic will not be prorcessed, comm identifier[%s], rank[%u], devType[%u], opretry enable[%u], "
                                                            "backup ip valid[%u], roce enable[%u].",
                                                            __func__, identifier_.c_str(), userRank_, deviceType_, GetExternalInputHcclAicpuUnfold() && GetExternalInputInterSuperPodRetryEnable(), !devBackupIpAddr_[0].IsInvalid(), IsEnableRoce()),
                    HCCL_SUCCESS);
        CHK_PRT_RET(resMap_.empty(), HCCL_ERROR("[HcclCommunicator][%s] "
                                                "no collective operation has been executed in this communication[%s] on rank[%u], "
                                                "which does not support to set working device nic.",
                                                __func__, identifier_.c_str(), userRank_),
                    HCCL_E_PARA);
        std::unordered_map<u32, bool> switchRanks;
        ChangeLinkInfo changeLinkInfo;
        ret = ParseSwitchRanks(nRanks, ranks, useBackup, switchRanks);
        if (ret == HCCL_SUCCESS) {
            ret = PrepareLinkForSwitchNic(switchRanks, changeLinkInfo);
        }
        changeLinkInfo.isChangeLinkFlag = ret == HCCL_SUCCESS; // 如果入参校验失败，则无需刷新链路；通知aicpu侧，防止其他卡超时等待

        switchNicWaitingResult_ = false;

        u32 changeLinkInfoStart = sizeof(KfcCommand) + sizeof(BackgroundCommand) + sizeof(HcclComSuspendingFlag) +
                                  sizeof(HcclOpIdentifier);
        CHK_RET(controlH2D->Put(changeLinkInfoStart, sizeof(ChangeLinkInfo),
                                reinterpret_cast<uint8_t *>(&changeLinkInfo)));

        KfcCommand switchNicCommand = KfcCommand::kSwitchNic;
        CHK_RET(controlH2D->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t *>(&switchNicCommand)));

        KfcExecStatus switchStatus;
        switchStatus.execStatus.kfcStatus = KfcStatus::kNull;
        u32 waitSwitchExecCmdTimeout = static_cast<u32>(GetExternalInputHcclLinkTimeOut() * 1000 * 2.5f);
        auto waitSwitchExecCmdTimeoutMs = std::chrono::milliseconds(waitSwitchExecCmdTimeout); // 等待2.5倍的建链超时时间，给快慢卡场景提供冗余
        auto startTime = std::chrono::steady_clock::now();
        while (true) {
            if (switchNicWaitingResult_) {
                CHK_RET(statusD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t *>(&switchStatus)));
            }
            if (switchStatus.execStatus.kfcStatus == KfcStatus::kSwitchSuccess) {
                HCCL_INFO("[HcclCommunicator][%s] comm identifier[%s], devicePhyId[%u], userRank[%u] switch nic success.",
                          __func__, identifier_.c_str(), devicePhyId_, userRank_);
                ret = HCCL_SUCCESS;
                break;
            } else if (switchStatus.execStatus.kfcStatus == KfcStatus::kSwitchFail) {
                HCCL_ERROR("[HcclCommunicator][%s] comm identifier[%s], devicePhyId[%u], userRank[%u] switch nic fail.",
                           __func__, identifier_.c_str(), devicePhyId_, userRank_);
                ret = HCCL_E_INTERNAL;
                break;
            }  else if ((std::chrono::steady_clock::now() - startTime) >= waitSwitchExecCmdTimeoutMs) {
                HCCL_ERROR("[HcclCommunicator][%s] comm identifier[%s], devicePhyId[%u], "
                           "userRank[%u] switch nic timeout[%u ms], the transport status is undefined. "
                           "Please search log with keywork [ErrToWarn] for detail.",
                           __func__, identifier_.c_str(), devicePhyId_, userRank_, waitSwitchExecCmdTimeout);
                ret = HCCL_E_TIMEOUT;
                break;
            } else {
                SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            }
        }
        KfcExecControl clearCommand{};
        CHK_RET(controlH2D->Put(0, sizeof(KfcExecControl), reinterpret_cast<uint8_t *>(&clearCommand)));
        KfcExecStatus clearStatus{};
        CHK_RET(controlH2D->Put(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t *>(&clearStatus)));
        switchRanksNum_ = 0;
        return ret;
    }

    HcclResult HcclCommunicator::GetSwitchRanks(u32 *distSwitchRankList, bool *distSwitchUseBackup, u32 &distSwitchRankNum,
                                                u8 *distRemoteRankNicStatus, u32 &distNicStatusNum, bool &needCheckDefaultNic, bool &needCheckBackupNic)
    {
        s32 ret = memcpy_s(distSwitchRankList, sizeof(u32) * AICPU_MAX_RANK_NUM, switchRankList_,
                           sizeof(u32) * switchRanksNum_);
        CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HcclCommunicator][GetSwitchRanks] mem copy switch rank list fail, ret[%u].", ret), HCCL_E_INTERNAL);
        ret = memcpy_s(distSwitchUseBackup, sizeof(bool) * AICPU_MAX_RANK_NUM, switchUseBackup_,
                       sizeof(bool) * switchRanksNum_);
        CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HcclCommunicator][GetSwitchRanks] mem copy switch use backup fail, ret[%u].", ret), HCCL_E_INTERNAL);
        distSwitchRankNum = switchRanksNum_;
        ret = memcpy_s(distRemoteRankNicStatus, sizeof(u8) * AICPU_MAX_RANK_NUM, remoteRankNicStatus_,
                       sizeof(u8) * userRankSize_);
        CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HcclCommunicator][GetSwitchRanks] mem copy remote rank nic status fail, "
                                           "ret[%u].",
                                           ret),
                    HCCL_E_INTERNAL);
        distNicStatusNum = userRankSize_;
        needCheckDefaultNic = needCheckDefaultNic_;
        needCheckBackupNic = needCheckBackupNic_;
        switchNicWaitingResult_ = true;
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::LoadCustomFile(const char *binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode,
                                                aclrtBinHandle &binHandle)
    {
        s64 isOpenCustomSwitch = 0;
        CHK_RET(hrtGetDeviceInfo(deviceLogicId_, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
                                 HcclRtDeviceInfoType::HCCL_INFO_TYPE_CUST_OP_ENHANCE, isOpenCustomSwitch));
        if (isOpenCustomSwitch == 1) {
            HcclResult ret = LoadBinaryFromFile(binPath, optionType, cpuKernelMode, binHandle);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[LoadCustomFile]errNo[0x%016llx]load custom file fail, path[%s] optionType[%u]"
                                   "cpuKernelMode[%u].",
                                   ret, binPath, optionType, cpuKernelMode),
                        ret);
        } else {
            binHandle = nullptr;
            HCCL_RUN_WARNING("[LoadCustomFile]custom switch is not open, please confirm the switch.");
        }
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicator::LoadBinaryFromFile(const char *binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode,
                                                    aclrtBinHandle &binHandle)
    {
        if (binPath == nullptr) {
            HCCL_ERROR("[Load][Binary]binary path is nullptr");
            return HCCL_E_PTR;
        }

        std::string libraryPath;
        HcclResult ret = ParseLibraryPath(libraryPath);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[LoadBinaryFromFile]errNo[0x%016llx]parse path fail.", ret), ret);

        std::string cannPath; // 存放cann安装路径
        std::string jsonPath(binPath);
        if (GetKeyWordPath(libraryPath, "/hccl", cannPath) == HCCL_SUCCESS) {
            cannPath += jsonPath;
        } else {
            HCCL_ERROR("[LoadBinaryFromFile]cannot found version file in %s.", libraryPath.c_str());
            return HCCL_E_PARA;
        }

        char realPath[PATH_MAX] = {0};
        if (realpath(cannPath.c_str(), realPath) == nullptr) {
            HCCL_ERROR("LoadBinaryFromFile: %s is not a valid real path, err[%d]", cannPath.c_str(), errno);
            return HCCL_E_INTERNAL;
        }

        HCCL_INFO("[LoadBinaryFromFile]realPath: %s", realPath);
        aclrtBinaryLoadOptions loadOptions = {0};
        aclrtBinaryLoadOption option;
        loadOptions.numOpt = 1;
        loadOptions.options = &option;
        option.type = optionType;
        option.value.cpuKernelMode = cpuKernelMode;
        aclError aclRet = aclrtBinaryLoadFromFile(realPath, &loadOptions, &binHandle); // ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE
        CHK_PRT_RET(aclRet != ACL_SUCCESS,
                    HCCL_ERROR("[LoadBinaryFromFile]errNo[0x%016llx] load binary from file error.", ret), HCCL_E_OPEN_FILE_FAILURE);
        return HCCL_SUCCESS;
    }

    void HcclCommunicator::UnloadBinary(aclrtBinHandle &binHandle)
    {
        if (binHandle != nullptr) {
            aclError ret = aclrtBinaryUnLoad(binHandle);
            if (ret != ACL_SUCCESS) {
                HCCL_ERROR("[UnloadBinary]errNo[0x%016llx] unload binary from file error.", ret);
            }
            binHandle = nullptr;
        }
        return;
    }
}
