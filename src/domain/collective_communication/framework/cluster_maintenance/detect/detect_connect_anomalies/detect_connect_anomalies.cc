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
#include "detect_connect_anomalies.h"
#include "hccl_common.h"
#include "hccl_socket.h"
#include "env_config.h"
#include "externalinput_pub.h"
#include <chrono>
#include <thread>

namespace hccl {
DetectConnectionAnomalies &DetectConnectionAnomalies::GetInstance(s32 deviceLogicID)
{
    static DetectConnectionAnomalies dca[MAX_MODULE_DEVICE_NUM];
    if (static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[DetectConnectionAnomalies][GetInstance]deviceLogicID[%d] is invalid", deviceLogicID);
        return dca[0];
    }
    return dca[deviceLogicID];
}

// 创建单例，保存所有RankInfoList中的Ip地址
void DetectConnectionAnomalies::Init(std::vector<RankInfo> &rankInfos, bool isNeedNic)
{
    if (isNeedNic) {
        isNeedNic_ =  isNeedNic;
    }
    // 直接用set保存，省掉查重
    int ref = initRef_.Ref();
    HCCL_INFO("DetectConnectionAnomalies[Init] initRef[%d]", ref);
    for (auto &rankInfo : rankInfos) {
        if (!rankInfo.nicIp[0].IsInvalid()) {
            uniqueIps_.insert(rankInfo.nicIp[0]);
        }

        if (!rankInfo.deviceVnicIp.IsInvalid()) {
            uniqueIps_.insert(rankInfo.deviceVnicIp);
        }
    }
    return;
}

// 添加ipQueue
void DetectConnectionAnomalies::AddIpQueue(RankInfo &localRankInfo, RankInfo &remoteRankInfo, NicType nicType,
    s32 deviceLogicId)
{
    // 检查是否需要进行连接异常检测
    if (GetExternalInputDfsConnectionFaultDetctionTime() == 0 || !threadExit_) {
        HCCL_RUN_INFO("[Add][IpQueue]GetExternalInputDfsConnectionFaultDetctionTime is 0, no need to detect");
        return;
    }

    // 检查设备类型是否支持
    if (localRankInfo.deviceType != DevType::DEV_TYPE_910_93 && localRankInfo.deviceType != DevType::DEV_TYPE_910B) {
        HCCL_WARNING("[AddIpQueue] not support deviceType[%d]", localRankInfo.deviceType);
        return;
    }

    // 检查是否需要进行连接异常检测
    HcclIpAddress localIp = (nicType == NicType::VNIC_TYPE) ? localRankInfo.deviceVnicIp : localRankInfo.nicIp[0];
    HcclIpAddress remoteIp = (nicType == NicType::DEVICE_NIC_TYPE) ? remoteRankInfo.nicIp[0] : remoteRankInfo.deviceVnicIp;
    if (localIp.IsInvalid() || remoteIp.IsInvalid()) {
        return;
    }

    // 多线程访问ipQueue需要加锁
    Detect();
    std::unique_lock<std::mutex> lock(ipNictypeQueueMutex_);
    if (!isPrint_) {
        HCCL_ERROR("-------------------CONNECT TIMEOUT DETECT RESULT-----------------------");
        HCCL_ERROR("if BELOW DETECT EVENT num ≥ 1:");
        HCCL_ERROR("    The error above was caused by a failure at the location in the cluster where the events happened.Please "\
            "confirm whether the link between SRCRANK and DSTRANK or the DSTRANK process is healthy.");
        HCCL_ERROR("if BELOW DETECT EVENT num = 0:");
        HCCL_ERROR("    please prioritize investigating the consistency of cluster script behaviors.");
        HCCL_ERROR("NOTE: The detection results are only used to assist in locating the problem and may not represent the "\
            "actual fault location in some complex scenarios. Please continue to analyze and confirm based on the current "\
            "detected fault location(if BELOW DETECT EVENT num ≥ 1).");
        HCCL_ERROR("-------------------------DETECT EVENT LIST-----------------------------------------");
        HCCL_ERROR("-----------------------------------------------------------------------------------");
        isPrint_ = true;
    }
    ErrInfo errInfo;
    auto ip = ipMap_.find(remoteIp);
    if (ip == ipMap_.end()) {
        ipMap_.insert(std::make_pair(remoteIp, localIp));
        HCCL_INFO("[Add][IpQueue]localIp[%s], remoteIp[%s], nicType[%d], deviceLogicId[%d]", 
            localIp.GetReadableAddress(), remoteIp.GetReadableAddress(), nicType, deviceLogicId);
        errInfo.localRankInfo = localRankInfo;
        errInfo.remoteRankInfo = remoteRankInfo;
        errInfo.nicType = nicType;
        errInfo.deviceLogicId = deviceLogicId;
        ipNictypeQueue_.push(errInfo); // 记录报错卡信息
    }
    lock.unlock();
    // 计算等待时间
    auto waitTime = std::chrono::seconds(GetExternalInputDfsConnectionFaultDetctionTime()) +
        std::chrono::seconds(broadCastTime);
    
    auto startTime = std::chrono::steady_clock::now();
    while (threadExit_ && (std::chrono::steady_clock::now() - startTime) <= waitTime) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 每次休眠100毫秒
    }
    HCCL_INFO("[Add][IpQueue]ipNictypeQueue size[%d]", ipNictypeQueue_.size());
    return;
}

// 检测连接异常
HcclResult DetectConnectionAnomalies::Detect()
{
    std::unique_lock<std::mutex> lock(detectThreadMutex_);
    if (!isInitThread_ && threadExit_) {
        // 初始化线程,轮询ipNictypeQueue_
        getIpNictypeQueue_.reset(new (std::nothrow) std::thread(&DetectConnectionAnomalies::DetectMonitor, this));
        CHK_SMART_PTR_NULL(getIpNictypeQueue_);
        isInitThread_  = true;
    }
    lock.unlock();
    return HCCL_SUCCESS;
}


void DetectConnectionAnomalies::DetectMonitor()
{
    while (threadExit_) {
        GetIpQueue();
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 每次休眠100毫秒
    }
    return;
}
// 心跳线程调用
HcclResult DetectConnectionAnomalies::GetIpQueue()
{
    if (ipNictypeQueue_.empty()) {
        return HCCL_SUCCESS;
    }

    HCCL_RUN_INFO("[GetIpQueue]ipNictypeQueue_ size[%d], start to detect", ipNictypeQueue_.size());
    std::unique_lock<std::mutex> lock(ipNictypeQueueMutex_);
    while (!ipNictypeQueue_.empty() && threadExit_) {
        auto& errInfo = ipNictypeQueue_.front();
        if (CreateServers(errInfo) != HCCL_SUCCESS ||
            CreateClients(errInfo, linkClientThreads_) != HCCL_SUCCESS) {
            ipNictypeQueue_.pop();
            HCCL_ERROR("[GetIpQueue]CreateServers or CreateClients fail");
            return HCCL_E_INTERNAL;
        }
        ipNictypeQueue_.pop();
    }
    HCCL_INFO("[GetIpQueue] compeleted[%d]", ipNictypeQueue_.size());
    lock.unlock();
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::CreateDetectVnicLinks(struct ErrInfo  errInfo)
{
    SetThreadName("Hccl_Detect_vnic");
    CHK_RET(HcclNetOpenDev(&vnicCtx_, NicType::VNIC_TYPE, errInfo.localRankInfo.devicePhyId,
        errInfo.deviceLogicId, errInfo.localRankInfo.deviceVnicIp));
    CHK_PTR_NULL(vnicCtx_);
    std::string tag = GetTag(errInfo.localRankInfo.deviceVnicIp);
    u32 port = (errInfo.localRankInfo.deviceVnicPort== HCCL_INVALID_PORT) ? HETEROG_CCL_PORT : port;

    // 创建vnic socket服务端
    EXECEPTION_CATCH((vnicSocket_ = std::make_shared<HcclSocket>(vnicCtx_, port)), return HCCL_E_PTR);
    HCCL_RUN_INFO("[CreateDetectVnicLinks]tag[%s], localIpAddr[%s], remoteIpAddr[%u], port[%u]", tag.c_str(),
        errInfo.localRankInfo.deviceVnicIp.GetReadableIP(),  errInfo.remoteRankInfo.deviceVnicIp.GetReadableIP(), port);

    CHK_RET(vnicSocket_->Init());
    CHK_RET(vnicSocket_->Listen());
    CHK_RET(AddWhiteList(vnicSocket_, NicType::VNIC_TYPE, tag));; // 添加白名单
    HCCL_INFO("[CreateDetectVnicLinks]AddWhiteList finshed");

    u32 acceptTimeOut = 1; // accecpt 超时1s
    std::shared_ptr<HcclSocket> acceptSuccessSocket;
    auto detectTimeOut = std::chrono::seconds(GetExternalInputDfsConnectionFaultDetctionTime());
    auto startTime = std::chrono::steady_clock::now();
    HcclResult ret;
    while (threadExit_ && (std::chrono::steady_clock::now() - startTime) < std::chrono::seconds(detectTimeOut)) {
        ret = vnicSocket_->Accept(tag, acceptSuccessSocket, acceptTimeOut);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("[CreateDetectVnicLinks]accept success, localIpAddr[%s], acceptSuccessSocket[%p]",
                errInfo.localRankInfo.deviceVnicIp.GetReadableIP(), acceptSuccessSocket.get());
            listenVnicVec_.push_back(acceptSuccessSocket);  // 保存accept成功的socket
        }
        usleep(ACCEPT_TIME_OF_USLEEP); // 休眠100毫秒
    }
    // 循环发送检测信息
    startTime = std::chrono::steady_clock::now();
    while (threadExit_ && (std::chrono::steady_clock::now() - startTime) <= std::chrono::seconds(broadCastTime)) {
        std::unique_lock<std::mutex> lock(readRecvErrtInfo_);
        for (auto &recvError : recvErrorInfoMap_) {
            auto it = sendErrorInfoMap_.find(recvError.first);
            if (it != sendErrorInfoMap_.end() && !sendErrorInfoMap_[recvError.first].isSendVnic) {
                for (auto &socket : listenVnicVec_) {
                    CHK_RET(socket->Send(&recvError.second, sizeof(recvError.second)));
                }
                // 给所有socket都发送完成后，才标记发送完成
                sendErrorInfoMap_[recvError.first].isSendVnic = true;
            }
        }
        lock.unlock();
        usleep(ACCEPT_TIME_OF_USLEEP); // 休眠1毫秒,将锁释放给IRecv
    }
    std::unique_lock<std::mutex> lock(whiteListMutex_);
    CHK_RET(DelWhiteList(errInfo.localRankInfo.deviceVnicIp, vnicWhiteListInfosVec_, vnicSocket_)); // 删除白名单
    lock.unlock();
    HCCL_INFO("[CreateDetectVnicLinks] compeleted");
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::CreateDetectNicLinks(struct ErrInfo errInfo)
{
    SetThreadName("Hccl_Detect_Nic");
    
    CHK_RET(HcclNetOpenDev(&nicCtx_, NicType::DEVICE_NIC_TYPE, errInfo.localRankInfo.devicePhyId,
        errInfo.deviceLogicId, errInfo.localRankInfo.nicIp[0]));
    CHK_PTR_NULL(nicCtx_);
    std::string tag = GetTag(errInfo.localRankInfo.nicIp[0]);

    u32 port = (errInfo.localRankInfo.deviceNicPort == HCCL_INVALID_PORT) ? HETEROG_CCL_PORT : port;
    EXECEPTION_CATCH((nicSocket_ = std::make_shared<HcclSocket>(nicCtx_, port)), return HCCL_E_PTR);

    HCCL_RUN_INFO("[CreateDetectNicLinks]tag[%s], localIp[%s], remoteIp[%u], port[%u]", tag.c_str(),
        errInfo.localRankInfo.nicIp[0].GetReadableIP(), errInfo.remoteRankInfo.nicIp[0].GetReadableIP(), port);
    CHK_RET(nicSocket_->Init());
    CHK_RET(nicSocket_->Listen());
    CHK_RET(AddWhiteList(nicSocket_, NicType::DEVICE_NIC_TYPE, tag)); // 添加白名单
    HCCL_INFO("[CreateDetectNicLinks]AddWhiteList finshed");

    u32 acceptTimeOutAccept = 1;
    auto acceptTimeOut = std::chrono::seconds(GetExternalInputDfsConnectionFaultDetctionTime());
    std::shared_ptr<HcclSocket> acceptSuccessSocket;
    auto startTime = std::chrono::steady_clock::now();
    HcclResult ret;
    while (threadExit_ && (std::chrono::steady_clock::now() - startTime) <= acceptTimeOut) {
        ret = nicSocket_->Accept(tag, acceptSuccessSocket, acceptTimeOutAccept);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("[CreateDetectNicLinks]accept success, localIpAddr[%s], acceptSuccessSocket[%p]",
                errInfo.localRankInfo.nicIp[0].GetReadableIP(), acceptSuccessSocket.get());
            listenNicVec_.push_back(acceptSuccessSocket);
        }
        usleep(ACCEPT_TIME_OF_USLEEP); // 休眠100毫秒
    }
    // 循环发送
    startTime = std::chrono::steady_clock::now();
    while (threadExit_ && (std::chrono::steady_clock::now() - startTime) <= std::chrono::seconds(broadCastTime)) {
        std::unique_lock<std::mutex> lock(readRecvErrtInfo_);
        for (auto &recvError : recvErrorInfoMap_) {
            auto it = sendErrorInfoMap_.find(recvError.first);
            if (it != sendErrorInfoMap_.end() && !sendErrorInfoMap_[recvError.first].isSendNic) {
                for (auto &socket : listenNicVec_) {
                    CHK_RET(socket->Send(&recvError.second, sizeof(recvError.second)));
                }
            // 给所有socket都发送完成后，才标记发送完成
            sendErrorInfoMap_[recvError.first].isSendNic = true;
            }
        }
        lock.unlock();
        usleep(ACCEPT_TIME_OF_USLEEP); // 休眠10毫秒,将锁释放给IRecv
    }
    std::unique_lock<std::mutex> lock(whiteListMutex_);
    CHK_RET(DelWhiteList(errInfo.localRankInfo.nicIp[0], nicWhiteListInfosVec_, nicSocket_)); // 删除白名单
    lock.unlock();
    HCCL_INFO("[CreateDetectNicLinks] compeleted");
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::CreateServers(struct ErrInfo errInfo)
{
    if (threadExit_) {
        if (!isCreateLink_) {
            detectVnicThread_.reset(new (std::nothrow) std::thread(&DetectConnectionAnomalies::CreateDetectVnicLinks,
                this, errInfo));
            CHK_SMART_PTR_NULL(detectVnicThread_);
            isCreateLink_ = true;
        }
        // 多机场景，且vnic失败时, 这里得用nicIp，否则添加白名单无效
        if (isNeedNic_ && !isCreateNicLink_) {
            detectNicThread_.reset(new (std::nothrow) std::thread(&DetectConnectionAnomalies::CreateDetectNicLinks,
                this, errInfo));
            CHK_SMART_PTR_NULL(detectNicThread_);
            isCreateNicLink_ = true;
        }
    }
    return HCCL_SUCCESS;
}

std::string DetectConnectionAnomalies::GetTag(HcclIpAddress &Ip, int i)
{
    return std::string(Ip.GetReadableIP()) + "_detect_" + std::to_string(i);
}

HcclResult DetectConnectionAnomalies::AddWhiteList(
    std::shared_ptr<HcclSocket> socket,
    NicType nicType, 
    std::string& tag)
{
    // 根据 NicType 处理白名单
    HcclResult ret;
    if (nicType == NicType::VNIC_TYPE) {
        for (const auto& ipAddr : uniqueIps_) {
            HcclResult res = AddWlistEntry(ipAddr, tag, whiteVnicSet_, vnicWhiteListInfosVec_);
            if (res != HCCL_SUCCESS) {
                return res;
            }
        }
        ret = socket->AddWhiteList(vnicWhiteListInfosVec_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AddWhiteList] fail");
            return HCCL_E_NOT_FOUND;
        }
    } else if (isNeedNic_) {
        for (const auto& ipAddr : uniqueIps_) {
            HcclResult res = AddWlistEntry(ipAddr, tag, whiteNicSet_, nicWhiteListInfosVec_);
            if (res != HCCL_SUCCESS) {
                return res;
            }
        }
        ret = socket->AddWhiteList(nicWhiteListInfosVec_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AddWhiteList] fail");
            return HCCL_E_NOT_FOUND;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::DelWhiteList(HcclIpAddress &localIpAddr, 
    std::vector<struct SocketWlistInfo> whiteListInfos, std::shared_ptr<HcclSocket> socket)
{
    if (!threadExit_ || whiteListInfos.size() == 0) {
        return HCCL_SUCCESS;
    }
    HcclResult ret = socket->DelWhiteList(whiteListInfos);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[DelWhiteList]ip[%s] DelWhiteList fail", localIpAddr.GetReadableIP()),
        HCCL_E_NOT_FOUND);
    whiteListInfos.clear();
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::ConstructErrorInfo(std::shared_ptr<HcclSocket> &clientSocket,
    RankInfo &localRankInfo, RankInfo &remoteRankInfo)
{
    DetectInfo  detectInfo{};
    detectInfo.localDeviceId = localRankInfo.devicePhyId;
    detectInfo.remoteDeviceId = remoteRankInfo.devicePhyId;

    // 获取本地设备IP并复制到错误信息中(直接获取，因为vnic场景从localRankInfo获得的IP可能是无效的)
    std::string localDeviceIp = clientSocket->GetLocalIp().GetReadableIP();
    CHK_SAFETY_FUNC_RET(memcpy_s(detectInfo.localDeviceIp, DEST_MAX_LEN, localDeviceIp.c_str(), localDeviceIp.size()));
    detectInfo.localDeviceIp[localDeviceIp.size()] = '\0';
    // 获取远程设备IP并复制到错误信息中
    std::string remoteDeviceIp = clientSocket->GetRemoteIp().GetReadableIP();
    CHK_SAFETY_FUNC_RET(
        memcpy_s(detectInfo.remoteDeviceIp, DEST_MAX_LEN, remoteDeviceIp.c_str(), remoteDeviceIp.size()));
    detectInfo.remoteDeviceIp[remoteDeviceIp.size()] = '\0';
    // 复制本地ServerId ID到错误信息中
    std::string localServerId = localRankInfo.serverId;
    CHK_SAFETY_FUNC_RET(memcpy_s(detectInfo.localServerId, DEST_MAX_LEN, localServerId.c_str(),
        localServerId.size()));
    detectInfo.localServerId[localServerId.size()] = '\0';

    // 复制远程serverId到错误信息中
    std::string remoteServerId = remoteRankInfo.serverId;
    CHK_SAFETY_FUNC_RET(memcpy_s(detectInfo.remoteServerId, DEST_MAX_LEN, remoteServerId.c_str(),
        remoteServerId.size()));
    detectInfo.remoteServerId[remoteServerId.size()] = '\0';

    std::unique_lock<std::mutex> lock(readRecvErrtInfo_);
    std::string ip = localDeviceIp + "-" + remoteDeviceIp;
    recvErrorInfoMap_.emplace(ip, detectInfo);
    sendErrorInfoMap_.emplace(ip, SendInfo{});
    PrintDetectInfo(detectInfo.localServerId, detectInfo.localDeviceId, detectInfo);
    lock.unlock();
    // 保存错误信息
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::GetStatus(struct ErrInfo errInfo, std::shared_ptr<HcclSocket> &clientSocket)
{
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputDfsConnectionFaultDetctionTime());
    // 等待时间不大于超时时间
    HcclSocketStatus status = HcclSocketStatus::SOCKET_INIT;

    while ((std::chrono::steady_clock::now() - startTime) < timeout) {
        status = clientSocket->GetStatus();
        if (status == HcclSocketStatus::SOCKET_OK) {
            HCCL_INFO("[Detect][ConnectionAnomalies]GetStatus success, remoteIpAddr[%s]",
                clientSocket->GetRemoteIp().GetReadableIP());
            return HCCL_SUCCESS;
        }
        SaluSleep(CLIENT_TIME_OF_USLEEP); // 休眠500毫秒
    }
    std::unique_lock<std::mutex> lock(ipConstuctMutex_);
    CHK_RET(ConstructErrorInfo(clientSocket, errInfo.localRankInfo, errInfo.remoteRankInfo));
    lock.unlock();
    return HCCL_E_TIMEOUT;
}

HcclResult DetectConnectionAnomalies::Connect(struct ErrInfo errInfo, std::shared_ptr<HcclSocket> &clientSocket)
{
   HcclIpAddress localIp = (errInfo.nicType == NicType::VNIC_TYPE) ? errInfo.localRankInfo.deviceVnicIp :
        errInfo.localRankInfo.nicIp[0];

    HcclNetDevCtx Ctx = (errInfo.nicType == NicType::VNIC_TYPE) ? vnicCtx_ : nicCtx_;
    if (Ctx == nullptr) {
        CHK_RET(HcclNetOpenDev(&Ctx, errInfo.nicType, errInfo.localRankInfo.devicePhyId,
            errInfo.deviceLogicId, localIp));
        CHK_PTR_NULL(Ctx);
        clientNicCtxs_.push_back(Ctx);
    }

    u32 port = (errInfo.nicType == NicType::VNIC_TYPE) ? errInfo.remoteRankInfo.deviceVnicPort : errInfo.remoteRankInfo.deviceNicPort;
    port = (port == HCCL_INVALID_PORT) ? HETEROG_CCL_PORT : port;
    HcclIpAddress remoteIpAddr = (errInfo.nicType == NicType::VNIC_TYPE) ? errInfo.remoteRankInfo.deviceVnicIp : errInfo.remoteRankInfo.nicIp[0];

    std::string tag = GetTag(remoteIpAddr);
    HCCL_INFO("[Connect]tag[%s], port[%u], nicCtx[%p], remoteIpAddr[%s], role[%d]", tag.c_str(),
        port, Ctx, remoteIpAddr.GetReadableIP(), HcclSocketRole::SOCKET_ROLE_CLIENT);
    EXECEPTION_CATCH((clientSocket = std::make_shared<HcclSocket>(tag, Ctx, remoteIpAddr, port, HcclSocketRole::SOCKET_ROLE_CLIENT)),
    return HCCL_E_PTR);
    CHK_RET(clientSocket->Init());

    HcclResult ret = clientSocket->Connect();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Detect][ConnectionAnomalies] connect fail, localIp[%s], remoteIp[%s]",
        localIp.GetReadableIP(), remoteIpAddr.GetReadableIP()), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::CreateClient(struct ErrInfo errInfo)
{
    SetThreadName("Hccl_Detect_Client");
    std::shared_ptr<HcclSocket> clientSocket;
    CHK_RET(Connect(errInfo, clientSocket));
    HcclResult ret = GetStatus(errInfo, clientSocket);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateClientConnect]GetStatus fail, ret[%d]", ret);
        return ret;
    }
    // 将localServerId转换char，以便打印函数统一
    char localServerId[DEST_MAX_LEN]{};
    CHK_SAFETY_FUNC_RET(memcpy_s(localServerId, DEST_MAX_LEN, errInfo.localRankInfo.serverId.c_str(),
        errInfo.localRankInfo.serverId.size()));
    localServerId[errInfo.localRankInfo.serverId.size()] = '\0';
    s32 localDeviceId = errInfo.localRankInfo.devicePhyId;

    // 保存clientSocket，在析构时join
    clientSockets_.push_back(clientSocket);

    // 开始计时
    auto waitTime = std::chrono::seconds(GetExternalInputDfsConnectionFaultDetctionTime()) +
        std::chrono::seconds(broadCastTime);
    auto startTime = std::chrono::steady_clock::now();

    DetectInfo detectInfo{};
    u64 totalSize = sizeof(detectInfo);
    void *recvBuffer = reinterpret_cast<void *>(&detectInfo);
    u64 recvSize = 0;
    while (threadExit_ && (std::chrono::steady_clock::now() - startTime) < waitTime) {
        u64 compSize = 0; // 本次接收长度
        void *recvBufferTmp = reinterpret_cast<u8 *>(recvBuffer) + recvSize; // 偏移
        ret = clientSocket->IRecv(recvBufferTmp, totalSize - recvSize, compSize);
        if (ret == HCCL_SUCCESS && compSize > 0) {
            recvSize += compSize;
        }
        if ((totalSize - recvSize) == 0) {
            recvSize = 0;
            std::string loaclIp(detectInfo.localDeviceIp);
            std::string remoteIp(detectInfo.remoteDeviceIp);
            std::string ip = loaclIp + "-" + remoteIp;
            std::unique_lock<std::mutex> lock(readRecvErrtInfo_);
            auto it  = recvErrorInfoMap_.find(ip);
            if (it == recvErrorInfoMap_.end()) {
                recvErrorInfoMap_.emplace(ip, detectInfo);
                sendErrorInfoMap_.emplace(ip, SendInfo{});
                PrintDetectInfo(localServerId, localDeviceId, detectInfo);
            }
            CHK_SAFETY_FUNC_RET(memset_s(&detectInfo, sizeof(DetectInfo), 0, sizeof(DetectInfo)));
            lock.unlock();
        }
        usleep(IRECV_TIME_OF_USLEEP); // 休眠500毫秒
    }
    HCCL_INFO("[CreateClient]compeleted");
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::CreateClients(struct ErrInfo errInfo, std::vector<std::unique_ptr<std::thread>> &linkClientThreads)
{
    std::unique_ptr<std::thread> linkClientThread;
    linkClientThread.reset(new (std::nothrow) std::thread(&DetectConnectionAnomalies::CreateClient, this, errInfo));
    CHK_SMART_PTR_NULL(linkClientThread);
    linkClientThreads.emplace_back(std::move(linkClientThread));
    return HCCL_SUCCESS;
}

void DetectConnectionAnomalies::PrintDetectInfo(const char *localServerId, s32 localDeviceId, DetectInfo &detectInfo)
{
    errorCount_++;
    HCCL_ERROR("DETECT EVENT[%d]:Rank[%s/%d]: srcRank[%s/%d] connect destRank[%s/%d] fail.", \
        errorCount_.load(), localServerId, localDeviceId, detectInfo.localServerId, detectInfo.localDeviceId, detectInfo.remoteServerId,
        detectInfo.remoteDeviceId);
    HCCL_ERROR("-----------------------------------------------------------------------------------------------------");
    return;
}

void DetectConnectionAnomalies::ThreadDestroy()
{
    HCCL_DEBUG("[DetectConnectionAnomalies]Destory");
    threadExit_ = false;

    // 销毁client线程
    std::unique_lock<std::mutex> lock(clientThreadMutex_);
    for (u32 index = 0; index < linkClientThreads_.size(); index++) {
        if (linkClientThreads_[index] != nullptr && linkClientThreads_[index]->joinable()) {
            HCCL_INFO("[DetectConnectionAnomalies]Destory linkClientThreads_[%p]", linkClientThreads_[index].get());
            linkClientThreads_[index]->join(); // 等待线程执行完毕
            linkClientThreads_[index] = nullptr;
        }
    }
    linkClientThreads_.clear();
    lock.unlock();

    // 先销毁线程，再释放资源
    // 销毁server线程
    if (detectVnicThread_ != nullptr && detectVnicThread_->joinable()) {
        detectVnicThread_->join();
        detectVnicThread_ = nullptr;
    }

    if (detectNicThread_ != nullptr && detectNicThread_->joinable()) {
        detectNicThread_->join();
        detectNicThread_ = nullptr;
    }

    // 销毁轮询线程
    if(getIpNictypeQueue_ != nullptr && getIpNictypeQueue_->joinable()) {
        getIpNictypeQueue_->join();
        getIpNictypeQueue_ = nullptr;
    }

    // 释放server侧资源
    if (vnicSocket_ != nullptr) {
        for (auto &socket : listenVnicVec_) {
            socket->DeInit();
            socket = nullptr;
        }
        vnicSocket_->DeInit();
        vnicSocket_ = nullptr;
    }

    if (nicSocket_ != nullptr) {
        for (auto &socket : listenNicVec_) {
            socket->DeInit();
        }
        nicSocket_->DeInit();
        nicSocket_ = nullptr;
    }

    // 释放client资源
    for (auto &socket : clientSockets_) {
        if (socket != nullptr) {
            socket->DeInit();
            socket = nullptr;
        }
    }

    // 销毁ctx
    if (nicCtx_ != nullptr) {
        HcclNetCloseDev(nicCtx_);
        nicCtx_ = nullptr;
    }

    if (vnicCtx_ != nullptr) {
        HcclNetCloseDev(vnicCtx_);
        vnicCtx_ = nullptr;
    }

    for (auto nicCtx : clientNicCtxs_) {
        if (nicCtx != nullptr) {
            HcclNetCloseDev(nicCtx);
        }
    }
    clientNicCtxs_.clear();
    clientSockets_.clear();
    listenVnicVec_.clear();
    listenNicVec_.clear();
}
void DetectConnectionAnomalies::Deinit()
{
    int count = initRef_.Unref();
    if (count > 0) {
        HCCL_INFO("[DetectConnectionAnomalies]Deinit initRef_[%d]", count);
        return;
    } else if (count < 0) {
        HCCL_WARNING("[DetectConnectionAnomalies]Deinit failed");
    }
    ThreadDestroy();
    HCCL_INFO("DetectConnectionAnomalies[Deinit] count[%d]", count);
    return;
}

void AddIpQueue(RankInfo &localRankInfo, RankInfo &remoteRankInfo, NicType nicType,
    s32 deviceLogicId)
{
    DetectConnectionAnomalies::GetInstance(deviceLogicId).AddIpQueue(localRankInfo, remoteRankInfo,
        nicType, deviceLogicId);
    return;
}

__attribute__((constructor)) void DetetcCallBackAddIpQueue()
{
    DetectCallBack(AddIpQueue);
}
} // namespace hccl
