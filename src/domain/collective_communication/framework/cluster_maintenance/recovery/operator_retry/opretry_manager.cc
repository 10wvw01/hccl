/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opretry_manager.h"
#include "opretry_link_manage.h"
#include "opretry_connection_pub.h"
#include "opretry_agent.h"
#include "opretry_server.h"
#include "adapter_rts_common.h"
#include "sal_pub.h"

namespace hccl {
HcclResult OpRetryManager::Init()
{
    CHK_PRT_RET(initialized_ == true, HCCL_WARNING("OpRetryManager has already initialized"), HCCL_SUCCESS);
    initialized_ = true;
    HCCL_INFO("OpRetryManager Init success");
    return HCCL_SUCCESS;
}

HcclResult OpRetryManager::RegisterOpRetryMachine(OpRetryAgentParam &agentParam, u32 rankSize, bool isRoot,
    std::map<u32, std::shared_ptr<HcclSocket> > &serverConnections, const OpRetryServerInfo& serverInfo)
{
    std::unique_lock<std::mutex> lock(ProcessLock_);
    CHK_SMART_PTR_NULL(agentParam.h2dPtr);
    CHK_SMART_PTR_NULL(agentParam.d2hPtr);
    CHK_SMART_PTR_NULL(agentParam.opStreamPtr);
    CHK_PRT_RET(agentParam.group.empty(),
        HCCL_ERROR("[OpRetryManager][RegisterOpRetryMachine]params invalid, group is empty"), HCCL_E_PARA);
    if (agentParam.agentConnection == nullptr && serverConnections.empty()) {
        CHK_RET(OpRetryConnectionPub::Init(agentParam.group, rankSize, serverInfo, agentParam.agentInfo));
        CHK_RET(OpRetryConnectionPub::GetConns(agentParam.group, isRoot, agentParam.agentConnection, serverConnections));
    }
    // 初始化
    if (initialized_ == false) {
        CHK_RET(Init());
    }

    // 注册agent状态机
    CHK_RET(RegisterAgentRetryMachine(agentParam));

    // 注册server状态机
    if (isRoot) {
        CHK_RET(RegisterServerRetryMachine(agentParam.group, serverConnections, agentParam.agentInfo));
    }
    HCCL_INFO("[Register][RetryMachine]group[%s] register success", agentParam.group.c_str());
    return HCCL_SUCCESS;
}

HcclResult OpRetryManager::RegisterAgentRetryMachine(OpRetryAgentParam &agentParam)
{
    std::string &group = agentParam.group;
    if (agentOpRetry_.find(group) != agentOpRetry_.end()) {
        HCCL_INFO("[Register][AgentRetryMachine]group[%s] has Registered to agentOpRetry, skip", group.c_str());
        return HCCL_SUCCESS;
    }

    RetryCtrl retryCtrl;
    agentOpRetry_.insert(std::make_pair(group, std::move(retryCtrl)));
    std::shared_ptr<OpRetryBase> retryPtr;
    EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryAgentRunning>()), return HCCL_E_PTR);
    EXECEPTION_CATCH((agentOpRetry_[group].retryCtx =
        std::make_shared<RetryContext>(agentParam, retryPtr)), return HCCL_E_PTR);
    agentOpRetry_[group].startExec = true;

    HcclRtContext ctx = nullptr;
    CHK_RET(hrtCtxGetCurrent(&ctx));
    agentOpRetry_[group].thread.reset(new (std::nothrow) std::thread(&OpRetryManager::RetryStateMonitor, this,
        group, agentOpRetry_[group].retryCtx, std::ref(agentOpRetry_[group].startExec), ctx));
    CHK_SMART_PTR_NULL(agentOpRetry_[group].thread);
    HCCL_INFO("[%s]group[%s] rank[%u], register to agentOpRetry success",
        __func__, group.c_str(), agentParam.agentInfo.userRank);
    return HCCL_SUCCESS;
}

HcclResult OpRetryManager::RegisterServerRetryMachine(const std::string& group,
    std::map<u32, std::shared_ptr<HcclSocket>> &serverConnections, const OpRetryAgentInfo& agentInfo)
{
    if (serverOpRetry.find(group) != serverOpRetry.end()) {
        HCCL_INFO("[Register][ServerRetryMachine]group[%s] has Registered to serverOpRetry, skip", group.c_str());
        return HCCL_SUCCESS;
    }
    for (auto it = serverConnections.begin(); it != serverConnections.end(); ++it) {
        CHK_SMART_PTR_NULL(it->second);
    }

    RetryCtrl retryCtrl;
    serverOpRetry.insert(std::make_pair(group, std::move(retryCtrl)));
    std::shared_ptr<OpRetryBase> retryPtr = nullptr;
    EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerRunning>()), return HCCL_E_PTR);

    EXECEPTION_CATCH((serverOpRetry[group].retryCtx =
        std::make_shared<RetryContext>(serverConnections, retryPtr, agentInfo)), return HCCL_E_PTR);
    serverOpRetry[group].startExec = true;

    HcclRtContext ctx = nullptr;
    CHK_RET(hrtCtxGetCurrent(&ctx));
    serverOpRetry[group].thread.reset(new (std::nothrow) std::thread(&OpRetryManager::RetryStateMonitor, this,
        group, serverOpRetry[group].retryCtx, std::ref(serverOpRetry[group].startExec), ctx));
    CHK_SMART_PTR_NULL(serverOpRetry[group].thread);
    HCCL_INFO("[%s]group[%s] rank[%u], register to serverOpRetry success", __func__, group.c_str(), agentInfo.userRank);
    return HCCL_SUCCESS;
}

HcclResult OpRetryManager::UnRegisterOpRetryManager(const std::string& group)
{
    std::unique_lock<std::mutex> lock(ProcessLock_);
    CHK_PRT_RET(group.empty(),
        HCCL_ERROR("[OpRetryManager][UnRegisterOpRetryManager]params invalid, group is empty"), HCCL_E_PARA);
    HCCL_INFO("[UnRegister][OpRetryManager]group[%s] unregister start", group.c_str());
    CHK_PRT_RET(initialized_ == false, HCCL_WARNING("OpRetryManager has been destroyed"), HCCL_SUCCESS);

    if (agentOpRetry_.find(group) != agentOpRetry_.end()) {
        agentOpRetry_[group].startExec = false;
        if (agentOpRetry_[group].thread != nullptr && agentOpRetry_[group].thread->joinable()) {
            agentOpRetry_[group].thread->join();
        }
        agentOpRetry_.erase(group);
        HCCL_INFO("[UnRegister][OpRetryManager]group[%s] unregister agentOpRetry success", group.c_str());
    }

    if (serverOpRetry.find(group) != serverOpRetry.end()) {
        serverOpRetry[group].startExec = false;
        if (serverOpRetry[group].thread != nullptr && serverOpRetry[group].thread->joinable()) {
            serverOpRetry[group].thread->join();
        }
        serverOpRetry.erase(group);
        HCCL_INFO("[UnRegister][OpRetryManager]group[%s] unregister serverOpRetry success", group.c_str());
    }
    HCCL_INFO("[UnRegister][OpRetryManager]group[%s] unregister success", group.c_str());
    OpRetryConnectionPub::DeInit(group);
    return HCCL_SUCCESS;
}

void OpRetryManager::RetryStateMonitor(const std::string &group, std::shared_ptr<RetryContext> retryCtx,
    const bool &startExec, HcclRtContext rtCtx)
{
    CHK_SMART_PTR_RET_NULL(retryCtx);
    CHK_SMART_PTR_RET_NULL(rtCtx);
    CHK_RET_NULL(hrtCtxSetCurrent(rtCtx));

    // 给当前线程添加名字
    SetThreadName("Hccl_OpRetry");

    HCCL_RUN_INFO("[%s]%s start, group[%s], rankId[%u], IpInfo[%s]", __func__, retryCtx->GetOpRetryMachineType(),
        group.c_str(), retryCtx->GetRankId(), retryCtx->GetDfxIpInfo());

    HcclResult ret = HCCL_SUCCESS;
    while(initialized_ && startExec) {
        ret = retryCtx->Request();
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("RetryStateMonitor group[%s] exec fail", group.c_str()), );
    }
    HCCL_INFO("RetryStateMonitor group[%s] exit, ret[%d], initialized_[%d], startExec[%d]",
        group.c_str(), ret, initialized_, startExec);
}

HcclResult OpRetryManager::AddLinkInfoByIdentifier(s32 deviceLogicID, const std::string &identifier, 
        const std::string &newTag, std::vector<u32> &remoteRankList, bool incre)
{
    return OpretryLinkManage::GetInstance(deviceLogicID).AddLinkInfoByIdentifier(identifier, newTag, remoteRankList, incre);
}
 
HcclResult OpRetryManager::GetLinkInfoByIdentifier(s32 deviceLogicID, const std::string &identifier, 
        const std::string &newTag, std::vector<u32> &remoteRankList)
{
    return OpretryLinkManage::GetInstance(deviceLogicID).GetLinkInfoByIdentifier(identifier, newTag, remoteRankList);
}
 
HcclResult OpRetryManager::DeleteLinkInfoByIdentifier(s32 deviceLogicID, const std::string &identifier)
{
    return OpretryLinkManage::GetInstance(deviceLogicID).DeleteLinkInfoByIdentifier(identifier);
}

HcclResult OpRetryManager::SetRetryStateToWaitResume(const std::string &group, bool isRoot)
{
    std::unique_lock<std::mutex> lock(ProcessLock_);
    std::chrono::seconds setTimeout = std::chrono::seconds(OP_RETRY_SWITCH_WAIT_RESUM);
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    if (agentOpRetry_.find(group) != agentOpRetry_.end()) {
        agentOpRetry_[group].retryCtx->isAgentStateWaitResume_ = true;
        agentOpRetry_[group].retryCtx->SetEnableSendRecv(false);
        while (agentOpRetry_[group].retryCtx->GetRetryState() != RETRY_STATE_AGENT_WAIT_RESUME) {
            std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
            const auto setTime = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
            if (setTime > setTimeout) {
                HCCL_ERROR("[OpRetryManager][SetRetryStateToWaitResume]group[%s], set agent state to wait resume timeout.", group.c_str());
                return HCCL_E_TIMEOUT;
            }
            if (agentOpRetry_[group].retryCtx->isOpRetryQuit) {
                agentOpRetry_[group].retryCtx->isOpRetryQuit = false;
                break;
            }
        }
        agentOpRetry_[group].retryCtx->SetEnableSendRecv(true);
    }

    if (isRoot && serverOpRetry.find(group) != serverOpRetry.end()) {
        serverOpRetry[group].retryCtx->isServerStateWaitResume_ = true;
        serverOpRetry[group].retryCtx->SetEnableSendRecv(false);
        while (serverOpRetry[group].retryCtx->GetRetryState() != RETRY_STATE_SERVER_WAIT_RESUME) {
            std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
            const auto setTime = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
            if (setTime > setTimeout) {
                HCCL_ERROR("[OpRetryManager][SetRetryStateToWaitResume]group[%s], set server state to wait resume timeout.", group.c_str());
                return HCCL_E_TIMEOUT;
            }
            if (serverOpRetry[group].retryCtx->isOpRetryQuit) {
                serverOpRetry[group].retryCtx->isOpRetryQuit = false;
                break;
            }
        }
        if (g_isRdmaError) {
            serverOpRetry[group].retryCtx->isRdmaError = true;
            HCCL_INFO("[OpRetryManager][SetRetryStateToWaitResume]group[%s], set server rdmaError to true.", group.c_str());
        }
        serverOpRetry[group].retryCtx->SetEnableSendRecv(true);
    }
    HCCL_INFO("[OpRetryManager][SetRetryStateToWaitResume]group[%s], set sate to wait resume success", group.c_str());
    return HCCL_SUCCESS;
}

HcclResult OpRetryManager::ExitWaitResumeState(const std::string &group, bool isRoot, bool &isChangedLink)
{
    HCCL_RUN_INFO("[OpRetryManager][ExitWaitResumeState]group[%s], exit wait resume state start", group.c_str());
    std::unique_lock<std::mutex> lock(ProcessLock_);
    std::chrono::seconds exitTimeout = std::chrono::seconds(OP_RETRY_SWITCH_WAIT_RESUM);
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    if (isRoot && serverOpRetry.find(group) != serverOpRetry.end()) {
        serverOpRetry[group].retryCtx->isServerStateWaitResume_ = false;
    }
    if (agentOpRetry_.find(group) != agentOpRetry_.end()) {
        agentOpRetry_[group].retryCtx->isAgentStateWaitResume_ = false;
    }
    while (isRoot && serverOpRetry.find(group) != serverOpRetry.end() && serverOpRetry[group].retryCtx->GetRetryState() != RETRY_STATE_SERVER_RUNNING) {
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto exitTime = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        if (exitTime > exitTimeout) {
            HCCL_ERROR("[OpRetryManager][ExitWaitResumeState]group[%s], server exit wait resume state timeout", group.c_str());
            return HCCL_E_TIMEOUT;
        }
        if (serverOpRetry[group].retryCtx->isRdmaError) {
            isChangedLink = true;
        }
    }
    while (agentOpRetry_.find(group) != agentOpRetry_.end() && agentOpRetry_[group].retryCtx->GetRetryState() != RETRY_STATE_AGENT_RUNNING) {
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto exitTime = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        if (exitTime > exitTimeout) {
            HCCL_ERROR("[OpRetryManager][ExitWaitResumeState]group[%s], agent exit wait resume state timeout", group.c_str());
            return HCCL_E_TIMEOUT;
        }
        if (agentOpRetry_[group].retryCtx->isRecivedCmdToCheckLink) {
            isChangedLink = true;
        }
    }
    HCCL_RUN_INFO("[OpRetryManager][ExitWaitResumeState]group[%s], exit wait resume state success, isChangedLink[%d]", group.c_str(), isChangedLink);
    return HCCL_SUCCESS;
}
}