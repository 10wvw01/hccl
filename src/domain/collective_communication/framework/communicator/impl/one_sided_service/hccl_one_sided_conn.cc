/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_one_sided_conn.h"
#include "sal_pub.h"
#include <numeric>

namespace hccl {
using namespace std;

HcclOneSidedConn::HcclOneSidedConn(const HcclNetDevCtx &netDevCtx, const HcclRankLinkInfo &localRankInfo,
    const HcclRankLinkInfo &remoteRankInfo, std::unique_ptr<HcclSocketManager> &socketManager,
    std::unique_ptr<NotifyPool> &notifyPool, const HcclDispatcher &dispatcher,
    const bool &useRdma, u32 sdid, u32 serverId, u32 trafficClass, u32 serviceLevel)
    : localRankInfo_(localRankInfo), socketManager_(socketManager),  notifyPool_(notifyPool)
{
    netDevCtx_ = netDevCtx;
    remoteRankInfo_ = remoteRankInfo;
    useRdma_ = useRdma;
    TransportMem::AttrInfo attrInfo = {localRankInfo.userRank, remoteRankInfo.userRank, sdid, serverId, trafficClass, serviceLevel};
    if (useRdma) {
        transportMemPtr_ = TransportMem::Create(TransportMem::TpType::ROCE, notifyPool, netDevCtx, 
            dispatcher, attrInfo);
    } else {
        transportMemPtr_ = TransportMem::Create(TransportMem::TpType::IPC, notifyPool, netDevCtx,
            dispatcher, attrInfo);
    }
    CHK_SMART_PTR_RET_NULL(transportMemPtr_);
}

HcclOneSidedConn::~HcclOneSidedConn()
{
}

HcclResult HcclOneSidedConn::Connect(const std::string &commIdentifier, s32 timeoutSec)
{
    const auto startTime = TIME_NOW();
    // 创建socket用于交换数据
    std::string newTag;
    if (localRankInfo_.userRank < remoteRankInfo_.userRank) {
        // 本端为SERVER，对端为CLIENT
        newTag = string(localRankInfo_.ip.GetReadableIP()) + "_" + to_string(localRankInfo_.port) + "_" +
            string(remoteRankInfo_.ip.GetReadableIP()) + "_" + to_string(remoteRankInfo_.port) + "_" + commIdentifier;
    } else {
        newTag = string(remoteRankInfo_.ip.GetReadableIP()) + "_" + to_string(remoteRankInfo_.port) + "_" +
            string(localRankInfo_.ip.GetReadableIP()) + "_" + to_string(localRankInfo_.port) + "_" + commIdentifier;
    }
    HCCL_DEBUG("[HcclOneSidedConn][Connect]socket tag:%s", newTag.c_str());
    std::vector<std::shared_ptr<HcclSocket>> connectSockets;
    CHK_RET(socketManager_->CreateSingleLinkSocket(newTag, netDevCtx_, remoteRankInfo_, connectSockets, true, true));
    CHK_RET(transportMemPtr_->SetDataSocket(connectSockets[0]));
    socket_ = connectSockets[0];

    if (useRdma_) {
        // 创建socket用于QP建链
        newTag += "_QP";
        CHK_RET(socketManager_->CreateSingleLinkSocket(newTag, netDevCtx_, remoteRankInfo_, connectSockets, true, true));
        CHK_RET(transportMemPtr_->SetSocket(connectSockets[0]));

        if (timeoutSec == -1) {
            // timeout为-1，超时时间设为最大值
            CHK_RET(transportMemPtr_->Connect(INT_MAX));
        } else {
            // 超时时间减去已消耗的时间，避免接口整体耗时超过入参的秒数
            const auto timeCostSec = std::chrono::duration_cast<std::chrono::seconds>(TIME_NOW() - startTime).count();
            const auto timeLeft = timeoutSec - timeCostSec;
            CHK_PRT_RET(timeLeft <= 0,
                HCCL_ERROR("[HcclOneSidedConn][Connect] Connect timeout. comm[%s], timeoutSec[%d s]",
                    commIdentifier.c_str(), timeoutSec), HCCL_E_TIMEOUT);
                    
            // Transport建链：notify资源创建+QP建链
            CHK_RET(transportMemPtr_->Connect(timeLeft));
        }
    }
    
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::ExchangeIpcProcessInfo(const ProcessInfo &localProcess, ProcessInfo &remoteProcess)
{
    HCCL_DEBUG("[HcclOneSidedConn][ExchangeIpcProcessInfo] localRank[%u] exchange process info", localRankInfo_.userRank);
    if (socket_->GetLocalRole() == HcclSocketRole::SOCKET_ROLE_CLIENT) {
        CHK_RET(socket_->Recv(&remoteProcess, sizeof(ProcessInfo)));
        CHK_RET(socket_->Send(&localProcess, sizeof(ProcessInfo)));
    } else {
        CHK_RET(socket_->Send(&localProcess, sizeof(ProcessInfo)));
        CHK_RET(socket_->Recv(&remoteProcess, sizeof(ProcessInfo)));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::ExchangeMemDesc(const HcclMemDescs &localMemDescs, HcclMemDescs &remoteMemDescs,
    u32 &actualNumOfRemote)
{
    TransportMem::RmaMemDesc *localMemDescArray = static_cast<TransportMem::RmaMemDesc *>(static_cast<void *>(localMemDescs.array));
    TransportMem::RmaMemDescs localRmaMemDescs = {localMemDescArray, localMemDescs.arrayLength};
    TransportMem::RmaMemDesc *remoteMemDescArray = static_cast<TransportMem::RmaMemDesc *>(static_cast<void *>(remoteMemDescs.array));
    TransportMem::RmaMemDescs remoteRmaMemDescs = {remoteMemDescArray, remoteMemDescs.arrayLength};

    return transportMemPtr_->ExchangeMemDesc(
        localRmaMemDescs, remoteRmaMemDescs, actualNumOfRemote);
}

HcclResult HcclOneSidedConn::GetMemType(const char *description, RmaMemType &memType)
{
    std::string tempDesc = std::string(description, TRANSPORT_EMD_ESC_SIZE);
    std::istringstream iss(tempDesc);
    // 定义需要跳过的变量的大小
    const std::vector<size_t> skip_sizes = {
        sizeof(u8),          // type
        sizeof(void*),       // addr
        sizeof(u64),         // size
        sizeof(void*)        // devAddr
    };
    // 计算偏移量
    size_t offset = std::accumulate(skip_sizes.begin(), skip_sizes.end(), 0);
    // 定位到 memType 的位置
    iss.seekg(offset);
    iss.read(reinterpret_cast<char_t *>(&memType), sizeof(memType));
    CHK_PRT_RET(memType >= RmaMemType::TYPE_NUM, HCCL_ERROR("[HcclOneSidedConn][GetMemType] get memType failed memType[%d]", static_cast<int>(memType)), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void HcclOneSidedConn::EnableMemAccess(const HcclMemDesc &remoteMemDesc, HcclMem &remoteMem)
{
    // 数据第一次转换
    HCCL_INFO("[HcclOneSidedConn][EnableMemAccess] Enable memory access.");
    const RmaMemDesc *remoteRmaMemDesc = static_cast<const RmaMemDesc *>(static_cast<const void *>(remoteMemDesc.desc));
    string tempStr = RmaMemDescCopyToStr(*remoteRmaMemDesc);
    RmaMemType memType;
    EXCEPTION_THROW_IF_ERR(GetMemType(remoteRmaMemDesc->memDesc, memType), "[HcclOneSidedConn][EnableMemAccess] get memType failed");
    auto iter = memDescMap_.find(tempStr);
    if (iter != memDescMap_.end()) {
        HcclBuf &outBuf = iter->second;
        BufferKey<uintptr_t, u64> tempKey(
            reinterpret_cast<uintptr_t>(outBuf.addr), outBuf.len);
        auto resultPair = remoteRmaBufferMgr_.Add(tempKey, outBuf.handle);
        EXCEPTION_THROW_IF_COND_ERR(resultPair.first == remoteRmaBufferMgr_.End(),
        "[HcclOneSidedConn][EnableMemAccess]The memory that is expected to enable"\
                " overlaps with the memory that has been enabled, please check params");
        remoteMem.type = static_cast<HcclMemType>(memType); //GE会检查这个字段先从字符串中获取
        remoteMem.addr = outBuf.addr;
        remoteMem.size = outBuf.len;
        return;
    }

    HcclBuf outBuf;
    EXCEPTION_THROW_IF_ERR(HcclMemImport(remoteRmaMemDesc->memDesc, TRANSPORT_EMD_ESC_SIZE, true, &outBuf, netDevCtx_),
        "[HcclOneSidedConn][EnableMemAccess] Enable memory access failed.");
    remoteMem.type = static_cast<HcclMemType>(memType); //GE会检查这个字段先从字符串中获取
    remoteMem.addr = outBuf.addr;
    remoteMem.size = outBuf.len;
    BufferKey<uintptr_t, u64> tempKey(
    reinterpret_cast<uintptr_t>(outBuf.addr), outBuf.len);
    auto resultPair = remoteRmaBufferMgr_.Add(tempKey, outBuf.handle);

    EXCEPTION_THROW_IF_COND_ERR(resultPair.first == remoteRmaBufferMgr_.End(),
    "[HcclOneSidedConn][EnableMemAccess]The memory that is expected to enable"\
            " overlaps with the memory that has been enabled, please check params");
    HCCL_INFO("[HcclOneSidedConn][EnableMemAccess] after insert remoteRmaBufferMgr_ size[%d]", remoteRmaBufferMgr_.size());
    
    HCCL_INFO("[HcclOneSidedConn][EnableMemAccess] before insert memDescMap_ size[%d]", memDescMap_.size());
    
    memDescMap_.emplace(tempStr, outBuf);
    HCCL_INFO("[HcclOneSidedConn][EnableMemAccess] after insert memDescMap_ size[%d]", memDescMap_.size());
    HCCL_INFO("[HcclOneSidedConn][EnableMemAccess] Enable memory access success.");
}

void HcclOneSidedConn::DisableMemAccess(const HcclMemDesc &remoteMemDesc)
{
    // 数据第一次转换
    const RmaMemDesc *remoteRmaMemDesc = static_cast<const RmaMemDesc *>(static_cast<const void *>(remoteMemDesc.desc));
    string tempStr = RmaMemDescCopyToStr(*remoteRmaMemDesc);
    auto it = memDescMap_.find(tempStr);
    EXCEPTION_THROW_IF_COND_ERR(it == memDescMap_.end(), "Can't find hcclmem by key");
    
    BufferKey<uintptr_t, u64> tempKey(
        reinterpret_cast<uintptr_t>(it->second.addr), it->second.len);
    HcclBuf &buf = it->second;
    try {
        if (remoteRmaBufferMgr_.Del(tempKey)) {
            EXCEPTION_THROW_IF_COND_ERR(HcclMemClose(&buf) != HCCL_SUCCESS, "Close remote memory failed.");
            HCCL_INFO("[HcclOneSidedConn][DisableMemAccess] before erase memDescMap_ size[%d]", memDescMap_.size());
            memDescMap_.erase(remoteRmaMemDesc->memDesc);
            HCCL_INFO("[HcclOneSidedConn][DisableMemAccess] after erase memDescMap_ size[%d]", memDescMap_.size());
            // 删除成功：输入key是表中某一最相近key的全集，计数-1后为0，返回true
            HCCL_INFO("[TransportIpcMem][DisableMemAccess]Memory reference count is 0, disable memory access.");
        } else {
            // 删除失败：输入key是表中某一最相近key的全集，计数不为0（存在其他remoteRank使用），返回false
            HCCL_INFO("[TransportIpcMem][DisableMemAccess]Memory reference count is larger than 0"\
                "(used by other RemoteRank), do not disable memory.");
        }
    } catch (std::out_of_range& e) {
        HCCL_ERROR("[TransportIpcMem][DisableMemAccess] catch RmaBufferMgr Del expection: %s", e.what());
        EXCEPTION_THROW_IF_COND_ERR(true, "[TransportIpcMem][DisableMemAccess] catch RmaBufferMgr Del expection");
    }
    HCCL_INFO("[HcclOneSidedConn][DisableMemAccess] Disable memory access success.");
}

void HcclOneSidedConn::BatchWrite(const HcclOneSideOpDesc* oneSideDescs, u32 descNum, const rtStream_t& stream)
{
    for (u32 i = 0; i < descNum; i++) {
        if (oneSideDescs[i].count == 0) {
            HCCL_WARNING("[HcclOneSidedConn][BatchWrite] Desc item[%u] count is 0.", i);
        }
        u32 unitSize;
        EXCEPTION_THROW_IF_ERR(SalGetDataTypeSize(oneSideDescs[i].dataType, unitSize),
            "[HcclOneSidedConn][BatchWrite] Get dataType size failed!");
        u64 byteSize = oneSideDescs[i].count * unitSize;
        HCCL_DEBUG("[HcclOneSidedConn][BatchWrite] Desc[%u], localMem[%p], remoteMem[%p], size[%llu]",
            i, oneSideDescs[i].localAddr, oneSideDescs[i].remoteAddr, byteSize);

        BufferKey<uintptr_t, u64> tempKey(
        reinterpret_cast<uintptr_t>(oneSideDescs[i].remoteAddr), byteSize);

        auto rmaBuffer = remoteRmaBufferMgr_.Find(tempKey);
        EXCEPTION_THROW_IF_COND_ERR(!rmaBuffer.first, "Can't find remoteBuffer by key");
        
        HcclBuf localMem = {oneSideDescs[i].localAddr, byteSize, nullptr}; 
        HcclBuf remoteMem = {oneSideDescs[i].remoteAddr, byteSize, rmaBuffer.second};
        EXCEPTION_THROW_IF_ERR(transportMemPtr_->Write(remoteMem, localMem, stream),
            "[HcclOneSidedConn][BatchWrite] transportMem WriteAsync failed.");
    }
    EXCEPTION_THROW_IF_ERR(transportMemPtr_->AddOpFence(stream), "[HcclOneSidedConn][BatchWrite] AddOpFence failed.");
}

void HcclOneSidedConn::BatchRead(const HcclOneSideOpDesc* oneSideDescs, u32 descNum, const rtStream_t& stream)
{
    for (u32 i = 0; i < descNum; i++) {
        if (oneSideDescs[i].count == 0) {
            HCCL_WARNING("[HcclOneSidedConn][BatchRead] Desc item[%u] count is 0.", i);
        }
        u32 unitSize;
        EXCEPTION_THROW_IF_ERR(SalGetDataTypeSize(oneSideDescs[i].dataType, unitSize),
            "[HcclOneSidedConn][BatchRead] Get dataType size failed!");
        u64 byteSize = oneSideDescs[i].count * unitSize;
        HCCL_DEBUG("[HcclOneSidedConn][BatchRead] Desc[%u], localMem[%p], remoteMem[%p], size[%llu]",
            i, oneSideDescs[i].localAddr, oneSideDescs[i].remoteAddr, byteSize);

        BufferKey<uintptr_t, u64> tempKey(
        reinterpret_cast<uintptr_t>(oneSideDescs[i].remoteAddr), byteSize);

        auto rmaBuffer = remoteRmaBufferMgr_.Find(tempKey);
        EXCEPTION_THROW_IF_COND_ERR(!rmaBuffer.first, "Can't find remoteBuffer by key");

        HcclBuf localMem = {oneSideDescs[i].localAddr, byteSize, nullptr}; 
        HcclBuf remoteMem = {oneSideDescs[i].remoteAddr, byteSize, rmaBuffer.second};
        EXCEPTION_THROW_IF_ERR(transportMemPtr_->Read(localMem, remoteMem, stream),
            "[HcclOneSidedConn][BatchRead] transportMem ReadAsync failed.");
    }
    EXCEPTION_THROW_IF_ERR(transportMemPtr_->AddOpFence(stream), "[HcclOneSidedConn][BatchRead] AddOpFence failed.");
}

HcclResult HcclOneSidedConn::ConnectWithRemote(const std::string &commIdentifier, ProcessInfo localProcess,
    s32 timeoutSec)
{
    CHK_RET(Connect(commIdentifier, timeoutSec));
    if (!useRdma_) {
        CHK_RET(ExchangeIpcProcessInfo(localProcess, remoteProcess_));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::GetRemoteProcessInfo(ProcessInfo& remoteProcess)
{
    remoteProcess = remoteProcess_;
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::ExchangeMemDesc(const HcclMemDescs &localMemDescs)
{
    remoteMemDescsVec_.resize(MAX_REMOTE_MEM_NUM);
    TransportMem::RmaMemDesc *localMemDescArray =
        static_cast<TransportMem::RmaMemDesc *>(static_cast<void *>(localMemDescs.array));
    TransportMem::RmaMemDescs localRmaMemDescs = {localMemDescArray, localMemDescs.arrayLength};
    TransportMem::RmaMemDesc *remoteMemDescArray =
        static_cast<TransportMem::RmaMemDesc *>(static_cast<void *>(remoteMemDescsVec_.data()));
    TransportMem::RmaMemDescs remoteRmaMemDescs = {remoteMemDescArray, MAX_REMOTE_MEM_NUM};

    return transportMemPtr_->ExchangeMemDesc(
        localRmaMemDescs, remoteRmaMemDescs, actualNumOfRemote_);
}

HcclResult HcclOneSidedConn::EnableMemAccess()
{
    CHK_PRT_RET(actualNumOfRemote_ > remoteMemDescsVec_.size(),
        HCCL_ERROR(
            "[HcclOneSidedConn][EnableMemAccess] actualNumOfRemote[%u] is larger than remoteMemDescsVec.size[%zu]",
            actualNumOfRemote_, remoteMemDescsVec_.size()),
        HCCL_E_INTERNAL);

    HcclMem remoteMem;
    for (u32 i = 0; i < actualNumOfRemote_; i++) {
        // 创建HcclMemDesc对象
        HcclMemDesc *remoteMemDesc = static_cast<HcclMemDesc *>(static_cast<void *>(&remoteMemDescsVec_.at(i)));
        this->EnableMemAccess(*remoteMemDesc, remoteMem);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::DisableMemAccess()
{
    CHK_PRT_RET(actualNumOfRemote_ > remoteMemDescsVec_.size(),
        HCCL_ERROR(
            "[HcclOneSidedConn][DisableMemAccess] actualNumOfRemote[%u] is larger than remoteMemDescsVec.size[%zu]",
            actualNumOfRemote_, remoteMemDescsVec_.size()),
        HCCL_E_INTERNAL);

    for (u32 i = 0; i < actualNumOfRemote_; i++) {
        // 创建HcclMemDesc对象
        HcclMemDesc *remoteMemDesc = static_cast<HcclMemDesc *>(static_cast<void *>(&remoteMemDescsVec_.at(i)));
        this->DisableMemAccess(*remoteMemDesc);
    }
    
    return HCCL_SUCCESS;
}
}