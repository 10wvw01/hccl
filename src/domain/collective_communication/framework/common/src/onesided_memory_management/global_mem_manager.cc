/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "global_mem_manager.h"

#include <string>
#include "adapter_pub.h"

namespace hccl {
GlobalMemRegMgr::~GlobalMemRegMgr()
{
}

GlobalMemRegMgr& GlobalMemRegMgr::GetInstance()
{
    static GlobalMemRegMgr instance;
    return instance;
}

HcclResult GlobalMemRegMgr::Destroy()
{
    HCCL_INFO("[GlobalMemRegMgr][%s] start.", __func__);
    for (auto& pair : netDevCtxMap_) {
        HcclNetCloseDev(pair.second.second);
        if (pair.second.first == NicType::DEVICE_NIC_TYPE) {
            socketManager_->ServerDeInit(pair.first, HETEROG_CCL_PORT);
        }
        HCCL_INFO("[GlobalMemRegMgr][%s] Close netdev[%p].", __func__, pair.second.second);
    }
    netDevCtxMap_.clear();
    CHK_RET(DeInitNic());
    HCCL_INFO("[GlobalMemRegMgr][%s] end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::CheckOverlapAndInsert(GlobalMemRecord& memRecord, void** memRecordHandle)
{
    // 由于每次插入都会保证不产生重叠，所以只需要检查最接近的两条记录是否有重叠即可
    const auto memInfo = memRecord.PrintInfo();

    auto it = memRecordSet_.lower_bound(memRecord);
    if (it != memRecordSet_.cend()) {
        if (memRecord == *it) {
            // 已经存在相同的记录，报错返回
            HCCL_ERROR("[GlobalMemRegMgr][CheckOverlapAndInsert] The memory[%s] has been registered already.",
                memInfo.c_str());
            return HCCL_E_PARA;
        }

        // 检查后一个记录
        if (memRecord.HasOverlap(*it)) {
            // 后一个记录有重叠，报错
            HCCL_ERROR(
                "[GlobalMemRegMgr][CheckOverlapAndInsert] The new memory[%s] overlaps with an existing memory[%s].",
                memInfo.c_str(), (*it).PrintInfo().c_str());
            return HCCL_E_PARA;
        }
    }

    // 检查前一个记录
    if (it != memRecordSet_.cbegin()) {
        auto prevIt = std::prev(it);
        if (memRecord.HasOverlap(*prevIt)) {
            // 前一个记录有重叠，报错
            HCCL_ERROR(
                "[GlobalMemRegMgr][CheckOverlapAndInsert] The new memory[%s] overlaps with an existing memory[%s].",
                memInfo.c_str(), (*prevIt).PrintInfo().c_str());
            return HCCL_E_PARA;
        }
    }

    // 没有重叠，插入在当前it附近的位置
    auto insertIt = memRecordSet_.insert(it, std::move(memRecord));

    // 取出地址作为handle
    *memRecordHandle = const_cast<GlobalMemRecord*>(&(*insertIt));

    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::Reg(const HcclMem* mem, void** memRecordHandle)
{
    // 不允许注册空内存，报错退出
    CHK_PTR_NULL(mem);
    CHK_PRT_RET(mem->addr == nullptr,
        HCCL_ERROR("[GlobalMemRegMgr][Reg] The address of mem[%p] to register is null.", mem),
        HCCL_E_PARA);
    CHK_PRT_RET(mem->size == 0,
        HCCL_ERROR("[GlobalMemRegMgr][Reg] The size of mem[%p] to register is 0.", mem),
        HCCL_E_PARA);

    // 进程粒度最多注册MAX_GLOBAL_MEM_REG_COUNT块独立的内存，若超过则报错退出
    CHK_PRT_RET(memRecordSet_.size() == MAX_GLOBAL_MEM_REG_COUNT,
        HCCL_ERROR("[GlobalMemRegMgr][Reg] The number of memory registered in the process has reached the maximum"
                   " value[%u]. Cannot register more memories.", MAX_GLOBAL_MEM_REG_COUNT),
        HCCL_E_UNAVAIL);

    GlobalMemRecord newRecord(mem);
    const auto memInfo = newRecord.PrintInfo();
    std::unique_lock<std::mutex> lock(lock_);
    CHK_RET(CheckOverlapAndInsert(newRecord, memRecordHandle));
    HCCL_INFO("[GlobalMemRegMgr][Reg] Added a new memory record[%s], handle[%p].",
        memInfo.c_str(), *memRecordHandle);
    
    // 记录地址，便于其他接口进行入参handle合法性校验
    validHandlePtrSet.emplace(*memRecordHandle);

    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::DeReg(void *memRecordHandle)
{
    const auto *memRecordPtr = static_cast<GlobalMemRecord *>(memRecordHandle);
    const auto memInfo = memRecordPtr->PrintInfo();
    std::unique_lock<std::mutex> lock(lock_);

    // 先找到指向这个记录的迭代器
    const auto it = memRecordSet_.find(*memRecordPtr);
    if (it == memRecordSet_.cend()) {
        // 找不到记录报错退出
        HCCL_ERROR("[GlobalMemRegMgr][DeReg] Cannot found the corresponding record of memory[%s].", memInfo.c_str());
        return HCCL_E_NOT_FOUND;
    }

    // 检查内存记录是否还与通信域绑定
    if (memRecordPtr->IsBeingBound()) {
        // 该内存还与一个或多个通信域绑定，报错并打印绑定的信息
        const auto boundComm = memRecordPtr->GetBoundComm();
        HCCL_ERROR(
            "[GlobalMemRegMgr][DeReg] Cannot deregistor memory[%s] since it is still bound to comm(s) listed below:",
            memInfo.c_str());

        for (const auto &commIdentifier : boundComm) {
            HCCL_ERROR("[GlobalMemRegMgr][DeReg][bound comm] %s", commIdentifier.c_str());
        }

        HCCL_ERROR("[GlobalMemRegMgr][DeReg] Please unbind from all bound comm first.");
        return HCCL_E_PARA;
    }

    // 清除记录，析构时会触发网络设备的解注册
    memRecordSet_.erase(it);
    HCCL_INFO("[GlobalMemRegMgr][DeReg] Memory[%s] has been deregistered.", memInfo.c_str());

    // 当内存全部解注册后，主动释放网络资源
    if (memRecordSet_.empty()) {
        CHK_RET(Destroy());
    }

    validHandlePtrSet.erase(memRecordHandle);
    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::InitNic()
{
    if (nicInited_) {
        HCCL_INFO("[InitNic] Nic has been inited. devicePhyId[%u], deviceLogicId[%d]", devicePhyId_, deviceLogicId_);
        return HCCL_SUCCESS;
    }

    if (devicePhyId_ == INVALID_UINT || deviceLogicId_ == INVALID_INT) {
        CHK_RET(hrtGetDevice(&deviceLogicId_));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_));
    }
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, static_cast<u32>(deviceLogicId_), false));
    nicInited_ = true;
    socketManager_.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, deviceLogicId_, devicePhyId_, 0));
    CHK_PTR_NULL(socketManager_);
    HCCL_INFO("[InitNic] Nic init success, devicePhyId[%u], deviceLogicId[%d]", devicePhyId_, deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::DeInitNic()
{
    if (!nicInited_) {
        HCCL_INFO(
            "[DeInitNic] Nic has been deinited. devicePhyId[%u], deviceLogicId[%d]", devicePhyId_, deviceLogicId_);
        return HCCL_SUCCESS;
    }

    if (devicePhyId_ == INVALID_UINT || deviceLogicId_ == INVALID_INT) {
        CHK_RET(hrtGetDevice(&deviceLogicId_));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_));
    }
    CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, static_cast<u32>(deviceLogicId_)));
    nicInited_ = false;
    HCCL_INFO("[DeInitNic] Nic deinit success. devicePhyId[%u], deviceLogicId[%d]", devicePhyId_, deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::GetNetDevCtx(NicType nicType, const HcclIpAddress &ipAddr, u32 port,
    HcclNetDevCtx &netDevCtx)
{
    if (devicePhyId_ == INVALID_UINT || deviceLogicId_ == INVALID_INT) {
        CHK_RET(hrtGetDevice(&deviceLogicId_));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_));
    }
    HCCL_INFO("[GlobalMemRegMgr][GetNetDevCtx] nicType[%d], ip[%s]", nicType, ipAddr.GetReadableAddress());

    // 进程粒度open dev，如果已open，直接复用
    if (netDevCtxMap_.find(ipAddr) != netDevCtxMap_.end()) {
        netDevCtx = netDevCtxMap_[ipAddr].second;
        CHK_PTR_NULL(netDevCtx);
        return HCCL_SUCCESS;
    }
    HcclNetDevCtx tempNetDevCtx;
    CHK_RET(HcclNetOpenDev(&tempNetDevCtx, nicType, devicePhyId_, deviceLogicId_, ipAddr));
    CHK_PTR_NULL(tempNetDevCtx);
    netDevCtxMap_.insert(std::make_pair(ipAddr, std::make_pair(nicType, tempNetDevCtx)));
    netDevCtx = tempNetDevCtx;
    if (nicType == NicType::DEVICE_NIC_TYPE) {
        CHK_RET(socketManager_->ServerInit(netDevCtx, port));
    }
    HCCL_INFO(
        "[GlobalMemRegMgr][GetNetDevCtx] nicType[%d] ip[%s] has been Init.", nicType, ipAddr.GetReadableAddress());
    return HCCL_SUCCESS;
}

} // namespace hccl