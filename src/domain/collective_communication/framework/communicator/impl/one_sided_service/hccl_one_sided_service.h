/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ONE_SIDED_SERVICE_H
#define HCCL_ONE_SIDED_SERVICE_H

#include <set>

#include "i_hccl_one_sided_service.h"
#include "hccl_one_sided_conn.h"
#include "hccl_common.h"
#include "common.h"
#include "externalinput_pub.h"
#include "hccl_mem.h"
#include "global_mem_record.h"

using HcclBatchData = struct HcclBatchDataDef {
    HcclComm comm;
    HcclCMDType cmdType;
    u32 remoteRank;
    HcclOneSideOpDesc* desc;
    u32 descNum;
    rtStream_t stream;
};

namespace hccl {
constexpr size_t HCCL_MEM_DESC_STR_LEN = HCCL_MEM_DESC_LENGTH + 1 - (sizeof(u32) * 2);
constexpr u32 MAX_COMM_MEM_BIND_COUNT = 256;

class HcclOneSidedService : public IHcclOneSidedService {
public:
    using RankId = u32;
    using ProcessInfo = HcclOneSidedConn::ProcessInfo;

    struct HcclMemDescData {
        u32 localRankId;
        u32 remoteRankId;
        char memDesc[HCCL_MEM_DESC_STR_LEN];
    };

    HcclOneSidedService(std::unique_ptr<HcclSocketManager> &socketManager,
        std::unique_ptr<NotifyPool> &notifyPool);

    // 父类Config()等已经完成必要参数的配置
    HcclOneSidedService() = default;
    ~HcclOneSidedService() override;

    HcclResult ReMapMem(HcclMem *memInfoArray, u64 arraySize);
    HcclResult RegMem(void* addr, u64 size, HcclMemType type, RankId remoteRankId, HcclMemDesc &localMemDesc);
    HcclResult DeregMem(const HcclMemDesc &localMemDesc);
    // 可能返回超时
    HcclResult ExchangeMemDesc(RankId remoteRankId, const HcclMemDescs &localMemDescs,
        HcclMemDescs &remoteMemDescs, u32 &actualNumOfRemote, const std::string &commIdentifier, s32 timeoutSec);

    void EnableMemAccess(const HcclMemDesc &remoteMemDesc, HcclMem &remoteMem);
    void DisableMemAccess(const HcclMemDesc &remoteMemDesc);

    void BatchPut(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t &stream);
    void BatchGet(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t &stream);

    HcclResult GetIsUsedRdma(RankId remoteRankId, bool &useRdma);

    // 主要完成通信域粒度的数据面建链和已绑定MR的交换、使能
    HcclResult Prepare(const std::string &commIdentifier, const HcclPrepareConfig* config, s32 timeoutSec);
    HcclResult InitIsUsedRdmaMap(bool& needInitNic, bool& needInitVnic);
    HcclResult DeInit() override;

    HcclResult BindMem(void* memRecordHandle, const std::string &commIdentifier);   // 绑定一块全局内存
    HcclResult UnbindMem(void *memRecordHandle, const std::string &commIdentifier); // 解绑一块全局内存
    
    inline bool HasBoundMem() const // 判断有没有绑定着的内存
    {
        return !boundMemPtrSet_.empty();
    }

private:
    u32 registedMemCnt_{0};
    HcclResult IsUsedRdma(RankId remoteRankId, bool &useRdma);

    HcclResult SetupRemoteRankInfo(RankId remoteRankId, HcclRankLinkInfo &remoteRankInfo);
    HcclResult CreateConnection(RankId remoteRankId, const HcclRankLinkInfo &remoteRankInfo,
        std::shared_ptr<HcclOneSidedConn> &tempConn);
    HcclResult Grant(const HcclMemDesc &localMemDesc, const ProcessInfo &remoteProcess);
    HcclBuf *GetHcclBufByDesc(std::string &descStr, bool useRdma);

    // Prepare新增函数
    void ConnectByThread(std::shared_ptr<HcclOneSidedConn>& conn, const std::string &commIdentifier, s32 timeoutSec);
    HcclResult CreateLinkFullmesh(const std::string &commIdentifier, s32 timeoutSec);
    HcclResult RegBoundMem(HcclNetDevCtx netDevCtx, const HcclMem& localMem,
        HcclMemDesc &localMemDesc, HcclBuf& buf);
    HcclResult RegisterBoundMems();
    HcclResult ExchangeMemDescFullMesh();
    HcclResult ExchangeMemDescByThread(std::shared_ptr<HcclOneSidedConn>& conn, bool isUseRdma);
    HcclResult EnableMemAccess();
    HcclResult DisableMemAccess();
    HcclResult Grant(HcclBuf& buf);
    HcclResult PrepareFullMesh(const std::string &commIdentifier, s32 timeoutSec);

    std::unordered_map<RankId, std::shared_ptr<HcclOneSidedConn>> oneSidedConns_{};
    std::unordered_map<RankId, bool> isUsedRdmaMap_;
    std::unordered_map<std::string, HcclBuf> desc2HcclBufMapIpc_{};
    std::unordered_map<std::string, HcclBuf> desc2HcclBufMapRoce_{};

    std::set<GlobalMemRecord*> boundMemPtrSet_{};
    s32 deviceLogicId_{HOST_DEVICE_ID};
    std::vector<HcclMemDesc> localMemIpcDescs_;
    std::vector<HcclMemDesc> localMemRoceDescs_;

    bool prepared_{false}; // 表示是否prepare过
    std::atomic<bool> hasErrorFlag_{false}; // 用于表示多线程操作是否出错
    bool needRegRoceMem_{false}; // 是否需要注册roce内存
    bool needRegIpcMem_{false}; // 是否需要注册ipc内存
    ProcessInfo localProcess_{};
};
}

#endif