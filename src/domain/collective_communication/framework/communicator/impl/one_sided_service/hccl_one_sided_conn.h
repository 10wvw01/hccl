/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ONE_SIDED_CONN_H
#define HCCL_ONE_SIDED_CONN_H

#include <hccl/hccl_types.h>
#include <hccl/base.h>
#include "hccl_socket_manager.h"
#include "hccl_network_pub.h"
#include "hccl_one_sided_services.h"
#include "notify_pool.h"
#include "transport_mem.h"
#include "exception_handler.h"
#include "rma_buffer_mgr.h"
#include "hccl_mem.h"
namespace hccl {
constexpr u32 MAX_REMOTE_MEM_NUM = 256;

using RemoteRmaBufferMgr = RmaBufferMgr<BufferKey<uintptr_t, u64>, void*>; // (addr, size) handle 
class HcclOneSidedConn {
public:
    struct ProcessInfo {
        u32 pid;
        u32 sdid;
        u32 serverId;
    };

    struct RmaMemDesc {
        u32 localRankId;
        u32 remoteRankId;
        char memDesc[TRANSPORT_EMD_ESC_SIZE];
    };

    // 参数超过5个，最终交付前完成优化
    HcclOneSidedConn(const HcclNetDevCtx &netDevCtx, const HcclRankLinkInfo &localRankInfo,
        const HcclRankLinkInfo &remoteRankInfo, std::unique_ptr<HcclSocketManager> &socketManager,
        std::unique_ptr<NotifyPool> &notifyPool, const HcclDispatcher &dispatcher,
        const bool &useRdma, u32 sdid, u32 serverId,
        u32 trafficClass = HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET,
        u32 serviceLevel = HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET);

    ~HcclOneSidedConn();

    HcclResult Connect(const std::string &commIdentifier, s32 timeoutSec);
    HcclResult ExchangeIpcProcessInfo(const ProcessInfo &localProcess, ProcessInfo &remoteProcess);
    HcclResult ExchangeMemDesc(const HcclMemDescs &localMemDescs, HcclMemDescs &remoteMemDescs, u32 &actualNumOfRemote);

    void EnableMemAccess(const HcclMemDesc &remoteMemDesc, HcclMem &remoteMem);
    void DisableMemAccess(const HcclMemDesc &remoteMemDesc);

    void BatchWrite(const HcclOneSideOpDesc* oneSideDescs, u32 descNum, const rtStream_t& stream);
    void BatchRead(const HcclOneSideOpDesc* oneSideDescs, u32 descNum, const rtStream_t& stream);

    HcclResult ConnectWithRemote(const std::string &commIdentifier, ProcessInfo localProcess, s32 timeoutSec);
    HcclResult GetRemoteProcessInfo(ProcessInfo& remoteProcess);

    HcclResult ExchangeMemDesc(const HcclMemDescs &localMemDescs);
    HcclResult EnableMemAccess();
    HcclResult DisableMemAccess();

private:
    std::string RmaMemDescCopyToStr(const RmaMemDesc &rmaMemDesc) const
    {
        return std::string(rmaMemDesc.memDesc, TRANSPORT_EMD_ESC_SIZE);
    }
    HcclResult GetMemType(const char *description, RmaMemType &memType);
    HcclNetDevCtx netDevCtx_{};

    const HcclRankLinkInfo &localRankInfo_;
    HcclRankLinkInfo remoteRankInfo_{};
    std::unique_ptr<HcclSocketManager> &socketManager_;

    std::shared_ptr<HcclSocket> socket_{};

    std::unique_ptr<NotifyPool> &notifyPool_;

    std::shared_ptr<TransportMem> transportMemPtr_{};

    RemoteRmaBufferMgr remoteRmaBufferMgr_{};
    std::unordered_map <std::string, HcclBuf> memDescMap_;
    bool useRdma_{true};

    ProcessInfo remoteProcess_{};
    std::vector<TransportMem::RmaMemDesc> remoteMemDescsVec_{};
    u32 actualNumOfRemote_;
};
}
#endif