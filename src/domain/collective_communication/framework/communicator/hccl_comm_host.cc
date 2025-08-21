/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <algorithm>
#include <arpa/inet.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <hccl/hccl_types.h>
#include "device_capacity.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "task_abort_handler_pub.h"
#include "coll_alg_utils.h"
#include "env_config.h"
#include "i_hccl_one_sided_service.h"

namespace hccl
{
    HcclResult hcclComm::AllReduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                   HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, SyncMode syncMode)
    {
        /* 增加输出日志关键字 */
        HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
                   tag.c_str(), inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
                   GetReduceOpEnumStr(op).c_str());

        /* * 入参检查 */
        CHK_PTR_NULL(stream);
        CHK_PTR_NULL(inputPtr);
        CHK_PTR_NULL(outputPtr);

        CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AllReduce]errNo[0x%016llx] all reduce tag length is 0", HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

        CHK_RET(communicator_->CheckCount(count));
        CHK_RET(communicator_->CheckDataType(dataType, true));
        CHK_RET(communicator_->CheckReduceDataType(dataType, op));
        CHK_RET(communicator_->CheckReductionOp(op));
        HcclResult ret = communicator_->AllReduce(tag, inputPtr, outputPtr, count, dataType, op, stream, syncMode);
        if (ret != HCCL_SUCCESS)
        {
            PrintSubmittedOpCnt(tag, ret);
            return ret;
        }

        return HCCL_SUCCESS;
    }

    HcclResult hcclComm::AllReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                           HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, SyncMode syncMode)
    {
        /* 增加输出日志关键字 */
        HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]", tag.c_str(),
                   inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());

        /* * 入参检查 */
        CHK_RET(communicator_->CheckDataType(dataType, true));
        CHK_RET(communicator_->CheckReduceDataType(dataType, op));
        HcclResult ret = communicator_->AllReduceOutPlace(tag, inputPtr, outputPtr, count, dataType, op, stream, syncMode);
        if (ret != HCCL_SUCCESS)
        {
            PrintSubmittedOpCnt(tag, ret);
            return ret;
        }

        return HCCL_SUCCESS;
    }

    HcclResult hcclComm::GetOneSidedService(IHcclOneSidedService **service)
    {
        CHK_RET(communicator_->GetOneSidedService(service));

        return HCCL_SUCCESS;
    }

    HcclResult hcclComm::InitOneSidedServiceNetDevCtx(u32 remoteRankId)
    {
        CHK_RET(communicator_->InitOneSidedServiceNetDevCtx(remoteRankId));
        return HCCL_SUCCESS;
    }

    HcclResult hcclComm::OneSidedServiceStartListen(NicType nicType, HcclNetDevCtx netDevCtx)
    {
        CHK_SMART_PTR_NULL(communicator_);
        CHK_RET(communicator_->OneSidedServiceStartListen(nicType, netDevCtx));
        return HCCL_SUCCESS;
    }

    HcclResult hcclComm::GetOneSidedServiceDevIpAndPort(NicType nicType, HcclIpAddress& ipAddress, u32& port)
    {
        CHK_SMART_PTR_NULL(communicator_);
        CHK_RET(communicator_->GetOneSidedServiceDevIpAndPort(nicType, ipAddress, port));
        return HCCL_SUCCESS;
    }

    HcclResult hcclComm::DeinitOneSidedService()
    {
        CHK_SMART_PTR_NULL(communicator_);
        CHK_RET(communicator_->DeinitOneSidedService());
        return HCCL_SUCCESS;
    }

    HcclResult hcclComm::RegistTaskAbortHandler() const
    {
        HCCL_INFO("RegistTaskAbortHandler begin");
        CHK_RET(TaskAbortHandler::Init(communicator_.get()));
        return HCCL_SUCCESS;
    }

    HcclResult hcclComm::UnRegistTaskAbortHandler() const
    {
        HCCL_INFO("UnRegistTaskAbortHandler begin");
        CHK_RET(TaskAbortHandler::DeInit(communicator_.get()));
        return HCCL_SUCCESS;
    }

} // namespace hccl
