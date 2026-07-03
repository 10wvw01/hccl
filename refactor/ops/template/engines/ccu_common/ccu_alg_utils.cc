/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_alg_utils.h"

#include "ccu_alg_template_base.h"
#include "ccu_res_dl.h"
#include "log.h"

namespace ops_hccl {
namespace ccu_alg_utils {

uint64_t PointerToAddr(void *pointer)
{
    if (pointer != nullptr) {
        return reinterpret_cast<uint64_t>(pointer);
    }
    return 0;
}

HcclResult GetToken(const BuffInfo &buffinfo, uint64_t &token)
{
    if (buffinfo.inputPtr != nullptr && buffinfo.inputSize != 0) {
        HcommCcuGetMemToken(PointerToAddr(buffinfo.inputPtr),
                            static_cast<uint64_t>(buffinfo.inputSize), &token);
        return HCCL_SUCCESS;
    } else if (buffinfo.outputPtr != nullptr && buffinfo.outputSize != 0) {
        HcommCcuGetMemToken(PointerToAddr(buffinfo.outputPtr),
                            static_cast<uint64_t>(buffinfo.outputSize), &token);
        return HCCL_SUCCESS;
    } else if (buffinfo.hcclBuff.addr != nullptr && buffinfo.hcclBuff.size != 0) {
        HcommCcuGetMemToken(PointerToAddr(buffinfo.hcclBuff.addr),
                            static_cast<uint64_t>(buffinfo.hcclBuff.size), &token);
        return HCCL_SUCCESS;
    }
    HCCL_WARNING("[ccu_alg_utils::GetToken] inputMem, outputMem and hcclBuff are all null");
    return HCCL_E_PTR;
}

HcclResult GetChannelDieId(HcclComm comm, uint32_t rankId, const HcclChannelDesc &channelDesc, uint32_t &dieId)
{
    return CcuAlgTemplateBase::GetChannelDieId(comm, rankId, channelDesc, dieId);
}

HcclResult GetChannelBwCoeff(HcclComm comm, uint32_t rankId, const HcclChannelDesc &channelDesc, uint32_t &bwCoeff)
{
    return CcuAlgTemplateBase::GetChannelBwCoeff(comm, rankId, channelDesc, bwCoeff);
}

HcclResult RestoreChannelMap(const std::vector<HcclChannelDesc> &channelDescs,
                             std::map<u32, std::vector<HcclChannelDesc>> &rankIdToChannelDesc)
{
    return CcuAlgTemplateBase::RestoreChannelMap(channelDescs, rankIdToChannelDesc);
}

HcclResult SelectChannelToVec(HcclComm comm, u32 myRankId, u32 rmtRankId,
                              const std::map<u32, std::vector<HcclChannelDesc>> &rankIdToChannelDesc, u32 dieId,
                              std::map<u32, u32> &rank2ChannelIdx, std::vector<HcclChannelDesc> &channels)
{
    return CcuAlgTemplateBase::SelectChannelToVec(comm, myRankId, rmtRankId, rankIdToChannelDesc, dieId,
                                                  rank2ChannelIdx, channels);
}

HcclResult ReverseChannelPerDieIfNeed(HcclComm comm, u32 myRankId,
                                      std::vector<std::vector<HcclChannelDesc>> &channelsPerDie)
{
    return CcuAlgTemplateBase::ReverseChannelPerDieIfNeed(comm, myRankId, channelsPerDie);
}

HcclResult GetDieInfoFromChannelDescs(HcclComm comm,
                                      const std::map<u32, std::vector<HcclChannelDesc>> &rankIdToChannelDesc,
                                      u32 myRankId, uint32_t &dieNum, uint32_t &dieId)
{
    return CcuAlgTemplateBase::GetDieInfoFromChannelDescs(comm, rankIdToChannelDesc, myRankId, dieNum, dieId);
}

}  // namespace ccu_alg_utils
}  // namespace ops_hccl
