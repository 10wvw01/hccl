/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_ALG_UTILS_H
#define OPS_HCCL_CCU_ALG_UTILS_H

#include <cstdint>
#include <map>
#include <vector>

#include "hccl/base.h"
#include "alg_param.h"
#include "template_utils.h"

namespace ops_hccl {
namespace ccu_alg_utils {

/**
 * CCU 算法工具函数集合
 * 本命名空间下的函数均为自由函数，迁移自 src/ops/op_common/template/ccu_alg_template_base.{h,cc}
 * 的 CcuAlgTemplateBase 静态/常量方法，供 refactor 侧的 CCU baseTemplate 与策略实现复用。
 * 不修改 src 侧任何文件，仅通过转调或等价实现复用业务逻辑。
 */

/** 将指针转换为 uint64_t 地址（0 if nullptr）。等价于 CcuAlgTemplateBase::PointerToAddr。 */
uint64_t PointerToAddr(void *pointer);

/**
 * 从 BuffInfo 中获取 CCU 内存 token。
 * 优先级：inputPtr -> outputPtr -> hcclBuff.addr，三者均空时返回 HCCL_E_PTR。
 * 等价于 CcuAlgTemplateBase::GetToken。
 */
HcclResult GetToken(const BuffInfo &buffinfo, uint64_t &token);

/** 获取 channel 对应的 dieId。转调 CcuAlgTemplateBase::GetChannelDieId。 */
HcclResult GetChannelDieId(HcclComm comm, uint32_t rankId, const HcclChannelDesc &channelDesc, uint32_t &dieId);

/** 获取 channel 的端口数（用于区分 2 die 出框链路）。转调 CcuAlgTemplateBase::GetChannelBwCoeff。 */
HcclResult GetChannelBwCoeff(HcclComm comm, uint32_t rankId, const HcclChannelDesc &channelDesc, uint32_t &bwCoeff);

/**
 * 将 channelDescs 按 remoteRank 重新组织为 map。
 * 转调 CcuAlgTemplateBase::RestoreChannelMap。
 */
HcclResult RestoreChannelMap(const std::vector<HcclChannelDesc> &channelDescs,
                             std::map<u32, std::vector<HcclChannelDesc>> &rankIdToChannelDesc);

/**
 * 从 map 中挑选指定 dieId 上的 channel 加入到 vec，并将 index 放入 rank2ChannelIdx。
 * 转调 CcuAlgTemplateBase::SelectChannelToVec。
 */
HcclResult SelectChannelToVec(HcclComm comm, u32 myRankId, u32 rmtRankId,
                              const std::map<u32, std::vector<HcclChannelDesc>> &rankIdToChannelDesc, u32 dieId,
                              std::map<u32, u32> &rank2ChannelIdx, std::vector<HcclChannelDesc> &channels);

/**
 * 当 2 个 die 出框链路端口数不一致时，使用端口数（而非 dieId）区分 channel，
 * 将端口数多的 channel 放在 channelsPerDie[0]。
 * 转调 CcuAlgTemplateBase::ReverseChannelPerDieIfNeed。
 */
HcclResult ReverseChannelPerDieIfNeed(HcclComm comm, u32 myRankId,
                                      std::vector<std::vector<HcclChannelDesc>> &channelsPerDie);

/**
 * 遍历每个对端 rank 的 channel 数量，决策 dieNum（1 或 2）与 myDieId。
 * - channels.size()==1：dieNum=1
 * - channels.size()==2 且 dieId 相同：dieNum=1
 * - channels.size()==2 且 dieId 不同：继续遍历，最终 dieNum=2
 * 转调 CcuAlgTemplateBase::GetDieInfoFromChannelDescs。
 */
HcclResult GetDieInfoFromChannelDescs(HcclComm comm,
                                      const std::map<u32, std::vector<HcclChannelDesc>> &rankIdToChannelDesc,
                                      u32 myRankId, uint32_t &dieNum, uint32_t &dieId);

}  // namespace ccu_alg_utils
}  // namespace ops_hccl

#endif  // OPS_HCCL_CCU_ALG_UTILS_H
