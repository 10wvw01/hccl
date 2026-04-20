/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_OMNI_TEMPLATE_PARAMS_GEN_H
#define HCCLV2_INS_OMNI_TEMPLATE_PARAMS_GEN_H

#include "template_utils.h"
#include "executor_base.h"
#include "utils.h"

namespace ops_hccl {
namespace omni {

// OMNI参数生成配置结构体
struct OmniParamGenConfig {
    u64 dataSize{0};
    u64 sliceNum{1};
    u64 dataTypeSize{0};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    u32 myRank{INVALID_VALUE_RANKID};
};

/**
 * @brief OMNI模板参数生成器
 * 负责为AICPU_TS引擎生成TemplateDataParams，支持多次执行的指令
 */
class InsOmniTemplateParamsGenerator {
public:
    explicit InsOmniTemplateParamsGenerator() = default;
    ~InsOmniTemplateParamsGenerator() = default;

    /**
     * @brief 开始处理一个指令
     * @param instructionInfo 指令信息
     * @param param 算子参数
     * @param resCtx 资源上下文
     * @param config 参数生成配置
     * @return HcclResult 执行结果
     */
    HcclResult StartInstruction(const OmniSendRecvInfo& instructionInfo,
                               const OpParam& param,
                               const AlgResourceCtxSerializable& resCtx,
                               const OmniParamGenConfig& config);

    /**
     * @brief 检查是否还有下一次执行
     * @return true 如果还有下一次执行
     */
    bool HasNext() const;

    /**
     * @brief 获取下一次执行的TemplateDataParams
     * @param[out] params 输出参数，返回下一次执行的参数
     * @return HcclResult 执行结果
     */
    HcclResult GetNext(TemplateDataParams& params);


private:
    // 当前指令状态
    const OmniSendRecvInfo* currentInstruction_{nullptr};
    const OpParam* currentParam_{nullptr};
    const AlgResourceCtxSerializable* currentResCtx_{nullptr};
    OmniParamGenConfig currentConfig_{};
    TemplateDataParams params_;
    size_t currentIndex_{0};
    bool isLast_{false};

    // 私有辅助方法
    HcclResult InitializeParams();
    HcclResult UpdateParamsForCurrentRun();
};

} // namespace omni
} // namespace ops_hccl

#endif // HCCLV2_INS_OMNI_TEMPLATE_PARAMS_GEN_H