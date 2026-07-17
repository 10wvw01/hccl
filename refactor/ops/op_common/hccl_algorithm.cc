/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_algorithm.h"

#include "log.h"
#include "base_engine.h"
#include "engine/aicpu/aicpu_engine.h"
#include "executor/ops_executor.h"

namespace ops_hccl {

/**
 * 根据算法的引擎类型（engineType）构造对应的 Engine。
 *   - AICPU: 返回 AiCpuEngine（传入 comm 供 CreateRes 使用）
 *   - CCU_MS: 返回 CcuMsEngine（传入 comm 供 CreateRes 使用）
 *   - CCU_SCHED: 返回 CcuScheEngine（传入 comm 供 CreateRes 使用）
 */
std::unique_ptr<BaseEngine> HcclAlgorithm::GetEngine(void)
{
    switch (engineType) {
        case HcclAlgEngineType::AICPU:
            return std::make_unique<AiCpuEngine>();
        default:
            HCCL_ERROR("[HcclAlgorithm][GetEngine] invalid engineType[%d]", static_cast<int>(engineType));
            return nullptr;
    }
}

/**
 * 构造统一Executor。
 * 输入参数：
 *   - param: 算子参数，传递给执行器构造函数用于初始化 rank/data 等信息
 */
std::unique_ptr<OpsExecutor> HcclAlgorithm::GetExecutor(OpParam &param)
{
    return std::make_unique<OpsExecutor>(*this, param);
}

/**
 * 打印算法基本信息，用于调试。
 */
void HcclAlgorithm::Dump()
{
    HCCL_INFO("[HcclAlgorithm][Dump] engineType[%d], hcclCmdType[%d], execPolicy[%d], childrenNum[%zu]",
        static_cast<int>(engineType), static_cast<int>(hcclCmdType), static_cast<int>(algoExecDesc.execPolicy),
        algoExecDesc.children.size());
}

// ─────────────────────────────────────────────────────────────────
// AlgoExecDesc 树的序列化/反序列化
// ─────────────────────────────────────────────────────────────────

void AlgoExecDesc::Serialize(BinaryStream &bs, const AlgoExecDesc &desc)
{
    bs << desc.execPolicy;

    // 序列化 children: 每个 child 是 variant<TemplateExecDesc, shared_ptr<AlgoExecDesc>>
    // 先写 children 数量，再逐个写 variant tag + data
    size_t childCount = desc.children.size();
    bs << childCount;
    for (const auto &child : desc.children) {
        // variant index: 0 = TemplateExecDesc, 1 = shared_ptr<AlgoExecDesc>
        uint32_t variantIndex = static_cast<uint32_t>(child.index());
        bs << variantIndex;
        if (variantIndex == 0) {
            const auto &tplExecDesc = std::get<TemplateExecDesc>(child);
            bs << tplExecDesc.templateDesc.hcclCmdType;
            bs << tplExecDesc.templateDesc.algType;
            bs << tplExecDesc.templateDesc.shotMode;
            bs << tplExecDesc.templateDesc.jettyMode;
            bs << tplExecDesc.subCommIndex;
        } else {
            const auto &subDesc = std::get<std::shared_ptr<AlgoExecDesc>>(child);
            // 递归序列化子树
            Serialize(bs, *subDesc);
        }
    }

    // 序列化 dataSplitRatio
    bs << desc.dataSplitRatio;
}

AlgoExecDesc AlgoExecDesc::Deserialize(BinaryStream &bs)
{
    AlgoExecDesc desc;
    bs >> desc.execPolicy;

    size_t childCount;
    bs >> childCount;
    desc.children.resize(childCount);
    for (size_t i = 0; i < childCount; i++) {
        uint32_t variantIndex;
        bs >> variantIndex;
        if (variantIndex == 0) {
            TemplateExecDesc tplExecDesc;
            bs >> tplExecDesc.templateDesc.hcclCmdType;
            bs >> tplExecDesc.templateDesc.algType;
            bs >> tplExecDesc.templateDesc.shotMode;
            bs >> tplExecDesc.templateDesc.jettyMode;
            bs >> tplExecDesc.subCommIndex;
            desc.children[i] = std::move(tplExecDesc);
        } else {
            auto subDesc = std::make_shared<AlgoExecDesc>(Deserialize(bs));
            desc.children[i] = std::move(subDesc);
        }
    }

    bs >> desc.dataSplitRatio;
    return desc;
}

// ─────────────────────────────────────────────────────────────────
// HcclAlgorithm 序列化/反序列化
// ─────────────────────────────────────────────────────────────────

void HcclAlgorithm::SerializeTo(BinaryStream &bs) const
{
    bs << hcclCmdType;
    bs << engineType;
    AlgoExecDesc::Serialize(bs, algoExecDesc);
}

void HcclAlgorithm::DeserializeFrom(BinaryStream &bs)
{
    bs >> hcclCmdType;
    bs >> engineType;
    algoExecDesc = AlgoExecDesc::Deserialize(bs);
}

} // namespace ops_hccl
