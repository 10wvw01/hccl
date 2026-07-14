/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_AICPU_BASE_TEMPLATE_H
#define OPS_HCCL_AICPU_BASE_TEMPLATE_H

#include <vector>

#include "base_template.h"
#include "alg_param.h"
#include "primitives/mesh_primitives.h"
#include "utils/utils.h"

namespace ops_hccl {

/**
 * AICPU 引擎模式 BaseTemplate。
 *
 * 职责：
 *   - 归一化 AICPU 模式的 KernelRun 主流程：PreCopy -> 通信原语 -> PostCopy；
 *   - 提供线程间前后同步、本地搬运、收发数据传输等公共 helper；
 *   - 子类化点：CalcRes、RunAlgorithm（具体的 Mesh/NHR 等通信编排）。
 *
 * 设计说明：
 *   - 不依赖 src/.../alg_data_trans_wrapper.h，因其会传递引入 src 的 template_utils.h，
 *     与 refactor 的 TemplateDataParams/TemplateResource/BuffInfo 产生重定义冲突；
 *   - 数据传输 helper（LocalCopy / PreSyncInterThreads / PostSyncInterThreads）
 *     统一在 ops/utils/utils.h 中实现，本类通过 include 引入。
 *
 * 参考实现：
 *   - src/ops/all_gather/template/aicpu/ins_temp_all_gather_mesh_1D.cc
 *   - src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.cc
 *   - refactor/ops/engine/aicpu/aicpu_engine.cc（数据传输 wrapper 实现参考）
 */
class AicpuBaseTemplate : public BaseTemplate {
public:
    AicpuBaseTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc)
        : BaseTemplate(myRank, std::move(ranks), templateDesc) {}
    ~AicpuBaseTemplate() = default;

    /**
     * AICPU 算法编排入口（Template Method）。
     * 流程：
     *   1. 保存 tempAlgParams 到成员 tempAlgParams_；
     *   2. 数据量为 0 直接返回；
     *   3. PreCopy：input -> output / ccl buffer 的本地数据预处理；
     *   4. 单 rank 时直接返回；
     *   5. 多线程时 PreSyncInterThreads；
     *   6. RunAlgorithm：子类实现具体通信原语编排；
     *   7. 多线程时 PostSyncInterThreads；
     *   8. PostCopy：ccl buffer -> output 的后处理（若需要）。
     */
    HcclResult KernelRun(BaseEngine &engine, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource, std::vector<u32> &ranksForOutputData) override;

protected:
    /**
     * 子类实现：具体的通信原语编排（如 RunMeshAllGather / RunNhrAllGather）。
     * 基类 KernelRun 在 PreCopy 后、PostCopy 前调用。
     * 输入参数：
     *   - templateResource: 通信资源（channels / threads）
     * 输出参数：
     *   - sendRecvInfos: 子类生成的收发描述列表，基类统一调 SendAll 执行
     *   - ranksForOutputData: 输出数据对应的 rank 列表
     * 返回值：
     *   - HCCL_SUCCESS: 通信成功
     *   - 其他: 通信失败错误码
     */
    virtual HcclResult RunAlgorithm(TemplateResource &templateResource,
                                    std::vector<SendRecvInfo> &sendRecvInfos,
                                    std::vector<u32> &ranksForOutputData) = 0;

    /**
     * 本地预处理（input -> output / ccl buffer）。
     * 默认实现：scratch 中内存布局为所有 rank 数据按顺序排列，
     * 每个 rank 数据大小为 sliceCount * dataTypeSize（即 dataCountPerLoop）。
     * 将本 rank 的 input 拷贝到 output 和 ccl buffer。
     * 子类可按算法语义覆盖。
     */
    virtual HcclResult PreCopy(const std::vector<ThreadHandle> &threads);

    /**
     * 本地后处理（ccl buffer -> output）。
     * 默认实现：将 ccl buffer 中其它 rank 的数据搬回 output。
     * scratch 布局同 PreCopy。子类可按算法语义覆盖。
     */
    virtual HcclResult PostCopy(const std::vector<ThreadHandle> &threads);

    /**
     * 统一逐个执行 SendRecv。
     * 由 KernelRun 在 RunAlgorithm 返回后调用。
     */
    HcclResult SendAll(BaseEngine &engine, const std::vector<SendRecvInfo> &sendRecvInfos,
                        TemplateResource &templateResource);

    /** 工具：判断 channels 是否为 PCIe 协议（决定 Read/Write 模式）。 */
    bool IsPcieProtocol(const std::map<u32, std::vector<ChannelInfo>> &channels) const
    {
        for (auto it = channels.begin(); it != channels.end(); ++it) {
            if (!it->second.empty() && it->second[0].protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
                return true;
            }
        }
        return false;
    }
    
    // ───────────── 公共成员（子类直接访问） ─────────────
    TemplateDataParams tempAlgParams_{};
    u32 templateRankSize_{0};
    HcclDataType dataType_{HcclDataType::HCCL_DATA_TYPE_RESERVED};
    bool enableRemoteMemAccess_{false};
    std::vector<u32> ranksForOutputData_{};
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_AICPU_BASE_TEMPLATE_H
