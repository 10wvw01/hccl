/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_BASE_TEMPLATE_H
#define OPS_HCCL_BASE_TEMPLATE_H

namespace ops_hccl {

/**
 * 通信模板基类
 * 职责：定义集合通信算法模板的通用接口，包括资源规划、资源计算、算法编排与kernel执行。
 * 子类（如 AllGatherMeshTemplate、AllGatherNhrTemplate）实现具体的算法逻辑。
 */
class BaseTemplate {
public:
    explicit BaseTemplate() {};
    explicit BaseTemplate(const u32 myRank_, const std::vector<u32> &ranks, HcclAlgEngineType engineType, TemplateDesc templateDesc);

    /**
     * 计算算法所需的资源请求（notify、channel、thread 等）。
     * 工作流程：
     *   1. 依据 alg 的算法类型（NHR/Mesh）选择对应的 rank 获取方式；
     *   2. NHR 算法调用 getNhrRanks 获取Clos拓扑连接的 rank 列表；
     *   3. Mesh 算法调用 getMeshRanks 获取 Mesh 拓扑连接的 rank 列表；
     *   4. 将结果保存到 ranks_ 成员中供后续 CalcRes/Orchestrate 使用。
     *   5. 基于 ranks_ 中已规划的 rank 列表确定通信规模；
     *   6. 根据 AlgType 计算所需线程数与 notify 数；
     *   7. 生成资源请求列表返回给 executor 汇总。
     * 输入参数：
     *   - comm：通信域上下文
     * 输出参数：
     *   - res: 资源请求列表，每项描述一个层级所需的 channel/notify/thread
     * 返回值：
     *   - HCCL_SUCCESS: 计算成功
     *   - HCCL_E_PARA: 参数非法
     */
    HcclResult CalcRes(HcclComm comm, std::vector<AlgResourceRequest> &res) {
        // TODO： 实现
        // 把Init放在CalcRes里面实现
    }

    /**
     * 算法编排入口，由 executor 调用，驱动模板完成数据通信。
     * 子类需实现具体的编排逻辑（如 PreCopy -> 通信原语 -> PostCopy）。
     * 输入参数：
     *   - params: 模板算法参数，包含 buffer 信息、slice 大小、repeat 次数等
     *   - templateResource: 可用资源
     * 返回值：
     *   - HCCL_SUCCESS: 编排成功
     *   - 其他: 编排失败错误码
     */
    virtual HcclResult KernelRun(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource) = 0;

protected:
    std::vector<HcclChannelDesc> &channels;              // 参与通信的 rank 列表
    u32 myRank_ = INVALID_VALUE_RANKID;
    std::vector<u32> ranks;
    HcclAlgEngineType engineType = HcclAlgEngineType::AICPU;
    TemplateDesc templateDesc;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_BASE_TEMPLATE_H
