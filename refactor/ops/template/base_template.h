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

#include <memory>

#include "hccl_algorithm.h"
#include "alg_param.h"
#include "channel.h"

namespace ops_hccl {

class BaseEngine;


/**
 * 通信模板基类
 * 职责：定义集合通信算法模板的通用接口，包括资源规划、资源计算、算法编排与kernel执行。
 * 子类（如 AllGatherMeshTemplate、AllGatherNhrTemplate）实现具体的算法逻辑。
 */
class BaseTemplate {
public:
    explicit BaseTemplate(const u32 myRank, const std::vector<u32> &ranks, TemplateDesc templateDesc)
        : myRank_(myRank), ranks_(ranks), templateDesc_(templateDesc) {}
    virtual ~BaseTemplate() = default;

    /**
     * 计算算法所需的资源请求（notify、channel、thread 等）。
     * 工作流程：
     *   1. 依据 alg 的算法类型（NHR/Mesh）选择对应的 channel 请求方式；
     *   2. NHR 算法调用 CalcChannelRequestNhr 获取 Clos 拓扑连接的 channel 列表；
     *   3. Mesh 算法调用 CalcChannelRequestMesh1D 获取 Mesh 拓扑连接的 channel 列表；
     *   4. 将结果保存到 channels_ 成员中供后续 KernelRun 使用；
     *   5. 调用 GetRes 计算所需线程数与 notify 数；
     *   6. 生成资源请求列表返回给 executor 汇总。
     * 输入参数：
     *   - comm：通信域上下文
     *   - engineType：引擎类型
     * 输出参数：
     *   - res: 资源请求，每项描述一个层级所需的 channel/notify/thread
     * 返回值：
     *   - HCCL_SUCCESS: 计算成功
     *   - HCCL_E_PARA: 参数非法
     */
    virtual HcclResult CalcRes(HcclComm comm, HcclAlgEngineType engineType, AlgResourceRequest &res) {
        const u32 rankSize = static_cast<u32>(ranks_.size());
        if (rankSize <= 1) {
            res.channels.emplace_back();
            return HCCL_SUCCESS;
        }
        // 构造 subcommInfo 和 topoInfo（对应原始 InsTempAllGather::CalcRes 的参数）。
        std::vector<std::vector<u32>> subcommInfo = {ranks_};

        std::vector<HcclChannelDesc> levelChannels;
        if (IsNhr()) {
            CHK_RET(CalcChannelRequestNhr(comm, engineType, myRank_, subcommInfo, levelChannels));
        } else {
            CHK_RET(CalcChannelRequestMesh1D(comm, engineType, myRank_, subcommInfo, levelChannels));
        }
        for (const auto &desc : levelChannels) {
            channels_.push_back(desc);
        }
        res.channels.push_back(levelChannels);

        // 计算 channelsPerRank 供 GetRes 使用。
        channelsPerRank_ = CalcChannelsPerRankInternal(levelChannels);

        // 根据算法类型计算线程数和 notify 数。
        CHK_RET(GetRes(res));
        return HCCL_SUCCESS;
    }

    /**
     * 计算所需线程数与 notify 数（对应原始 InsTempAllGatherNHR::GetRes / InsTempReduceScatterNHR::GetRes）。
     * 基类默认实现：
     *   - NHR: threadNum = channelsPerRank（普通 NHR，不 DMA 消减）, notifyPerThread = 1
     *   - Mesh: threadNum = rankSize - 1, notifyPerThread = 1
     * 子类可按算法语义覆盖（如 AllGather NHR 使用 DMA 消减算法，threadNum = channelsPerRank * 2, notifyPerThread = 2）。
     */
    virtual HcclResult GetRes(AlgResourceRequest &res) const {
        u32 threadNum = 0;
        u32 notifyPerThread = 0;
        if (IsNhr()) {
            threadNum = channelsPerRank_;
            notifyPerThread = 1;
        } else {
            const u32 rankSize = static_cast<u32>(ranks_.size());
            threadNum = (rankSize > 1) ? rankSize - 1 : 1;
            notifyPerThread = 1;
        }
        res.slaveThreadNum = threadNum - 1;
        res.notifyNumPerThread.assign(res.slaveThreadNum, notifyPerThread);
        res.notifyNumOnMainThread = threadNum - 1;
        return HCCL_SUCCESS;
    }

    /**
     * 算法编排入口，由 executor 调用，驱动模板完成数据通信。
     * 子类需实现具体的编排逻辑（如 PreCopy -> 通信原语 -> PostCopy）。
     * 输入参数：
     *   - params: 模板算法参数，包含 buffer 信息、slice 大小、repeat 次数等
     *   - templateResource: 可用资源
     * 输出参数：
     *   - ranksForOutputData: 输出数据对应的 rank 列表
     * 返回值：
     *   - HCCL_SUCCESS: 编排成功
     *   - 其他: 编排失败错误码
     */
    virtual HcclResult KernelRun(BaseEngine &engine, const TemplateDataParams &tempAlgParams,
                                TemplateResource &templateResource, std::vector<u32> &ranksForOutputData) {
        ranksForOutputData.clear();
        return HCCL_SUCCESS;
    }

protected:
    bool IsNhr() const {
        return templateDesc_.algType == HcclAlgoType::HCCL_ALGO_TYPE_NHR ||
               templateDesc_.algType == HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1;
    }

    // 计算每个对端 rank 的 channel 数最大值（等价于 template_utils.h 中的 CalcChannelsPerRank）。
    static u32 CalcChannelsPerRankInternal(const std::vector<HcclChannelDesc> &channels) {
        u32 channelsPerRank = 1;
        u32 currentRank = INVALID_VALUE_RANKID;
        u32 currentCount = 0;
        for (const auto &channel : channels) {
            if (channel.remoteRank == currentRank) {
                currentCount++;
            } else {
                if (currentCount > channelsPerRank) {
                    channelsPerRank = currentCount;
                }
                currentRank = channel.remoteRank;
                currentCount = 1;
            }
        }
        if (currentCount > channelsPerRank) {
            channelsPerRank = currentCount;
        }
        return channelsPerRank;
    }

    std::vector<HcclChannelDesc> channels_;              // 参与通信的 rank 列表
    u32 myRank_ = INVALID_VALUE_RANKID;
    std::vector<u32> ranks_;
    TemplateDesc templateDesc_;
    u32 channelsPerRank_ = 1;                             // 每个对端 rank 的 channel 数最大值
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_BASE_TEMPLATE_H
