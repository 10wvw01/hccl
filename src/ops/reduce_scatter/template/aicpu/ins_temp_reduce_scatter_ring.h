/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_RING_H
#define INS_TEMP_REDUCE_SCATTER_RING_H

#include "alg_v2_template_base.h"

namespace ops_hccl {

// Ring ReduceScatter — mini task 学习用模板
// 算法: N 个 rank 排列为逻辑环, 经过 N-1 轮 SendRecvWrite + LocalReduce,
//       每个 rank 最终得到一份完全归约的数据分片
//
// Rank i (ring rank = algRank):
//   - 输入: rankSize 个等大 chunk (每 chunk count 个元素)
//   - 输出: 1 个 chunk (count 个元素), 所有 rank 对应 chunk 的归约结果
//
// 环结构: 按 subCommRanks_[0] 的顺序排列, 首尾相连
//   发送方向: rank_i → rank_{i+1}
//   接收方向: rank_i ← rank_{i-1}
class InsTempReduceScatterRing : public InsAlgTemplateBase {
public:
    InsTempReduceScatterRing() = default;
    explicit InsTempReduceScatterRing(const OpParam &param, const u32 rankId,
                                      const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempReduceScatterRing() override = default;

    std::string Describe() const override
    {
        return "Template of ReduceScatter Ring, rankSize=" + std::to_string(templateRankSize_);
    }

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    // 执行 ring 的 N-1 轮 send/recv + reduce
    HcclResult RunRingLoop(const OpParam &param, const TemplateDataParams &tempAlgParams,
                           const std::vector<ThreadHandle> &threads,
                           const std::map<u32, std::vector<ChannelInfo>> &channels);
};

}  // namespace ops_hccl

#endif  // INS_TEMP_REDUCE_SCATTER_RING_H
