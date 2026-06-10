/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_V_MESH_OCS_H
#define INS_TEMP_ALL_TO_ALL_V_MESH_OCS_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

constexpr u32 NET_NUM = 2;

class InsTempAlltoAllVMeshOcs : public InsAlgTemplateBase {
public:
    InsTempAlltoAllVMeshOcs() = default;

    /* 构造函数：传入算子参数、rank ID、子通信域划分结果
     * 由 Executor 通过模板工厂创建，subCommRanks 由 TopoMatch1D 生成 */
    explicit InsTempAlltoAllVMeshOcs(const OpParam& param, const u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);

    ~InsTempAlltoAllVMeshOcs() override;    

    std::string Describe() const override
    {
        std::string info = "Template of alltoallv Mesh OCS with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    /* 算法执行入口：由 Executor 在每个 loop 中调用
     * 负责：初始化 OCS 参数 → 查找逻辑 rank → 调用 RunLimitedConcurrencyOCS */
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& tempAlgParams,
                         TemplateResource& templateResource) override;

    /* 资源计算：由 Executor 在编排前调用
     * 负责：通道申请 → OCS 参数初始化 → 并发度计算 → 从线程数/notify 数确定 */
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                        AlgResourceRequest& resourceRequest) override;

    /* CCL Buffer 倍率计算：返回槽位数 concurrentSendRecvNum_
     * Executor 据此计算 inputSliceStride = hcclBuff.size / concurrentSendRecvNum_
     * AICPU 路径无 CalcRes，需在此再读 env/topo 填充 numGroups_ */
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    /* 主线程→从线程的 notify 索引：所有从线程使用 notify index 0（广播唤醒） */
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub) override;

    /* 从线程→主线程的 notify 索引：从线程 i 使用 notify index i（逐个确认） */
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

    void GetOcsGroupNumInfoFromTopo(const TopoInfoWithNetLayerDetails* topoInfo) override;

private:
    /* OCS 有限并发算法主入口               
     * 流程：LocalCopy(自环) → 多轮通信(PreSync → SendRecv → PostSync) */
    HcclResult RunLimitedConcurrencyOcs(const std::map<u32, std::vector<ChannelInfo>> &channels,
        const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams, u32 myAlgRank);

    /* 计算某轮次的发送目标 rank 列表和接收源 rank 列表
     * 输出大小均为 concurrentPlanesPerRound × numGroups_
     * sendToRanks[i] 和 recvFromRanks[i] 构成第 i 路通信对 */
    HcclResult CalcCommRankSetforOneRoundOcs(u32 roundIdx, u32 concurrentPlanesPerRound,
        std::vector<u32> &sendToRanks, std::vector<u32> &recvFromRanks);

    /* 计算发送目标 rank 对应的 CCL Buffer 槽位索引（写入对端 CCL）
     * 基于：对端 plane 与本端 plane 的顺时针距离 → planeIdx → planeIdx×numGroups + groupId */
    u32 CalcSendDstIdxOcs(u32 dstRank, u32 concurrentPlanesPerRound) const;

    /* 计算接收源 rank 对应的 CCL Buffer 槽位索引（从本端 CCL 读出）
     * 基于：本端 plane 与源端 plane 的顺时针距离 → planeIdx → planeIdx×numGroups + groupId */
    u32 CalcRecvSrcIdxOcs(u32 srcRank, u32 concurrentPlanesPerRound) const;

    /* 单通道数据路径：根据 doSend/doRecv 执行 SendWrite/RecvWrite/SendRecvWrite
     * recv 后自动调用 PostCopyOCS 将数据从 CCL Buffer 搬到用户 Output */
    HcclResult RunSendRecvByChannelOcs(const TemplateDataParams &tempAlgParams,
        const std::vector<ChannelInfo> &sendChannels, const std::vector<ChannelInfo> &recvChannels,
        u32 sendRank, u32 recvRank, u32 dstIdx, u32 srcIdx, const ThreadHandle &thread, u32 channelId,
        bool doSend, bool doRecv) const;

    /* 从 CCL Buffer 槽位拷贝到用户输出 buffer
     * srcIdx：CCL 槽位索引；recvRank：用于计算 output 偏移 */
    HcclResult PostCopyOcs(const TemplateDataParams &tempAlgParams, const ThreadHandle &thread,
        u32 srcIdx, u32 recvRank, u64 recvSize, u64 recvCount, u64 recvOffset) const;

    /* ---- 成员变量 ---- */

    /* OCS 分组数：rank 按 plane×group 二维分解，numGroups_ = group 维度大小
     * 由 InitOcsParamsFromTopo 填充，默认 1（非 OCS 场景） */
    u32 numGroups_{1};

    /* 对齐后的最大并发 SendRecv 对数 = (min(16, N-1) / numGroups_) × numGroups_
     * 即每轮最多同时进行的 SendRecv 通信路数，也等于 CCL Buffer 槽位数 */
    u32 concurrentSendRecvNum_{1};

    /* 数据类型字节大小（如 FP32=4, FP16=2），用于偏移/大小计算 */
    u64 dataTypeSize_{0};

    /* 以下 6 个数组由 CalcDataSplitByPortGroup 填充，按 channel（Port Group）分片：
     * sendCountsSplit_[ch] = 第 ch 个 channel 的发送 count（元素数）
     * sendSizeSplit_[ch]   = 第 ch 个 channel 的发送字节数
     * sendOffsetSplit_[ch] = 第 ch 个 channel 的发送偏移（字节）
     * recvCountsSplit_[ch] = 第 ch 个 channel 的接收 count（元素数）
     * recvSizeSplit_[ch]   = 第 ch 个 channel 的接收字节数
     * recvOffsetSplit_[ch] = 第 ch 个 channel 的接收偏移（字节） */
    std::vector<u64> sendCountsSplit_;
    std::vector<u64> sendSizeSplit_;
    std::vector<u64> sendOffsetSplit_;
    std::vector<u64> recvCountsSplit_;
    std::vector<u64> recvSizeSplit_;
    std::vector<u64> recvOffsetSplit_;
};

}
#endif