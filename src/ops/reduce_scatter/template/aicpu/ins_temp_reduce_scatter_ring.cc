/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "ins_temp_reduce_scatter_ring.h"
#include "alg_data_trans_wrapper.h"
#include "channel.h"
#include "alg_v2_template_register.h"
#include "hccl_common.h"

namespace ops_hccl {

InsTempReduceScatterRing::InsTempReduceScatterRing(const OpParam &param, const u32 rankId,
                                                   const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

// ============================================================
// CalcRes: 声明资源需求
// ============================================================
HcclResult InsTempReduceScatterRing::CalcRes(HcclComm comm, const OpParam &param,
                                             const TopoInfoWithNetLayerDetails *topoInfo,
                                             AlgResourceRequest &resourceRequest)
{
    u32 rankSize = templateRankSize_;
    HCCL_INFO("[ReduceScatterRing][CalcRes] rankSize=%u", rankSize);

    // 单线程执行 (ring 每轮依赖上一轮结果, 天然串行)
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    // 申请 Mesh 1D channel (可以从中取 ring 相邻 rank 的 channel)
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    HCCL_INFO("[ReduceScatterRing][CalcRes] channels=%zu", level0Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterRing::GetThreadNum() const
{
    return 1; // 单线程
}

u64 InsTempReduceScatterRing::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // 需要一个 scratch buffer: 大小 = 1 个 chunk (接收远端数据用)
    return 1;
}

// ============================================================
// KernelRun: 模板入口
// ============================================================
HcclResult InsTempReduceScatterRing::KernelRun(const OpParam &param,
                                               const TemplateDataParams &tempAlgParams,
                                               TemplateResource &templateResource)
{
    u32 rankSize = templateRankSize_;
    HCCL_INFO("[ReduceScatterRing] KernelRun start, rank=%u rankSize=%u", myRank_, rankSize);

    // 单 rank: 直接拷贝 input[0..count] → output
    if (rankSize == 1) {
        u64 dataSize = param.DataDes.count * DATATYPE_SIZE_TABLE[param.DataDes.dataType];
        DataSlice src(tempAlgParams.buffInfo.inputPtr, 0, dataSize, param.DataDes.count);
        DataSlice dst(tempAlgParams.buffInfo.outputPtr, 0, dataSize, param.DataDes.count);
        return LocalCopy(templateResource.threads[0], src, dst);
    }

    dataType_ = param.DataDes.dataType;
    threadNum_ = templateResource.threads.size();

    // 执行 Ring 循环
    CHK_RET(RunRingLoop(param, tempAlgParams, templateResource.threads, templateResource.channels));

    HCCL_INFO("[ReduceScatterRing] KernelRun End, rank=%u", myRank_);
    return HCCL_SUCCESS;
}

// ============================================================
// RunRingLoop: Ring ReduceScatter 核心 — N-1 轮 send/recv + reduce
// ============================================================
HcclResult InsTempReduceScatterRing::RunRingLoop(
    const OpParam &param, const TemplateDataParams &tempAlgParams,
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    u32 rankSize = templateRankSize_;
    u32 unitSize = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    u64 totalCount = param.DataDes.count;          // 每个 rank 的输入元素总数
    u64 chunkCount = totalCount / rankSize;         // 每个 chunk 的元素数
    u64 chunkBytes = chunkCount * unitSize;         // 每个 chunk 的字节数

    void *inputPtr  = tempAlgParams.buffInfo.inputPtr;
    void *outputPtr = tempAlgParams.buffInfo.outputPtr;
    HcclMem cclMem  = tempAlgParams.buffInfo.hcclBuff;

    // 计算本 rank 在环中的位置
    u32 algRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], algRank));

    // 环邻居
    u32 nextRank = subCommRanks_[0][(algRank + 1) % rankSize];
    u32 prevRank = subCommRanks_[0][(algRank - 1 + rankSize) % rankSize];

    CHK_PRT_RET(channels.count(nextRank) == 0 || channels.at(nextRank).empty() ||
                channels.count(prevRank) == 0 || channels.at(prevRank).empty(),
                HCCL_ERROR("[ReduceScatterRing] rank=%u missing channel to next=%u or prev=%u",
                           myRank_, nextRank, prevRank),
                HCCL_E_INTERNAL);

    const ChannelInfo &chanToNext = channels.at(nextRank)[0];
    const ChannelInfo &chanFromPrev = channels.at(prevRank)[0];

    ThreadHandle thread = threads[0];

    // N-1 轮 send/recv
    for (u32 step = 0; step < rankSize - 1; step++) {
        // 本轮要发送的 chunk 索引 (在 input 中的位置)
        u32 sendIdx = (algRank - step + rankSize) % rankSize;
        // 本轮要接收后归约到的 chunk 索引
        u32 recvIdx = (algRank - step - 1 + rankSize) % rankSize;

        HCCL_DEBUG("[ReduceScatterRing] rank=%u step=%u sendIdx=%u recvIdx=%u",
                   myRank_, step, sendIdx, recvIdx);

        // 发送: input[sendIdx * chunkCount] → next rank
        DataSlice sendSlice(inputPtr, sendIdx * chunkBytes, chunkBytes, chunkCount);

        // 接收: 写入 scratch buffer
        DataSlice recvSlice(cclMem.addr, 0, chunkBytes, chunkCount);

        SlicesList txSlices({sendSlice}, {});
        SlicesList rxSlices({recvSlice}, {});
        TxRxSlicesList srl(txSlices, rxSlices);
        TxRxChannels srChannels(chanToNext, chanFromPrev);
        SendRecvInfo srInfo(srChannels, srl);
        CHK_RET(SendRecvWrite(srInfo, thread));

        // 归约: 将 scratch 中的数据 reduce 到 input[recvIdx * chunkCount]
        DataSlice localSlice(inputPtr, recvIdx * chunkBytes, chunkBytes, chunkCount);
        CHK_RET(LocalReduce(thread, recvSlice, localSlice, param.DataDes.dataType, param.reduceType));
    }

    // 最终: 将完全归约的 chunk 拷贝到 output
    // rank i 最后拥有 chunk[(algRank + 1) % rankSize] (经历 N-1 轮后的 recvIdx)
    u32 resultIdx = (algRank + 1) % rankSize;
    DataSlice resultSlice(inputPtr, resultIdx * chunkBytes, chunkBytes, chunkCount);
    DataSlice outputSlice(outputPtr, 0, chunkBytes, chunkCount);
    CHK_RET(LocalCopy(thread, resultSlice, outputSlice));

    HCCL_INFO("[ReduceScatterRing] Loop End, rank=%u resultIdx=%u", myRank_, resultIdx);
    return HCCL_SUCCESS;
}

// ============================================================
// Notify 索引 (单线程, 无主从同步, 返回空)
// ============================================================
void InsTempReduceScatterRing::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
}

void InsTempReduceScatterRing::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
}

// 注册模板
REGISTER_TEMPLATE_V2("InsTempReduceScatterRing", InsTempReduceScatterRing);

}  // namespace ops_hccl
