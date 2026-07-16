/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_OPS_EXECUTOR_H
#define HCCL_OPS_EXECUTOR_H

#include <map>
#include <memory>
#include <vector>
#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <hccl/hccl_res.h>
#include "hccl_algorithm.h"
#include "template/base_template.h"
#include "template/template_factory.h"
#include "utils/utils.h"

namespace ops_hccl {

class BaseEngine;

struct BufferInfo {
    void *ptr = nullptr;
    u64 size = 0;
    BufferType bufferType = BufferType::HCCL_BUFFER;
};

struct ExecDataInfo {
    void *inputPtr = nullptr;
    u64 inputSize = 0;
    void *outputPtr = nullptr;
    u64 outputSize = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_RESERVED;
};

struct AlgoExecDataDesc {
    u64 dataOffset{0};
    u64 dataStride{0};
    u64 sliceCount{0};
    u64 sliceOffset{0};
    u64 scratchSize{0}; // 输出参数
    u64 scratchStride{0};
    u64 tailCount{0};
    std::vector<u32> ranksForInputData;
    std::vector<u32> ranksForOutputData; // 输出参数
    BufferType inputBufferType{BufferType::INPUT};
    BufferType outputBufferType{BufferType::OUTPUT};
    BufferType cclBufferType{BufferType::HCCL_BUFFER};
};

class OpsExecutor {
public:
    OpsExecutor(HcclAlgorithm &algo, OpParam &param);
    ~OpsExecutor();
    // 供Host侧调用
    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo);
    // 供Host侧调用，使用之前必须先调用CalcAlgHierarchyInfo
    HcclResult CalcRes(HcclComm comm, AlgResourceRequest &resReq);
    // 供Kernel侧调用，使用之前必须先将计算好的algHierarchyInfo和thread信息填到resCtx
    HcclResult Orchestrate(AlgResourceCtxSerializable &resCtx);

private:
    HcclResult GetRes(AlgResourceRequest &resReq);
    // 调用template CalcRes获取Channel资源，Host侧才可以使用,需要知道comm
    HcclResult CalcChannelResRecursion(HcclComm comm, AlgoExecDesc &algoExecDesc);
    // 调用template CalcRes获取Channel资源，Host侧和Kernel都可以使用，不需要知道comm
    HcclResult GetResRecursion(AlgoExecDesc &algoExecDesc, u32 &subCommMask);
    HcclResult CalcTemplateChannelRes(HcclComm comm, const TemplateExecDesc &templateExeDes);
    HcclResult GetTemplateRes(const TemplateExecDesc &templateExeDes);
    HcclResult OrchestrateLoop(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc);
    HcclResult GenTemplateRes(const u32 subCommIndex, TemplateResource &templateResource);
    inline void GenTemplateDataParams(AlgoExecDataDesc &algoExecDataDesc, TemplateDataParams &templateDataParams);
    inline void UpdateSubCommMaskMap(AlgoExecDesc &algoExecDesc, const u32 subCommMask);
    HcclResult PreSyncBySubCommMask(const AlgoExecDesc &execDesc);
    HcclResult PostSyncBySubCommMask(const AlgoExecDesc &execDesc);
    inline void InitAlgoExecDataDesc(
        AlgoExecDataDesc &algoExecDataDesc, u64 dataOffset, u64 dataCount, u64 tailCount, u64 dataStride);
    inline void UpdateDataSplitParallel(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc, u32 childrenId,
        std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc);
    inline void UpdateDataSplitSequence(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc, u32 childrenId,
        std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc);
    inline void MergeChildrenOutput(const AlgoExecDesc &algoExecDesc,
        const std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc, AlgoExecDataDesc &algoExecDataDesc);
    HcclResult RunTemplateDesc(TemplateExecDesc *templateExeDes, AlgoExecDataDesc &algoExecDataDesc);
    HcclResult InitRes(const AlgResourceCtxSerializable &resCtx);
    std::vector<std::map<u32, std::vector<ChannelInfo>>> RestoreChannelMap(const AlgResourceCtxSerializable &resCtx);
    u64 GetMaxProcCntPerLoop(u64 dataCount);

    // 引擎指针，由外部通过 SetEngine 注入
    BaseEngine *engine_ = nullptr;
    // algo
    HcclAlgorithm algo_;

    // rankInfo
    u32 myRank_ = INVALID_VALUE_RANKID;
    u32 rankSize_ = 0;
    u32 root_ = INVALID_VALUE_RANKID;
    // dataInfo
    ExecDataInfo dataInfo_;
    u64 dataTypeSize_ = 0;
    u32 scratchMultiple_ = 0;
    // config
    OpMode opMode_;

    // 拓扑分级信息
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;

    // 资源信息
    // [Buffer资源]
    BufferInfo cclBufferInfo_;
    // [线程资源]
    ThreadHandle mainThread_ = 0;
    std::vector<ThreadHandle> threads_;
    std::vector<std::vector<ThreadHandle>> subThreads_;
    // [Notify资源]
    std::vector<u32> notifyNumOnSubMainThread_;
    // [Channel资源]
    std::vector<std::map<u32, std::vector<ChannelInfo>>> channelTable_;
    std::vector<std::vector<HcclChannelDesc>> requestChannels_;

    std::vector<u32> maxSlaveThreadNum_;
    std::vector<u32> maxNotifyNumOnMainThread_;
    std::vector<u32> maxNotifyNumPerThread_;

    // 递归后用于保存算法执行所需要的流同步信息
    std::map<const AlgoExecDesc *, u32> execDescSubCommMaskMap_;
};

} // namespace ops_hccl

#endif