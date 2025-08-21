/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_NATIVE_EXECUTOR_BASE_H
#define COLL_NATIVE_EXECUTOR_BASE_H

#include "coll_executor_base.h"
#include "device_capacity.h"
#include "dispatcher.h"
#include "stream_active_manager.h"
#include "comm_factory_pub.h"
#include "rank_consistentcy_checker.h"
#include "hccl_aiv.h"
#include "config_log.h"

namespace hccl {
constexpr u64 HCCL_INPLACE_MEMCOPY_SIZE = 131072; // 128K数据量 = 131072B数据量

struct ExecMem {
    u64 count = 0;
    DeviceMem inputMem;         /* 单算子模式时是InCCLMem, 图模式时是InUserMem */
    DeviceMem outputMem;        /* 单算子模式时是OutCCLMem, 图模式时是OutUserMem */
    DeviceMem scratchMem;
    void *inputPtr = nullptr;   /* InUserMem的地址，图模式时与inputMem的地址相同 */
    void *outputPtr = nullptr;  /* OutUserMem的地址，图模式时与outputMem的地址相同 */
};

class CollNativeExecutorBase : public CollExecutorBase {
public:
    CollNativeExecutorBase(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollNativeExecutorBase() = default;

    HcclResult CalcResRequest(const OpParam& param, AlgResourceRequest &resourceRequest) override;

protected:
    /* *************** 资源计算 *************** */
    virtual void ParseParam(const OpParam& param);
    virtual HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport);
    virtual HcclResult CalcLevel0CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport);
    virtual HcclResult CalcLevel1CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport); // 默认情况下可根据algType_支持NHR、NHRV1、NB、HD、Ring算法。
    virtual HcclResult CalcLevel2CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport);
    virtual HcclResult CalcStreamNum(u32& streamNum);
    virtual HcclResult CalcScratchMemSize(u64& scratchMemSize);
    virtual HcclResult CalcNotifyNum(u32 streamNum, u32 &notifyNum);
    virtual HcclResult CalcAivBufferRequest(u64 &aivBufferRequest);

    // 考虑新建一个资源计算类ResourceCalculator，将资源推导、资源解析的都放进去。
    // 推导通信域信息的公用函数，不同Executor的在计算Level0、Level1、Level2时使用。
    HcclResult CalcCommPlaneInfo(const std::string &tag, const CommParaInfo &commParaInfo,
        std::vector<SingleSubCommTransport> &commTransport, TransportMemType inPutMemType,
        TransportMemType outPutMemType);
    HcclResult BuildResourceRequest(u64 scratchMemSize, u32 streamNum, u32 notifyNum, u64 aivBufferRequest,
        std::vector<LevelNSubCommTransport>& opTransport, AlgResourceRequest& resourceRequest);
    HcclResult PrintTransportRequest(AlgResourceRequest& resourceRequest);
    /* *************** 算法编排 *************** */
    // 非零拷贝场景走KernelRun
    virtual HcclResult KernelRun(const OpParam &param, ExecMem &execMem);
    // 零拷贝场景走KernelRunIntraServerPre、KernelRunInterServer、KernelRunIntraServerPost
    virtual HcclResult KernelRunInterServer(const OpParam &param, ExecMem &execMem) {return HCCL_SUCCESS;}
    virtual HcclResult KernelRunIntraServerPre(const OpParam &param, ExecMem &execMem) {return HCCL_SUCCESS;}
    virtual HcclResult KernelRunIntraServerPost(const OpParam &param, ExecMem &execMem) {return HCCL_SUCCESS;}
    virtual HcclResult Getlevel1CommRank(SubCommInfo& level1CommInfo);
    virtual HcclResult SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize);
    virtual HcclResult GetDevNumInlocalPod(u32& devNumInlocalPod);

    // 图模式下激活从流
    HcclResult ActiveSlaveStreams(const Stream &stream);
    // 将从流添加至Profiling
    HcclResult AddSubStreamToProfiling();
    // 检查通信域大小
    HcclResult CheckCommSize(const CommPlane levelIndex, const u32 subLevelIndex);

    // 获取不同类型通信域中的 transport 信息
    // 为了避免循环调用时反复校验Range引发性能问题，此处不做Range校验，建议调用该接口前先调用CheckCommSize避免OutOfRange问题
    SubCommInfo GetSubCommInfo(const CommPlane levelIndex, const u32 subLevelIndex);

    HcclResult GetRankByUserRank(CommPlane levelIndex, u32 subLevelIndex, u32 userRank, u32 &rank);
    HcclResult GetUserRankByRank(CommPlane levelIndex, u32 subLevelIndex, u32 rank, u32 &userRank);
    HcclResult SendRecvSignalOnLinks(OpParam &param, ExecMem &execMem, std::vector<LINK> links);
    bool OpSyncCheckCommSize(const CommPlane levelIndex, const u32 expectedSize);
    HcclResult InplaceOpSync(OpParam &param, ExecMem &execMem);

    HcclResult CopyAivCommInfoToDevice(const CommPlane levelIndex, const u32 subLevelIndex,
        AlgResourceResponse& algResource);

    /* ---------------以下为 protected 成员变量定义领域-------------------------- */
    std::string tag_;
    u32 root_ = INVALID_VALUE_RANKID;
    AlgResourceResponse *algResResp_ = nullptr;
    HcclCMDType opType_ = HcclCMDType::HCCL_CMD_INVALID;

    // Infos got from topoMatcher_
    const HcclTopoInfo topoAttr_;
    const HcclAlgoInfo algoAttr_;
    TopoType topoType_;
    bool is310P3Common_ = false;
    bool aicpuUnfoldMode_ = false;
    HcclWorkflowMode workflowMode_;
};
}
#endif
