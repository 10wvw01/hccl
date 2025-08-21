/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_DIRECT_FULLMESH_PUB_H
#define ALLTOALL_V_DIRECT_FULLMESH_PUB_H

#include "mc2_handler_pub.h"
#include "alg_template_register.h"

namespace hccl {
class AlltoAllVDirectFullMesh : public AlgTemplateBase {
public:
    explicit AlltoAllVDirectFullMesh(const HcclDispatcher dispatcher);
    ~AlltoAllVDirectFullMesh() override;
    HcclResult RunAsync() override;
    HcclResult Prepare(PrepareData &param) override;
    HcclResult GetNslbAdjInfo(const u32 rank, const u32 rankSize,
                              const std::vector<LINK> &links, AdjInfo& nslbAdjInfo) override;
protected:
private:
    HcclResult GenerateSubStreamInfo(const std::vector<Stream> &subStreams,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain);
    std::string GetStreamIndexString();
    u64 CalcMaxSendLen();
    HcclResult NotifySubStreamStart();
    HcclResult WaitSubStreamFinish();
    HcclResult NotifyLocalSubStreamStart();
    HcclResult WaitLocalSubStreamFinish();
    u32 CalcNumSubStep();
    HcclResult NotifyRemoteRankStart(u32 step);
    HcclResult SDMAwithRemoteRankAndNotifyEnd(u32 step);
    HcclResult SendRecvData(u32 step, u32 roundIdx);

    void UpdateCurrRankSendInfo(u32 roundIdx, u32 side, u32 destRank, std::vector<SendDataBlock>& sendInfo, u32 maxSendStep);
    void UpdateCurrRankRecvInfo(u32 roundIdx, u32 side, u32 destRank, std::vector<ReadDataBlock>& readInfo, u32 maxRecvStep);
    void UpdateOpBaseSubStreamInfo(u32 roundIdx);
    void UpdateRemoteRankSet(u32 roundIdx, u32 groupRankSize);
    void UpdatePartialCommunicationRankSetPairWise(u32 roundIdx, u32 groupRankSize);
    void UpdatePartialCommunicationRankSet(u32 roundIdx, u32 groupRankSize,
        std::vector<std::vector<std::pair<u32,u32>>> &partialCommRankSet);
    HcclResult PrepareIntraData(u32 step, std::unordered_map<u32, std::vector<SendDataBlock>> &subStreamSendInfo);
    void UpdateSendRecvInfo(u32 roundIdx,  std::unordered_map<u32, std::vector<ReadDataBlock>> &subStreamReadInfo,
        std::unordered_map<u32, std::vector<SendDataBlock>> &subStreamSendInfo,
        const std::vector<std::vector<std::pair<u32,u32>>> &partialCommRankSet);
    HcclResult LocalCopy();
    HcclResult RunGroupFullMeshAlltoall(u32 roundIdx, u32 step);
    HcclResult RunSDMA(HcclOpMetaInfoDef &opMeta);
    HcclResult RunSDMATasks(u32 roundIdx, u32 step, u32 groupRankSize, u32 leftRankSize);
    HcclResult RunSDMAFineGrained(u32 totalStep, HcclOpMetaInfoDef& opMeta);

    // RDMA处理相关函数
    HcclResult MainNotifyRdmaControlStart();
    HcclResult RdmaControlNotifyMainFinish();
    HcclResult RdmaControlNotifySubStart();
    HcclResult SubNotifyRdmaControlFinish();
    u32 GetNextDstRank(u32& curDstRank);
    u32 GetPreSrcRank(u32& curDstRank);
    void GenRdmaSendInfo(u32 dstRank, std::vector<SendDataBlock>& sendInfo);
    void GenRdmaRecvInfo(u32 srcRank, std::vector<RecvDataBlock>& recvInfo);
    HcclResult CopyDataForSend(u32 dstRank, std::vector<SendDataBlock>& sendInfo, u32 curStep, Stream strem);
    HcclResult SendRecvRdmaData(u32 dstRank, u32 srcRank, std::vector<SendDataBlock>& sendInfo,
        std::vector<RecvDataBlock>& recvInfo, u32 curStep, Stream strem);
    HcclResult CopyRecvDataToOutput(u32 srcRank, std::vector<RecvDataBlock>& recvInfo,
        u32 curStep, Stream strem);
    HcclResult ProcessSingleGroupRdmaData(std::vector<u32>& dstRanks, std::vector<u32>& srcRanks);
    HcclResult ProcessRdmaData();
    HcclResult RunRDMA();

    Stream mainStream_;
    u32 userRank_;
    u32 userRankSize_;
    u32 podStartRank_;  // 表示一个pod内起始的userRankId
    u32 podEndRank_; // 表示一个pod内结束的userRankId
    std::vector<LINK> links_;
    const SendRecvInfo* localSendRecvInfoPtr_;
    u32 podNum_;
    u32 devNumInlocalPod_;
    u32 rankIdxInPod_;
    u32 totalRdmaRankNum_; // 需要通信的rdma对端
    bool isSuPodAsym_;
    HcclCMDType opType_;
    bool isBigCount_;

    DeviceMem userInput_;
    DeviceMem userOutput_;
    DeviceMem cclInMem_;
    DeviceMem cclOutMem_;
    HcclWorkflowMode workMode_;
    u64 sdmaDataBlockSize_ = 0;

    bool islocalCpyDone_ = false;
    std::unordered_map<u32, std::vector<SendDataBlock>> subStreamSendInfo_; // 从流当前发送长度和发送的本地偏移
    std::unordered_map<u32, std::vector<ReadDataBlock>> subStreamReadInfo_; // 从流当前接收长度和接收到的本地偏移
    std::unordered_map<u32, std::vector<SendDataBlock>> nextSubStreamSendInfo_; // 下一轮从流发送长度和发送的本地偏移
    std::unordered_map<u32, std::vector<ReadDataBlock>> nextSubStreamReadInfo_; // 下一轮从流接收长度和接收到的本地偏移
    std::unordered_map<u32, u32> sendNumSubStep_;                       // 需要向对应对端rank发几次数据
    std::unordered_map<u32, u32> recvNumSubStep_;                       // 需要从对应对端rank收几次数据
    u32 sdmaConcurrentNum_; // 分组mesh-每组group的ranksize
    std::vector<std::vector<std::pair<u32,u32>>> partialCommRankSet_;  // 参与通信的rank组合, 第0、1、2个vector分别存放左、右、中的rank
    std::vector<std::vector<std::pair<u32,u32>>> nextPartialCommRankSet_;  // 下一轮参与通信的rank组合
    u64 commRounds_ = 0; // 每个rank分组fullmesh后需要通信的轮次

    // 本地拷贝处理相关
    std::vector<Stream> localSubStream_;
    std::vector<std::shared_ptr<LocalNotify>> localSignalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> localSignalSubToMain_;
    // SDMA处理相关
    std::vector<Stream> sdmaSubStream_;
    std::vector<std::shared_ptr<LocalNotify>> sdmaMeshSignalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> sdmaMeshSignalSubToMain_;
    // RDMA处理相关
    u64 rdmaDataBlockSize_ = 0;
    // RDMA并发数量
    u32 rdmaConcurrentNum_;
    std::shared_ptr<LocalNotify> main2RdmaControlStreamNotify_;
    std::shared_ptr<LocalNotify> rdmaControl2MainStreamNotify_;
    // RDMA从流，以及RDMA控制流与从流同步的notify
    std::vector<Stream> rdmaSubStreams_;
    std::vector<std::shared_ptr<LocalNotify>> rdmaControl2SubNotifies_;
    std::vector<std::shared_ptr<LocalNotify>> rdmaSub2ControlNotifies_;
    Mc2HandlerPub mc2HandlerPub;
};
} // namespace hccl
#endif /* ALLTOALL_V_MESH_READ_ONLY_PUB_H */