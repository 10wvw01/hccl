/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTER_RING_FOR_910_93_EXECUTOR_H
#define COLL_REDUCESCATTER_RING_FOR_910_93_EXECUTOR_H
#include "coll_reduce_scatter_executor.h"

namespace hccl {
class CollReduceScatterRingFor91093Executor : public CollReduceScatterExecutor {
public:
    explicit CollReduceScatterRingFor91093Executor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollReduceScatterRingFor91093Executor() = default;

protected:
    u64 CalcTotalCount(const OpParam &param) const;

private:
    void ParseParam(const OpParam& param) override;
    /* *************** 资源计算 *************** */
    HcclResult CalcScratchMemSize(u64& scratchMemSize) override;
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);

    /* *************** 算法编排 *************** */
    u64 CalcLoopMaxCount(const u32 unitSize) override;
    bool IsHugeData(const u64 curSize, OpParam *param = nullptr) override;
    bool IsDataSplitForRdmaSdmaConcurrent(const u64 curSize) override;
    virtual HcclResult RunIntraSeverReduceScatter(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        const u64 count, const HcclDataType &dataType, const HcclReduceOp &reductionOp,
        const std::vector<std::vector<Slice>> &multRingsSliceZero, const Stream &stream, s32 profStage,
        const u64 baseOffset = 0, const HcomCollOpInfo *opInfo = nullptr,
        const std::vector<std::vector<Slice>> &multRingsUserMemSlice = std::vector<std::vector<Slice>>(0),
        const bool disableDMAReduce = false);
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
    HcclResult Getlevel1CommRank(SubCommInfo& level1CommInfo) override;
    HcclResult SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize) override;
    virtual bool IsEnableRdmaSdmaConcurrent() const;
    virtual bool IsUnifiedMarch(const OpParam &param) const;
    HcomCollOpInfo GetHcomCollOpInfo(const OpParam &param, const ExecMem &execMem) const;
    u64 CalcSrcMemOffset(const ExecMem &execMem, const OpParam &param, u32 perDataSize) const;
    /* **************** 数据准备*************** */
    virtual void FillMultiRingSlice(const ExecMem &execMem, const std::vector<std::vector<Slice>> &multiStreamSlice,
        u32 sliceNum, u32 level1RankSize, u32 level2RankSize, const u32 ringIndex, std::vector<Slice> &dataSlice);
    virtual HcclResult CalLevel0DataSegsSlice(const ExecMem &execMem, std::vector<std::vector<Slice>> &multiStreamSlice,
        const OpParam &param, u32 ringNum, u32 sliceNum, u32 level1RankSize, u32 level2RankSize, HcclDataType dataType,
        std::vector<std::vector<Slice>> &level0DataSegsSlice);
    virtual HcclResult CalUserMemDataSegsSlice(const ExecMem &execMem,
        const std::vector<std::vector<Slice>> &level0DataSegsSlice,
        const std::vector<std::vector<Slice>> &multiStreamSlice, const OpParam &param, u32 ringNum, u32 sliceNum,
        u32 level1RankSize, u32 level2RankSize, HcclDataType dataType, u32 perDataSize, HcomCollOpInfo *opInfoPtr,
        bool disableDMAReduce, std::vector<std::vector<Slice>> &multRingsUserMemSlice);
    virtual HcclResult CalLevel1DataSegsSlice(const ExecMem &execMem, const OpParam &param, CommPlane commPlaneLevel,
        const u32 &commIndex, u32 sliceNum, u32 level1RankSize, u32 level2RankSize, u32 perDataSize,
        std::vector<Slice> &level1DataSegsSlice);
    virtual HcclResult CalLevel2DataSegsSlice(const ExecMem &execMem, const OpParam &param, u32 level2RankSize,
        u32 perDataSize, std::vector<Slice> &level2DataSegsSlice);

    using Level0SlicesCalculator = void(*)(const OpParam &param, u32 sliceNum, u32 level1RankSize, u32 level1Index,
        u32 level2Index, u32 perDataSize, std::vector<Slice> &segSlices);
    static void PrepareLevel0Slices(const OpParam &param, u32 sliceNum, u32 level1RankSize, u32 level1Index,
        u32 level2Index, u32 perDataSize, std::vector<Slice> &cclSegSlices);
    static void PrepareLevel0UserSlices(const OpParam &param, u32 sliceNum, u32 level1RankSize, u32 level1Index,
        u32 level2Index, u32 perDataSize, std::vector<Slice> &userSegSlices);
    bool IsCceReduceAligned(const std::vector<Slice> &dataSlices) const;
    HcclResult FillMultiRingSliceV(const ExecMem &execMem, const OpParam &param, u32 ringNum, u32 sliceNum,
        u32 level1RankSize, u32 level2RankSize, HcclDataType dataType,
        std::vector<std::vector<Slice>> &level0DataSegsSlice,
        std::vector<std::vector<std::vector<Slice>>> &serverSlices, const Level0SlicesCalculator &calcLevel0Slices);
    virtual HcclResult CalUserMemDataSegsSliceV(const ExecMem &execMem, const OpParam &param, u32 ringNum, u32 sliceNum,
        u32 level1RankSize, u32 level2RankSize, HcclDataType dataType,
        std::vector<std::vector<Slice>> &multRingsUserMemSlice);
    virtual HcclResult CalLevel0DataSegsSliceV(const ExecMem &execMem,
        std::vector<std::vector<Slice>> &multiStreamSlice, const OpParam &param, u32 ringNum, u32 sliceNum,
        u32 level1RankSize, u32 level2RankSize, HcclDataType dataType,
        std::vector<std::vector<Slice>> &level0DataSegsSlice);
    virtual HcclResult CalLevel1DataSegsSliceV(const OpParam &param, CommPlane commPlaneLevel, const u32 &commIndex,
        u32 sliceNum, u32 level1RankSize, u32 level2RankSize, u32 perDataSize, std::vector<Slice> &level1DataSegsSlice);
    virtual HcclResult CalLevel2DataSegsSliceV(const OpParam &param, u32 level2RankSize, u32 perDataSize,
        std::vector<Slice> &level2DataSegsSlice);
};

} // namespace hccl

#endif