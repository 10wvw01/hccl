/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#include "hcom_all_reduce_op.h"

#include "acl/acl_rt.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/shape.h"
#include "hccl/hccl_comm.h"
#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "log.h"

extern "C" {
HcclResult HcclCreateOpParamGraphMode(void **opParam);
HcclResult HcclDestroyOpParamGraphMode(void *opParam);
HcclResult HcclSetOpParamGraphModeOpType(void *opParam, const char *opType);
HcclResult HcclSetOpParamGraphModeDataCount(void *opParam, const uint64_t *dataCount);
HcclResult HcclSetOpParamGraphModeRankSize(void *opParam, const uint32_t *rankSize);
HcclResult HcclSetOpParamGraphModeHCCLBufferSize(void *opParam, const uint64_t *hcclBufferSize);
HcclResult HcclSetOpParamGraphModeDataType(void *opParam, HcclDataType dataType);
HcclResult HcclSetAivSelectOpParamGraphMode(void *opParam, uint32_t aivCoreLimit);
HcclResult HcclSelectAlgGraphMode(const char *group, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                                  HcclCMDType opType, uint32_t aivCoreLimit, bool *ifAiv, char *algName);
HcclResult HcclCalcOpResOfflineGraphMode(void *opParam, uint64_t *opMemSize, uint32_t *streamNum,
                                         uint32_t *taskNum, uint32_t *aivCoreNum);
HcclResult HcclAllReduceGraphMode(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                                  HcclReduceOp op, const char *group, aclrtStream stream, const char *tag,
                                  void **streams, size_t streamCount, void *scratchMemAddr,
                                  uint64_t scratchMemSize);
}

HcomAllReduceOp::~HcomAllReduceOp()
{
    if (indirectInCCLbuf_ != nullptr) {
        aclrtFree(indirectInCCLbuf_);
        indirectInCCLbuf_ = nullptr;
    }
    if (indirectOutCCLbuf_ != nullptr) {
        aclrtFree(indirectOutCCLbuf_);
        indirectOutCCLbuf_ = nullptr;
    }
}

ge::graphStatus HcomAllReduceOp::ExtractParams(hccl::HcclOpState &st)
{
    const gert::Tensor *firstInput = st.ctx->GetInputTensor(INPUT_INDEX);
    if (firstInput == nullptr) {
        HCCL_ERROR("HcomAllReduceOp::ExtractParams: input tensor is null.");
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("HcomAllReduceOp::ExtractParams: attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const char *reduction = attrs->GetStr(ATTR_REDUCTION);
    const char *group = attrs->GetStr(ATTR_GROUP);
    const int64_t *fusion = attrs->GetInt(ATTR_FUSION);
    const int64_t *fusionId = attrs->GetInt(ATTR_FUSION_ID);

    st.dataType = hccl::GeDataTypeToHccl(firstInput->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR("HcomAllReduceOp::ExtractParams: unsupported data type %d.",
                   static_cast<int>(firstInput->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    st.reduceOp = hccl::StringToReduceOp(reduction);
    if (st.reduceOp == HCCL_REDUCE_RESERVED) {
        HCCL_ERROR("HcomAllReduceOp::ExtractParams: unsupported reduction '%s'.",
                   reduction ? reduction : "(null)");
        return ge::GRAPH_FAILED;
    }

    // 遍历所有 input（融合场景多输入），按 512B 对齐累加计算总 count
    // 对应 InfoStore CalcCountForAlignedOp: 每个 input 的 GetSize() 向上对齐到 512B 后累加，再除以 dataTypeSize
    constexpr uint64_t ALIGNED_SIZE = 512;
    uint32_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    uint64_t totalAlignedSize = 0;
    size_t inputCount = 0;

    for (size_t i = 0;; i++) {
        const gert::Tensor *inTensor = st.ctx->GetInputTensor(i);
        if (inTensor == nullptr) {
            break;
        }
        inputCount++;

        gert::Tensor *outTensor = st.ctx->MallocOutputTensor(i, inTensor->GetShape(), inTensor->GetFormat(),
                                                              inTensor->GetDataType());
        if (outTensor == nullptr) {
            HCCL_ERROR("HcomAllReduceOp::ExtractParams: failed to allocate output tensor %zu.", i);
            return ge::GRAPH_FAILED;
        }

        uint64_t tensorSize = static_cast<uint64_t>(inTensor->GetSize());
        uint64_t alignedSize = (tensorSize + ALIGNED_SIZE - 1) / ALIGNED_SIZE * ALIGNED_SIZE;
        totalAlignedSize += alignedSize;

        st.inputPtrs.push_back(const_cast<void *>(inTensor->GetAddr()));
        st.outputPtrs.push_back(outTensor->GetAddr());
    }

    if (inputCount == 0) {
        HCCL_ERROR("HcomAllReduceOp::ExtractParams: no input tensors.");
        return ge::GRAPH_FAILED;
    }

    st.count = totalAlignedSize / unitSize;
    st.group = group;
    st.stream = st.ctx->GetStream();

    if (inputCount > 1) {
        HCCL_WARNING("HcomAllReduceOp::ExtractParams: fusion scenario (%zu inputs) — contiguous buffer not yet handled, "
                     "using first input/output address only.", inputCount);
    }

    HCCL_INFO("HcomAllReduceOp::ExtractParams: inputCount=%zu input[0]=%p output[0]=%p count=%lu dataType=%d reduceOp=%d "
              "group=%s stream=%p fusion=%ld fusionId=%ld.",
              inputCount, st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.reduceOp,
              st.group ? st.group : "(null)", st.stream, fusion ? *fusion : -1,
              fusionId ? *fusionId : -1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::CalcResources(hccl::HcclOpState &st)
{
    uint32_t rankSize = 0;
    HcclResult ret = HcclGetRankSize(st.comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CalcResources: HcclGetRankSize failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    uint64_t cclBuffSize = 0;
    ret = HcomGetCommCCLBufferSize(st.group, cclBuffSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CalcResources: HcomGetCommCCLBufferSize failed for group '%s', ret=%d.",
                   st.group, static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    void *opParam = nullptr;
    ret = HcclCreateOpParamGraphMode(&opParam);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CalcResources: HcclCreateOpParamGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HcclSetAivSelectOpParamGraphMode(opParam, 0);
    HcclSetOpParamGraphModeOpType(opParam, HCCL_KERNEL_OP_TYPE_ALLREDUCE.c_str());
    HcclSetOpParamGraphModeDataType(opParam, st.dataType);
    HcclSetOpParamGraphModeRankSize(opParam, &rankSize);
    HcclSetOpParamGraphModeDataCount(opParam, &st.count);
    HcclSetOpParamGraphModeHCCLBufferSize(opParam, &cclBuffSize);

    uint64_t opMemSize = 0;
    uint32_t taskNum = 0;
    uint32_t aivCoreNum = 0;
    ret = HcclCalcOpResOfflineGraphMode(opParam, &opMemSize, &st.streamNum, &taskNum, &aivCoreNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CalcResources: HcclCalcOpResOfflineGraphMode failed, ret=%d.", static_cast<int>(ret));
        HcclDestroyOpParamGraphMode(opParam);
        return ge::GRAPH_FAILED;
    }

    char algName[ALG_NAME_MAX_LEN] = {0};
    ret = HcclSelectAlgGraphMode(st.group, st.count, st.dataType, st.reduceOp, HCCL_CMD_ALLREDUCE, 0, &st.ifAiv, algName);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CalcResources: HcclSelectAlgGraphMode failed, ret=%d.", static_cast<int>(ret));
        HcclDestroyOpParamGraphMode(opParam);
        return ge::GRAPH_FAILED;
    }

    if (st.ifAiv) {
        constexpr uint32_t AIV_WORKSPACE_MEM_SIZE = 512;
        constexpr uint32_t AIV_TASK_NUM = 3;
        st.streamNum = 0;
        opMemSize = AIV_WORKSPACE_MEM_SIZE;
        taskNum = AIV_TASK_NUM;
    }

    HcclDestroyOpParamGraphMode(opParam);

    st.scratchMemSize = opMemSize;
    st.cclBuffSize = cclBuffSize;

    HCCL_INFO("HcomAllReduceOp::CalcResources: rankSize=%u scratchMemSize=%lu streamNum=%u taskNum=%u ifAiv=%d "
              "cclBuffSize=%lu.",
              rankSize, st.scratchMemSize, st.streamNum, taskNum, st.ifAiv, st.cclBuffSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::LaunchHcclOp(hccl::HcclOpState &st)
{
    if (st.scratchMemSize > 0) {
        st.scratchMem = st.ctx->MallocWorkSpace(st.scratchMemSize);
        if (st.scratchMem == nullptr) {
            HCCL_ERROR("HcomAllReduceOp::LaunchHcclOp: MallocWorkSpace failed, size=%lu.", st.scratchMemSize);
            return ge::GRAPH_FAILED;
        }
    }

    // TODO: attached 子流 — GE 根据 ATTR_NAME_ATTACHED_STREAM_INFO_LIST 创建后通过 task.rt_attached_streams 传回，
    //       调 HcomSetAttachedStream 设置到通信域
    // TODO: used_stream_num 子流 — GE 根据属性创建后通过 hcclInfo.hcclStreamList 传回，
    //       传给 HcclAllReduceGraphMode 的 streams[] 参数

    // TODO: 解析 needRefresh（对应 InfoStore task.needRefresh，
    //       由 IsFeatureBaseRefreshable + IsStaticAddrFixed + is_refresh_addr_op_ 决定）
    st.needRefresh = false;

    if (!st.needRefresh) {
        return LaunchDirect(st);
    }
    return LaunchLoop(st);
}

ge::graphStatus HcomAllReduceOp::LaunchDirect(hccl::HcclOpState &st)
{
    HCCL_INFO("HcomAllReduceOp::LaunchDirect: input=%p output=%p count=%lu (non-refresh, user buffers directly).",
              st.inputPtrs[0], st.outputPtrs[0], st.count);

    HCCL_GE_CHK_RET(CleanCracks(st.inputPtrs[0], 0));

    HcclResult ret = HcclAllReduceGraphMode(
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.reduceOp, st.group,
        static_cast<aclrtStream>(st.stream), HCCL_KERNEL_OP_TYPE_ALLREDUCE.c_str(),
        nullptr, 0, st.scratchMem, st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::LaunchDirect: HcclAllReduceGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::LaunchLoop(hccl::HcclOpState &st)
{
    HCCL_GE_CHK_RET(CreateIndirectCCLbuf());

    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    uint64_t maxCountPerLoop = st.cclBuffSize / unitSize;
    uint64_t curCount = 0;

    // 数据量 <= CCL buffer: 单轮，二级指针无偏移拷贝
    // 数据量 > CCL buffer: 多 loop，二级指针有偏移拷贝
    bool secAddrCopyWithoutOffset = (st.count * unitSize <= st.cclBuffSize);
    HCCL_INFO("HcomAllReduceOp::LaunchLoop: secAddrCopyWithoutOffset=%d.", secAddrCopyWithoutOffset);

    for (uint64_t countLeft = st.count, inputOffset = 0, outputOffset = 0, loopTime = 0;
         countLeft > 0; countLeft -= curCount) {
        curCount = (countLeft * unitSize > st.cclBuffSize) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * unitSize;

        HCCL_INFO("HcomAllReduceOp::LaunchLoop: loop=%lu inputOffset=%lu countLeft=%lu curCount=%lu curSize=%lu.",
                  loopTime, inputOffset, countLeft, curCount, curSize);

        HCCL_GE_CHK_RET(RefreshInputAddr(st, inputOffset, curSize, secAddrCopyWithoutOffset));

        void *commInputPtr = nullptr;
        u64 commInputSize = 0;
        HcclResult ret = HcomGetInCCLbuffer(st.group, &commInputPtr, &commInputSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HcomAllReduceOp::LaunchLoop: HcomGetInCCLbuffer failed, ret=%d.", static_cast<int>(ret));
            return ge::GRAPH_FAILED;
        }
        void *commOutputPtr = nullptr;
        u64 commOutputSize = 0;
        ret = HcomGetOutCCLbuffer(st.group, &commOutputPtr, &commOutputSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HcomAllReduceOp::LaunchLoop: HcomGetOutCCLbuffer failed, ret=%d.", static_cast<int>(ret));
            return ge::GRAPH_FAILED;
        }

        if (curCount == countLeft) {
            HCCL_GE_CHK_RET(CleanCracks(commInputPtr, inputOffset));
        }

        ret = HcclAllReduceGraphMode(
            commInputPtr, commOutputPtr, curCount, st.dataType, st.reduceOp, st.group,
            static_cast<aclrtStream>(st.stream), HCCL_KERNEL_OP_TYPE_ALLREDUCE.c_str(),
            nullptr, 0, st.scratchMem, st.scratchMemSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HcomAllReduceOp::LaunchLoop: HcclAllReduceGraphMode failed, ret=%d.", static_cast<int>(ret));
            return ge::GRAPH_FAILED;
        }

        HCCL_GE_CHK_RET(RefreshOutputAddr(st, outputOffset, curSize, secAddrCopyWithoutOffset));

        inputOffset += curSize;
        outputOffset += curSize;
        loopTime++;
    }

    HCCL_INFO("HcomAllReduceOp::LaunchLoop: success, input=%p output=%p count=%lu scratchMem=%p scratchMemSize=%lu.",
              st.inputPtrs[0], st.outputPtrs[0], st.count, st.scratchMem, st.scratchMemSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::CreateIndirectCCLbuf()
{
    if (indirectBufInited_) {
        return ge::GRAPH_SUCCESS;
    }
    aclError aclRet = aclrtMalloc(&indirectInCCLbuf_, sizeof(uintptr_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CreateIndirectCCLbuf: aclrtMalloc indirectIn failed, ret=%d.",
                   static_cast<int>(aclRet));
        return ge::GRAPH_FAILED;
    }
    aclRet = aclrtMalloc(&indirectOutCCLbuf_, sizeof(uintptr_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::CreateIndirectCCLbuf: aclrtMalloc indirectOut failed, ret=%d.",
                   static_cast<int>(aclRet));
        aclrtFree(indirectInCCLbuf_);
        indirectInCCLbuf_ = nullptr;
        return ge::GRAPH_FAILED;
    }
    indirectBufInited_ = true;
    HCCL_INFO("HcomAllReduceOp::CreateIndirectCCLbuf: indirectIn=%p indirectOut=%p.",
              indirectInCCLbuf_, indirectOutCCLbuf_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::CleanCracks(void *baseAddr, uint64_t inputOffset)
{
    // TODO: 按照 InfoStore CleanIntervalMemoryOpKernel 逻辑实现清缝
    // 当前打桩：检测到多输入融合场景时打印警告
    HCCL_DEBUG("HcomAllReduceOp::CleanCracks: stub, baseAddr=%p inputOffset=%lu.", baseAddr, inputOffset);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::RefreshInputAddr(hccl::HcclOpState &st, uint64_t inputOffset, uint64_t curSize,
                                                   bool secAddrCopyWithoutOffset)
{
    void *commInputPtr = nullptr;
    u64 commInputSize = 0;
    HcclResult ret = HcomGetInCCLbuffer(st.group, &commInputPtr, &commInputSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::RefreshInputAddr: HcomGetInCCLbuffer failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    if (commInputPtr == nullptr || commInputSize == 0) {
        ret = HcomCreateCommCCLbuffer(st.group);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HcomAllReduceOp::RefreshInputAddr: HcomCreateCommCCLbuffer failed, ret=%d.",
                       static_cast<int>(ret));
            return ge::GRAPH_FAILED;
        }
        ret = HcomGetInCCLbuffer(st.group, &commInputPtr, &commInputSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HcomAllReduceOp::RefreshInputAddr: HcomGetInCCLbuffer retry failed, ret=%d.",
                       static_cast<int>(ret));
            return ge::GRAPH_FAILED;
        }
    }
    if (curSize > commInputSize) {
        HCCL_ERROR("HcomAllReduceOp::RefreshInputAddr: curSize=%lu > commInputSize=%llu.", curSize, commInputSize);
        return ge::GRAPH_FAILED;
    }

    // H2D 同步: 把 CCL buffer 地址写入 indirect buffer
    aclError aclRet = aclrtMemcpy(indirectInCCLbuf_, sizeof(void *), &commInputPtr, sizeof(void *),
                                  ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::RefreshInputAddr: aclrtMemcpy H2D failed, ret=%d.", static_cast<int>(aclRet));
        return ge::GRAPH_FAILED;
    }

    // D2D 异步: 用户数据 → CCL buffer
    // secAddrCopyWithoutOffset==true: 无偏移（单轮，inputOffset 恒为 0）
    // secAddrCopyWithoutOffset==false: 有偏移（多 loop）
    void *src = secAddrCopyWithoutOffset ? st.inputPtrs[0]
                                         : static_cast<char *>(st.inputPtrs[0]) + inputOffset;
    aclRet = aclrtMemcpyAsync(commInputPtr, commInputSize, src, curSize,
                              ACL_MEMCPY_DEVICE_TO_DEVICE, static_cast<aclrtStream>(st.stream));
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::RefreshInputAddr: aclrtMemcpyAsync D2D failed, ret=%d.", static_cast<int>(aclRet));
        return ge::GRAPH_FAILED;
    }

    HCCL_DEBUG("HcomAllReduceOp::RefreshInputAddr: commInputPtr=%p commInputSize=%llu inputOffset=%lu curSize=%lu "
               "secAddrCopyWithoutOffset=%d.", commInputPtr, commInputSize, inputOffset, curSize,
               secAddrCopyWithoutOffset);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::RefreshOutputAddr(hccl::HcclOpState &st, uint64_t outputOffset, uint64_t curSize,
                                                    bool secAddrCopyWithoutOffset)
{
    void *commOutputPtr = nullptr;
    u64 commOutputSize = 0;
    HcclResult ret = HcomGetOutCCLbuffer(st.group, &commOutputPtr, &commOutputSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::RefreshOutputAddr: HcomGetOutCCLbuffer failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    if (commOutputPtr == nullptr || commOutputSize == 0) {
        ret = HcomCreateCommCCLbuffer(st.group);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HcomAllReduceOp::RefreshOutputAddr: HcomCreateCommCCLbuffer failed, ret=%d.",
                       static_cast<int>(ret));
            return ge::GRAPH_FAILED;
        }
        ret = HcomGetOutCCLbuffer(st.group, &commOutputPtr, &commOutputSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HcomAllReduceOp::RefreshOutputAddr: HcomGetOutCCLbuffer retry failed, ret=%d.",
                       static_cast<int>(ret));
            return ge::GRAPH_FAILED;
        }
    }
    if (curSize > commOutputSize) {
        HCCL_ERROR("HcomAllReduceOp::RefreshOutputAddr: curSize=%lu > commOutputSize=%llu.", curSize, commOutputSize);
        return ge::GRAPH_FAILED;
    }

    // H2D 同步: 把 CCL output buffer 地址写入 indirect output buffer
    aclError aclRet = aclrtMemcpy(indirectOutCCLbuf_, sizeof(void *), &commOutputPtr, sizeof(void *),
                                  ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::RefreshOutputAddr: aclrtMemcpy H2D failed, ret=%d.", static_cast<int>(aclRet));
        return ge::GRAPH_FAILED;
    }

    // D2D 异步: CCL buffer → 用户输出
    // secAddrCopyWithoutOffset==true: 无偏移（单轮，outputOffset 恒为 0）
    // secAddrCopyWithoutOffset==false: 有偏移（多 loop）
    void *dst = secAddrCopyWithoutOffset ? st.outputPtrs[0]
                                         : static_cast<char *>(st.outputPtrs[0]) + outputOffset;
    aclRet = aclrtMemcpyAsync(dst, curSize, commOutputPtr, curSize,
                              ACL_MEMCPY_DEVICE_TO_DEVICE, static_cast<aclrtStream>(st.stream));
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("HcomAllReduceOp::RefreshOutputAddr: aclrtMemcpyAsync D2D failed, ret=%d.",
                   static_cast<int>(aclRet));
        return ge::GRAPH_FAILED;
    }

    HCCL_DEBUG("HcomAllReduceOp::RefreshOutputAddr: commOutputPtr=%p commOutputSize=%llu outputOffset=%lu curSize=%lu "
               "secAddrCopyWithoutOffset=%d.", commOutputPtr, commOutputSize, outputOffset, curSize,
               secAddrCopyWithoutOffset);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::InferShape(gert::InferShapeContext *ctx)
{
    const gert::Shape *inputShape = ctx->GetInputShape(INPUT_INDEX);
    if (inputShape == nullptr) {
        HCCL_ERROR("HcomAllReduceOp::InferShape: input shape is null.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape *outputShape = ctx->GetOutputShape(OUTPUT_INDEX);
    if (outputShape == nullptr) {
        HCCL_ERROR("HcomAllReduceOp::InferShape: output shape is null.");
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::InferDataType(gert::InferDataTypeContext *ctx)
{
    ge::DataType dataType = ctx->GetInputDataType(INPUT_INDEX);
    if (dataType == ge::DT_UNDEFINED) {
        HCCL_ERROR("HcomAllReduceOp::InferDataType: input data type is undefined.");
        return ge::GRAPH_FAILED;
    }
    return ctx->SetOutputDataType(OUTPUT_INDEX, dataType);
}

// TODO: 等 GE 提供支持指定 op type 字符串的注册宏后替换此手动注册
static ge::BaseCustomOp *CreateHcomAllReduceOp() { return new HcomAllReduceOp(); }
static const ge::CustomOpCreatorRegister g_hcom_all_reduce_register(
    HCCL_KERNEL_OP_TYPE_ALLREDUCE.c_str(), CreateHcomAllReduceOp);
