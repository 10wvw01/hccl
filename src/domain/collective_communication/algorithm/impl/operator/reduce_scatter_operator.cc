/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_operator.h"
#include "device_capacity.h"
#include "rank_consistentcy_checker.h"
#include "executor_impl.h"
#include "coll_alg_utils.h"
#include "stream_active_manager.h"
#include "hccl_aiv.h"
#include "coll_alg_op_registry.h"

constexpr u32 MODULE_NUM_FOUR = 4;
constexpr u32 HCCL_310P_DATA_SIZE_MID_COUNT = 320 * 1024;
constexpr u32 HCCL_310P_DATA_SIZE_SMALL_COUNT = 1024;
constexpr u32 HCCL_310P_SLIM_RING_MAX_SIZE = 8;

namespace hccl {
ReduceScatterOperator::ReduceScatterOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher) :
    CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE_SCATTER)
{
}

ReduceScatterOperator::~ReduceScatterOperator()
{
}

HcclResult ReduceScatterOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
    std::string& newTag)
{
    if (userRankSize_ == 1) {
        algName = "ReduceScatterSingleExecutor";
        return HCCL_SUCCESS;
    }
    HcclResult ret;
    if (isDiffDeviceType_) {
        ret = SelectAlgforMix(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        ret = SelectAlgfor310P3(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    }  else {
        HCCL_ERROR("[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterSelector][SelectAlg]tag[%s], reduce_scatter fsailed, retrun[%d]",
            tag.c_str(), ret), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else {
        if (deviceType_ == DevType::DEV_TYPE_310P3) {
            newTag = tag + algName;
        } else {
            auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType_.algoLevel1);
        CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
            algType_.algoLevel1), HCCL_E_INTERNAL);
            newTag = tag + level1Iter->second + algName;
        }

        bool isInlineReduce = IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
            cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType);
        bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);
        const std::string REDUCE_SCATTER_NO_INLINE = "_no_inline";
        newTag = (isInlineReduce && isRdmaReduce) ? newTag : newTag + REDUCE_SCATTER_NO_INLINE;
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_INFO("[SelectAlg] reduce_scatter newTag is [%s]", newTag.c_str());

    if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
        HCCL_CONFIG_INFO(HCCL_ALG, 
            "[ReduceScatterOperator][SelectAlg]userRank_[%u], algName[%s] actual level1 algo[%d], level2 algo[%d]",
            userRank_, algName.c_str(), algType_.algoLevel1, algType_.algoLevel2);
    }

    return ret;
}

HcclResult ReduceScatterOperator::SelectAlgforMix(const OpParam& param, std::string& algName)
{

    if (gcdDeviceNumPerAggregation_ > 1) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING("[ReduceScatterOperator][SelectAlgforMix] only support NHR in AlgoLevel1 yet, "\
            "default is algType=NHR.");
        algName = "ReduceScatterMixExecutor";
    } else {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;;
        HCCL_WARNING("[ReduceScatterOperator][SelectAlgforMix] only support ring in AlgoComm yet, "\
            "default is algType=ring.");
        algName = "ReduceScatterComm";
    }

    HCCL_INFO("[SelectAlgforMix] reduce_scatter SelectAlgforMix is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    if(HCCL_310P_DATA_SIZE_SMALL_COUNT< param.DataDes.count &&param.DataDes.count <= HCCL_310P_DATA_SIZE_MID_COUNT && userRankSize_ <= HCCL_310P_SLIM_RING_MAX_SIZE){
        algName = "ReduceScatterSlimRing";
    }
    else {
        algName = "ReduceScatterRing";
    }

    HCCL_INFO("[SelectAlgfor310P3] reduce_scatter SelectAlgfor310P3 is algName [%s] DataDesCount [%llu]", algName.c_str(), param.DataDes.count);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "ReduceScatterMeshExecutor";
    } else if (isRingTopo) {
        algName = "ReduceScatterRingExecutor";
    } else {
        algName = "ReduceScatterComm";
    }
    HCCL_INFO("[SelectAlgfor910A] reduce_scatter SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    u64 cclBufferSize = cclBufferManager_.GetInCCLbufferSize() / userRankSize_;

    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize = 0;
    u64 commOutputSize = 0;

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
    bool isServNumPowOfTwo = (serverNum_ > 0) && ((serverNum_ & (serverNum_ - 1)) == 0);
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_STRICT) {
        if (!isMeshTopo || multiModuleDiffDeviceNumMode_) {
            // 保序规约场景（多batch一致），当前不支持A2标卡（ring拓扑场景）/ 非对称场景
            HCCL_ERROR("[SelectAlgfor910B] reduce order preservation only support MeshTopo(isMeshTopo:[%d]) "
                "and Symmetry(multiModuleDiffDeviceNumMode_[%d]).", isMeshTopo, multiModuleDiffDeviceNumMode_);
            return HCCL_E_NOT_SUPPORT;
        }
        if (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32
            || param.DataDes.dataType == HCCL_DATA_TYPE_BFP16) {
            // 只有浮点数存在多batch不一致的可能，整数天然一致
            if (param.aicpuUnfoldMode || topoMatcher_->GetAivModeConfig()) {
                // AIV / AICPU场景，规约保序优先级更高
                HCCL_WARNING("[SelectAlgfor910B]aicpuMode[%d], AivModeConfig[%d], "
                    "the AIV/AICPU mode does not support when the reduce order preservation is enabled.",
                    param.aicpuUnfoldMode, topoMatcher_->GetAivModeConfig());
            }
            algName = "ReduceScatterOrderPreservedExecutor";
            HCCL_INFO("[SelectAlgfor910B] ReduceScatterSelectAlgfor910B is algName [%s]", algName.c_str());
            return HCCL_SUCCESS;
        }
    }

    // 暂只支持单算子模式
    bool isCCLBufferGE16M = isOpbase &&
        (commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB);
    bool isSupportAivRdmaCount = !isSingleMeshAggregation_ && !multiModuleDiffDeviceNumMode_ && isMeshTopo &&
        (isServNumPowOfTwo || dataSize <= HCCL_SMALL_COUNT_128_KB) &&
        dataSize * userRankSize_ <= HCCL_MID_COUNT_16_MB && isCCLBufferGE16M && dataSize <= HCCL_SMALL_COUNT_256_KB;
    
    bool isSupportAivDeter = isSingleMeshAggregation_ && isOpbase &&
                            (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE); // 单机确定性场景
    isSupportAivDeter = isSupportAivDeter && (dataSize * userRankSize_ < HCCL_SMALL_COUNT_8_MB);
    bool isAivMode = topoMatcher_->GetAivModeConfig() && IsSupportAIVReduce(param.DataDes.dataType, param.reduceType) &&
        (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE || isSupportAivDeter )  &&
        ((isSingleMeshAggregation_ && dataSize <= AIV_BIG_SIZE) || isSupportAivRdmaCount);
    if (isAivMode) {
        if (isSupportAivDeter) {
            if (dataSize * userRankSize_ < HCCL_SMALL_COUNT_8_MB){
                algName = "ReduceScatterAivDeterSmallExecutor"; 
            }else{
                algName = "ReduceScatterAivDeterExecutor"; 
            }
            HCCL_INFO("[SelectAlgfor910B] all_reduce SelectAlgfor910B is algName [%s].", algName.c_str());
            return HCCL_SUCCESS;
        }
        if (isSupportAivRdmaCount) {
            algName = "ReduceScatterAivRdmaExecutor";
        } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && dataSize <= AIV_REDUCE_SCATTER_MID_SIZE) {
            algName = "ReduceScatterMeshAivSmallCountExecutor"; 
        } else {
            algName = "ReduceScatterMeshAivExecutor"; 
        }
        HCCL_INFO("[SelectAlgfor910BAIV] ReduceScatterSelectAlgfor910B is algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        bool isInlineReduce = IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
            cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType);
        bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);

        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, dataSize, cclBufferSize, algTypeLevel1Tag,
            isInlineReduce, isRdmaReduce));
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // AHC 算法选择逻辑
    if (((algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) ||
         (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE))) {
        CHK_RET(SelectAlgforAHC(dataSize, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER));
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到HD算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
            HCCL_WARNING("[ReduceScatterOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+"
                "graph[%u], reset algorithm to HD.", contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    if (isMeshTopo) {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            if (SingleMeshInlineReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
                cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType)) {
                if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE) {
                    algName = "ReduceScatterDeterExecutor";
                } else {
                    algName = "ReduceScatterMeshDmaEliminationExecutor";
                }
            } else if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE &&
                algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE &&
                IsMultiMeshInlineReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
                cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType)) {
                algName = "ReduceScatterMeshOpbasePipelineExecutor";
            }
        } else {
            if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
                if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE &&
                    deviceNumPerAggregation_ > DEVICE_TWO) {
                    algName = "ReduceScatterDeterExecutor";
                } else {
                    if (dataSize <= HCCL_SMALL_COUNT_1_MB &&
                        workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                        algName = "ReduceScatterMeshGraphExecutor";
                    } else {
                        algName = "ReduceScatterMeshExecutor";
                    }
                }
            }
        }
        if (algName.empty()) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && moduleNum_ > 1 &&
                deviceNumPerAggregation_ > 1 &&
                IsMultiMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType) &&
                (dataSize > HCCL_SMALL_COUNT_1_MB || moduleNum_ <= MODULE_NUM_FOUR ||
                    algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE)) {
                algName = "ReduceScatterMeshGraphPipelineExecutor";
            } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ||
                       dataSize > HCCL_SMALL_COUNT_1_MB) {
                algName = "ReduceScatterMeshExecutor";
            } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                algName = "ReduceScatterMeshGraphExecutor";
            }
        }
    } else if (isRingTopo) {
        algName = "ReduceScatterRingExecutor";
    } else {
        algName = "ReduceScatterComm";
    }
    HCCL_INFO("[SelectAlgfor910B] reduce_scatter SelectAlgfor910B is algName [%s], current mode is [%u]", algName.c_str(), workflowMode_);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    if (dataSize >= cclBufferManager_.GetInCCLbufferSize()) {
        HCCL_WARNING("The current inCCLbufferSize is [%llu] bytes, change the HCCL_BUFFSIZE environment variable "\
            "to be greater than the current data volume[%llu] bytes to improve the performance of the 91093 environment.",
            cclBufferManager_.GetInCCLbufferSize(), dataSize);
    }

    bool isOpbase = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    bool isAivCrossnodeMode = (superPodNum_ == 1 && serverNum_ > 1 && !GetExternalInputInterHccsDisable()) && (
                (userRankSize_ <= ONE_EIGHTH_MAX_BLOCK_DIM && dataSize <= AIV_REDUCE_SCATTER_A3_SMALL_RANKSIZE_ENTRY_SIZE) ||
                (userRankSize_ <= ONE_THIRD_MAX_BLOCK_DIM && dataSize <= AIV_REDUCE_SCATTER_A3_MID_RANKSIZE_ENTRY_SIZE) ||
                (dataSize <= AIV_REDUCE_SCATTER_A3_LARGE_RANKSIZE_ENTRY_SIZE));
    bool isAivMode = topoMatcher_->GetAivModeConfig() && IsSupportAIVReduce(param.DataDes.dataType, param.reduceType) &&
        ((serverNum_ == 1 && ((isOpbase && dataSize <= AIV_REDUCE_SCATTER_A3_ENTRY_SIZE) ||
        (!isOpbase && dataSize <= AIV_REDUCE_SCATTER_A3_GRAPH_ENTRY_SIZE))) || isAivCrossnodeMode) &&
        topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE;
    if (isAivMode) {
        if (isAivCrossnodeMode) {
            algName = "ReduceScatterMeshAivFor91093Executor"; 
        }
        else if ((isOpbase && dataSize <= AIV_REDUCE_SCATTER_MID_SIZE) 
            || (!isOpbase && dataSize <= AIV_A3_REDUCE_SCATTER_GRAPH_GUIYI_SIZE
            && userRankSize_ <= MAX_BLOCK_DIM / BLOCK_DIM_FOUR_PER_RANK_A3)) {
            algName = "ReduceScatterMeshAivSmallCountExecutor"; 
        } else {
            algName = "ReduceScatterMeshAivExecutor"; 
        }
        HCCL_INFO("[SelectAlgfor91093] reduce_scatter SelectAlgfor91093 is algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }

    // AHC 算法选择逻辑
    bool isAHCAlgo = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) || (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    if (isAHCAlgo) {
        CHK_RET(SelectAlgforAHC(dataSize, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER));
    }

    bool isSupportInlineReduce = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    bool isPowOfTwo = ((userRankSize_ - 1) & userRankSize_) == 0;
    void *commInputPtr = nullptr;
    u64 commInputSize = 0;
    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    bool dmaReduceLimit = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && isPowOfTwo &&
        ((commInputSize * HCCL_DEVICE_NUM_TWO < param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] * userRankSize_) ||
        retryEnable_);
    bool smallCountOptimSingleServer =
        (!retryEnable_) &&
        (serverNum_ == 1) &&
        ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) ||
        (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !param.aicpuUnfoldMode)) &&
        isSupportInlineReduce &&
        (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) &&
        (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_512_KB) &&
        !GetExternalInputInterHccsDisable() && !dmaReduceLimit;
    bool smallCountOptimMultiServer =
        isSupportInlineReduce &&
        (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) && (serverNum_ != 1) && (superPodNum_ == 1) &&
        !dmaReduceLimit && !GetExternalInputInterHccsDisable();
    bool isHccsPlusSio = userRankSize_ == 2 && pairLinkCounter_[static_cast<u32>(LinkTypeInServer::SIO_TYPE)] == 1;
    isHccsPlusSio = false;  // rts未适配，不启用

    if (isHccsPlusSio && param.supportZeroCopy) {
        algName = "AllGatherSioHccsExecutor";
    } else if (multiModuleDiffDeviceNumMode_) {
        algName = "ReduceScatterComm";
    } else if (smallCountOptimMultiServer && !isPowOfTwo &&
        (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_256_KB)) {
        algName = "ReduceScatterComm";
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
    } else if (smallCountOptimSingleServer ||
        (smallCountOptimMultiServer && isPowOfTwo &&
        (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] * serverNum_ <= HCCL_SMALL_COUNT_1_MB))) {
        algName = "ReduceScatterDeterExecutor";
    } else if (param.supportZeroCopy && isSupportInlineReduce) {    // 不申请scratch ==> 不支持非InlineReduce
        const u32 SEVER_NUM_FOUR = 4;
        if (serverNum_ < SEVER_NUM_FOUR || isAHCAlgo) {
            algName = "ReduceScatterRingZerocopyExecutor";      // 非连续数据通信（限制Server数，避免数据切太碎）
        } else {
            algName = "ReduceScatterRingZerocopyExchangeExecutor";      // 连续数据通信+数据交换（AHC不支持）
        }
    } else {
        if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
            algName = "ReduceScatterRingFor91093Executor";
        } else if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            if (IsSupportUnifiedMarch(param, topoType_, serverNum_, superPodNum_)) {
                algName = "ReduceScatterSemiRingExecutor";
            } else if (GetExternalInputEnableRdmaSdmaConcurrent() && !param.aicpuUnfoldMode &&
                (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
                if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)) {
                    algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
                    HCCL_WARNING("[ReduceScatterOperator][SelectAlgfor91093] env HCCL_CONCURRENT_ENABLE is set, "
                        "set interserver algo to ring.");
                }
                algName = "ReduceScatterDoubleRingConcurrentExecutor";
            } else {
                s32 HCCS_PORT_NUM_910_93_7 = 7;
                if (hccsPortNum_ == HCCS_PORT_NUM_910_93_7) {
                    algName = "ReduceScatterFastDoubleRingFor91093Executor";
                } else {
                    algName = "AlignedReduceScatterDoubleRingFor91093Executor";
                }
            }
        } else {
            algName = "ReduceScatterComm";
        }
    }

    if (GetExternalInputEnableRdmaSdmaConcurrent()) {
        if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)) {
                algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        }
    } else if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB || 
            (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING) ||
             algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||  algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE)
        && (algName != "ReduceScatterComm" && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD)) {
        // 910_93超节点只支持server间ring,NB和NHR，默认需继续使用NHR
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING("[ReduceScatterOperator][SelectAlgfor91093] only support ring, NB AHC and NHR in AlgoLevel1 yet, "\
            "default is algType=NHR.");
    }

    HCCL_INFO("[SelectAlgfor91093] reduce_scatter SelectAlgfor91093 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, ReduceScatter, ReduceScatterOperator);

}