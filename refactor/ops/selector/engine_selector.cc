EngineType Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    EngineType &engineType) {
        //根据环境变量、机型、数据量选出合适的引擎
}

//TODO: 对OpParam进一步进行正交分解，Selector仅选择必要信息对象作为参数
SelectorStatus AutoSelectorBase::Select(OpParam &opParam, TopoInfoWithNetLayerDetails* topoInfo,
                                        EngineType &engineType) const
{
    HCCL_DEBUG("[AutoSelectorBase][%s] start, OpExecuteConfig is %d.", __func__, opParam.opExecuteConfig);
    std::map<HcclCMDType, std::vector<HcclAlgoType>> configAlgMap = GetExternalInputHcclAlgoConfigAllType();

    // 1. HostDPUOnly 优先判断
    bool hostDPUOnly = false;
    if ((CheckHostDPUOnly(opParam.hcclComm, topoInfo, hostDPUOnly) == HCCL_SUCCESS) && hostDPUOnly) {
        opParam.opExecuteConfig = OpExecuteConfig::HOSTCPU;
        opParam.engine = CommEngine::COMM_ENGINE_CPU;
        return SelectDPUAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
    }

    // 2. 获取引擎优先级列表，遍历查找第一个可用的引擎
    std::vector<EngineCandidate> priorityList = GetEnginePriorityList(
        opParam.opExecuteConfig, topoInfo, opParam.opType, opParam.hcclComm);

    for (const auto& candidate : priorityList) {
        opParam.opExecuteConfig = candidate.opExecuteConfig;
        opParam.engine = candidate.engine;

        if (!CanEngineWork(candidate.opExecuteConfig, topoInfo, opParam)) {
            continue;
        }
        if (candidate.isHardFail) {
            HCCL_ERROR("[Algo][AutoSelectorBase] Hard fail for OpExecuteConfig=%d, abort.",
                opParam.opExecuteConfig);
            return SelectorStatus::NOT_MATCH;
        }
    }

    HCCL_INFO("[Algo][AutoSelectorBase] The selected algo is %s, OpExecuteConfig is %d.",
        selectAlgName.c_str(), opParam.opExecuteConfig);
    return SelectorStatus::NOT_MATCH;
}

bool AutoSelectorBase::CanEngineWork(OpExecuteConfig config, const TopoInfoWithNetLayerDetails* topoInfo,
                                      const OpParam& opParam) const
{
    switch (config) {
    case OpExecuteConfig::CCU_MS:     return CanCcuMsWork(topoInfo, opParam);
    case OpExecuteConfig::CCU_SCHED:  return CanCcuScheduleWork(topoInfo, opParam);
    case OpExecuteConfig::AICPU_TS:   return CanAicpuWork(topoInfo, opParam);
    case OpExecuteConfig::AIV:        return CanAivWork(topoInfo, opParam);
    case OpExecuteConfig::AIV_ONLY:   return CanAivWork(topoInfo, opParam);
    case OpExecuteConfig::HOSTCPU:    return CanDPUWork(topoInfo, opParam);
    case OpExecuteConfig::HOSTCPU_TS: return CanAicpuWork(topoInfo, opParam);
    default: return false;
    }
}

// ===== GetEnginePriorityList: 根据展开模式生成引擎优先级列表 =====

std::vector<EngineCandidate> AutoSelectorBase::GetEnginePriorityList(
    const OpExecuteConfig expansionMode, const TopoInfoWithNetLayerDetails* topoInfo,
    const HcclCMDType opType, HcclComm comm)
{
    // Stars 特殊分支：PCIE-SW混合拓扑 + CLOS规模>8 + AlltoAll类算子 → AIV_ONLY 硬失败
    if (topoInfo != nullptr && topoInfo->level0PcieMix && topoInfo->level0BigClosRange &&
        (opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
         opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
         opType == HcclCMDType::HCCL_CMD_ALLTOALLVC)) {
        return {
            {OpExecuteConfig::AIV_ONLY, CommEngine::COMM_ENGINE_AIV, true}
        };
    }

    switch (expansionMode) {
    case OpExecuteConfig::CCU_MS:
        return {
            {OpExecuteConfig::CCU_MS, CommEngine::COMM_ENGINE_CCU, false},
            {OpExecuteConfig::CCU_SCHED, CommEngine::COMM_ENGINE_CCU, false},
            {OpExecuteConfig::AICPU_TS, CommEngine::COMM_ENGINE_AICPU_TS, false},
        };
    case OpExecuteConfig::CCU_SCHED:
        return {
            {OpExecuteConfig::CCU_SCHED, CommEngine::COMM_ENGINE_CCU, false},
            {OpExecuteConfig::AICPU_TS, CommEngine::COMM_ENGINE_AICPU_TS, false},
        };
    case OpExecuteConfig::AIV:
        return {
            {OpExecuteConfig::AIV, CommEngine::COMM_ENGINE_AIV, false},
            {OpExecuteConfig::AICPU_TS, CommEngine::COMM_ENGINE_AICPU_TS, false},
        };
    case OpExecuteConfig::AIV_ONLY:
        return {
            {OpExecuteConfig::AIV_ONLY, CommEngine::COMM_ENGINE_AIV, true},
        };
    case OpExecuteConfig::AICPU_TS:
    case OpExecuteConfig::HOSTCPU_TS:
    case OpExecuteConfig::CCU_FAIL:
        return {
            {OpExecuteConfig::AICPU_TS, CommEngine::COMM_ENGINE_AICPU_TS, false},
        };
    default:
        return {};
    }
}