HcclResult Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    HcclAlgorithm &alg) {
    engineSelector.Select() //选出执行引擎
    HcclAlgorithm alg = opSelector.SelectAicpuAlg() //根据算子、拓扑选算法（数据结构包含顶层调度逻辑Executor，及算法Template等信息）
}


SelectorStatus AutoSelectorBase::SelectAlgForEngine(OpExecuteConfig config,
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, 
    EngineType engineType,
    HcclAlgorithm &alg) const
{
    switch (config) {
    case OpExecuteConfig::CCU_MS:     return SelectCcuMsAlgo(topoInfo, opParam, configAlgMap, alg);
    case OpExecuteConfig::CCU_SCHED:  return SelectCcuScheduleAlgo(topoInfo, opParam, configAlgMap, alg);
    case OpExecuteConfig::AICPU_TS:   return SelectAicpuAlgo(topoInfo, opParam, configAlgMap, alg);
    case OpExecuteConfig::AIV:        return SelectAivAlgo(topoInfo, opParam, configAlgMap, alg);
    case OpExecuteConfig::AIV_ONLY:   return SelectAivAlgo(topoInfo, opParam, configAlgMap, alg);
    case OpExecuteConfig::HOSTCPU:    return SelectDPUAlgo(topoInfo, opParam, configAlgMap, alg);
    case OpExecuteConfig::HOSTCPU_TS: return SelectAicpuAlgo(topoInfo, opParam, configAlgMap, alg);
    default: return SelectorStatus::NOT_MATCH;
    }
}
