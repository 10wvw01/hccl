HcclResult SelectAicpuAlg(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    std::string &algName) { 
        //根据拓扑、数据量选算法（数据结构包含顶层调度逻辑Executor，及算法Template等信息）
        if (topo == mesh) {
            new HCCLAlgorithm()
        }
}

HcclResult SelectCcuAlg(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    std::string &algName) { 
        //根据拓扑、数据量选算法（数据结构包含顶层调度逻辑Executor，及算法Template等信息）(单Die/2Die、CCU_MS/CCU_SCHE)
}

HcclResult SelectAivAlg(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    std::string &algName) { 
        //根据拓扑、数据量选算法（数据结构包含顶层调度逻辑Executor，及算法Template等信息）
}


bool AllGatherAutoSelector::CanCcuMsWork(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam) const
{
    (void)opParam;
    ConditionChain chain;
    chain.Add("topoLevelNums<=1", [&](const TopoInfoWithNetLayerDetails* t, const OpParam& o) -> bool {
        (void)o;
        return t->topoLevelNums <= 1;
    });
    return chain.Evaluate(topoInfo, opParam);
}

bool AllGatherAutoSelector::CanCcuScheduleWork(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam) const
{
    ConditionChain chain;
    chain.Add("!IsInputOutputOverlap", [&](const TopoInfoWithNetLayerDetails* t, const OpParam& o) -> bool {
        (void)t;
        return !IsInputOutputOverlap(o);
    });
    chain.Add("level0Topo supported", [&](const TopoInfoWithNetLayerDetails* t, const OpParam& o) -> bool {
        (void)o;
        if (t->topoLevelNums > 1) {
            return t->level0Topo == Level0Shape::MESH_1D || t->level0Topo == Level0Shape::CLOS;
        }
        if (t->topoLevelNums <= 1) {
            return t->level0Topo == Level0Shape::MESH_1D ||
                   t->level0Topo == Level0Shape::MESH_1D_CLOS ||
                   t->level0Topo == Level0Shape::CLOS;
        }
        return false;
    });
    return chain.Evaluate(topoInfo, opParam);
}

bool AllGatherAutoSelector::CanAicpuWork(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam) const
{
    (void)topoInfo;
    (void)opParam;
    return true;
}

bool AllGatherAutoSelector::CanAivWork(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam) const
{
    ConditionChain chain;
    chain.Add("userRankSize<=MAX_RANK_SIZE", [&](const TopoInfoWithNetLayerDetails* t, const OpParam& o) -> bool {
        (void)o;
        return t->userRankSize <= MAX_RANK_SIZE;
    });
    chain.Add("totalSize<=cclBufferSize*AIV_MAX_CCL_LOOP_NUM", [&](const TopoInfoWithNetLayerDetails* t, const OpParam& o) -> bool {
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        if (HcclGetHcclBuffer(o.hcclComm, &cclBufferAddr, &cclBufferSize) != HCCL_SUCCESS) {
            return false;
        }
        u64 perDataSize = DATATYPE_SIZE_TABLE[o.DataDes.dataType];
        u64 totalSize = o.DataDes.count * perDataSize * t->userRankSize;
        return totalSize <= cclBufferSize * AIV_MAX_CCL_LOOP_NUM;
    });
    return chain.Evaluate(topoInfo, opParam);
}

bool AllGatherAutoSelector::CanDPUWork(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam) const
{
    (void)opParam;
    ConditionChain chain;
    chain.Add("topoLevelNums>1", [&](const TopoInfoWithNetLayerDetails* t, const OpParam& o) -> bool {
        (void)o;
        return t->topoLevelNums > 1;
    });
    chain.Add("localNetInsSizeOfLayer[0]==1 || MESH_1D || MESH_1D_CLOS", [&](const TopoInfoWithNetLayerDetails* t, const OpParam& o) -> bool {
        (void)o;
        return t->netLayerDetails.localNetInsSizeOfLayer[0] == 1 ||
               t->level0Topo == Level0Shape::MESH_1D ||
               t->level0Topo == Level0Shape::MESH_1D_CLOS;
    });
    return chain.Evaluate(topoInfo, opParam);
}

SelectorStatus AllGatherAutoSelector::SelectCcuMsAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    
    return SelectorStatus::MATCH;
}


SelectorStatus AllGatherAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
{

    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectAivAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectDPUAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{

    return SelectorStatus::NOT_MATCH;
}