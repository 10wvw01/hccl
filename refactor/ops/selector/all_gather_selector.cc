HcclResult SelectAicpuAlg(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    std::string &algName) { 
        //根据拓扑、数据量选算法（数据结构包含顶层调度逻辑Executor，及算法Template等信息）
}

HcclResult SelectCcuAlg(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    std::string &algName) { 
        //根据拓扑、数据量选算法（数据结构包含顶层调度逻辑Executor，及算法Template等信息）(单Die/2Die、CCU_MS/CCU_SCHE)
}

HcclResult SelectAivAlg(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    std::string &algName) { 
        //根据拓扑、数据量选算法（数据结构包含顶层调度逻辑Executor，及算法Template等信息）
}
