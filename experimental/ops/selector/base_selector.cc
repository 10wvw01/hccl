HcclResult Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    HcclAlgorithm &alg) {
    engineSelector.Select() //选出执行引擎
    HcclAlgorithm alg = opSelector.SelectAicpuAlg() //根据算子、拓扑选算法（数据结构包含顶层调度逻辑Executor，及算法Template等信息）
}