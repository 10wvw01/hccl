EngineType Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    EngineType &engineType) {
        //根据环境变量、机型、数据量选出合适的引擎
}

//TODO: 对OpParam进一步进行正交分解，Selector仅选择必要信息对象作为参数