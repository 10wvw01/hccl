class HcclAlgorithm {

    HcclAlgorithm() {
    }

    GetEngine() {
        switch (engineType) {
            case aicpu:
                new AicpuLauncher()
                break;
            default:
                statements3
                break;
        }
    }

    GetExecutor(OpParam &param) {
        switch (op, executorType) {
            case AllGather, PARALLEL:
                new AllGatherParallelExecutor(alg, param)
                break;
            default:
                statements3
                break;
        }
    }

    //打印基本信息
    Dump()

    //Todo: 算法选择是否需要在多机之间一致性校验？
    Serialize()

    
}