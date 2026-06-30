HcclResult HcclExecOp(HcclComm comm, OpParam &param,
                      std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo, HcclAlgorithm &alg, const ResPackGraphMode &resPack) {

    // Todo： Kernel缓存

     engineType = alg.GetEngine()
     executor = alg.GetExecutor(param)
     executor.CalcAlgHierarchyInfo(comm, topoInfo)
     AlgResourceRequest resReq;
     executor.CalcRes(resReq)  // executor.CalcRes -> template1: BaseTemplate.CalcRes + template2: BaseTemplate.CalcRes
     engine.CreateRes(resReq)  // 不同引擎创建资源的方式不同 aicpu -> Channel、notify、 thread  ;  aiv -> channel 、共享内存;   ccu -> cclMem 、notify、thread、channel
     engine.LaunchKernel(param, executor)  //aicpuEngine.LaunchKernel()
}