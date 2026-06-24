HcclResult HcclExecOp(HcclComm comm, OpParam &param,
                      std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo, HcclAlgorithm &alg, const ResPackGraphMode &resPack) {

    // Todo： Kernel缓存

     executor = alg.GetExecutor()
     engine = alg.GetEngine()
     AlgHierarchyInfoForAllLevel algHierarchyInfo = executor.CalcAlgHierarchyInfo(alg.topoMatch)
     executor.Plan(algHierarchyInfo, alg)  // executor.Plan -> template1: BaseTemplate.Plan + template2: BaseTemplate.Plan
     vector<Res> res = executor.CalcRes(algHierarchyInfo)  // executor.CalcRes -> template1: BaseTemplate.CalcRes + template2: BaseTemplate.CalcRes
     engine.CreateRes(res)  // 不同引擎创建资源的方式不同 aicpu -> Channel、notify、 thread  ;  aiv -> channel 、共享内存;   ccu -> cclMem 、notify、thread、channel
     engine.LaunchKernel(param, executor)  //aicpuEngine.LaunchKernel()
}