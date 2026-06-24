BaseExecutor {
    protected AlgHierarchyInfoForAllLevel algHierarchyInfo

    CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgTopoMatch topoMatch)
    {
        CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
        return HCCL_SUCCESS;
    }

    Plan(AlgHierarchyInfoForAllLevel &algHierarchyInfo, HcclAlgorithm &alg) {
        template0.plan()
        template1.plan()
    }

    virtual CalcRes(AlgHierarchyInfoForAllLevel &algHierarchyInfo);

    virtual Orchestrate();

    //TODO:
    FastLaunch()
    
}