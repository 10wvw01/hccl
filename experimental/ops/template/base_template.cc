    private vector<RankInfo> ranks
    
    Plan(AlgHierarchyInfoForAllLevel &algHierarchyInfo, HcclAlgorithm &alg) {
        nhr -> getNhrRanks
        mesh -> getMeshRanks

        ranks = getNhrRanks() / getMeshRanks()
    }

    CalcRes(AlgHierarchyInfoForAllLevel &algHierarchyInfo, Policy) {
        ranks = this.ranks
        return vector<Res>  // notify channel
    }

    virtual Orchestrate();