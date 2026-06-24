BaseParallerExecutor : BaseExecutor {
    protected vector<BaseTemplate> templates;

    CalcRes(ranks) {
        templates[0].CalcRes(policy1) + templates[1].CalcRes(policy2)
    }

    Orchestrate(HcclComm comm, OpParam &param) {
        TemplateAlgParams0 = GenTemplateAlgParams0
        TemplateAlgParams1 = GenTemplateAlgParams1

        // 按CCLBuffer大小切分loop并对齐
        loops = calcLoops
        for (loop : loops) {
            presync()
            templates0.kernelRun(TemplateAlgParams0)
            templates1.kernelRun(TemplateAlgParams1)
            postsync()
        }
    }

}