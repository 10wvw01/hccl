AllGatherParallelExecutor : BaseParallerExecutor {
    private BaseTemplate template1;
    private BaseTemplate template2;

    AllGatherParallelExecutor(HcclAlgorithm alg) {
        vector<TemplateType> templates = alg.templates
        switch (engine, alg, customValue) {
            case aicpu, mesh:
                template1 = new AllGatherMesh1dTemplate()
        }

        switch (engine, alg, customValue) {
            case aicpu, nhr:
                template2 = new AllGatherNhrTemplate()
        }

        templates.insert(template1, template2)
    }


}