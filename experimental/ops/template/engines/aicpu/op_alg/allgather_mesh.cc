AllGatherMeshTemplate: BaseTemplate {
    kernelRun(TemplateAlgParams params) {
        utils::PreCopy()
        nhr_primitivces::RunMeshAllGather()
    }
}