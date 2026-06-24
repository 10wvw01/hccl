AllGatherNhrTemplate : BaseTemplate {
    kernelRun(TemplateAlgParams params) {
        utils::PreCopy()
        nhr_primitivces::RunNhrAllGather()
        utils::PostCopy()
    }
}