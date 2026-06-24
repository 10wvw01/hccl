kernelRun(TemplateAlgParams params) {
    utils::PreCopy()
    nhr_primitivces::RunNhrReduceScatter()
    utils::LocalReduce()
    nhr_primitivces::RunNhrAllGather()
    utils::PostCopy()
}