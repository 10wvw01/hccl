
class HcclAlgorithm {
    TopoMatchBase topoMatch,
    OpType op,
    EngineType engine,
    ExecutorType executor,
    vector<TemplateDesc> templates
}
// REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, CcuAllReduceParallelMesh1DNHR, InsAllReduceParallelExecutor,
//    TopoMatchMultilevel, CcuTempReduceScatterMesh1DMem2Mem, CcuTempReduceScatterNHR1DMem2Mem, CcuTempAllGatherMesh1DMem2Mem, 
//    CcuTempAllGatherNHR1DMem2Mem);

HcclAlgorithm {op = AllReduce, engine = ccu_ms, executor = Parallel, templates = [
    {op = ReduceScatter, alg = mesh, CustomFeatures= {}}, 
    {op = ReduceScatter, alg = nhr, CustomFeatures={MultiJetty: True}}, 
    {op = AllGather, alg = mesh, CustomFeatures= {}},
    {op = AllGather, alg = nhr, CustomFeatures= {}}
]
}

//REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLREDUCE, InsAllReduceMesh1DTwoShotMeshChunk, InsV2AllReduceSoleExecutor, 
//    TopoMatch1D, InsTempAllReduceMesh1DTwoShotMeshChunk);
HcclAlgorithm {op = AllReduce, engine = aicpu, executor = Sole, templates = [
    {op = AllReduce, alg = mesh, CustomFeatures= {ShotType: TwoShotMeshChunk}},
]
}

enum EngineType {
    AICPU,
    CCU_MS,
    CCU_SCHED
}

enum ExecutorType {
    PARALLEL,
    CONCURRENT,
    OMINIPIPE,
}

enum AlgType {
    MESH,
    NHR
}

enum ShotType {
    ONE,
    TWO
}

enum JettyType {
    SINGLE,
    MULTIPLE
}

struct TemplateDesc {
    OpType op,
    AlgType alg,  // Mesh / NHR
    //hashmap<CustomType, CustomValue> CustomFeatures // MultiJetty::Multi | ShotType::Oneshot/TwoShot/TwoShotMeshChunk
    ShotType,
    JetttyType,
}