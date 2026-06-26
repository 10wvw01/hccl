
class TopoMatchBase {
public:
    explicit TopoMatchBase();
    virtual ~TopoMatchBase();

    virtual std::string Describe() const = 0;

    virtual HcclResult MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                 AlgHierarchyInfoForAllLevel& algHierarchyInfo);
};


class HcclAlgorithm {
    TopoMatchBase topoMatch;
    HcclCMDType hcclCmdType;
    EngineType engineType;
    ExecutorType executorType;
    vector<TemplateDesc> templates;

    // vector<vector<TemplateDesc>> templateDescs;   // 第一层表示stage，第二层表示数据part, 
    //                                               //  parallel: [[stage0_part0, stage0_part1],[stage1_part0, stage1_part1]] 
    //                                               //  concurrent: [[stage0_part0, stage0_part1]]
    //                                               //  omnipipe: [[stage0_part], [stage1_part0, stage1_part1]]
    //                                               //  sequece: [[stage0_part], [stage1_part], [stage2_part]]
    //                                               //  partConcurrent: [[stage0_part0, stage0_part1], [stage1_part0]]

    //                             Orch {
    //                                 thread (stage0_part0, stage0_part1)
    //                                 thread (stage1_part0)
    //                                 sync
    //                             }

    // vector<vector<int>> templateTopoIndex;        // 当前：topoIndex0表示板内的通信子域，1表示框内的通信子域，2表示超节点间的通信子域
                                                
    // 3: 
    // 串行Executor: 2 -> 1 -> 0  templateDescs: [[NHR], [NHR], [MESH]]    templateTopoIndex :[[2],[1],[0]]
    // OmniPipe 2 -> 1,0  templateDescs: [[NHR], [NHR, MESH]]    templateTopoIndex:[[2], [1, 0]]


    // [[NHR], [NHR]]
}


// REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherParallelMesh1DNHR,
//                                InsV2AllGatherParallelExecutor, TopoMatchMultilevel, InsTempAllGatherMesh1D,
//                                InsTempAllGatherNHR);

struct TemplateDesc {
    OpType op,
    AlgType alg,  // Mesh / NHR
    ShotType,
    JetttyType,
}


// HcclAlgorithm { hcclCmdType = HCCL_CMD_ALLGATHER, engineType = AICPU, executorType = PARALLEL, templates = 
//     [[{hcclCmdType = HCCL_CMD_ALLGATHER, alg = mesh, ShotType = DEFAULT, JetttyType = DEFAULT}, 
//             {hcclCmdType = HCCL_CMD_ALLGATHER, alg = nhr, ShotType = DEFAULT, JetttyType = DEFAULT}],
//         [{hcclCmdType = HCCL_CMD_ALLGATHER, alg = nhr, ShotType = DEFAULT, JetttyType = DEFAULT}, 
//         {hcclCmdType = HCCL_CMD_ALLGATHER, alg = mesh, ShotType = DEFAULT, JetttyType = DEFAULT}]]
// }

// templateTopoIndex[[0, 1], [1, 0]]  // topoIndex0表示板内的通信子域，1表示框内的通信子域，2表示超节点间的通信子域
// template[[0, 1, 2]]  [1][3]




//REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLREDUCE, InsAllReduceMesh1DTwoShotMeshChunk, InsV2AllReduceSoleExecutor, 
//    TopoMatch1D, InsTempAllReduceMesh1DTwoShotMeshChunk);
// HcclAlgorithm {op = AllReduce, engine = aicpu, executor = Sole, templates = [[
//     {op = AllReduce, alg = mesh, CustomFeatures= {ShotType: TwoShotMeshChunk}},
// ],

// ]
// }

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