
struct SelectorParams {
    HcclCMDType opType;
    HcclDataType dataType;       // 从 DataDes/vDataDes/all2AllVDataDes 中标准化
    u64 dataCount;               // 同上
    void* inputPtr;
    void* outputPtr;
    u64 inputSize;
    u64 outputSize;
    u32 userRankSize;
    DevType deviceType;
    void* hcclComm;              // 仅用于AIV算法校验cclBuffer等
    bool enableDetour;
    OpMode opMode;
    OpExecuteConfig userForcedConfig;  // 用户通过环境变量强制指定的引擎配置

    static SelectorParams FromOpParam(const OpParam& param, u32 rankSize);
};

// 新文件: src/ops/op_common/selector/engine_selector.h

// ============================================================
// 引擎可用性条件位掩码（分类分层设计，一个类别一个独立的 u32）
// ============================================================
// 设计原则:
//   1. 每个类别使用独立的 u32，互不干扰，天然隔离
//   2. 每类最多支持 32 个条件，远超过实际需求
//   3. 跨类别条件组合由 ConditionMask 结构体承载
//   4. 每个枚举内部 bit 从 0 开始编号，无需维护全局 bit 偏移

// ---- 类别A: netLayer层级和范围 ----
enum CondNetLayer : u32 {
    COND_LAYER_NONE         = 0,
    COND_LAYER_SINGLE       = 1 << 0,  // 单层拓扑: topoLevelNums <= 1
    COND_LAYER_MULTI        = 1 << 1,  // 多层拓扑: topoLevelNums > 1
    COND_LAYER_HAS_LEVEL1   = 1 << 2,  // netLayerNum > 1 (存在level1网络层)
    COND_LAYER_LEVEL1_CLOS  = 1 << 3,  // level1存在CLOS拓扑
};

// ---- 类别B: 数据量大小 ----
enum CondDataSize : u32 {
    COND_DATA_NONE          = 0,
    COND_DATA_SMALL         = 1 << 0,  // dataSize < 64KB
    COND_DATA_MEDIUM        = 1 << 1,  // 64KB <= dataSize < 512KB
    COND_DATA_LARGE         = 1 << 2,  // 512KB <= dataSize < 8MB
    COND_DATA_HUGE          = 1 << 3,  // dataSize >= 8MB
};

// ---- 类别C: 物理topo类型 ----
enum CondPhysicalTopo : u32 {
    COND_TOPO_NONE              = 0,
    COND_TOPO_MESH_1D           = 1 << 0,   // level0Topo == MESH_1D
    COND_TOPO_MESH_1D_CLOS      = 1 << 1,   // level0Topo == MESH_1D_CLOS
    COND_TOPO_CLOS              = 1 << 2,   // level0Topo == CLOS
    COND_TOPO_UBX               = 1 << 3,   // UBX机型: clos%mesh==0 或 clos==mesh
    COND_TOPO_PCIE_MIX          = 1 << 4,   // level0PcieMix == true
    COND_TOPO_TWO_DIE_REGULAR   = 1 << 5,   // level0MeshType == TWO_DIE_REGULAR
    COND_TOPO_TWO_DIE_NONREG    = 1 << 6,   // level0MeshType == TWO_DIE_NOT_REGULAR
    COND_TOPO_NHR               = 1 << 7,   // Level1Nhr 或 Level0Nhr
};

// ---- 类别D: 其他条件 ----
enum CondOther : u32 {
    COND_OTHER_NONE         = 0,
    COND_OTHER_IO_OVERLAP   = 1 << 0,  // input/output 指针重叠
    COND_OTHER_RANK_LE_4    = 1 << 1,  // userRankSize <= 4
    COND_OTHER_RANK_LE_32   = 1 << 2,  // userRankSize <= 32
    COND_OTHER_DPU_OK       = 1 << 3,  // HostDPU可用
    COND_OTHER_AIV_BUF_OK   = 1 << 4,  // cclBuffer充足(AIV可用)
    COND_OTHER_AIV_ONLY     = 1 << 5,  // 用户通过环境变量强制AIV_ONLY
};

// ============================================================
// 分类条件掩码结构体（每类独占一个 u32）
// ============================================================
struct ConditionMask {
    u32 netLayer;      // 类别A: CondNetLayer 的组合
    u32 dataSize;      // 类别B: CondDataSize 的组合
    u32 physicalTopo;  // 类别C: CondPhysicalTopo 的组合
    u32 other;         // 类别D: CondOther 的组合

    bool IsEmpty() const {
        return netLayer == 0 && dataSize == 0 && physicalTopo == 0 && other == 0;
    }
};

// 跨类别复合排斥条件（两个不同类别的条件同时满足才排除）
struct CompoundReject {
    bool (*checkA)(const ConditionMask& mask);  // 条件A检查函数
    bool (*checkB)(const ConditionMask& mask);  // 条件B检查函数
};

// 引擎候选条目
struct EngineEntry {
    CommEngine      engine;
    OpExecuteConfig config;
    ConditionMask   requireMask;   // 每类都必须满足的条件
    ConditionMask   rejectMask;    // 任一类别命中即排除
    std::vector<CompoundReject> compoundRejects;  // 跨类别复合排斥
};

class EngineSelector {
public:
    // 计算当前场景的分类条件掩码
    static ConditionMask ComputeConditionMask(
        const TopoInfoWithNetLayerDetails* topo,
        const SelectorParams& params);

    // 按优先级顺序返回第一个满足条件的引擎
    HcclResult Select(const TopoInfoWithNetLayerDetails* topoInfo,
                      const SelectorParams& params,
                      CommEngine& outEngine,
                      OpExecuteConfig& outConfig) const;

    // 各算子对应的引擎优先级链（按算子类型获取）
    static const std::vector<EngineEntry>& GetEngineChain(HcclCMDType opType);

    // 辅助: 将 OpExecuteConfig 映射到 CommEngine
    static CommEngine ConfigToEngine(OpExecuteConfig config);

    // 调试: 将条件掩码转换为分类可读字符串
    static std::string MaskToString(const ConditionMask& mask);
};