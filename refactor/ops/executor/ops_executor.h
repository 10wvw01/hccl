#include "hccl_algorithm.h"

class OpsExecutor {
public:
    OpsExecutor(HcclAlgorithm &algo, OpParam &param);
    ~OpsExecutor();

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo);

    virtual HcclResult CalcRes(AlgResourceRequest &resReq);

    HcclResult Orchestrate(const OpsExecutorParam &baseExecutorParam,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceCtxSerializable &resCtx);

private:
    HcclResult CalcResRecursion(AlgoExecDesc &algoExecDesc, u32 &subCommMask, u32 &scratchMutiple);
    HcclResult PrepareResForTemplate();
    HcclResult OrchestrateLoop(const AlgResourceCtxSerializable &resCtx, AlgoExecDesc &algoExecDesc,
        AlgoExecDataDesc &algoExecDataDesc);
    HcclResult GenTemplateRes(
        const AlgResourceCtxSerializable &resCtx, const u32 subCommIndex, TemplateResource &templateResource);
    inline void GenTemplateDataParams(const AlgResourceCtxSerializable &resCtx, AlgoExecDataDesc &algoExecDataDesc,
        TemplateDataParams &templateDataParams);
    inline void UpdateSubCommMask(AlgoExecDesc &algoExecDesc, const u32 subCommMask);
    HcclResult PreSyncBySubCommMask(const AlgoExecDesc &execDesc);
    HcclResult PostSyncBySubCommMask(const AlgoExecDesc &execDesc);
    inline void InitAlgoExecDataDesc(AlgoExecDataDesc &algoExecDataDesc, u64 dataOffset, u64 dataCount);
    inline void UpdateDataSplit(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc,
    u32 childrenId, std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc);
    u32 MergeScratchMutiple(AlgoExecDesc &algoExecDesc, std::vector<u32> &childrenScrachMutilple);
    // 处理单个 TemplateExecDesc 子节点：实例化template、生成资源/数据参数、计算stride、KernelRun
    HcclResult RunTemplateDesc(
        const AlgResourceCtxSerializable &resCtx, TemplateExecDesc *templateExeDes, AlgoExecDataDesc &algoExecDataDesc);

protected:
    HcclResult InitRes(const AlgResourceCtxSerializable &resCtx);

    std::vector < std::map<u32, std::vector<ChannelInfo>> RestoreChannelMap(const AlgResourceCtxSerializable &resCtx);

    virtual u64 GetMaxProcCntPerLoop();

    virtual HcclResult OrchestrateLoop(u64 processCount, u64 offsetCount);

    // 通信域指针
    HcclComm hcclComm_;

    // algo
    HcclAlgorithm algo_;

    // rankInfo
    u32 myRank_ = INVALID_VALUE_RANKID;
    u32 rankSize_ = 0;
    u32 root_ = INVALID_VALUE_RANKID;
    // dataInfo
    DataInfo dataInfo_;
    u32 scratchMultiple_;
    // config
    OpMode opMode_;

    // 拓扑分级信息
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    // vector中第一个元素表示intra，第二个元素表示inter，后续可扩展
    std::vector<u32> subRankSize_;
    std::vector<u32> subRankIdx_;

    // 资源信息
    // [Buffer资源]
    BufferInfo bufferInfo_;
    // [线程资源]
    ThreadHandle mainThread_;
    std::vector<ThreadHandle> threads_;
    std::vector<std::vector<ThreadHandle>> subThreads_;
    // [Notify资源]
    std::vector<u32> notifyNumOnSubMainThread_;
    // [Channel资源]
    // Channel资源表，vector层表示不同拓扑层级，map层key表示remoteRank，value为channel信息
    std::vector < std::map<u32, std::vector<ChannelInfo>> channelTable_;

    std::vector<u32> maxSlaveThreadNum_;
    std::vector<u32> maxNotifyNumOnMainThread_;
    std::vector<u32> maxNotifyNumPerThread_;

    // 递归后用于保存算法执行所需要的流同步信息
    std::map<AlgoExecDesc, u32> execDescSubCommMask_;
};

struct OpsExecutorParam {
    BaseOpParam baseOpParam;
    ConfigParam configParam;
    BufferParam bufferParam;
};

struct BaseOpParam {
    u32 myRank = INVALID_VALUE_RANKID;
    u32 rankSize = 0;

    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    u64 dataCount = 0;

    HcclReduceOp reduceOp = HCCL_REDUCE_RESERVED; // reduce类型，搬运类算子使用默认值
    u32 root = INVALID_VALUE_RANKID;              // root节点所在rank，不涉及root算子使用默认值

    // TODO：针对带V的算子，需要额外传入数组
    u8 varData = 0;
};

struct ConfigParam {
    OpMode opMode;
    // TODO：绕路参数
};

struct BufferInfo {
    Buffer inputBuffer;
    Buffer outputBuffer;
    Buffer cclBuffer;

    // cclBuffer统一用ptr和size来描述，template里面也是用这些参数，不使用HcclMem
    // 删除冗余maxTmpMemSize_参数
};

struct Buffer {
    void *ptr;
    u64 size;
    BufferType bufferType;
};

struct DataInfo {
    void *inputPtr = nullptr;
    u64 inputSize = 0;
    void *outputPtr = nullptr;
    u64 outputSize = 0;
    DataDesUnion dataDesUnion;
    HcclReduceOp reduceOp_ = HCCL_REDUCE_RESERVED;
};

union DataDesUnion {
    struct {
        u64 count;
        HcclDataType dataType;
        HcclDataType outputType;
        u64 strideCount;
    } DataDes = {0, HCCL_DATA_TYPE_RESERVED, HCCL_DATA_TYPE_RESERVED, 0};
    struct {
        HcclDataType sendType;
        HcclDataType recvType;
        u64 sendCount;
        u64 recvCount;
    } all2AllDataDes;
    struct {
        void *counts;
        void *displs;
        HcclDataType dataType;
    } vDataDes;
    struct {
        HcclDataType sendType;
        HcclDataType recvType;
        void *sendCounts;
        void *recvCounts;
        void *sdispls;
        void *rdispls; // 指向变长区指针
    } all2AllVDataDes;
    struct {
        HcclDataType sendType;
        HcclDataType recvType;
        void *sendCountMatrix;
    } all2AllVCDataDes;
    struct {
        HcclSendRecvItem *sendRecvItemsPtr;
        u32 itemNum;
    } batchSendRecvDataDes;
};

struct AlgoExecDataDesc {
    u64 dataOffset{0};
    u64 dataCount{0};
    std::vector<u32> ranksForInputData;
    std::vector<u32> ranksForOutputData;
    BufferType inputBufferType{BufferType::INPUT};
    BufferType outputBufferType{BufferType::OUTPUT};
    BufferType cclBufferType{BufferType::HCCL_BUFFER};
};

struct TemplateDataParam {
    void *inputBufferPtr;
    void *outputBufferPtr;
    void *cclBufferPtr;
    BufferType inputBufferType;
    BufferType outputBufferType;
    BufferType cclBufferType;

    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    u64 sliceCount{0}; // 传入根节点的每个loop的count，后续不变
    u64 tailCount{0};

    u64 dataOffset{0};
    u64 cclBufferOffset{0};

    HcclReduceOp reduceOp{HCCL_REDUCE_RESERVED}; // reduce类型，搬运类算子使用默认值
    u32 root{INVALID_VALUE_RANKID};              // root节点所在rank，不涉及root算子使用默认值

    bool enableRemoteMemAccess{false};

    std::vector<u32> ranksForInputData;

    // TODO：变长
};
