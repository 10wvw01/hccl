#include "hccl_algorithm.h"

class BaseExecutor {
public:
    BaseExecutor(HcclAlgorithm &algo);
    // TODO：是否需要2种构造函数，一种用于CalcRes，另一种用于Orchestrate
    BaseExecutor(HcclAlgorithm &algo, BaseExecutorParam &param);
    ~BaseExecutor();

protected:
    // Describe函数完全没用到，可以删除
    // std::string Describe();

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
        AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgTopoMatch topoMatch);

    HcclResult Plan(AlgHierarchyInfoForAllLevel &algHierarchyInfo);

    virtual HcclResult CalcRes(AlgHierarchyInfoForAllLevel &algHierarchyInfo);

    virtual HcclResult Orchestrate(const BaseExecutorParam &baseExecutorParam, ConfigParam &configParam,
        const BufferParam &bufferParam, const CommInfoList &CommInfoList);

    //TODO: 目前仅用于CCU，理论上可扩展至所有模式
    HcclResult FastLaunch();

    // algo
    HcclAlgorithm algo_;

    // rankInfo
    u32 myRank_ = INVALID_VALUE_RANKID;
    u32 rankSize_ = 0;

    // dataInfo
    HcclDataType dataType_;
    u64 dataTypeSize_ = 0;
    u64 dataCount_ = 0;
    u64 dataSize_ = 0;

    // opInfo
    HcclReduceOp reduceOp_;
    u32 root_ = INVALID_VALUE_RANKID;
    
    // TODO：分析下使用场合
    DevType devType_ = DevType::DEV_TYPE_COUNT;

    std::vector<ThreadHandle> threads_;

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    
    // vector中第一个元素表示intra，第二个元素表示inter，后续可扩展
    std::vector<u32> subRankSize_;
    std::vector<u32> subRankIdx_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> subChannels_;
    std::vector<std::vector<ThreadHandle>> subThreads_;
    std::vector<u32> syncNotifyOnMain_;
    
    // vector外层表示stage，内层第一个元素表示intra的最后一个notifyid，第二个元素表示inter的最后一个notifyid，内层可扩展
    std::vector<std::vector<u32>> syncNotifyOnTemplates_;
};


struct BaseExecutorParam {
    u32 myRank = INVALID_VALUE_RANKID;
    u32 rankSize = 0;

    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    u64 dataTypeSize = 0;
    u64 dataCount = 0;
    u64 dataSize = 0;

    HcclReduceOp reduceOp = HCCL_REDUCE_RESERVED;  // reduce类型，搬运类算子使用默认值
    u32 root = INVALID_VALUE_RANKID;  // root节点所在rank，不涉及root算子使用默认值

    // TODO：针对带V的算子，需要额外传入数组
    u8 varData = 0;
};

struct ConfigParam {
    OpMode opMode;
    // TODO：绕路参数
};

struct BufferParam {
    Buffer inputBuffer;
    Buffer outputBuffer;
    Buffer cclBuffer;

    // cclBuffer统一用ptr和size来描述，template里面也是用这些参数，不使用HcclMem
    // 删除冗余maxTmpMemSize_参数
};

struct Buffer {
    void* ptr;
    u64 size;
    BufferType bufferType;
};

