#include <map>
#include <memory>
#include <vector>
#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <hccl/hccl_res.h>
#include "hccl_algorithm.h"
#include "template/base_template.h"
#include "template/template_factory.h"
#include "utils/utils.h"

namespace ops_hccl {

struct BufferInfo {
    void *ptr = nullptr;
    u64 size = 0;
    BufferType bufferType = BufferType::HCCL_BUFFER;
};

struct DataInfo {
    void *inputPtr = nullptr;
    u64 inputSize = 0;
    void *outputPtr = nullptr;
    u64 outputSize = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_RESERVED;
};

struct AlgoExecDataDesc {
    u64 dataOffset{0};
    u64 sliceCount{0};
    u64 scratchOffset{0};
    u64 scratchSize{0}; // 输出参数
    u64 tailCount{0};
    std::vector<u32> ranksForInputData;
    std::vector<u32> ranksForOutputData; // 输出参数
    BufferType inputBufferType{BufferType::INPUT};
    BufferType outputBufferType{BufferType::OUTPUT};
    BufferType cclBufferType{BufferType::HCCL_BUFFER};
};

class OpsExecutor {
public:
    OpsExecutor(HcclAlgorithm &algo, OpParam &param);
    ~OpsExecutor();

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo);

    virtual HcclResult CalcRes(AlgResourceRequest &resReq);

    HcclResult Orchestrate(const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceCtxSerializable &resCtx);

private:
    HcclResult CalcResRecursion(AlgoExecDesc &algoExecDesc, float inputRatio, float &outputRatio, u32 &subCommMask);
    HcclResult CalcTemplateRes(const TemplateExecDesc &templateExeDes, float inputRatio, float &outputRatio);
    HcclResult OrchestrateLoop(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc);
    HcclResult GenTemplateRes(const u32 subCommIndex, TemplateResource &templateResource);
    inline void GenTemplateDataParams(AlgoExecDataDesc &algoExecDataDesc, TemplateDataParams &templateDataParams);
    inline void UpdateSubCommMaskMap(AlgoExecDesc &algoExecDesc, const u32 subCommMask);
    HcclResult PreSyncBySubCommMask(const AlgoExecDesc &execDesc);
    HcclResult PostSyncBySubCommMask(const AlgoExecDesc &execDesc);
    inline void InitAlgoExecDataDesc(AlgoExecDataDesc &algoExecDataDesc, u64 dataOffset, u64 dataCount, u64 tailCount);
    inline void UpdateDataSplitParallel(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc, u32 childrenId,
        std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc);
    inline void UpdateDataSplitSequence(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc, u32 childrenId,
        std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc);
    inline void MergeChildrenOutput(const AlgoExecDesc &algoExecDesc,
        const std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc, AlgoExecDataDesc &algoExecDataDesc);
    HcclResult RunTemplateDesc(TemplateExecDesc *templateExeDes, AlgoExecDataDesc &algoExecDataDesc);

protected:
    HcclResult InitRes(const AlgResourceCtxSerializable &resCtx);

    std::vector<std::map<u32, std::vector<ChannelInfo>>> RestoreChannelMap(const AlgResourceCtxSerializable &resCtx);

    u64 GetMaxProcCntPerLoop(u64 dataCount);

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
    u64 dataTypeSize_ = 0;
    u32 scratchMultiple_ = 0;
    // vector中第一个元素表示intra，第二个元素表示inter，后续可扩展
    std::vector<float> maxSubScratchMutiple_;
    // config
    OpMode opMode_;

    // 拓扑分级信息
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;

    // 资源信息
    // [Buffer资源]
    BufferInfo cclBufferInfo_;
    // [线程资源]
    ThreadHandle mainThread_ = 0;
    std::vector<ThreadHandle> threads_;
    std::vector<std::vector<ThreadHandle>> subThreads_;
    // [Notify资源]
    std::vector<u32> notifyNumOnSubMainThread_;
    // [Channel资源]
    std::vector<std::map<u32, std::vector<ChannelInfo>>> channelTable_;

    std::vector<std::vector<HcclChannelDesc>> requestChannels_;

    std::vector<u32> maxSlaveThreadNum_;
    std::vector<u32> maxNotifyNumOnMainThread_;
    std::vector<u32> maxNotifyNumPerThread_;

    // 递归后用于保存算法执行所需要的流同步信息
    std::map<const AlgoExecDesc *, u32> execDescSubCommMask_;
};

} // namespace ops_hccl
