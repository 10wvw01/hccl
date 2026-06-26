# include "alg_selector.h"
# include "engine_selector.h"
// HcclResult Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
//     HcclAlgorithm &alg) {
//     engineSelector.Select() //选出执行引擎
//     HcclAlgorithm alg = opSelector.SelectAicpuAlg() //根据算子、拓扑选算法（数据结构包含顶层调度逻辑Executor，及算法Template等信息）
// }

// Selector()  ← 对外接口，保持不变
//   ├── EngineSelector  ← 第一阶段：决定引擎（不再回退）
//   │     └── 输入: SelectorParams + TopoInfo, 输出: CommEngine
//   │
//   ├── AlgorithmSelector  ← 第二阶段：在已选引擎下选算法
//   │     └── 输入: Engine + SelectorParams + TopoInfo, 输出: algName
//   │
//   └── PostProcess  ← 第三阶段：设置algTag, kernel加载等
HcclResult Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    HcclAlgorithm &alg) {
    EngineType engineType = EngineSelector.Select(comm, param); //选出执行引擎
    AlgorithmSelector opSelector(comm, param, engineType);
    HcclAlgorithm alg = opSelector.Selector(); //根据算子、拓扑选算法（数据结构包含顶层调度逻辑Executor，及算法Template等信息）
}



static constexpr int kMaxConfig = 7; // 根据 OpExecuteConfig 枚举值范围

static constexpr AlgoSelectFunc kAlgoDispatchArray[] = {
    /* [CCU_MS=0]     */ &AutoSelectorBase::SelectCcuMsAlgo,
    /* [CCU_SCHED=1]  */ &AutoSelectorBase::SelectCcuScheduleAlgo,
    /* [AICPU_TS=2]   */ &AutoSelectorBase::SelectAicpuAlgo,
    /* [AIV=3]        */ &AutoSelectorBase::SelectAivAlgo,
    /* [AIV_ONLY=4]   */ &AutoSelectorBase::SelectAivAlgo,
    /* [HOSTCPU=5]    */ &AutoSelectorBase::SelectDPUAlgo,
    /* [HOSTCPU_TS=6] */ &AutoSelectorBase::SelectAicpuAlgo,
};

SelectorStatus AutoSelectorBase::SelectAlgForEngine(
    OpExecuteConfig config,
    const TopoInfoWithNetLayerDetails* topoInfo,
    const SelectorParams& params,
    HcclAlgorithm &algDes) const
{
    auto idx = static_cast<int>(config);
    if (idx < 0 || idx >= static_cast<int>(std::size(kAlgoDispatchArray))) {
        return SelectorStatus::NOT_MATCH;
    }
    return (this->*kAlgoDispatchArray[idx])(topoInfo, params, algDes);
}

