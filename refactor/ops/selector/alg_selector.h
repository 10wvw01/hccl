

// 新文件: src/ops/op_common/selector/algorithm_selector.h

class AlgorithmSelector {
public:
    // 核心接口：给定引擎，选择最佳算法
    HcclResult Select(const TopoInfoWithNetLayerDetails* topoInfo,
                      const SelectorParams& params,
                      CommEngine engine,
                      OpExecuteConfig executeConfig,
                      std::string& outAlgName) const;

private:
    // 分引擎派发
    SelectorStatus SelectCcuMs(const TopoInfoWithNetLayerDetails* topoInfo,
                               const SelectorParams& params,
                               std::string& algName) const;
    SelectorStatus SelectCcuSched(const TopoInfoWithNetLayerDetails* topoInfo,
                                  const SelectorParams& params,
                                  std::string& algName) const;
    SelectorStatus SelectAiv(const TopoInfoWithNetLayerDetails* topoInfo,
                             const SelectorParams& params,
                             std::string& algName) const;
    SelectorStatus SelectAicpu(const TopoInfoWithNetLayerDetails* topoInfo,
                               const SelectorParams& params,
                               std::string& algName) const;
    SelectorStatus SelectHostDPU(const TopoInfoWithNetLayerDetails* topoInfo,
                                 const SelectorParams& params,
                                 std::string& algName) const;
};