/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_SELECTOR_COST_MODEL
#define HCCLV2_COLL_ALG_SELECTOR_COST_MODEL

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "alg_param.h"
#include "log.h"

namespace ops_hccl {

typedef struct {
    const char *algName;
    const char *executorName;
    const char **templateName;
    int templateNum;
    HcclCMDType opType;
} AlgElement;

typedef struct {
    AlgElement *algElements;
    int count;
    int capacity;
} AllAlgos;

AllAlgos *GetAllAlgos();

HcclResult AddAlgToAllAlgos(HcclCMDType opType, const char *algName, const char *executorName,
                            const char **templateName, int templateNum);

typedef struct {
    float A;  // 用来描述跨卡传输的时间随DataSize变化的趋势，会受到UB带宽利用率的影响
    float B;  // 用来描述本地传输的时间随DataSize变化的趋势，不受到UB带宽利用率的影响
    float C;  // 用来描述一些基本时延的常数项
} CostModelParam;

typedef struct {
    const char *algName;         //
    const CostModelParam *param; //
    int count;                //
} CostAlgoParams;

typedef struct {
    CostAlgoParams *costAlgoParams;
    int count;
} CostModel;

class CostModelManager {
public:
    CostModelManager();
    ~CostModelManager();

    HcclResult Load();
    HcclResult InitCostModel(const AllAlgos &allAlgos);
    void InitBandwidth();
    double Estimate(const std::string &algName, u64 dataSize) const;

    // n: 每次发送数据量占总数据量的比例
    // netType: 组网类型（mesh组网或clos组网）
    // portNum: clos组网下使用的端口数量，mesh组网时为0
    // A: 出参，接收计算得到的A值
    // B: 出参，接收计算得到的B值
    // 计算Mesh算法的A、B参数
    static void CalcMeshParam(float n, int netType, int portNum, float &A, float &B);
    // 计算NHR算法的A、B参数
    static void CalcNHRParams(float n, int netType, int portNum, float &A, float &B);
    // 计算Latency参数, taskNum需要写算法的人预估
    static void CalcLatencyParams(int taskNum, float &C);

private:
    void FreeCostModel();
    CostModel costModel_{nullptr, 0};

    // 带宽的单位都是GB/s
    float localCopyBw_{};       // 本地拷贝带宽
    float localReduceBw_{};     // 本地reduce带宽
    float crossChipBw_{};       // 跨片带宽
    float crossChipReduceBw_{}; // 跨片reduce带宽
};

enum class AlgNetType : int {
    MESH = 0, // mesh 组网
    CLOS = 1, // clos 组网
};

enum class CostAggMode : int {
    SUM = 0, // 多组 cost 求和
    MAX = 1, // 多组 cost 取最大值
};

struct AlgNetMeta {
    std::vector<AlgNetType> netTypes; // 每个 template 一个，顺序与 costmodel 中 A/B/C 一致
    CostAggMode aggMode = CostAggMode::SUM;
};

class AlgNetMetaRegistry {
public:
    static AlgNetMetaRegistry *Global();
    void Register(const std::string &algName, AlgNetMeta meta);
    bool Query(const std::string &algName, AlgNetMeta &meta) const;

private:
    std::map<std::string, AlgNetMeta> metas_;
    mutable std::mutex mu_;
};

} // namespace ops_hccl

#endif
