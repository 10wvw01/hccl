/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALGO_NAME_MAPPER_H
#define ALGO_NAME_MAPPER_H

#include <string>
#include <unordered_map>
#include <utility>

#include "hccl_tuner_plugin.h"

namespace ops_hccl {

/* 查询结果：算法的 3D 用户名 */
struct AlgoDims {
    const char *engineUser;     /* "aicpu" */
    const char *executorUser;   /* "sole" */
    const char *templateUser;   /* "mesh_one_shot" */
};

/* 算法名 + opType 对（供 Init 使用，解耦 AllAlgos） */
struct AlgoRegInfo {
    std::string algName;
    int opType;  /* HcclCMDType 枚举值 */
};

class AlgoNameMapper {
public:
    static AlgoNameMapper *Global();

    /* init：建 2D 表 + 缓存所有算法（HCCL 启动时调用一次） */
    void Init(const AlgoRegInfo *algos, int count);

    /* enrich：填 3D 名到 entry 数组（每次 op，CostTableGen 之后调用） */
    void Enrich(hcclTunerAlgoEntry_t *entries, int count);

    /* query：按需查单个算法（其他模块用） */
    bool Query(const std::string &algName, AlgoDims &dims) const;

private:
    AlgoNameMapper() = default;

    /* 维度值表：PascalCase 前缀 + 用户侧名 */
    struct DimEntry { const char *pascal; const char *user; };
    static const DimEntry g_engines[];
    static const int g_engineCount;
    static const DimEntry g_executors[];
    static const int g_executorCount;
    static const DimEntry g_templates[];
    static const int g_templateCount;
    static const char * const g_opTypes[];
    static const int g_opTypeCount;

    /* 2D 预计算表（init 时构建，30 条） */
    std::unordered_map<std::string,
        std::pair<const char*, const char*>> map2D_;

    /* 算法缓存（init 时填充，Query/Enrich 直接读） */
    std::unordered_map<std::string, AlgoDims> cache_;

    void BuildMap2D();
    bool Lookup2D(const std::string &algName,
                  const std::string &opTypePascal,
                  AlgoDims &dims) const;
    static std::string OpTypeToPascal(int opType);
};

} /* namespace ops_hccl */

#endif /* ALGO_NAME_MAPPER_H */
