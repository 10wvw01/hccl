/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_parse.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <set>
#include <map>

#include "sal.h"
#include "op_common.h"
#include "alg_env_config.h"

namespace ops_hccl {

// ===========================================================================
// 工具函数
// ===========================================================================
std::string UnderscoreToCamelCase(const std::string &name)
{
    std::string result;
    bool nextUpper = false;
    for (char c : name) {
        if (c == '_') {
            nextUpper = true;
        } else {
            if (nextUpper) {
                result += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                nextUpper = false;
            } else {
                result += c;
            }
        }
    }
    return result;
}

// ===========================================================================
// 用户算法配置解析器（内部实现类）
// ===========================================================================

class AlgoParserImpl {
public:
    explicit AlgoParserImpl(const std::string &input) : input_(input), pos_(0) {}

    HcclResult Parse(std::vector<HcclAlgoExecutor> &result)
    {
        SkipWs();
        while (!AtEnd()) {
            HcclAlgoExecutor exec;
            CHK_RET(ParseSegment(exec));
            CompactAlgoList(exec.algoList);
            result.push_back(std::move(exec));
            SkipWs();
            if (Eat(';')) {
                SkipWs();
                continue;
            }
            break;
        }
        SkipWs();
        if (!AtEnd()) {
            HCCL_ERROR("[HcclAlgoParser] trailing chars at pos %zu: [%s]", pos_,
                       input_.substr(pos_).c_str());
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

private:
    const std::string &input_;
    size_t pos_;

    // 哨兵值：标记 level 未显式指定
    static constexpr uint32_t LEVEL_UNSPECIFIED = UINT32_MAX;

    // ---- 辅助函数 ----

    // 小写字符串转换
    static std::string ToLowerStr(const std::string &s)
    {
        std::string r = s;
        std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return std::tolower(c); });
        return r;
    }

    // 有序插入：将 algo 按 level 插入到 algoList 的正确位置
    // level 由 algoList 的 index 隐式表达，插入后保持 index 与 level 对应
    void InsertAlgoOrdered(std::vector<HcclAlgo> &algoList, HcclAlgo algo, uint32_t level) const
    {
        if (level == LEVEL_UNSPECIFIED) {
            // 未指定 level，追加到末尾
            algoList.push_back(std::move(algo));
            return;
        }
        // 扩展到足够大小（中间空位用默认 HcclAlgo 填充，后续会被覆盖或保留）
        if (level >= algoList.size()) {
            algoList.resize(level + 1);
        }
        algoList[level] = std::move(algo);
    }

    // 补全 algoList：消除空位，确保 index 连续且按 level 升序
    // 空位是指 algoType 为空的默认 HcclAlgo 条目，由 resize 产生
    void CompactAlgoList(std::vector<HcclAlgo> &algoList) const
    {
        // 先将所有有效条目（algoType 非空）按 index(=level) 收集
        std::vector<std::pair<uint32_t, HcclAlgo>> validItems;
        for (uint32_t i = 0; i < static_cast<uint32_t>(algoList.size()); i++) {
            if (!algoList[i].algoType.empty()) {
                validItems.emplace_back(i, std::move(algoList[i]));
            }
        }
        // 已按 index 升序（遍历顺序），直接紧凑写入
        algoList.clear();
        for (auto &p : validItems) {
            algoList.push_back(std::move(p.second));
        }
    }

    // ---- 基础词法 ----
    char Peek() const { return pos_ < input_.size() ? input_[pos_] : '\0'; }
    char Peek2() const { return pos_ + 1 < input_.size() ? input_[pos_ + 1] : '\0'; }
    // 尝试消费字符 c：当前字符匹配则前进一步返回 true，否则返回 false
    bool Eat(char c)
    {
        if (Peek() == c) {
            pos_++;
            return true;
        }
        return false;
    }
    bool AtEnd() const { return pos_ >= input_.size(); }

    void SkipWs()
    {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            pos_++;
        }
    }

    // 解析标识符：字母/数字/下划线/连字符（规则8）
    bool ParseIdentifier(std::string &name)
    {
        SkipWs();
        size_t start = pos_;
        while (pos_ < input_.size()) {
            char c = input_[pos_];
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
                pos_++;
            } else {
                break;
            }
        }
        if (pos_ == start) return false;
        name = input_.substr(start, pos_ - start);
        return true;
    }

    // ---- 文法规则 ----

    // segment := [opType ':'] executor_expr
    HcclResult ParseSegment(HcclAlgoExecutor &exec)
    {
        size_t savePos = pos_;
        std::string name;
        if (ParseIdentifier(name)) {
            SkipWs();
            if (Peek() == ':' && Peek2() != ':') {
                Eat(':');
                exec.opType = UnderscoreToCamelCase(name);
                SkipWs();
            } else {
                pos_ = savePos;
            }
        }
        return ParseExecutorExpr(exec);
    }

    // executor_expr := 'not' '(' inner ')' | executor_unit | template_name(shorthand)
    HcclResult ParseExecutorExpr(HcclAlgoExecutor &exec)
    {
        SkipWs();
        size_t savePos = pos_;
        std::string name;
        if (ParseIdentifier(name)) {
            SkipWs();
            if (ToLowerStr(name) == "not" && Peek() == '(') {
                Eat('(');
                SkipWs();
                CHK_RET(ParseExecutorUnitOrAtom(exec));
                SkipWs();
                if (!Eat(')')) {
                    HCCL_ERROR("[HcclAlgoParser] expected ')' after not(...) at pos %zu", pos_);
                    return HCCL_E_PARA;
                }
                exec.enable = false;
                return HCCL_SUCCESS;
            }
            pos_ = savePos;
        }
        return ParseExecutorUnitOrAtom(exec);
    }

    // executor_unit | template_name(shorthand)
    HcclResult ParseExecutorUnitOrAtom(HcclAlgoExecutor &exec)
    {
        SkipWs();
        std::string name;
        if (!ParseIdentifier(name)) {
            HCCL_ERROR("[HcclAlgoParser] expected executor or template at pos %zu", pos_);
            return HCCL_E_PARA;
        }
        SkipWs();
        if (Peek() == '{') {
            exec.executorType = UnderscoreToCamelCase(name);
            Eat('{');
            CHK_RET(ParseTemplateList(exec.algoList));
            SkipWs();
            if (!Eat('}')) {
                HCCL_ERROR("[HcclAlgoParser] expected '}' at pos %zu", pos_);
                return HCCL_E_PARA;
            }
            return HCCL_SUCCESS;
        }
        // template shorthand: name => sole{name}
        exec.executorType = "sole";
        HcclAlgo algo;
        algo.algoType = UnderscoreToCamelCase(name);
        algo.enable = true;
        InsertAlgoOrdered(exec.algoList, std::move(algo), LEVEL_UNSPECIFIED);
        return HCCL_SUCCESS;
    }

    // tpl_list := tpl_item (',' tpl_item)*
    HcclResult ParseTemplateList(std::vector<HcclAlgo> &algoList)
    {
        SkipWs();
        if (Peek() == '}') return HCCL_SUCCESS;
        while (true) {
            HcclAlgo algo;
            uint32_t level = LEVEL_UNSPECIFIED;
            CHK_RET(ParseTemplateItem(algo, level));
            InsertAlgoOrdered(algoList, std::move(algo), level);
            SkipWs();
            if (!Eat(',')) break;
            SkipWs();
        }
        return HCCL_SUCCESS;
    }

    // tpl_item := ['level' digit '=' ] tpl_expr
    HcclResult ParseTemplateItem(HcclAlgo &algo, uint32_t &level)
    {
        SkipWs();
        size_t savePos = pos_;
        std::string name;
        if (ParseIdentifier(name)) {
            // 判定 "levelN"
            if (name.size() > 5 && ToLowerStr(name.substr(0, 5)) == "level") {
                bool allDigit = true;
                for (size_t i = 5; i < name.size(); i++) {
                    if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
                        allDigit = false;
                        break;
                    }
                }
                if (allDigit) {
                    SkipWs();
                    if (Eat('=')) {
                        try {
                            level = static_cast<uint32_t>(std::stoul(name.substr(5)));
                        } catch (...) {
                            HCCL_ERROR("[HcclAlgoParser] invalid level: %s", name.c_str());
                            return HCCL_E_PARA;
                        }
                        SkipWs();
                        return ParseTemplateExpr(algo);
                    }
                }
            }
            pos_ = savePos;
        }
        return ParseTemplateExpr(algo);
    }

    // tpl_expr := 'not' '(' template_name ')' | template_name
    HcclResult ParseTemplateExpr(HcclAlgo &algo)
    {
        SkipWs();
        size_t savePos = pos_;
        std::string name;
        if (ParseIdentifier(name)) {
            SkipWs();
            if (ToLowerStr(name) == "not" && Peek() == '(') {
                Eat('(');
                SkipWs();
                CHK_RET(ParseTemplateAtom(algo));
                algo.enable = false;
                SkipWs();
                if (!Eat(')')) {
                    HCCL_ERROR("[HcclAlgoParser] expected ')' after not(template) at pos %zu", pos_);
                    return HCCL_E_PARA;
                }
                return HCCL_SUCCESS;
            }
            pos_ = savePos;
        }
        return ParseTemplateAtom(algo);
    }

    // template_name
    HcclResult ParseTemplateAtom(HcclAlgo &algo)
    {
        SkipWs();
        std::string name;
        if (!ParseIdentifier(name)) {
            HCCL_ERROR("[HcclAlgoParser] expected template name at pos %zu", pos_);
            return HCCL_E_PARA;
        }
        algo.algoType = UnderscoreToCamelCase(name);
        return HCCL_SUCCESS;
    }
};

// ===========================================================================
// HcclAlgoExecutorParser 实现
// ===========================================================================

HcclResult HcclAlgoParser::Parser(const std::string &algoConfig)
{
    executorList.clear();
    if (algoConfig.empty()) {
        return HCCL_SUCCESS;
    }
    AlgoParserImpl parser(algoConfig);
    HcclResult ret = parser.Parse(executorList);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclAlgoParser] parse failed, input=[%s]", algoConfig.c_str());
        executorList.clear();
        return ret;
    }
    HCCL_DEBUG("[HcclAlgoParser] parse ok, %s", ToString().c_str());
    return HCCL_SUCCESS;
}

std::string HcclAlgoParser::ToString() const
{
    std::string s = "HcclAlgoExecutorParser{ executorList=[";
    for (size_t i = 0; i < executorList.size(); i++) {
        if (i > 0) s += "; ";
        const auto &exec = executorList[i];
        if (!exec.opType.empty()) s += exec.opType + ":";
        if (!exec.enable) s += "not(";
        s += exec.executorType + "{";
        for (size_t j = 0; j < exec.algoList.size(); j++) {
            if (j > 0) s += ",";
            s += "level" + std::to_string(j) + "=";
            if (!exec.algoList[j].enable) s += "not(";
            s += exec.algoList[j].algoType;
            if (!exec.algoList[j].enable) s += ")";
        }
        s += "}";
        if (!exec.enable) s += ")";
    }
    s += "] }";
    return s;
}

// ===========================================================================
// CostModel 刷新：UpdateCostModelWithAlgo
// ===========================================================================
// 首字母大写（用于驼峰命名拼接）
static std::string CapitalizeFirst(const std::string &s)
{
    if (s.empty()) return s;
    std::string r = s;
    r[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(r[0])));
    return r;
}

// 拼接算法名：[EngineType][OpType(cap)][ExecutorType(cap)][AlgoType0(cap)][AlgoType1(cap)]...
// 驼峰命名：首字段（EngineType）小写开头，后续字段首字母大写
static std::string ComposeAlgoName(const std::string &engineType, const std::string &opType,
                                   const std::string &executorType, const std::vector<HcclAlgo> &algoList)
{
    std::string name = engineType; // 首字段保持小写（camelCase）
    name += CapitalizeFirst(opType);
    name += CapitalizeFirst(executorType);
    for (const auto &algo : algoList) {
        name += CapitalizeFirst(algo.algoType);
    }
    return name;
}

// 拼接算法名前缀（algoList 为空时使用）：[EngineType][OpType(cap)][ExecutorType(cap)]
static std::string ComposeAlgoPrefix(const std::string &engineType, const std::string &opType,
                                     const std::string &executorType)
{
    return engineType + CapitalizeFirst(opType) + CapitalizeFirst(executorType);
}

// 判断算法名是否属于指定 OpType
static bool IsAlgoOfOpType(const std::string &algoKey, const std::string &opType)
{
    for (const auto &engine : ENGINE_TYPES) {
        std::string prefix = engine + CapitalizeFirst(opType);
        if (algoKey.size() >= prefix.size() && algoKey.compare(0, prefix.size(), prefix) == 0) {
            return true;
        }
    }
    return false;
}

// 前缀匹配
static bool StartsWith(const std::string &str, const std::string &prefix)
{
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

// 判断算法名是否包含 send 或 recv（大小写不敏感）
// 规则 4.6：send/recv 算法名不参与 param_count=-1 的排除逻辑
static bool ContainsSendRecv(const std::string &algoKey)
{
    std::string lower;
    lower.reserve(algoKey.size());
    for (char c : algoKey) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return lower.find("send") != std::string::npos || lower.find("recv") != std::string::npos;
}

// 验证值是否在合法表中，不在则打印 warning 并返回 false
static bool ValidateAgainstTable(const std::string &value, const std::vector<std::string> &table)
{
    for (const auto &item : table) {
        if (value == item) return true;
    }
    std::string validList;
    for (size_t i = 0; i < table.size(); i++) {
        if (i > 0) validList += ", ";
        validList += table[i];
    }
    HCCL_WARNING("[UpdateCostModelWithAlgo] value [%s] not in valid list [%s], skip.",
                 value.c_str(), validList.c_str());
    return false;
}

// ---------------------------------------------------------------------------
// 主函数：UpdateCostModelWithAlgo
// ---------------------------------------------------------------------------
HcclResult UpdateCostModelWithAlgo(const HcclAlgoParser &algoParser, costModel &model,
                                    const std::vector<std::string> &engineTypes)
{
    // 构建查找集合
    std::set<std::string> opSet(OP_TYPES.begin(), OP_TYPES.end());
    std::set<std::string> executorSet(EXECUTOR_TYPES.begin(), EXECUTOR_TYPES.end());
    std::set<std::string> algoSet(ALGO_TYPES.begin(), ALGO_TYPES.end());

    // 已匹配成功的 OpType 集合
    std::set<std::string> matchedOpTypes;

    // 构建 key → index 映射，加速精确查找
    std::map<std::string, int> keyToIdx;
    for (int i = 0; i < model.algorithm_count; i++) {
        if (model.algorithms[i].key != nullptr) {
            keyToIdx[model.algorithms[i].key] = i;
        }
    }

    // 判断所有 OpType 是否都已匹配
    auto allOpTypesMatched = [&]() -> bool {
        for (const auto &op : OP_TYPES) {
            if (matchedOpTypes.find(op) == matchedOpTypes.end()) return false;
        }
        return true;
    };

    // 反向遍历 executorList（后面的优先级高）
    for (int idx = static_cast<int>(algoParser.executorList.size()) - 1; idx >= 0; idx--) {
        const auto &exec = algoParser.executorList[idx];

        // 校验 executorType 不能为空（规则 4.3）
        if (exec.executorType.empty()) {
            HCCL_WARNING("[UpdateCostModelWithAlgo] executorType is empty, skip current executor.");
            continue;
        }
        // 校验 executorType 在合法表中
        if (!ValidateAgainstTable(exec.executorType, EXECUTOR_TYPES)) {
            continue;
        }

        // 校验 algoList 中的 algoType
        bool algoValid = true;
        for (const auto &algo : exec.algoList) {
            if (!ValidateAgainstTable(algo.algoType, ALGO_TYPES)) {
                algoValid = false;
                break;
            }
        }
        if (!algoValid) continue;

        // 确定目标 OpType 列表
        std::vector<std::string> targetOpTypes;
        if (exec.opType.empty()) {
            targetOpTypes = OP_TYPES; // 对所有 OpType 生效
        } else {
            if (!ValidateAgainstTable(exec.opType, OP_TYPES)) {
                continue;
            }
            targetOpTypes.push_back(exec.opType);
        }

        // 过滤掉已匹配的 OpType（已匹配的不参与后续匹配，包括反向排除）
        std::vector<std::string> unprocessedOps;
        for (const auto &op : targetOpTypes) {
            if (matchedOpTypes.find(op) == matchedOpTypes.end()) {
                unprocessedOps.push_back(op);
            }
        }
        if (unprocessedOps.empty()) continue;

        // 确定匹配模式（规则 4.5）
        bool isExecNegated = !exec.enable;           // executor 整体取非
        bool hasNegatedAlgo = false;                  // algoList 中有单项取非
        for (const auto &algo : exec.algoList) {
            if (!algo.enable) {
                hasNegatedAlgo = true;
                break;
            }
        }
        // 规则 4.5.3：executor.enable=false 且有 algo.enable=false 时，忽略 algo 级别取非
        if (isExecNegated && hasNegatedAlgo) {
            hasNegatedAlgo = false;
        }

        // 逐 OpType 处理
        for (const auto &opType : unprocessedOps) {
            // 收集本次匹配到的算法名（正向匹配用）
            std::vector<std::string> matchedNames;

            for (const auto &engine : engineTypes) {
                if (exec.algoList.empty()) {
                    // 规则 4.4：algoList 为空 → 前缀匹配
                    std::string prefix = ComposeAlgoPrefix(engine, opType, exec.executorType);
                    for (int i = 0; i < model.algorithm_count; i++) {
                        std::string key(model.algorithms[i].key ? model.algorithms[i].key : "");
                        if (StartsWith(key, prefix)) {
                            if (isExecNegated) {
                                // 反向匹配：设置 param_count=-1
                                model.algorithms[i].param_count = -1;
                            } else if (model.algorithms[i].param_count != -1) {
                                // 正向匹配：加入匹配列表
                                matchedNames.push_back(key);
                            }
                        }
                    }
                } else {
                    // 精确匹配：拼接完整算法名
                    std::string fullName = ComposeAlgoName(engine, opType, exec.executorType, exec.algoList);
                    auto it = keyToIdx.find(fullName);
                    if (it != keyToIdx.end()) {
                        int algoIdx = it->second;
                        if (isExecNegated || hasNegatedAlgo) {
                            // 反向匹配（4.5.1 / 4.5.2）：设置 param_count=-1
                            model.algorithms[algoIdx].param_count = -1;
                        } else if (model.algorithms[algoIdx].param_count != -1) {
                            // 正向匹配：加入匹配列表
                            matchedNames.push_back(fullName);
                        }
                        // param_count==-1 说明已被之前的反向排除跳过
                    }
                }
            }

            // 正向匹配处理：如果匹配到算法，标记 OpType 并排除未匹配算法
            if (!isExecNegated && !hasNegatedAlgo && !matchedNames.empty()) {
                matchedOpTypes.insert(opType);
                // 将该 OpType 下未匹配到的算法的 param_count 设为 -1
                for (int i = 0; i < model.algorithm_count; i++) {
                    std::string key(model.algorithms[i].key ? model.algorithms[i].key : "");
                    if (IsAlgoOfOpType(key, opType)) {
                        bool isMatched = false;
                        for (const auto &name : matchedNames) {
                            if (key == name) {
                                isMatched = true;
                                break;
                            }
                        }
                        if (!isMatched && !ContainsSendRecv(key)) {
                            model.algorithms[i].param_count = -1;
                        }
                    }
                }
                // 提前退出检查：所有 OpType 都已匹配
                if (allOpTypesMatched()) {
                    HCCL_DEBUG("[UpdateCostModelWithAlgo] all OpTypes matched, exit early.");
                    return HCCL_SUCCESS;
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

// ===========================================================================
// 提供给selector的costModel刷新接口
// ===========================================================================
HcclResult FilterCmByHcclAlgo(HcclComm comm, costModel &cm)
{
    //获取当前可用的引擎类型列表
    std::vector<std::string> engineTypes = GetAvaliableEngineTypes();

    //获取配置：通信域 hcclAlgo 优先，其次环境变量 HCCL_ALGO
    std::string algoConfig;
    HcclResult ret = HcclGetHcclAlgo(comm, algoConfig);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[FilterCmByHcclAlgo] HcclGetHcclAlgo failed, ret[%d], try env variable.", ret);
        algoConfig.clear();
    }

    if (algoConfig.empty()) {
        algoConfig = GetEnv("HCCL_ALGO");
        if (algoConfig == "EmptyString") {
            HCCL_DEBUG("[FilterCmByHcclAlgo] both hcclAlgo and HCCL_ALGO env are empty, skip filtering.");
            return HCCL_SUCCESS;
        }
    }
    HCCL_DEBUG("[FilterCmByHcclAlgo] use algo config: [%s]", algoConfig.c_str());

    //解析算法配置
    HcclAlgoParser algoParser;
    ret = algoParser.Parser(algoConfig);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[FilterCmByHcclAlgo] parse algo config [%s] failed, ret[%d].", algoConfig.c_str(), ret);
        return ret;
    }

    //刷新 CostModel
    ret = UpdateCostModelWithAlgo(algoParser, cm, engineTypes);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[FilterCmByHcclAlgo] UpdateCostModelWithAlgo failed, ret[%d].", ret);
        return ret;
    }

    HCCL_DEBUG("[FilterCmByHcclAlgo] filter costModel success.");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
