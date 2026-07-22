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

#include "sal.h"

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
// 递归下降解析器（内部实现类）
//
// 文法:
//   algo_config  := segment (';' segment)*
//   segment      := [opType ':'] executor_expr
//   executor_expr:= 'not' '(' executor_unit_or_atom ')' | executor_unit_or_atom
//   executor_unit_or_atom := executor_name '{' tpl_list '}' | template_name(shorthand)
//   tpl_list     := tpl_item (',' tpl_item)*
//   tpl_item     := ['level' digit '=' ] tpl_expr
//   tpl_expr     := 'not' '(' template_name ')' | template_name
//   template_name:= identifier (含数字/下划线/连字符，下划线→驼峰)
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

HcclResult HcclAlgoExecutorParser::Parser(const std::string &algoConfig)
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

std::string HcclAlgoExecutorParser::ToString() const
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

} // namespace ops_hccl
