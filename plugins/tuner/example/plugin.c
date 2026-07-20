/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* HCCL Tuner Plugin 示例实现。
 *
 * 功能：读取 JSON 配置文件，按 match 条件匹配集合通信调用，修改 3D cost table
 * （engine[5] × executor[7] × template[16] = 560 floats）以影响 Selector 算法选择。
 *
 * 编译：make（链接为 hccl_tuner_example.so）
 * 使用：export HCCL_TUNER_PLUGIN=/path/to/hccl_tuner_example.so
 *       export HCCL_TUNER_CONFIG_FILE=/path/to/hccl_tuner_config.json
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hccl_tuner_plugin.h"

/* ===== 常量 ===== */
#define MAX_RULES 64
#define MAX_OP_TYPES 8
#define MAX_STR_LEN 64
#define MAX_FILE_SIZE (256 * 1024)
#define COMM_NAME_BUF_LEN 128

/* ===== 数据结构 ===== */
typedef struct {
    int hasMinRanks;
    int hasMaxRanks;
    int hasMinBytes;
    int hasMaxBytes;
    int hasDataType;
    int hasCommName;
    int hasMinNpus;
    int hasMaxNpus;
    int hasMinServers;
    uint32_t minRanks;
    uint32_t maxRanks;
    size_t minBytes;
    size_t maxBytes;
    char dataType[MAX_STR_LEN];
    char commName[MAX_STR_LEN];
    uint32_t minNpus;
    uint32_t maxNpus;
    uint32_t minServers;
} MatchCond;

typedef struct {
    MatchCond match;
    int engine;
    int executor;
    int tmpl;
    float cost;
    int hasCost;
} Rule;

typedef struct {
    hcclOpType_t opType;
    Rule rules[MAX_RULES];
    int ruleCount;
} OpRuleSet;

/* per-comm 存储上下文（经 hostFuncs->ctxCreate 持久化到通信域 host 内存） */
typedef struct {
    OpRuleSet opSets[MAX_OP_TYPES];
    int opSetCount;
    hcclTunerCommInfo_t commInfo;
    char commNameBuf[COMM_NAME_BUF_LEN];
} StoredContext;

/* ===== 全局 hostFuncs（函数表，全进程相同）===== */
static hcclTunerHostFunctions_t g_hostFuncs;
static int g_hostFuncsReady = 0;

/* ===== 算子类型 / 数据类型字符串映射 ===== */
static const struct {
    const char *name;
    hcclOpType_t type;
} g_opTypeMap[] = {
    {"allreduce", HCCL_OP_ALLREDUCE},     {"allgather", HCCL_OP_ALLGATHER},     {"broadcast", HCCL_OP_BROADCAST},
    {"reduce", HCCL_OP_REDUCE},           {"reduce_scatter", HCCL_OP_REDUCE_SCATTER},
    {"scatter", HCCL_OP_SCATTER},         {"alltoall", HCCL_OP_ALLTOALL},       {"alltoallv", HCCL_OP_ALLTOALLV},
};

static const struct {
    const char *name;
    HcclDataType type;
} g_dataTypeMap[] = {
    {"int8", HCCL_DATA_TYPE_INT8},         {"int16", HCCL_DATA_TYPE_INT16},       {"int32", HCCL_DATA_TYPE_INT32},
    {"int64", HCCL_DATA_TYPE_INT64},       {"uint8", HCCL_DATA_TYPE_UINT8},       {"uint16", HCCL_DATA_TYPE_UINT16},
    {"uint32", HCCL_DATA_TYPE_UINT32},     {"uint64", HCCL_DATA_TYPE_UINT64},     {"fp16", HCCL_DATA_TYPE_FP16},
    {"float16", HCCL_DATA_TYPE_FP16},      {"fp32", HCCL_DATA_TYPE_FP32},         {"float32", HCCL_DATA_TYPE_FP32},
    {"fp64", HCCL_DATA_TYPE_FP64},         {"float64", HCCL_DATA_TYPE_FP64},     {"bfp16", HCCL_DATA_TYPE_BFP16},
    {"bfloat16", HCCL_DATA_TYPE_BFP16},
};

static hcclOpType_t LookupOpType(const char *name)
{
    for (size_t i = 0; i < sizeof(g_opTypeMap) / sizeof(g_opTypeMap[0]); i++) {
        if (strcmp(name, g_opTypeMap[i].name) == 0) {
            return g_opTypeMap[i].type;
        }
    }
    return HCCL_OP_INVALID;
}

static HcclDataType ParseDataType(const char *name)
{
    for (size_t i = 0; i < sizeof(g_dataTypeMap) / sizeof(g_dataTypeMap[0]); i++) {
        if (strcmp(name, g_dataTypeMap[i].name) == 0) {
            return g_dataTypeMap[i].type;
        }
    }
    return HCCL_DATA_TYPE_RESERVED;
}

/* ===== 极简 JSON 解析器（仅支持本插件配置格式，无外部依赖）===== */
typedef struct {
    const char *json;
    size_t pos;
    size_t len;
} JsonParser;

static void JsonSkipWs(JsonParser *p)
{
    while (p->pos < p->len && isspace((unsigned char)p->json[p->pos])) {
        p->pos++;
    }
}

static int JsonMatch(JsonParser *p, char c)
{
    JsonSkipWs(p);
    if (p->pos < p->len && p->json[p->pos] == c) {
        p->pos++;
        return 1;
    }
    return 0;
}

static int JsonPeek(JsonParser *p)
{
    JsonSkipWs(p);
    return (p->pos < p->len) ? (unsigned char)p->json[p->pos] : -1;
}

static int JsonReadString(JsonParser *p, char *buf, size_t bufSize)
{
    JsonSkipWs(p);
    if (p->pos >= p->len || p->json[p->pos] != '"') {
        return -1;
    }
    p->pos++;
    size_t i = 0;
    while (p->pos < p->len && p->json[p->pos] != '"') {
        if (i + 1 < bufSize) {
            buf[i++] = p->json[p->pos];
        }
        p->pos++;
    }
    if (p->pos < p->len) {
        p->pos++;
    }
    buf[i] = '\0';
    return (int)i;
}

static int JsonReadNumber(JsonParser *p, double *out)
{
    JsonSkipWs(p);
    size_t start = p->pos;
    if (p->pos < p->len && (p->json[p->pos] == '-' || p->json[p->pos] == '+')) {
        p->pos++;
    }
    int hasDigit = 0;
    while (p->pos < p->len && (isdigit((unsigned char)p->json[p->pos]) || p->json[p->pos] == '.')) {
        p->pos++;
        hasDigit = 1;
    }
    if (!hasDigit) {
        return -1;
    }
    char numBuf[64] = {0};
    size_t n = p->pos - start;
    if (n >= sizeof(numBuf)) {
        n = sizeof(numBuf) - 1;
    }
    memcpy(numBuf, p->json + start, n);
    *out = strtod(numBuf, NULL);
    return 0;
}

static void JsonSkipValue(JsonParser *p)
{
    int peek = JsonPeek(p);
    if (peek == '"') {
        char buf[MAX_STR_LEN] = {0};
        JsonReadString(p, buf, sizeof(buf));
    } else if (peek == '{' || peek == '[') {
        int depth = 0;
        while (p->pos < p->len) {
            char c = p->json[p->pos++];
            if (c == '{' || c == '[') {
                depth++;
            } else if (c == '}' || c == ']') {
                depth--;
                if (depth == 0) {
                    break;
                }
            }
        }
    } else {
        double v = 0;
        JsonReadNumber(p, &v);
    }
}

static void ParseMatchField(JsonParser *p, const char *key, Rule *r)
{
    double v = 0;
    char buf[MAX_STR_LEN] = {0};
    if (strcmp(key, "min_ranks") == 0) {
        if (JsonReadNumber(p, &v) == 0) {
            r->match.hasMinRanks = 1;
            r->match.minRanks = (uint32_t)v;
        }
    } else if (strcmp(key, "max_ranks") == 0) {
        if (JsonReadNumber(p, &v) == 0) {
            r->match.hasMaxRanks = 1;
            r->match.maxRanks = (uint32_t)v;
        }
    } else if (strcmp(key, "min_bytes") == 0) {
        if (JsonReadNumber(p, &v) == 0) {
            r->match.hasMinBytes = 1;
            r->match.minBytes = (size_t)v;
        }
    } else if (strcmp(key, "max_bytes") == 0) {
        if (JsonReadNumber(p, &v) == 0) {
            r->match.hasMaxBytes = 1;
            r->match.maxBytes = (size_t)v;
        }
    } else if (strcmp(key, "data_type") == 0) {
        if (JsonReadString(p, buf, sizeof(buf)) > 0) {
            strncpy(r->match.dataType, buf, sizeof(r->match.dataType) - 1);
            r->match.hasDataType = 1;
        }
    } else if (strcmp(key, "comm_name") == 0) {
        if (JsonReadString(p, buf, sizeof(buf)) > 0) {
            strncpy(r->match.commName, buf, sizeof(r->match.commName) - 1);
            r->match.hasCommName = 1;
        }
    } else if (strcmp(key, "min_npus") == 0) {
        if (JsonReadNumber(p, &v) == 0) {
            r->match.hasMinNpus = 1;
            r->match.minNpus = (uint32_t)v;
        }
    } else if (strcmp(key, "max_npus") == 0) {
        if (JsonReadNumber(p, &v) == 0) {
            r->match.hasMaxNpus = 1;
            r->match.maxNpus = (uint32_t)v;
        }
    } else if (strcmp(key, "min_servers") == 0) {
        if (JsonReadNumber(p, &v) == 0) {
            r->match.hasMinServers = 1;
            r->match.minServers = (uint32_t)v;
        }
    } else {
        JsonSkipValue(p);
    }
}

static void ParseRule(JsonParser *p, Rule *r)
{
    memset(r, 0, sizeof(*r));
    r->engine = -1;
    r->executor = -1;
    r->tmpl = -1;
    if (!JsonMatch(p, '{')) {
        return;
    }
    while (p->pos < p->len) {
        if (JsonMatch(p, '}')) {
            break;
        }
        JsonMatch(p, ',');
        char key[MAX_STR_LEN] = {0};
        if (JsonReadString(p, key, sizeof(key)) <= 0) {
            break;
        }
        JsonMatch(p, ':');
        if (strcmp(key, "match") == 0) {
            if (JsonMatch(p, '{')) {
                while (p->pos < p->len) {
                    if (JsonMatch(p, '}')) {
                        break;
                    }
                    JsonMatch(p, ',');
                    char mkey[MAX_STR_LEN] = {0};
                    if (JsonReadString(p, mkey, sizeof(mkey)) <= 0) {
                        break;
                    }
                    JsonMatch(p, ':');
                    ParseMatchField(p, mkey, r);
                }
            }
        } else if (strcmp(key, "engine") == 0) {
            double v = 0;
            if (JsonReadNumber(p, &v) == 0) {
                r->engine = (int)v;
            }
        } else if (strcmp(key, "executor") == 0) {
            double v = 0;
            if (JsonReadNumber(p, &v) == 0) {
                r->executor = (int)v;
            }
        } else if (strcmp(key, "template") == 0) {
            double v = 0;
            if (JsonReadNumber(p, &v) == 0) {
                r->tmpl = (int)v;
            }
        } else if (strcmp(key, "cost") == 0) {
            double v = 0;
            if (JsonReadNumber(p, &v) == 0) {
                r->cost = (float)v;
                r->hasCost = 1;
            }
        } else {
            JsonSkipValue(p);
        }
    }
}

static void ParseOpRules(JsonParser *p, StoredContext *ctx, const char *opName)
{
    hcclOpType_t opType = LookupOpType(opName);
    if (opType == HCCL_OP_INVALID || ctx->opSetCount >= MAX_OP_TYPES) {
        JsonSkipValue(p);
        return;
    }
    OpRuleSet *set = &ctx->opSets[ctx->opSetCount];
    set->opType = opType;
    set->ruleCount = 0;
    if (!JsonMatch(p, '{')) {
        return;
    }
    while (p->pos < p->len) {
        if (JsonMatch(p, '}')) {
            break;
        }
        JsonMatch(p, ',');
        char key[MAX_STR_LEN] = {0};
        if (JsonReadString(p, key, sizeof(key)) <= 0) {
            break;
        }
        JsonMatch(p, ':');
        if (strcmp(key, "rules") == 0) {
            if (JsonMatch(p, '[')) {
                while (p->pos < p->len) {
                    if (JsonMatch(p, ']')) {
                        break;
                    }
                    JsonMatch(p, ',');
                    if (set->ruleCount < MAX_RULES) {
                        ParseRule(p, &set->rules[set->ruleCount]);
                        set->ruleCount++;
                    } else {
                        JsonSkipValue(p);
                    }
                }
            }
        } else {
            JsonSkipValue(p);
        }
    }
    ctx->opSetCount++;
}

static void ParseConfig(JsonParser *p, StoredContext *ctx)
{
    if (!JsonMatch(p, '{')) {
        return;
    }
    while (p->pos < p->len) {
        if (JsonMatch(p, '}')) {
            break;
        }
        JsonMatch(p, ',');
        char key[MAX_STR_LEN] = {0};
        if (JsonReadString(p, key, sizeof(key)) <= 0) {
            break;
        }
        JsonMatch(p, ':');
        if (strcmp(key, "op_types") == 0) {
            if (JsonMatch(p, '{')) {
                while (p->pos < p->len) {
                    if (JsonMatch(p, '}')) {
                        break;
                    }
                    JsonMatch(p, ',');
                    char opName[MAX_STR_LEN] = {0};
                    if (JsonReadString(p, opName, sizeof(opName)) <= 0) {
                        break;
                    }
                    JsonMatch(p, ':');
                    ParseOpRules(p, ctx, opName);
                }
            }
        } else {
            JsonSkipValue(p);
        }
    }
}

/* ===== 配置文件加载（多级 fallback）===== */
static char *ReadFile(const char *path, size_t *outLen)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return NULL;
    }
    (void)fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    (void)fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz > MAX_FILE_SIZE) {
        fclose(fp);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (buf == NULL) {
        fclose(fp);
        return NULL;
    }
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[rd] = '\0';
    *outLen = rd;
    return buf;
}

static int LoadConfig(StoredContext *ctx)
{
    const char *envPath = getenv("HCCL_TUNER_CONFIG_FILE");
    const char *paths[] = {envPath, "./hccl_tuner_config.json", "/etc/hccl/hccl_tuner_config.json"};
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        if (paths[i] == NULL || paths[i][0] == '\0') {
            continue;
        }
        size_t len = 0;
        char *content = ReadFile(paths[i], &len);
        if (content != NULL) {
            JsonParser p = {content, 0, len};
            ParseConfig(&p, ctx);
            free(content);
            if (g_hostFuncsReady && g_hostFuncs.logFunction != NULL) {
                g_hostFuncs.logFunction(HCCL_TUNER_LOG_INFO, __FILE__, __LINE__,
                                        "tuner config loaded from %s, opSetCount=%d", paths[i], ctx->opSetCount);
            }
            return 1;
        }
    }
    return 0;
}

/* ===== 规则匹配 ===== */
static int MatchRule(const Rule *r, const hcclTunerCollInfo_t *collInfo, const hcclTunerCommInfo_t *commInfo)
{
    if (r->match.hasMinRanks && commInfo->nRanks < r->match.minRanks) {
        return 0;
    }
    if (r->match.hasMaxRanks && commInfo->nRanks > r->match.maxRanks) {
        return 0;
    }
    if (r->match.hasMinBytes && collInfo->nBytes < r->match.minBytes) {
        return 0;
    }
    if (r->match.hasMaxBytes && collInfo->nBytes > r->match.maxBytes) {
        return 0;
    }
    if (r->match.hasDataType) {
        if (ParseDataType(r->match.dataType) != collInfo->dataType) {
            return 0;
        }
    }
    if (r->match.hasCommName) {
        if (commInfo->commName == NULL || strstr(commInfo->commName, r->match.commName) == NULL) {
            return 0;
        }
    }
    if (r->match.hasMinNpus && commInfo->nNpusPerServer < r->match.minNpus) {
        return 0;
    }
    if (r->match.hasMaxNpus && commInfo->nNpusPerServer > r->match.maxNpus) {
        return 0;
    }
    if (r->match.hasMinServers && commInfo->nServers < r->match.minServers) {
        return 0;
    }
    return 1;
}

static void ApplyRule(const Rule *r, float *costTable)
{
    if (r->engine >= 0 && r->engine < HCCL_NUM_ENGINES && r->executor >= 0 && r->executor < HCCL_NUM_EXECUTORS &&
        r->tmpl >= 0 && r->tmpl < HCCL_NUM_TEMPLATES) {
        int idx =
            r->engine * HCCL_NUM_EXECUTORS * HCCL_NUM_TEMPLATES + r->executor * HCCL_NUM_TEMPLATES + r->tmpl;
        costTable[idx] = r->hasCost ? r->cost : 0.0f;
    }
}

/* ===== 插件接口实现 ===== */
static HcclResult MyInit(HcclComm comm, const hcclTunerCommInfo_t *commInfo, const hcclTunerHostFunctions_t *hostFuncs)
{
    if (commInfo == NULL || hostFuncs == NULL) {
        return HCCL_E_PTR;
    }
    /* hostFuncs 是函数表，全进程相同，存全局 */
    g_hostFuncs = *hostFuncs;
    g_hostFuncsReady = 1;

    /* 分配并填充 StoredContext */
    StoredContext *tmp = (StoredContext *)calloc(1, sizeof(StoredContext));
    if (tmp == NULL) {
        return HCCL_E_INTERNAL;
    }
    if (!LoadConfig(tmp)) {
        if (g_hostFuncs.logFunction != NULL) {
            g_hostFuncs.logFunction(HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "no tuner config loaded, plugin inactive");
        }
        free(tmp);
        return HCCL_SUCCESS;
    }
    /* 拷贝 commInfo（commName 需拷贝到持久缓冲，init 返回后 commInfo 指针失效） */
    tmp->commInfo = *commInfo;
    if (commInfo->commName != NULL) {
        strncpy(tmp->commNameBuf, commInfo->commName, sizeof(tmp->commNameBuf) - 1);
        tmp->commInfo.commName = tmp->commNameBuf;
    }
    /* 持久化到通信域 host 内存（经 __tuner_ 前缀的 ctxCreate） */
    void *storedCtx = NULL;
    if (hostFuncs->ctxCreate(comm, "main", sizeof(StoredContext), &storedCtx) != HCCL_SUCCESS || storedCtx == NULL) {
        if (g_hostFuncs.logFunction != NULL) {
            g_hostFuncs.logFunction(HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "ctxCreate failed, plugin inactive");
        }
        free(tmp);
        return HCCL_SUCCESS;
    }
    memcpy(storedCtx, tmp, sizeof(StoredContext));
    free(tmp);
    if (g_hostFuncs.logFunction != NULL) {
        g_hostFuncs.logFunction(HCCL_TUNER_LOG_INFO, __FILE__, __LINE__,
                                "tuner init done, comm[%p] nRanks[%u] opSetCount[%d]", comm, commInfo->nRanks,
                                ((StoredContext *)storedCtx)->opSetCount);
    }
    return HCCL_SUCCESS;
}

static HcclResult MyGetCollInfo(HcclComm comm, const hcclTunerCollInfo_t *collInfo, float *collCostTable)
{
    if (collInfo == NULL || collCostTable == NULL) {
        return HCCL_E_PTR;
    }
    if (!g_hostFuncsReady || g_hostFuncs.ctxGet == NULL) {
        return HCCL_SUCCESS;
    }
    void *ctxPtr = NULL;
    uint64_t ctxSize = 0;
    if (g_hostFuncs.ctxGet(comm, "main", &ctxPtr, &ctxSize) != HCCL_SUCCESS || ctxPtr == NULL) {
        return HCCL_SUCCESS;
    }
    StoredContext *ctx = (StoredContext *)ctxPtr;

    for (int i = 0; i < ctx->opSetCount; i++) {
        OpRuleSet *set = &ctx->opSets[i];
        if (set->opType != collInfo->collType) {
            continue;
        }
        for (int j = 0; j < set->ruleCount; j++) { /* first-match-wins */
            if (MatchRule(&set->rules[j], collInfo, &ctx->commInfo)) {
                ApplyRule(&set->rules[j], collCostTable);
                return HCCL_SUCCESS;
            }
        }
    }
    return HCCL_SUCCESS;
}

/* ===== 插件描述符与函数表 ===== */
hcclPluginDescriptor_t hcclTunerPlugin = {HCCL_TUNER_API_VERSION, 0, "HCCL Tuner Example", 1};

HcclResult hcclTunerGetFuncs(hcclTunerFuncs_t *funcs)
{
    if (funcs == NULL) {
        return HCCL_E_PTR;
    }
    funcs->init = MyInit;
    funcs->getCollInfo = MyGetCollInfo;
    return HCCL_SUCCESS;
}

/* 供 test_plugin.c 通过 #include "../plugin.c" 方式单元测试 */
#ifdef HCCL_TUNER_TESTING
StoredContext *TunerGetStoredCtx(HcclComm comm)
{
    if (!g_hostFuncsReady || g_hostFuncs.ctxGet == NULL) {
        return NULL;
    }
    void *ctxPtr = NULL;
    uint64_t ctxSize = 0;
    if (g_hostFuncs.ctxGet(comm, "main", &ctxPtr, &ctxSize) != HCCL_SUCCESS) {
        return NULL;
    }
    return (StoredContext *)ctxPtr;
}
#endif
