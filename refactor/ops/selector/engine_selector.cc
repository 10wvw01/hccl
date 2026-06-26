#include "engine_selector.h"

EngineType Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    EngineType &engineType) {
        //根据环境变量、机型、数据量选出合适的引擎
}

// 单类别 require/reject 匹配
inline bool FieldMatch(u32 maskField, u32 requireField, u32 rejectField) {
    return (maskField & requireField) == requireField &&
           (maskField & rejectField) == 0;
}

// 全类别匹配
inline bool IsMatch(const EngineEntry& entry, const ConditionMask& mask) {
    // 四个类别独立匹配
    if (!FieldMatch(mask.netLayer,     entry.requireMask.netLayer,     entry.rejectMask.netLayer))     return false;
    if (!FieldMatch(mask.dataSize,     entry.requireMask.dataSize,     entry.rejectMask.dataSize))     return false;
    if (!FieldMatch(mask.physicalTopo, entry.requireMask.physicalTopo, entry.rejectMask.physicalTopo)) return false;
    if (!FieldMatch(mask.other,        entry.requireMask.other,        entry.rejectMask.other))        return false;

    // 跨类别复合排斥检查
    for (const auto& cr : entry.compoundRejects) {
        if (cr.checkA(mask) && cr.checkB(mask)) return false;
    }
    return true;
}

HcclResult EngineSelector::Select(
    const TopoInfoWithNetLayerDetails* topo,
    const SelectorParams& params,
    CommEngine& outEngine,
    OpExecuteConfig& outConfig) const
{
    ConditionMask mask = ComputeConditionMask(topo, params);

    const auto& chain = GetEngineChain(params.opType);
    for (const auto& entry : chain) {
        if (IsMatch(entry, mask)) {
            outEngine = entry.engine;
            outConfig = entry.config;
            HCCL_INFO("[EngineSelector] matched engine=%d, config=%d, %s",
                      static_cast<int>(outEngine), static_cast<int>(outConfig),
                      MaskToString(mask).c_str());
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[EngineSelector] no engine matched, %s", MaskToString(mask).c_str());
    return HCCL_E_NOT_SUPPORT;
}

ConditionMask EngineSelector::ComputeConditionMask(
    const TopoInfoWithNetLayerDetails* topo,
    const SelectorParams& params)
{
    ConditionMask mask = {};

    // ============================================================
    // 类别A: netLayer层级和范围
    // ============================================================
    if (topo->topoLevelNums <= 1) {
        mask.netLayer |= COND_LAYER_SINGLE;
    } else {
        mask.netLayer |= COND_LAYER_MULTI;
    }

    if (topo->netLayerDetails.netLayerNum > 1) {
        mask.netLayer |= COND_LAYER_HAS_LEVEL1;

        u32 level1Idx = topo->netLayerDetails.netLayers[1];
        if (topo->topoInstDetailsOfLayer.size() > level1Idx &&
            topo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.find(COMM_TOPO_CLOS) !=
                topo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.end()) {
            mask.netLayer |= COND_LAYER_LEVEL1_CLOS;
        }
    }

    // ============================================================
    // 类别B: 数据量大小
    // ============================================================
    u64 dataSize = params.dataCount * DATATYPE_SIZE_TABLE[params.dataType];
    if (dataSize < DATA_64KB) {
        mask.dataSize |= COND_DATA_SMALL;
    } else if (dataSize < DATA_512KB) {
        mask.dataSize |= COND_DATA_MEDIUM;
    } else if (dataSize < DATA_8MB) {
        mask.dataSize |= COND_DATA_LARGE;
    } else {
        mask.dataSize |= COND_DATA_HUGE;
    }

    // ============================================================
    // 类别C: 物理topo类型
    // ============================================================
    if (topo->level0Topo == Level0Shape::MESH_1D)
        mask.physicalTopo |= COND_TOPO_MESH_1D;
    else if (topo->level0Topo == Level0Shape::MESH_1D_CLOS)
        mask.physicalTopo |= COND_TOPO_MESH_1D_CLOS;
    else if (topo->level0Topo == Level0Shape::CLOS)
        mask.physicalTopo |= COND_TOPO_CLOS;

    bool isUbx = false;
    if (CheckClosNumMultipleOfMeshNum(topo, isUbx) == HCCL_SUCCESS && isUbx)
        mask.physicalTopo |= COND_TOPO_UBX;

    if (topo->level0PcieMix)
        mask.physicalTopo |= COND_TOPO_PCIE_MIX;

    if (topo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR)
        mask.physicalTopo |= COND_TOPO_TWO_DIE_REGULAR;
    else if (topo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR)
        mask.physicalTopo |= COND_TOPO_TWO_DIE_NONREG;

    if (topo->Level0Nhr || topo->Level1Nhr)
        mask.physicalTopo |= COND_TOPO_NHR;

    // ============================================================
    // 类别D: 其他条件
    // ============================================================
    if (IsInputOutputOverlap(params))
        mask.other |= COND_OTHER_IO_OVERLAP;

    if (params.userRankSize <= 4)
        mask.other |= COND_OTHER_RANK_LE_4;
    if (params.userRankSize <= 32)
        mask.other |= COND_OTHER_RANK_LE_32;

    bool dpuOk = false;
    if (CheckHostDPUOnly(params.hcclComm, topo, dpuOk) == HCCL_SUCCESS && dpuOk)
        mask.other |= COND_OTHER_DPU_OK;

    void* cclBufAddr;
    uint64_t cclBufSize;
    if (HcclGetHcclBuffer(params.hcclComm, &cclBufAddr, &cclBufSize) == HCCL_SUCCESS) {
        u64 totalSize = dataSize * params.userRankSize;
        if (totalSize <= cclBufSize * AIV_MAX_CCL_LOOP_NUM)
            mask.other |= COND_OTHER_AIV_BUF_OK;
    }

    if (params.userForcedConfig == OpExecuteConfig::AIV ||
        params.userForcedConfig == OpExecuteConfig::AIV_ONLY)
        mask.other |= COND_OTHER_AIV_ONLY;

    return mask;
}

const std::vector<EngineEntry>& EngineSelector::GetEngineChain(
    HcclCMDType opType, OpExecuteConfig userForcedConfig)
{
    // 辅助宏：构造 ConditionMask 字面量
    #define MASK(net, data, topo, oth) ConditionMask{net, data, topo, oth}

    // ============================================================
    // AIV / AIV_ONLY: 用户强制走 AIV 链 → 仅 AIV → AICPU
    // AIV 不满足条件时自动回退到 AICPU 兜底
    // ============================================================
    if (userForcedConfig == OpExecuteConfig::AIV ||
        userForcedConfig == OpExecuteConfig::AIV_ONLY) {
        static const std::vector<EngineEntry> aivChain = {
            // 优先级1: AIV
            {
                COMM_ENGINE_AIV, OpExecuteConfig::AIV,
                /* require */ MASK(0, 0, 0, COND_OTHER_AIV_BUF_OK),
                /* reject  */ MASK(0, 0, 0, 0),
            },
            // 优先级2: AICPU (AIV buffer 不足时兜底)
            {
                COMM_ENGINE_AICPU_TS, OpExecuteConfig::AICPU_TS,
                /* require */ MASK(0, 0, 0, 0),
                /* reject  */ MASK(0, 0, 0, 0),
            },
        };
        return aivChain;
    }

    // ============================================================
    // 正常链: 按 opType 返回对应的引擎优先级表
    // ============================================================
    static const std::map<HcclCMDType, std::vector<EngineEntry>> chains = {
        { HCCL_CMD_ALLGATHER, {
            // 优先级1: HOSTCPU (DPU模式)
            {
                COMM_ENGINE_CPU, OpExecuteConfig::HOSTCPU,
                /* require */ MASK(0, 0, 0, COND_OTHER_DPU_OK),
                /* reject  */ MASK(0, 0, 0, 0),
            },
            // 优先级2: CCU_MS
            //   要求: 单层 + (MESH_1D 或 MESH_1D_CLOS)
            //   排斥: IO重叠, pcie_mix, 不规则双die
            {
                COMM_ENGINE_CCU, OpExecuteConfig::CCU_MS,
                /* require */ MASK(COND_LAYER_SINGLE, 0,
                                   COND_TOPO_MESH_1D | COND_TOPO_MESH_1D_CLOS, 0),
                /* reject  */ MASK(0, 0,
                                   COND_TOPO_PCIE_MIX | COND_TOPO_TWO_DIE_NONREG,
                                   COND_OTHER_IO_OVERLAP),
            },
            // 优先级3: CCU_SCHED
            //   排斥: IO重叠, pcie_mix, 不规则双die,
            //         跨类别: (MESH_1D_CLOS 且 多层)
            {
                COMM_ENGINE_CCU, OpExecuteConfig::CCU_SCHED,
                /* require */ MASK(0, 0, 0, 0),
                /* reject  */ MASK(0, 0,
                                   COND_TOPO_PCIE_MIX | COND_TOPO_TWO_DIE_NONREG,
                                   COND_OTHER_IO_OVERLAP),
                /* compoundRejects */ {
                    { /* MESH_1D_CLOS 且 MULTI */
                      [](const ConditionMask& m) { return m.physicalTopo & COND_TOPO_MESH_1D_CLOS; },
                      [](const ConditionMask& m) { return m.netLayer & COND_LAYER_MULTI; }
                    },
                },
            },
            // 优先级4: AICPU (兜底，无约束)
            //   注意: 正常链路无 AIV，CCU_SCHED 失败直接跳到 AICPU
            {
                COMM_ENGINE_AICPU_TS, OpExecuteConfig::AICPU_TS,
                /* require */ MASK(0, 0, 0, 0),
                /* reject  */ MASK(0, 0, 0, 0),
            },
        }},
        // ... 其他算子类似定义
    };
    #undef MASK

    return chains.at(opType);
}