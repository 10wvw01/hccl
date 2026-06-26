EngineType Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    EngineType &engineType) {
        //根据环境变量、机型、数据量选出合适的引擎
}

//TODO: 对OpParam进一步进行正交分解，Selector仅选择必要信息对象作为参数


HcclResult EngineSelector::Select(
    const TopoInfoWithNetLayerDetails* topo,
    const SelectorParams& params,
    CommEngine& outEngine,
    OpExecuteConfig& outConfig) const
{
    u32 mask = ComputeConditionMask(topo, params);

    // 如果用户通过环境变量指定了引擎，直接返回，跳过优先级链
    if (params.userForcedConfig != OpExecuteConfig::DEFAULT) {
        outConfig = params.userForcedConfig;
        outEngine = ConfigToEngine(outConfig);
        return HCCL_SUCCESS;
    }

    const auto& chain = GetEngineChain(params.opType);
    for (const auto& entry : chain) {
        // 位运算：require全部满足 AND reject全部不命中
        if ((mask & entry.requireMask) == entry.requireMask &&
            (mask & entry.rejectMask) == 0) {
            outEngine = entry.engine;
            outConfig = entry.config;
            HCCL_INFO("[EngineSelector] matched engine=%d, config=%d, mask=0x%08x",
                      static_cast<int>(outEngine), static_cast<int>(outConfig), mask);
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[EngineSelector] no engine matched, mask=0x%08x", mask);
    return HCCL_E_NOT_SUPPORT;
}

u32 EngineSelector::ComputeConditionMask(
    const TopoInfoWithNetLayerDetails* topo,
    const SelectorParams& params)
{
    u32 mask = COND_NONE;

    // === 拓扑层级 ===
    if (topo->topoLevelNums <= 1) {
        mask |= COND_SINGLE_LEVEL;
    } else {
        mask |= COND_MULTI_LEVEL;
    }

    // === level0 拓扑形状 ===
    if (topo->level0Topo == Level0Shape::MESH_1D)       mask |= COND_MESH_1D;
    if (topo->level0Topo == Level0Shape::MESH_1D_CLOS)  mask |= COND_MESH_1D_CLOS;
    if (topo->level0Topo == Level0Shape::CLOS)           mask |= COND_CLOS;

    // === UBX机型 ===
    bool isUbx = false;
    if (CheckClosNumMultipleOfMeshNum(topo, isUbx) == HCCL_SUCCESS && isUbx) {
        mask |= COND_UBX;
    }

    // === PCIE混合 ===
    if (topo->level0PcieMix) mask |= COND_PCIE_MIX;

    // === IO重叠 ===
    if (IsInputOutputOverlap(params)) mask |= COND_IO_OVERLAP;

    // === 数据量 ===
    u64 dataSize = params.dataCount * DATATYPE_SIZE_TABLE[params.dataType];
    if (dataSize < SMALL_COUNT_512KB) mask |= COND_SMALL_DATA;
    else                              mask |= COND_LARGE_DATA;

    // === Mesh类型 ===
    if (topo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR)
        mask |= COND_TWO_DIE_REG;
    if (topo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR)
        mask |= COND_TWO_DIE_NONREG;

    // === NHR ===
    if (topo->Level0Nhr || topo->Level1Nhr) mask |= COND_NHR;

    // === rank数 ===
    if (params.userRankSize <= 4) mask |= COND_RANK_LE_4;
    if (params.userRankSize > MAX_RANK_SIZE) mask |= COND_RANK_GT_MAX;

    // === DPU ===
    bool dpuOk = false;
    if (CheckHostDPUOnly(params.hcclComm, topo, dpuOk) == HCCL_SUCCESS && dpuOk)
        mask |= COND_DPU_AVAILABLE;

    // === AIV Buffer ===
    void* cclBufAddr;
    uint64_t cclBufSize;
    if (HcclGetHcclBuffer(params.hcclComm, &cclBufAddr, &cclBufSize) == HCCL_SUCCESS) {
        u64 totalSize = dataSize * params.userRankSize;
        if (totalSize <= cclBufSize * AIV_MAX_CCL_LOOP_NUM)
            mask |= COND_AIV_BUF_OK;
    }

    // === Level1 CLOS ===
    if (topo->netLayerDetails.netLayerNum > 1) {
        u32 level1Idx = topo->netLayerDetails.netLayers[1];
        if (topo->topoInstDetailsOfLayer.size() > level1Idx &&
            topo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.find(COMM_TOPO_CLOS) !=
                topo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.end()) {
            mask |= COND_HAS_LEVEL1_CLOS;
        }
    }

    return mask;
}

const std::vector<EngineEntry>& EngineSelector::GetEngineChain(HcclCMDType opType)
{
    static const std::map<HcclCMDType, std::vector<EngineEntry>> chains = {
        { HCCL_CMD_ALLGATHER, {
            // 优先级1: HOSTCPU (DPU模式)
            {
                COMM_ENGINE_CPU, OpExecuteConfig::HOSTCPU,
                /* require */ COND_DPU_AVAILABLE,
                /* reject  */ COND_NONE
            },
            // 优先级2: CCU_MS (单层MESH，不允许重叠IO，不允许pcie_mix，不允许不规则双die)
            {
                COMM_ENGINE_CCU, OpExecuteConfig::CCU_MS,
                /* require */ COND_SINGLE_LEVEL | (COND_MESH_1D | COND_MESH_1D_CLOS),
                /* reject  */ COND_IO_OVERLAP | COND_PCIE_MIX | COND_TWO_DIE_NONREG
            },
            // 优先级3: CCU_SCHED (不允许重叠IO和pcie_mix)
            {
                COMM_ENGINE_CCU, OpExecuteConfig::CCU_SCHED,
                /* require */ COND_NONE,
                /* reject  */ COND_IO_OVERLAP | COND_PCIE_MIX | COND_TWO_DIE_NONREG
                             | (COND_MESH_1D_CLOS & COND_MULTI_LEVEL)
            },
            // 优先级4: AIV
            {
                COMM_ENGINE_AIV, OpExecuteConfig::AIV,
                /* require */ COND_AIV_BUF_OK,
                /* reject  */ COND_RANK_GT_MAX
            },
            // 优先级5: AICPU (兜底，无约束)
            {
                COMM_ENGINE_AICPU_TS, OpExecuteConfig::AICPU_TS,
                /* require */ COND_NONE,
                /* reject  */ COND_NONE
            },
        }},
        // ... 其他算子类似定义
    };

    return chains.at(opType);
}