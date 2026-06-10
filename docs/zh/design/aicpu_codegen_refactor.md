# AICPU 后端「注册层代码生成」重构设计

> 状态：试点(Pilot)进行中 · 范围：AICPU 后端 · 首个落地：`all_gather`
> 关联分支：`feat/aicpu-codegen-allgather`

## 0. 背景与动机

HCCL 每个算子的「算法变体」是如下元组的一个取值：

```
(opType, algName, ExecutorTemplate, TopoMatch, Tmpl0, Tmpl1[, Tmpl2, Tmpl3])
```

它当前以手写的注册宏，散落在各 executor `.cc` 末尾，例如
`src/ops/all_gather/executor/ins_v2_all_gather_sequence_executor_aicpu.cc`：

```cpp
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER,
                               InsAllGatherSequenceNHRMesh1D,            // algName 字符串
                               InsV2AllGatherSequenceExecutorAicpu,      // 执行器模板
                               TopoMatchMultilevel,                      // 形状/拓扑
                               InsTempAllGatherMesh1D1DZAxisDetour,      // 框内模板
                               InsTempAllGatherNHR);                     // 框间模板
```

注册机制(`ops/op_common/executor/registry/coll_alg_v2_exec_registry.h`)把
`(opType, algName) → []{ return new Executor<TopoMatch, Tmpl...>(); }` 塞进一张全局表；
运行时 selector 产出一个 `algName` 字符串，`GetAlgExec` 查表实例化执行器。

### 现状量化(实仓统计)

| 项 | 数量 |
|---|---|
| 被注册的算法变体(去 helper) | ≈ 40(`Ins*`/AICPU **19** + `Ccu*`/CCU 21) |
| AICPU template(`InsTemp*`,`.cc`/`.h`) | 各 47,分布 9 个 op |
| executor 模板类 | 52(被复用的 C++ 模板,非每组合一个) |

### 现状最大痛点：`algName` 是「双份手写真值」

selector 里(`all_gather_auto_selector.cc:247`)：

```cpp
selectAlgName = (dataSize*rankSize > 阈值) ? "InsAllGatherSequenceNHRMesh1D"
                                           : "InsAllGatherParallelMesh1DNHR";
```

注册宏里又写一遍同一个字符串。两处靠人肉对齐 —— 对不上就在运行时报
`Fail to find executor for algName[...]` 或静默回退。这是结构性的 bug 来源。

## 1. 范围与边界(为什么先做 AICPU)

- **边界天然清晰**：AICPU 变体的模板族统一是 `InsTemp*`/算法名 `Ins*` 前缀,与 `Ccu*`(CCU)、AIV 物理隔离,只动 19 条注册。
- **零风险增量**：注册表是运行时「字符串 → 创建器」的 map,生成代码只是「往 map 里多塞条目」,可与手写宏共存,出错就关掉开关回退手写。
- **验证闭环已打通**：`build.sh --pkg` → 安装 `.run` 覆盖 CANN → `build.sh --st`(190 用例)可做二进制级回归基线。

## 2. 明确「不做」什么

- **不生成算法本体**。`KernelRun`/`CalcRes`/`Orchestrate` 是真正的 IP,47 个 AICPU 模板绝大多数是不同算法,逐个手写合理。所以「自动生成 334 个文件」不成立。
- **不改 selector 决策逻辑**。本期只把「selector 产出的名字」变成生成常量(单一真值源),决策树本身不动。

## 3. 要生成的产物

| 产物 | 取代现状 | 价值 |
|---|---|---|
| ① `alg_names.h`(algName 常量) | 散落字符串字面量 | **单一真值源**,消灭第 0 节的对齐 bug |
| ② `<op>_aicpu_reg.cc`(生成的 `Register()` 调用) | executor.cc 末尾手写宏 | 「加一行配置」代替「改 .cc + 手对字符串」 |
| ③ 执行器显式实例化 `template class Executor<...>;` | 宏隐式触发的实例化 | 保证符号落在 `libhccl.so`,控制代码膨胀 |

## 4. spec 形态

每个 op 一个 `src/ops/<op>/aicpu.spec.yaml`：

```yaml
op: all_gather
cmd: HCCL_CMD_ALLGATHER
includes:
  - topo_match_multilevel.h
  - ins_v2_all_gather_sequence_executor_aicpu.h
  - ins_temp_all_gather_mesh_1D_Z_axis_detour.h
  - ins_temp_all_gather_nhr.h
variants:
  - name: InsAllGatherSequenceNHRMesh1D
    executor: InsV2AllGatherSequenceExecutorAicpu
    topo:     TopoMatchMultilevel
    templates: [InsTempAllGatherMesh1D1DZAxisDetour, InsTempAllGatherNHR]
```

> spec 不是凭空写 —— 直接从现有 19 条 `Ins*` 注册宏机械抽取即可。

## 5. 生成器

`tools/codegen/gen_aicpu.py`(仿 NCCL `generate.py`),输入 spec,输出到
`${CMAKE_BINARY_DIR}/generated/aicpu/`：

- `alg_names.h`：`constexpr char <name>[] = "<name>";`(C++14,namespace 作用域 constexpr 即内部链接,各 TU 私有拷贝,ODR 安全)。
- `<op>_aicpu_reg.cc`：对每个 variant 生成一条静态注册：

```cpp
static HcclResult g_reg_<op>_<name> = CollAlgExecRegistryV2::Instance().Register(
    HcclCMDType::<CMD>, std::string(algnames::<name>),
    DefaultExecCreatorV2<Executor<Topo, Tmpl0, Tmpl1...>>);
```

模板个数(1/2/4)决定展开形式。

## 6. CMake 接入

在 `src/ops/<op>/CMakeLists.txt`(本试点为 all_gather)：

```cmake
option(HCCL_AICPU_CODEGEN "Use generated AICPU registrations" ON)
if(HCCL_AICPU_CODEGEN)
  set(AICPU_GEN_DIR ${CMAKE_BINARY_DIR}/generated/aicpu)
  add_custom_command(
    OUTPUT ${AICPU_GEN_DIR}/alg_names.h ${AICPU_GEN_DIR}/all_gather_aicpu_reg.cc
    COMMAND ${HI_PYTHON} ${OPS_BASE_DIR}/tools/codegen/gen_aicpu.py
            --out ${AICPU_GEN_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/aicpu.spec.yaml
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/aicpu.spec.yaml ${OPS_BASE_DIR}/tools/codegen/gen_aicpu.py)
  add_custom_target(gen_aicpu_all_gather DEPENDS ${AICPU_GEN_DIR}/all_gather_aicpu_reg.cc)
  add_dependencies(hccl gen_aicpu_all_gather)
  target_sources(hccl PRIVATE ${AICPU_GEN_DIR}/all_gather_aicpu_reg.cc)
  target_include_directories(hccl PRIVATE ${AICPU_GEN_DIR})
  target_compile_definitions(hccl PRIVATE HCCL_AICPU_CODEGEN)
endif()
```

`INCLUDE_LIST` 已包含所有 op 子目录(executor/template/aicpu/topo/registry),生成的 `.cc` 加入 `hccl` 目标即可解析全部头文件。

## 7. executor.cc 的最小改动

把手写宏用开关包住；开启 codegen 时改为显式实例化(保证符号仍在本 TU)：

```cpp
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#ifdef HCCL_AICPU_CODEGEN
template class InsV2AllGatherSequenceExecutorAicpu<TopoMatchMultilevel,
    InsTempAllGatherMesh1D1DZAxisDetour, InsTempAllGatherNHR>;   // 注册由生成文件负责
#else
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherSequenceNHRMesh1D,
    InsV2AllGatherSequenceExecutorAicpu, TopoMatchMultilevel,
    InsTempAllGatherMesh1D1DZAxisDetour, InsTempAllGatherNHR);
#endif
#endif
```

> 链接关系：生成 `reg.cc` 里的 `new Executor<...>()` 需要执行器的 vtable/虚函数符号,这些由
> executor.cc 的**显式实例化**提供;`reg.cc` 只 include 头文件即可。

## 8. 迁移路径(增量 / 可回滚 / 每步 ST 验证)

- **Pilot(本分支)**：只迁 `all_gather` 的 `InsAllGatherSequenceNHRMesh1D` 一条。
  1. 写 `all_gather/aicpu.spec.yaml` + 生成器 + CMake 接入。
  2. executor.cc 宏改为开关 + 显式实例化。
  3. 验证：`build.sh --pkg` → 安装 → `build.sh --st`,期望仍 **190 全过**;`nm -DC libhccl.so` 比对执行器符号一致。
- **铺开**：每个 op 一个 spec/PR,同一套生成器,逐个迁完 19 条。
- **Phase 1b**：selector 字面量换成 `algnames::*` 常量(需 `alg_names.h` 无条件生成),彻底消灭对齐 bug;并加 CI 校验「selector 出现的每个 `Ins*` 名都在 spec 里」。
- **Phase 2(可选,LOC 大头)**：op 入口的逐参数 `RPT_INPUT_ERR/CHK_PTR_NULL`、entry-log、tag 拼装样板也从 op-spec 生成。

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| 重复注册(`already registered`) | 开关二选一,绝不同时编译宏与生成文件 |
| 实例化落错 .so | ③ 显式实例化在 executor.cc(编进 hccl);`nm` 校验 |
| 生成符号冲突 | 生成器给静态变量名拼 `op+name` 唯一后缀 |
| spec 与本体脱节(改了模板签名) | spec 只列类名,签名仍由 C++ 编译期检查,编不过立刻暴露 |

## 10. 本期收益(诚实版)

- 不是砍几万行(算法本体不动)。
- 是：**散落注册 + 双份 algName → 一份 spec**;新增 AICPU 变体从「改 .cc + 手对字符串」变成「spec 加一行」;为 CCU/AIV 复用同一生成器、及 Phase 2 入口样板生成铺路。
