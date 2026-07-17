# Refactor CMake 方案 · 交接文档

> 给接手 refactor CMake 工作的 agent / 同事。
> 创建时间：2026-06-29
> 工作分支：`refactor`
> 涉及模块：refactor/ops/{api, c_adaptor, executor, op_common, selector, template, utils}

---

## 1. 背景与目标

`refactor/` 下是从 `src/ops/` 搬出来重构的新代码（按 `分工.txt` 安排），最终目的是替代 `src/ops/`。

**问题**：
- refactor 路径下大量文件还是骨架/伪代码/空文件，直接 `target_sources(hccl ...)` 会让主构建挂掉
- 不同人改不同模块，进度独立
- 仓库原有 `hccl` target 是在 `src/hccl.cmake` 里通过 `add_library(hccl SHARED)` 创建的
- 仓库有大量已有的 CMake 基础设施（fetch_cann_cmake、config.cmake、package.cmake 等），重写代价大

**目标**：让 refactor 路径能**渐进式接入主构建**，每个模块独立可控，骨架文件不影响其他人。

---

## 2. 现有 CMake 体系（5 层结构）

```
第 0 层  CMakeLists.txt                  入口：定义选项 + 模式分发
第 1 层  cmake/*.cmake                   基础设施：变量、函数、工具
第 2 层  src/CMakeLists.txt              编排：INCLUDE_LIST + 引入子目录
第 3 层  src/hccl.cmake                  主体目标：add_library(hccl SHARED)
第 4 层  refactor/.../CMakeLists.txt     你的代码：target_sources(hccl ...)
```

**关键约束**：第 4 层任何 `target_sources(hccl ...)` 都必须在第 3 层（`src/hccl.cmake`）的 `add_library(hccl ...)` 之后执行。

---

## 3. 落地方案

**3 个改动 + 7 个新文件 = 整套接入**。

### 改动 1 · 顶层 `CMakeLists.txt` 加开关（已推）

```cmake
option(REFACTOR_OPS "Enable refactored ops path (refactor/ops/)" OFF)
```

- **位置**：第 0 层 `CMakeLists.txt:16`，紧接 `ENABLE_BUILD_AARCH` 之后
- **默认 OFF**：refactor 路径默认不参与构建，团队无感知
- **commit**：`a50fd9e build: add REFACTOR_OPS option to gate refactored ops path`

### 改动 2 · `src/CMakeLists.txt` 加 include 守卫（已推）

```cmake
# refactor 路径接入（由顶层 REFACTOR_OPS 开关控制）
# 必须在 include(hccl.cmake) 之后，因为下面的 CMakeLists 会 target_sources(hccl ...)
if(REFACTOR_OPS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../refactor/CMakeLists.txt)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../refactor/CMakeLists.txt)
endif()
```

- **位置**：`src/CMakeLists.txt:301`，紧接 `add_subdirectory(ops)` 之后
- **EXISTS 守卫**：兼容 main 分支（没有 refactor/ 目录时不报错）
- **commit**：`36ae538 build: wire refactor path into src/CMakeLists with REFACTOR_OPS gate`

### 改动 3 · 新建 `refactor/CMakeLists.txt`（已推）

这是 refactor 自己的"第 3 层编排器"，结构：

```cmake
# 6 个模块独立 option，默认全 OFF
option(REFACTOR_API       "..." OFF)
option(REFACTOR_EXECUTOR  "..." OFF)
option(REFACTOR_OP_COMMON "..." OFF)
option(REFACTOR_SELECTOR  "..." OFF)
option(REFACTOR_TEMPLATE  "..." OFF)
option(REFACTOR_UTILS     "..." OFF)

# c_adaptor/common 是"直接拷贝"稳定代码，无 option，无条件
add_subdirectory(ops/c_adaptor/common)

# 各模块：option=ON 且 CMakeLists 存在时才进入
if(REFACTOR_API AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ops/api/CMakeLists.txt)
    add_subdirectory(ops/api)
endif()
# ... 其余 5 个同理
```

- **commit**：`91273be build: add module-level CMakeLists for refactor/ops/...`（同一个）

---

## 4. 6 个模块级 CMakeLists.txt（已建，已推）

每个都是 `target_sources(hccl PRIVATE <源文件>)` 的极简模板，未完成的 .cc 用 `#` 注释掉。

| 文件路径 | 实际编进 hccl 的 .cc | 注释掉的（待补完）|
|---|---|---|
| `refactor/ops/api/CMakeLists.txt` | `all_gather_op.cc` | （无）|
| `refactor/ops/executor/CMakeLists.txt` | （无）| `base_executor.cc` `sole/*` `parallel/*` `concurrent/*` `omnipipe/*` |
| `refactor/ops/op_common/CMakeLists.txt` | （无）| `op_common.cc` `hccl_algorithm.cc` |
| `refactor/ops/selector/CMakeLists.txt` | `engine_selector.cc` | `base_selector.cc` `all_gather_selector.cc` |
| `refactor/ops/template/CMakeLists.txt` | `primitives/mesh_primitives.cc`<br>`primitives/nhr_primitives.cc`<br>`engines/aicpu/launcher/load_kernel.cc`<br>`engines/aicpu/launcher/aicpu_launcher.cc`<br>`engines/aicpu/launcher/kernel_launch.cc` | `engines/aicpu/op_alg/allgather_mesh.cc`<br>`engines/aicpu/op_alg/allgather_nhr.cc`<br>`engines/aiv/launcher/load_kernel.cc`<br>`engines/aiv/launcher/kernel_launch.cc` |
| `refactor/ops/utils/CMakeLists.txt` | （无）| `data_transfer.cc` `plan_ranks.cc` `data_copy.cc` |

---

## 5. 当前状态

### 文件清单（3 个 commit 推上去）

```
a50fd9e build: add REFACTOR_OPS option to gate refactored ops path
36ae538 build: wire refactor path into src/CMakeLists with REFACTOR_OPS gate
91273be build: add module-level CMakeLists for refactor/ops/{...}
```

| 文件 | 状态 |
|---|---|
| `CMakeLists.txt` | 修改：加 1 行 `option(REFACTOR_OPS ... OFF)` |
| `src/CMakeLists.txt` | 修改：加 5 行 `include()` 守卫 |
| `refactor/CMakeLists.txt` | **新建**：顶层 refactor 编排器 |
| `refactor/ops/api/CMakeLists.txt` | **新建** |
| `refactor/ops/executor/CMakeLists.txt` | **新建** |
| `refactor/ops/op_common/CMakeLists.txt` | **新建** |
| `refactor/ops/selector/CMakeLists.txt` | **新建** |
| `refactor/ops/template/CMakeLists.txt` | **新建** |
| `refactor/ops/utils/CMakeLists.txt` | **新建** |
| `refactor/ops/c_adaptor/common/CMakeLists.txt` | 已存在，不动 |
| `refactor/ops/c_adaptor/common/hcomm_dlsym/CMakeLists.txt` | 已存在，不动 |

### 实际行为

| 场景 | 行为 |
|---|---|
| 默认（什么开关都不动） | `REFACTOR_OPS=OFF` → refactor/ 不参与构建，跟改之前完全一样 |
| `cmake -DREFACTOR_OPS=ON ..` | refactor/CMakeLists.txt 被 include，但 6 个模块 option 全 OFF → 只有 c_adaptor/common 编进 hccl |
| `cmake -DREFACTOR_OPS=ON -DREFACTOR_SELECTOR=ON ..` | c_adaptor/common + selector 模块进 hccl |
| 在 main 分支编译 | refactor/ 目录不存在 → EXISTS 守卫跳过 → 完全不报错 |

---

## 6. 工作流（每个模块的负责人怎么做）

### 场景 A：我负责的模块代码写完了

1. 打开 `refactor/ops/<我的模块>/CMakeLists.txt`
2. 找到要加进 build 的 .cc 那行，**删掉行首的 `#`**
3. 打开 `refactor/CMakeLists.txt`
4. 把 `option(REFACTOR_<我的模块> ... OFF)` 改成 `ON`
5. 编译验证：
   ```bash
   cmake -B build -S . -DREFACTOR_OPS=ON -DREFACTOR_<我的模块>=ON
   cmake --build build -j
   ```
6. 提交

### 场景 B：我负责的模块没写完（写崩了）

不用动 CMake——它的 option 默认就是 OFF，编译时自动跳过。

### 场景 C：临时回退到旧版本

```bash
cmake -B build -S . -DREFACTOR_OPS=OFF
```

整条 refactor 路径完全不参与构建。

---

## 7. 关键设计决策

### 决策 1：把开关放在顶层 `CMakeLists.txt` 而不是 src/

CMake 变量是**单向流动**的——子目录能读父目录，反过来不行。顶层 `BUILD_OPEN_PROJECT` 分支在 include src/ 之前就要用 `${REFACTOR_OPS}`，所以**全局开关必须在顶层**。

### 决策 2：用 `target_sources(hccl PRIVATE ...)` 而不是新建 `hccl_refactor` 库

- ABI 兼容性：新建 .so 会有符号表、soname、DT_NEEDED 一致性问题
- 链接依赖：原 `hccl` 还是要存在，最后等于两套
- 集成简洁：refactor 目标是替代 `src/ops/`，最终还是一个 `libhccl.so`
- 跟同事的 `refactor/ops/c_adaptor/common/CMakeLists.txt` 风格一致

### 决策 3：每个模块独立 CMakeLists + 独立 option

- 跟仓库惯例一致（`src/ops/` 下有 50+ 个 CMakeLists）
- 各模块独立 owner，独立进度
- 编译失败时定位快（`-DREFACTOR_<X>=OFF` 即可屏蔽某个模块）

### 决策 4：option 门控在 `refactor/CMakeLists.txt`（不在模块 CMakeLists 里）

```cmake
# 顶层：门控进入
if(REFACTOR_API AND EXISTS ...)
    add_subdirectory(ops/api)
endif()
```

**好处**：模块 CMakeLists 极简，将来重构完替代 `src/ops/` 时直接 `rm -rf refactor/`，没有散落的 `if(REFACTOR_API)` 要清理。

---

## 8. 已知问题 & 后续工作

### 缺失的头文件（refactor 引用但仓库里没有）

- `alg_primitive_types.h` — 应该是 refactor 自己要新建的
- `aicpu_base_template.h` — 同上
- `workflow.h` — 同上
- `hccl_log.h`、`mmpa_api.h`、`hccl_diag.h` — 原仓库里位置待确认

这些头不补上，refactor 完整编不过。

### 骨架文件清单（注释掉等补完）

- `executor/concurrent/base_concurrent_executor.cc`（空）
- `executor/omnipipe/base_omnipipe_executor.cc`（空）
- `executor/parallel/parallel_executor.cc`（语法错误）
- `executor/sole/sole_executor.h`（7 行空类）
- `op_common/op_common.h`（0 行）
- `op_common/op_common.cc`（伪代码）
- `op_common/hccl_algorithm.cc`（伪代码）
- `selector/base_selector.cc`（缺返回值）
- `selector/all_gather_selector.cc`（多次直接 return）
- `selector/alg_selector.h`（只有声明，无 .cc）
- `template/base_template.h`（方法空体）
- `template/engines/aicpu/op_alg/allgather_mesh.cc`（伪代码）
- `template/engines/aicpu/op_alg/allgather_nhr.cc`（伪代码）
- `template/engines/aiv/launcher/*`（直接复制 aicpu 的内容，是占位）
- `utils/*`（全部骨架）

### 建议的下一步

**最快验证全链路的方法**：

```bash
# 1. 打开 selector 模块
# 编辑 refactor/CMakeLists.txt: option(REFACTOR_SELECTOR ... OFF) → ON

# 2. 配置 + 编译
cmake -B build -S . -DREFACTOR_OPS=ON
cmake --build build -j

# 3. 验证
nm build/libhccl.so | grep EngineSelector
```

如果能看到 `EngineSelector::Select` 符号，说明从顶层开关到 `target_sources(hccl ...)` 整条链路打通。

---

## 9. 调试 / 排错速查

| 问题 | 原因 | 解决 |
|---|---|---|
| 编译时报 `Cannot add sources to target 'hccl' which does not exist` | `target_sources(hccl ...)` 在 `include(hccl.cmake)` 之前执行了 | 检查 refactor/CMakeLists.txt 的 include 位置，必须在 `src/CMakeLists.txt:301` 之后 |
| main 分支编译挂 | refactor/ 目录不存在 | EXISTS 守卫应该跳过；检查守卫语法 |
| `REFACTOR_OPS=ON` 但模块没编进去 | 模块 option 仍为 OFF | 确认 `refactor/CMakeLists.txt` 里对应 option 是 ON |
| 模块编进去了但找不到符号 | 模块的 .cc 还在 `#` 注释里 | 删除 `#` |
| 头文件找不到 | 缺失头（见第 8 节）| 从 `src/ops/` 对应路径 include，或新建头文件 |
| 工作树有别人未提交的 mesh_primitives.cc TODO 注释 | 跟我这次的 commit 无关 | 推之前需要 `git stash` + `git pull --rebase` + `git stash pop` |

---

## 10. 后续替代 src/ops/ 的迁移路径

等所有 6 个模块的代码都写完：

1. 把 `refactor/CMakeLists.txt` 里 6 个 option 全置为 ON
2. 把 `refactor/ops/<模块>/CMakeLists.txt` 里所有 `#` 全删掉
3. 测试 `cmake --build build -j` 编过
4. 删除 `src/ops/` 对应目录（all_gather、all_reduce 等）
5. 删 `refactor/CMakeLists.txt`、`refactor/ops/<模块>/CMakeLists.txt`
6. 删顶层 `option(REFACTOR_OPS ...)` 和 `src/CMakeLists.txt` 里的 `if(REFACTOR_OPS)` 守卫
7. 删 `refactor/` 整个目录

**目标：最终只一个 `libhccl.so`，目录结构从 `src/ops/` 直接变成 `refactor/ops/`（顶层切换）。**

---

## 11. 联系 / 上下文

- 仓库：HCCL（华为集合通信库）
- 平台：x86_64 + aarch64（cann-toolkit）
- 编译环境：CMake ≥ 3.16，C++14，host/device 双侧
- CANN 包路径：`${ASCEND_CANN_PACKAGE_PATH}/latest`
- 链接选项：`-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack`、`-Wl,--no-as-needed`
- 编译选项：`-Werror -fno-common -fno-strict-aliasing -fstack-protector-all -std=c++14`
- ABI 兼容：host 侧必须 `_GLIBCXX_USE_CXX11_ABI=0`

---

## 12. 改动文件清单速查

```bash
# 1. 顶层 CMakeLists.txt（+1 行）
CMakeLists.txt:16  option(REFACTOR_OPS "..." OFF)

# 2. src/CMakeLists.txt（+5 行）
src/CMakeLists.txt:301  if(REFACTOR_OPS AND EXISTS ...) include(...)

# 3. refactor/CMakeLists.txt（新建，约 60 行）
#    6 个 option + c_adaptor/common 强制 add_subdirectory + 6 个带 EXISTS 守卫的 add_subdirectory

# 4. 6 个模块 CMakeLists.txt（新建，每个 20-40 行）
refactor/ops/api/CMakeLists.txt
refactor/ops/executor/CMakeLists.txt
refactor/ops/op_common/CMakeLists.txt
refactor/ops/selector/CMakeLists.txt
refactor/ops/template/CMakeLists.txt
refactor/ops/utils/CMakeLists.txt
```
