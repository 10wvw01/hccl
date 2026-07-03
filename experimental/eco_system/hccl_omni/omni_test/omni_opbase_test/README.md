# omni_opbase_test / run.sh 使用说明

`run.sh` 位于 `experimental/eco_system/hccl_omni/omni_test/omni_opbase_test/`，用于在**真机 NPU** 上通过 `torchrun` 启动 HCCL Omni 集成测试。当前主要验证 `@hccl_omni.jit` 高级 API 能否正确调用 `HcclOmniRun`。

---

## 1. 目录结构

```
omni_opbase_test/
├── run.sh              # 测试启动脚本（本文档说明对象）
├── env.sh              # CANN / HCCL / 日志环境变量（需按本机修改）
├── test_jit_api.py     # JIT API 真机测试（run.sh 选项 1）
└── xml_example/
    └── 2pfullmesh.xml  # 2 卡 fullmesh AlltoAllV 算法 XML（Vanilla 模式输入）
```

---

## 2. 用法

```bash
cd experimental/eco_system/hccl_omni/omni_test/omni_opbase_test
chmod +x run.sh
./run.sh {0|1}
```

| 参数 | 测试项 | 状态 |
|------|--------|------|
| `0` | `alltoallv_aicpu_ts_test`（底层 AlltoAllV AICPU 测试） | **未启用**，执行后仅打印 `alltoallv test not supported` |
| `1` | `test_jit_api.py`（`@hccl_omni.jit` + `HcclOmniRun` 集成） | **可用**，固定 `torchrun --nproc_per_node=2` |

示例：

```bash
./run.sh 1
```

参数错误或缺失时输出：

```
Usage: ./run.sh {0|1}
  0: alltoallv_aicpu_ts_test (torchrun)
  1: test_jit_api.py — hccl_omni JIT API (torchrun)
```

---

## 3. run.sh 执行流程

```
run.sh
  ├─ source /home/yhb/local/torch-venv/bin/activate   ← 需改为本机 Python 虚拟环境
  ├─ export HCCL_SOCKET_IFNAME=lo
  ├─ export HCCL_IF_BASE_PORT=50033
  ├─ source env.sh                                    ← 加载 CANN / LD_LIBRARY_PATH 等
  └─ case $1
       0 → 占位（不支持）
       1 → torchrun --nproc_per_node=2 test_jit_api.py
```

脚本使用 `set -e`，任一步失败即退出。

---

## 4. 运行前准备

### 4.1 HCCL 与 AICPU 包

需已用 `--experimental --full` 编译并安装 HCCL，且 Device 侧 AICPU 包已部署：

```bash
bash build.sh --experimental --full --pkg -p /usr/local/Ascend/cann
# 安装 run 包、关闭自编译 AICPU 验签、重启 NPU 业务进程
```

详见 `experimental/eco_system/HcclOmniRun_使用说明.md`。

### 4.2 Python 依赖

```bash
cd ../hccl-omni-py
python -m build --wheel
pip install dist/hccl_omni-*.whl
```

需要：`torch`、`torch_npu`、`hccl_omni`（含 JIT 编译的 C++ 扩展）。

### 4.3 硬件

- 至少 **2 张 NPU**（脚本写死 `--nproc_per_node=2`，与 `2pfullmesh.xml` 对应）
- `env.sh` 中 `HCCL_TEST_USE_DEVS="0,1"` 指定使用 device 0、1

### 4.4 修改本机路径（必做）

脚本内含开发者环境硬编码路径，**首次使用前必须修改**：

**`run.sh` 第 5 行** — Python 虚拟环境：

```bash
# 改前
source /home/yhb/local/torch-venv/bin/activate

# 改后（示例）
source /path/to/your/venv/bin/activate
```

**`env.sh`** — CANN 与运行目录：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOME_PATH` | `/home/yhb/local` | CANN 安装根目录的父路径 |
| `RUN_PATH` | `/home/yhb/omni_opbase_test` | 日志输出目录（`ASCEND_PROCESS_LOG_PATH`） |
| `MPI_PATH` | `/home/hjh/mpich` | MPI 库路径（可选，若不用 mpirun 可忽略 PATH 部分） |

典型修改示例（CANN 默认安装）：

```bash
HOME_PATH="/usr/local/Ascend"
RUN_PATH="$(pwd)"
# source $HOME_PATH/ascend-toolkit/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**`test_jit_api.py` 第 40 行** — XML 备用路径（本地找不到 `xml_example/2pfullmesh.xml` 时使用）：

```python
xml_path = '/home/yhb/omni_opbase_test/xml_example/2pfullmesh.xml'
```

一般无需修改：脚本会优先使用同目录下 `xml_example/2pfullmesh.xml`。

---

## 5. env.sh 环境变量说明

| 变量 | 作用 |
|------|------|
| `LD_LIBRARY_PATH` | 追加 CANN lib64、devlib、HCCL 实验库、MPI lib |
| `ASCEND_DIR` | CANN 设备侧路径 |
| `source .../set_env.sh` | 加载 Ascend 工具链环境 |
| `ASCEND_GLOBAL_LOG_LEVEL=0` | Ascend 全局日志级别 |
| `ASCEND_PROCESS_LOG_PATH` | 进程日志目录（`$RUN_PATH/log`） |
| `ASCEND_MODULE_LOG_LEVEL=HCCL=1` | 打开 HCCL 模块 INFO 日志 |
| `HCCL_TEST_USE_DEVS="0,1"` | 测试使用的 NPU 设备 ID |

**run.sh 额外设置：**

| 变量 | 值 | 说明 |
|------|-----|------|
| `HCCL_SOCKET_IFNAME` | `lo` | 集合通信网卡；多机或物理网卡环境需改为实际接口（如 `eth0`） |
| `HCCL_IF_BASE_PORT` | `50033` | HCCL 通信端口基址，避免与其他任务冲突 |

**test_jit_api.py 内设置：**

| 变量 | 值 | 说明 |
|------|-----|------|
| `HCCL_OP_EXPANSION_MODE` | `AI_CPU` | 走 AICPU 展开执行 |
| `HCCL_BUFFSIZE` | `200` | 通信 buffer 大小（未设置时默认 200） |

---

## 6. test_jit_api.py 测试内容（选项 1）

1. `dist.init_process_group(backend='hccl')`，每 rank 绑定 `torch.npu.set_device(rank)`
2. `HcclOmniConfig(ins_gen_mode='Vanilla')`，读取 `xml_example/2pfullmesh.xml`
3. `@hccl_omni.jit` 装饰空函数 `alltoallv_op`
4. 构造 AlltoAllV 等分参数：每 rank 1024 元素，2 卡共 2048
5. 调用 `alltoallv_op[kernel_config](cluster_config, op_param)`，内部完成 XML→bin→`HcclOmniRun`

### 成功输出示例

```
[Rank 0] Initialized: rank=0, world_size=2
[Rank 1] Initialized: rank=1, world_size=2
[Rank 0] JIT API call completed successfully
[Rank 1] JIT API call completed successfully
=== test_jit_api.py Done ===
```

任 rank 异常会打印 `EXCEPTION` 与 traceback，但仍会执行 `destroy_process_group`。

---

## 7. 手动运行（不通过 run.sh）

若已自行配置环境，可直接：

```bash
source /path/to/venv/bin/activate
source env.sh   # 或手动 export 等价变量

export HCCL_SOCKET_IFNAME=lo
export HCCL_IF_BASE_PORT=50033
export HCCL_OP_EXPANSION_MODE=AI_CPU

torchrun --nproc_per_node=2 test_jit_api.py
```

多机时需为 `torchrun` 增加 `--nnodes`、`--node_rank`、`--master_addr`、`--master_port` 等参数，并同步修改 `HCCL_SOCKET_IFNAME`；当前 `run.sh` **仅支持单机 2 卡**。

---

## 8. 常见问题

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `source: ... torch-venv: No such file` | `run.sh` 中 venv 路径未改 | 修改第 5 行为本机 venv |
| `Cannot find libhccl.so` | 未安装 experimental HCCL 或未 `source set_env.sh` | 安装 run 包并加载 CANN 环境 |
| `HcclOmniRun: undefined symbol` | `libhccl.so` 非 experimental 构建 | `build.sh --experimental` 重编安装 |
| `device type not supported` | NPU 非 910_95/950 | 更换支持的硬件 |
| HCCL 初始化失败 | 网卡/端口冲突 | 调整 `HCCL_SOCKET_IFNAME`、`HCCL_IF_BASE_PORT` |
| 仅 1 张 NPU | 脚本要求 2 卡 | 改用 1 卡需同时改 XML、`--nproc_per_node` 与 op_param |
| AICPU 加载失败 | 验签未关或 tar 未装 | 见 `docs/zh/build/build.md` 关闭验签章节 |

日志位置：`$RUN_PATH/log`（默认 `env.sh` 中配置的目录）。

---

## 9. 与 hccl-omni-py 测试的区别

| 入口 | 位置 | 是否需要 NPU | 说明 |
|------|------|--------------|------|
| `omni_opbase_test/run.sh` | 本目录 | **是** | 真机 JIT + `HcclOmniRun` 端到端 |
| `hccl-omni-py/tests/run_test.py` | `../hccl-omni-py/tests/` | fake 模式否 / real 模式是 | 单元与集成测试 |
| `pytest tests/fake_torch/` | `../hccl-omni-py/tests/` | **否** | mock torch，无需 CANN |

`run.sh` 面向**部署验证**；开发阶段可先在 `hccl-omni-py` 跑 fake_torch 单测，再在真机执行 `./run.sh 1`。

---

## 10. 快速检查清单

运行 `./run.sh 1` 前确认：

- [ ] 已修改 `run.sh` 中 Python venv 路径
- [ ] 已修改 `env.sh` 中 `HOME_PATH`、`RUN_PATH` 并 `source set_env.sh`
- [ ] 已安装 `--experimental --full` 构建的 HCCL 与 AICPU 包
- [ ] 已 `pip install hccl_omni` wheel
- [ ] 机器上至少有 2 张可用 NPU（0、1）
- [ ] `xml_example/2pfullmesh.xml` 存在
- [ ] 自编译 AICPU 包已按文档关闭验签
