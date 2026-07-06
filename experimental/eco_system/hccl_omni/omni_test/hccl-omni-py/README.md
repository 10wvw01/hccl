# HCCL-OMNI 框架

基于 PyTorch 的通信算子框架，依赖 Ascend torch_npu 能力，提供灵活的算子定义与执行机制。

## 设计概述

框架采用三层架构：
- **前端**：Python 装饰器接口，负责用户函数包装
- **中端**：JIT 编译与缓存管理，生成执行指令
- **后端**：C 代码执行

## 安装与打包

### 打包
```bash
python -m build --wheel
```

生成的 wheel 文件位于 `dist/hccl_omni-0.1.0-py3-none-any.whl`。

### 安装
```bash
pip install dist/hccl_omni-0.1.0-py3-none-any.whl
```

## 用户接口

### `hccl_omni.jit` 装饰器

将 Python 函数包装为通信算子。

```python
import hccl_omni
from hccl_omni import HcclOmniConfig

# 基础用法
@hccl_omni.jit
def user_op(cluster_config, op_param):
    pass

# 带配置用法
config = HcclOmniConfig(op_key="my_op", ins_gen_mode="DSL")

@hccl_omni.jit(version=1, hccl_omni_cfg=config, debug=False)
def user_op(cluster_config, op_param):
    pass
```

**装饰器参数**：
- `version` (int): 接口版本，默认 1
- `hccl_omni_cfg` (HcclOmniConfig): 配置对象
- `debug` (bool): 调试模式，默认 False

### `HcclOmniConfig` 配置类

类型安全的配置对象。

**字段**：
- `op_key` (str): 算子标识，用于缓存键生成
- `ins_file_path` (str): 指令文件路径（Vanilla 模式必需）
- `ins_gen_mode` (str): 指令生成模式，可选 `AlgSynthesizer`、`DSL`、`Vanilla`

### 算子调用

装饰器返回 `JITFunction` 实例，支持索引调用：

```python
operator = user_op[kernel_config]
result = operator(cluster_config, op_param)
```

**参数**：
- `kernel_config` (dict): 内核配置
- `cluster_config` (dict): 集群拓扑描述
- `op_param` (dict): 算子参数，包含算子类型、输入输出张量及切分配置

### `op_param` 参数说明

`op_param` 描述待执行算子的类型、数据及切分方式。

**字段**：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `op_name` | `OpName` | 是 | 算子类型枚举 |
| `input` | `torch.Tensor` | 是 | 输入张量 |
| `output` | `torch.Tensor \| list[torch.Tensor]` | 是 | 输出张量，Allgather 为 list |
| `reduce_op` | `ReduceOp` | 否 | 归约操作，默认 SUM |
| `input_split_sizes` | `list[int]` | 否 | 输入切分大小（ReduceScatter / Alltoall） |
| `output_split_sizes` | `list[int]` | 否 | 输出切分大小（Alltoall） |

**各算子要求**：

**Allgather**

| 字段 | 要求 |
|------|------|
| `input` | 单个 tensor，size = N |
| `output` | `list[tensor]`，长度 = world_size，每个 tensor 的 size = N |
| `split_sizes` | 不使用 |

每个 rank 持有 N 个元素的本地数据，收集所有 rank 的数据后得到 world_size × N 的结果。

**ReduceScatter**

| 字段 | 要求 |
|------|------|
| `input` | 单个 tensor，size = N |
| `output` | 单个 tensor，size = 每个 rank 分得的 chunk |
| `input_split_sizes` | 可选 `list[int]`，长度 = world_size。未提供时 N 必须被 world_size 整除，自动均分 |

输入 tensor 按 `input_split_sizes` 切分为 world_size 份，各份归约后散射到对应 rank。框架根据 `input_split_sizes` 自动计算 `send_counts` 和 `sdispls`。

**Allreduce**

| 字段 | 要求 |
|------|------|
| `input` | 单个 tensor，size = N |
| `output` | 单个 tensor，size 必须与 input 相同 |
| `split_sizes` | 不使用 |

所有 rank 的输入进行归约，结果广播到所有 rank。

**Alltoall**

| 字段 | 要求 |
|------|------|
| `input` | 单个 tensor，size = sum(input_split_sizes) |
| `output` | 单个 tensor，size = sum(output_split_sizes) |
| `input_split_sizes` | 可选 `list[int]`，长度 = world_size。定义发给每个 rank 的元素数。未提供时 input size 必须被 world_size 整除，自动均分 |
| `output_split_sizes` | 可选 `list[int]`，长度 = world_size。定义从每个 rank 接收的元素数。未提供时等于 `input_split_sizes` |

每个 rank 按 `input_split_sizes` 切分自身数据发给对应 rank，同时按 `output_split_sizes` 从各 rank 接收数据。

**示例**：

```python
from hccl_omni import OpName, ReduceOp

# Allgather
op_param = {
    'op_name': OpName.Allgather,
    'input': local_tensor,
    'output': [t1, t2, t3, t4],  # world_size 个 tensor
}

# ReduceScatter with explicit split
op_param = {
    'op_name': OpName.ReduceScatter,
    'input': input_tensor,
    'output': output_tensor,
    'input_split_sizes': [256, 512, 256],
    'reduce_op': ReduceOp.SUM,
}

# Allreduce
op_param = {
    'op_name': OpName.Allreduce,
    'input': input_tensor,
    'output': output_tensor,
    'reduce_op': ReduceOp.SUM,
}

# Alltoall with explicit split
op_param = {
    'op_name': OpName.Alltoall,
    'input': input_tensor,
    'output': output_tensor,
    'input_split_sizes': [2, 4, 6, 8],
    'output_split_sizes': [4, 4, 4, 4],
}
```

## 设计约束

### 指令生成模式

| 模式 | 说明 | 广播需求 |
|------|------|----------|
| AlgSynthesizer | root rank 调用求解器生成 XML | 是 |
| DSL | 各 rank 独立将 DSL 转换为 XML | 否 |
| Vanilla | 读取用户提供的 XML 文件 | 是 |

### 缓存键生成算法

1. 若配置了 `op_key`，直接使用 `f"{op_key}.{rank}"` 作为缓存键
2. 否则：
   - 对 `op_param` 字典计算哈希得到 `op_hash`
   - 对 `cluster_config` 字典打平后计算哈希得到 `cluster_hash`
   - 组合为 `f"{cluster_hash}-{op_hash}.{rank}"`

每个 rank 拥有独立的缓存文件，支持分布式环境下各 rank 独立缓存。

### XML 切分算法

缓存未命中时，XML 生成后按 rank 切分：
1. root rank 生成完整 XML 后广播至所有 rank
2. 各 rank 根据 `root/NPU` 标签的 `rankId` 属性提取对应内容
3. 返回的 XML 仅包含提取的内容，不含 NPU 标签本身

### 缓存路径解析

缓存路径通过拓扑配置文件自动确定：
1. 检查 `HCCL_TOPO_FILE_PATH` 环境变量查找 `hccl_rootinfo.json` 文件位置
2. 若未设置，尝试读取 `/etc/hccl_rootinfo.json`
3. 解析其中 `{ "topo_file_path": "path of TOPO file", ... }` 字段，取 `topo_file_path` 父目录作为缓存路径
4. 路径不存在时抛出 `RuntimeError`

### 分布式通信

- AlgSynthesizer 和 Vanilla 模式在 root rank 生成/读取 XML 后广播至所有 rank
- DSL 模式各 rank 独立生成，无需广播
- 缓存读写在各 rank 上独立进行，无需跨 rank 同步

## 环境要求

- Python 3.10+
- PyTorch 2.9.0+（Ascend v7.3.1）
- Ascend torch_npu

## 测试

测试分为两种模式：
- **fake_torch**：使用 mock 打桩，无需真实 PyTorch 环境，适用于单元测试
- **real_torch**：使用真实 PyTorch，支持分布式测试，适用于集成测试

### 使用 pytest 运行

```bash
# 运行所有 fake torch 测试
pytest tests/fake_torch/ -v

# 运行单个测试文件
pytest tests/fake_torch/test_basic.py -v
```

### real_torch 分布式测试

### real_torch 测试

验证 HCCL-OMNI 算子与 `torch.distributed` 原生接口的数值一致性，覆盖 Allgather、ReduceScatter、Allreduce、Alltoall。

需要在 Ascend NPU 环境下运行，依赖 `torch_npu` 和 HCCL 通信库。

```bash
# 单进程（world_size=1，部分测试会跳过）
pytest tests/real_torch/test_operators.py -v

# 分布式（4 卡，pytest 报告由各 rank 独立输出）
torchrun --nproc_per_node=4 tests/real_torch/test_operators.py -v
```