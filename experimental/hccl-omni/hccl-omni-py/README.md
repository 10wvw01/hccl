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
- `op_param` (dict): 算子参数

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

### 使用 pytest 运行测试（推荐）

```bash
# 安装 pytest
pip install pytest

# 运行所有 fake torch 测试
pytest tests/fake_torch/ -v

# 运行单个测试文件
pytest tests/fake_torch/test_basic.py -v
```

### 使用测试运行器

```bash
# 运行 fake torch 测试
python tests/run_test.py --mode fake

# 运行 real torch 测试（单进程）
python tests/run_test.py --mode real

# 运行 real torch 测试（分布式）
torchrun --nproc_per_node=4 tests/run_test.py --mode real
```

### 执行单个测试文件

**Fake torch 测试**（无需真实 PyTorch 环境）：
```bash
python tests/fake_torch/test_<test_name>.py
```

**Real torch 测试**：
```bash
# 单进程模式
python tests/real_torch/test_<test_name>.py

# 分布式模式
torchrun --nproc_per_node=4 tests/real_torch/test_<test_name>.py
```

### XML 到 BIN 流程测试

验证从 XML 文件读取、切分到 bin 生成的完整流程：

```bash
# 使用默认输入文件和临时输出目录
python tests/fake_torch/test_xml_to_bin.py

# 指定输入文件和输出目录
python tests/fake_torch/test_xml_to_bin.py --input-file xml_example/4pfullmesh.xml --output-dir /path/to/output
```

### 测试环境说明

测试框架通过 `tests/conftest.py` 自动管理：
- 自动添加项目根目录到 Python 路径
- 自动设置 mock torch 环境（fake_torch 测试）

测试文件无需手动处理路径设置，直接导入即可：

```python
# tests/fake_torch/test_example.py
import hccl_omni
from test_utils import setup_unified_test_environment

# 设置测试配置
temp_dir, config_file = setup_unified_test_environment()

def test_example():
    # 测试代码
    pass
```

### 编写测试

- **Fake torch 测试**：在 `fake_torch/` 目录下创建测试文件，使用测试工具设置 mock 环境。
- **Real torch 测试**：在 `real_torch/` 目录下创建测试文件，手动初始化分布式环境。

使用 `torchrun` 时，测试代码应处理多 rank 情况。