# hcclgen

`hcclgen` 是面向 HCCL 的精简合成器原型，只保留两部分：

1. 稳定输入 JSON 解析。
2. SCIP 完整 MILP AllGather 求解。

这里不包含 Gurobi、step-MILP、beam search、rolling-window、历史实验脚本。

## 安装

自动安装,可以直接使用已有环境：

```bash
cd hccl_omni\experimental\eco_system\hccl-omni\omni_test\hccl-omni-py\hccl_omni
pip install -e .
```

依赖：

```text
pyscipopt
```

手动安装scip

```bash
git clone https://github.com/scipopt/SCIPpp.git
cd SCIPpp
cmake . -DCMAKE_PREFIX_PATH=/path/to/SCIP # change the path of SCIP, e.g., /usr/local
make ScipPP
make install
export LD_LIBRARY_PATH=/usr/local/scipoptsuite-10.0.1/build/lib:$LD_LIBRARY_PATH
```


## 输入格式

示例在：

```text
examples/input_4x4_1dmesh_clos.json
examples/input_15rank_split.json
```

当前固定约定：

```text
rank_base = 0
level 从 level0 开始连续递增
同一个 level 内，一个 rank 不能出现在多个 group 中
1DMESH = group 内 fullmesh 直连
Clos = group 内 rank 连接到同一个交换机层
netlayer = 后续 HCCL XML 使用的 netLayerId
Clos 默认参考上一层 group 做 degree=1 balanced b-matching 剪枝
output.hccl_xml 可选；相对路径按输入 JSON 所在目录解析
```

Clos 资源命名：

```text
l{level}_rank{rank}_out
l{level}_rank{rank}_in
```

例如 level1 上 `0 -> 5` 会消耗：

```text
l1_rank0_out
l1_rank5_in
```

## 代码地图

```text
hcclgen/input_schema.py  输入 JSON 校验，1DMESH/Clos 展开，默认 Clos 剪枝
hcclgen/sketch.py        balanced b-matching 剪枝
hcclgen/topology.py      Edge/GridTopology、资源容量、T 下界、带宽计算
hcclgen/global_milp.py   SCIP 完整 MILP 求解和 CLI
hcclgen/hccl_xml.py      pair-exchange HCCL XML 输出
hcclgen/api.py           对外一键式生成接口
```

## 函数接口

推荐入口是 `generate_hccl_xml()`，调用方只需要传输入 JSON。
默认开启 `require_pair_exchange=True`，也就是要求同一个 step、同一个 netLayer、
同一对 rank 的两个方向发送数量相等，方便保守 lowering 到 HCCL `SendRecvWrite`。

```python
from hcclgen.api import generate_hccl_xml

artifact = generate_hccl_xml(
    "examples/input_4x4_1dmesh_clos.json",
    chunks=1,
    threads=4,
)
```

返回值包含输入、拓扑、求解结果和 XML 路径：

```python
artifact.spec
artifact.topology
artifact.result
artifact.xml_path
```

## 运行

如果不手动指定 `--total-epochs`，程序会用接收端口容量下界作为默认 T。
如果不手动指定 `--max-steps`，程序会默认使用 `max_steps = T`。
路由剪枝默认使用 `--route-dag-mode spdag`：

```bash
/home/lw/CCL/msccl-tools/.venv/bin/python -m hcclgen.global_milp \
  --input-json examples/input_4x4_1dmesh_clos.json \
  --threads 4 \
  --quiet
```

也可以显式指定：

```bash
/home/lw/CCL/msccl-tools/.venv/bin/python -m hcclgen.global_milp \
  --input-json examples/input_15rank_split.json \
  --total-epochs 3 \
  --threads 4 \
  --quiet
```

输出 JSON：

```bash
/home/lw/CCL/msccl-tools/.venv/bin/python -m hcclgen.global_milp \
  --input-json examples/input_4x4_1dmesh_clos.json \
  --solution-json results/4x4_solution.json \
  --threads 4 \
  --quiet
```

输出 HCCL XML：

```bash
/home/lw/CCL/msccl-tools/.venv/bin/python -m hcclgen.global_milp \
  --input-json examples/input_4x4_1dmesh_clos.json \
  --hccl-xml results/4x4_hccl.xml \
  --threads 4 \
  --quiet
```

也可以直接在输入 JSON 中设置：

```json
"output": {
  "hccl_xml": "../results/4x4_hccl.xml"
}
```

如果同时设置 JSON 字段和 `--hccl-xml`，命令行参数优先。

## 验证

```bash
cd /home/lw/CCL/hcclgen
/home/lw/CCL/msccl-tools/.venv/bin/pytest tests -q --no-cov
```
