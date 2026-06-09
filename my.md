# HCCL Data Flow SVG Generator

从 HCCL 运行日志生成数据流可视化 SVG 图，支持三级拓扑结构（Server / Pod / 集群）的多层次数据流展示。

## 快速开始

```bash
python3 gen_svg.py <log_file> --l0 <devices_per_server> --l1 <servers_per_pod> --l2 <num_pods>
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `log_file` | HCCL 日志文件路径（必填） |
| `-o, --output-dir` | SVG 输出目录（默认与 log 文件同目录） |
| `--l0` | 每个 Server 的设备数（覆盖自动推断） |
| `--l1` | 每个 Pod 的 Server 数（覆盖自动推断） |
| `--l2` | Pod 数量（覆盖自动推断） |
| `--levels` | 生成的图级别，逗号分隔，默认 `summary` |

## 拓扑层级

```
集群
├── Pod0 (L2=0)
│   ├── Server0 (L1=0)
│   │   ├── R0 (L0=0)
│   │   ├── R1 (L0=1)
│   │   ├── R2 (L0=2)
│   │   └── R3 (L0=3)
│   └── Server1 (L1=1)
│       ├── R4 ── R7
├── Pod1 (L2=1)
│   └── ...
```

- **L0**：同一 Server 内的设备数
- **L1**：同一 Pod 内的 Server 数
- **L2**：Pod 数量
- 总 rank 数 = L0 × L1 × L2

## 图级别说明

| 级别 | 文件名 | 说明 |
|------|--------|------|
| `0` | `data_flow_level0.svg` | Server 内数据流：INPUT / CCL / OUTPUT 三行缓冲区 |
| `1` | `data_flow_level1.svg` | Server 间数据流（同 Pod）：CCL 缓冲区 + 跨 Server 箭头 |
| `2` | `data_flow_level2.svg` | Pod 间数据流：CCL 缓冲区 + 跨 Pod 箭头 |
| `summary` | `data_flow_summary.svg` | 汇总图：INPUT / CCL / OUTPUT + 所有操作箭头 |

## 支持的操作类型

| 操作 | 颜色 | 说明 |
|------|------|------|
| LocalCopy | 绿色 `#2E7D32` | INPUT→CCL（Server 内拷贝） |
| LocalReduce | 紫色 `#7B1FA2` | CCL→CCL（Server 内 reduce） |
| Write (INPUT→CCL) | 红色 `#D32F2F` | 本地 INPUT→远端 CCL（跨 rank 写） |
| Write (CCL→CCL) | 红色 `#D32F2F` | 本地 CCL→远端 CCL（跨 rank 写） |
| Read (CCL→CCL) | 蓝色 `#1976D2` | 远端 CCL→本地 CCL（跨 rank 读） |
| Read (CCL→INPUT) | 蓝色 `#1976D2` | 远端 CCL→本地 INPUT（跨 rank 读） |
| WriteReduce L1 | 蓝色 `#1565C0` | CCL→远端 CCL（跨 Server 同 Pod reduce） |
| WriteReduce L2 | 深橙 `#BF360C` | CCL→远端 CCL（跨 Pod reduce） |
| Output | 青色 `#00695C` | CCL→OUTPUT（最终输出） |

## 使用示例

```bash
# 8-rank log，只生成汇总图
python3 gen_svg.py log202606081422.log --l0 4 --l1 1 --l2 2

# 16-rank log，生成所有级别
python3 gen_svg.py log202606081040.log --l0 4 --l1 2 --l2 2 --levels 0,1,2,summary

# 24-rank log，只生成 Level1 和 Level2
python3 gen_svg.py log202606081700432.log --l0 4 --l1 3 --l2 2 --levels 1,2

# 含 Read 操作的 log
python3 gen_svg.py log202606081900_br_parall.log --l0 4 --l1 2 --l2 1

# 指定输出目录
python3 gen_svg.py log202606081040.log --l0 4 --l1 2 --l2 2 -o ./output/
```

## 日志格式要求
将checker 输出日志放入即可，会自动解析里面的编排任务
工具解析 HCCL 日志中的以下行格式：

```
rankIdx:<rank>, threadIdx:<thread>, [<Operation>]:[...]
```

支持的操作关键字：
- `[LocalCopy]:` — 本地拷贝
- `[LocalReduce]:` — 本地 reduce
- `[Write]:` — 跨 rank 写
- `[WriteReduce]:` — 跨 rank reduce 写
- `[Read]:` — 跨 rank 读
- `LocalCopy ... OUTPUT` — 最终输出

每个操作需包含 `DataSlice[BufferType::<TYPE>, offset=<hex>, size=<hex>]` 格式的缓冲区描述。

## 输出文件

默认输出到 log 文件所在目录，文件名格式：

```
data_flow_level0.svg
data_flow_level1.svg
data_flow_level2.svg
data_flow_summary.svg
```

在浏览器或 VS Code 预览中打开 SVG 文件即可查看。
