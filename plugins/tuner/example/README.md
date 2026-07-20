# HCCL Tuner Plugin 示例

## 概述

本目录提供 HCCL Tuner Plugin 的参考实现，演示如何通过外部 `.so` 插件读取 JSON 配置并修改 3D cost table，影响 Selector 的算法选择。

## 编译

```bash
source /usr/local/Ascend/cann/set_env.sh
make
```

产物：`hccl_tuner_example.so`

## 使用

```bash
export HCCL_TUNER_PLUGIN=/path/to/hccl_tuner_example.so
export HCCL_TUNER_CONFIG_FILE=/path/to/hccl_tuner_config.json
```

环境变量未设置时，插件按以下顺序查找配置文件：

1. `$HCCL_TUNER_CONFIG_FILE`
2. `./hccl_tuner_config.json`
3. `/etc/hccl/hccl_tuner_config.json`

## JSON 配置格式

```json
{
  "version": 1,
  "op_types": {
    "allreduce": {
      "rules": [
        {
          "match": {
            "min_ranks": 8, "max_ranks": 8,
            "min_bytes": 0, "max_bytes": 65536,
            "data_type": "fp16",
            "comm_name": "world",
            "min_npus": 8, "max_npus": 8,
            "min_servers": 1
          },
          "engine": 2,
          "executor": 0,
          "template": 6,
          "cost": 0.0
        }
      ]
    }
  }
}
```

### match 条件（全部 AND，first-match-wins）

| 字段 | 类型 | 说明 |
|------|------|------|
| `min_ranks` / `max_ranks` | uint32 | 通信域 rank 数范围 |
| `min_bytes` / `max_bytes` | size_t | 数据量范围（字节） |
| `data_type` | string | 数据类型（fp16/fp32/int8/...） |
| `comm_name` | string | 通信域名（子串匹配） |
| `min_npus` / `max_npus` | uint32 | 每服务器 NPU 数范围 |
| `min_servers` | uint32 | 最小服务器数 |

### 命中行为

- `engine` / `executor` / `template` 指定 cost table 中的目标位置
- `cost` 设置该位置的 cost 值（省略时为 0.0，即最优）
- Selector 从 560 个 cost 中选最小值确定算法

### 支持的 op_type

`allreduce` / `allgather` / `broadcast` / `reduce` / `reduce_scatter` / `scatter` / `alltoall` / `alltoallv`

### cost table 维度

- engine: 5（CPU / CPU_TS / AICPU / AICPU_TS / AIV / CCU 中的 5 个）
- executor: 7
- template: 16
- 总计：5 × 7 × 16 = 560 floats

## 测试

```bash
cd test
make
./test_plugin
```

## 插件接口

插件需导出两个符号：

| 符号 | 类型 | 说明 |
|------|------|------|
| `hcclTunerPlugin` | `hcclPluginDescriptor_t` | 描述符（含 apiVersion） |
| `hcclTunerGetFuncs` | 函数 | 返回 init + getCollInfo 函数表 |

详见 `include/hccl_tuner_plugin.h`。
