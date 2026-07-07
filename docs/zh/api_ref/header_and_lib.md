# HCCL 对外头文件与库文件说明

## 概述

HCCL（Huawei Collective Communication Library，华为集合通信库）基于昇腾 AI 处理器，提供单机/多机环境下的高性能集合通信与点对点通信能力，是 CANN 的核心组件之一。

本文档列举 HCCL **正式对外发布**的头文件与库文件，供框架集成、算子开发与部署安装时参考。所谓"正式对外"，以 `CMakeLists.txt` 中 `install(FILES ... DESTINATION ${INSTALL_INCLUDE_DIR}/hccl/)` 与 `install(TARGETS ... LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR})` 的安装规则为准；其余仅用于仓内编译、不安装到对外路径的头文件不在本文档范围内。

HCCL 与通信基础库 HCOMM 通过 `dlsym` 动态加载解耦、独立编译演进，HCOMM 的对外头文件与库文件请参考 `cann/hcomm` 仓库的对应文档。

## 头文件

HCCL 对外头文件共 2 个，安装到 CANN 安装路径的 `include/hccl/` 目录下。

| 头文件 | 安装路径 | 功能描述 |
|--------|---------|---------|
| `hccl.h` | `include/hccl/` | HCCL 主头文件，声明 14 个通信算子 C 接口（11 个集合通信算子 + 3 个点对点通信算子） |
| `hccl_mc2.h` | `include/hccl/` | MC2（Kernel Fusion Custom）自定义算子框架头文件，声明 9 个 OpArgs 参数对象与通信资源上下文（OpResCtx）管理接口 |

### hccl.h

声明 HCCL 全部对外通信算子 C 接口，是 AI 框架（PyTorch、TensorFlow 等）进行单算子模式适配的主要入口。

- **集合通信算子（11 个）**：`HcclAllReduce`、`HcclBroadcast`、`HcclAllGather`、`HcclAllGatherV`、`HcclReduceScatter`、`HcclReduceScatterV`、`HcclScatter`、`HcclReduce`、`HcclAlltoAll`、`HcclAlltoAllV`、`HcclAlltoAllVC`
- **点对点通信算子（3 个）**：`HcclSend`、`HcclRecv`、`HcclBatchSendRecv`

各算子的函数原型、参数说明、数据类型支持与约束详见 [通信算子接口](./comm_op_interface/README.md)。

### hccl_mc2.h

声明 MC2 自定义算子框架的 C 接口，支撑开发者基于 HCCL 通信资源构建自定义融合算子。

- **OpArgs 参数对象（8 个）**：`HcclKfcAllocOpArgs` / `HcclKfcFreeOpArgs` 负责参数对象分配与释放；`HcclKfcOpArgsSetSrcDataType` / `HcclKfcOpArgsSetDstDataType` / `HcclKfcOpArgsSetReduceType` / `HcclKfcOpArgsSetCount` / `HcclKfcOpArgsSetAlgConfig` / `HcclKfcOpArgsSetCommEngine` 负责设置源/目的数据类型、归约类型、元素数、算法配置与通信引擎
- **通信资源上下文（1 个）**：`HcclCreateOpResCtx` 创建自定义算子的通信资源上下文

## 库文件

| 库文件 | 类型 | 说明 |
|--------|------|------|
| `libhccl.so` | 动态库 | HCCL 主库，默认构建产物，AI 框架与上层应用运行时链接 |
| `libhccl_static.a` | 静态库 | 静态构建产物，由 `bash build.sh --static`（`-DSTATIC_MODE=ON`）触发 |
| `cann-hccl_<version>_linux-<arch>.run` | 安装包 | 一键安装包，包含 `libhccl.so`、对外头文件与 `aicpu_hccl.tar.gz` 等组件 |
| `cann-hccl-static_<version>_linux-<arch>.tar.gz` | 静态库 tar 包 | 静态库归档包，由静态构建模式产出 |
| `aicpu_hccl.tar.gz` | AICPU 算子包 | HCCL AICPU 算子包，随 `.run` 安装包发布，含 Device 侧 `libscatter_aicpu_kernel.so` |

- `<version>`：HCCL 版本号，由 `version.cmake` 的 `set_cann_package(hccl VERSION ...)` 决定。
- `<arch>`：目标架构，`x86_64` 或 `aarch64`。

> 自定义算子构建（`bash build.sh --custom_ops_path=<PATH>`）额外产出 `cann-hccl_custom_<name>_linux-<arch>.run`，属开发者定制产物，不在标准对外发布范围内。

## 安装后目录结构

通过 `cann-hccl_<version>_linux-<arch>.run` 安装后，对外头文件与库文件落入 CANN 安装路径下 `<arch>-linux/` 子目录（`${INSTALL_PATH}` 默认为 `/usr/local/Ascend/ascend-toolkit/<version>`，`<arch>` 为 `x86_64` 或 `aarch64`）：

```text
${INSTALL_PATH}/<arch>-linux/
├── include/
│   └── hccl/
│       ├── hccl.h              # 14 个通信算子 C 接口
│       └── hccl_mc2.h          # MC2 自定义算子框架接口
└── lib64/
    └── libhccl.so              # HCCL 主动态库
```

静态构建模式下额外生成 `libhccl_static.a`。`aicpu_hccl.tar.gz` 由安装包按需释放到 Device 侧算子目录。

## 版本与依赖

- **当前版本**：9.1.0（见 `version.cmake`）
- **构建依赖**：`hcomm >= 8.5`、`runtime >= 8.5`、`metadef >= 8.5`、`bisheng-compiler >= 8.5`、`asc-devkit >= 8.5`
- **运行依赖**：`hcomm >= 8.5`、`runtime >= 8.5`、`metadef >= 8.5`

> HCCL 与 HCOMM 版本独立演进。当 `CANN_VERSION_NUM < 90000000` 时，HCCL 启用 `HCCL_CANN_COMPAT_850=ON` 前向兼容模式。源码编译、安装与上板测试流程详见 [构建指南](../build/build.md)。
