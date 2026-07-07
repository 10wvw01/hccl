# 概述

HCCL（Huawei Collective Communication Library，华为集合通信库）基于昇腾 AI 处理器，提供单机/多机环境下的高性能集合通信与点对点通信能力，是 CANN 的核心组件之一。HCCL 与通信基础库 HCOMM 通过 `dlsym` 动态加载解耦、独立编译、独立版本演进，HCOMM 的对外头文件与库文件请参考 `cann/hcomm` 仓库的对应文档。

本章节介绍 HCCL 对外接口的头文件与库文件说明。

## 头文件和库文件说明

### 接口分类

HCCL 对外接口按功能分为以下类别：

**表 1** 接口分类

| 接口类别 | 描述 |
| --- | --- |
| 集合通信算子 | AllReduce、Broadcast、AllGather(V)、ReduceScatter(V)、Scatter、Reduce、AlltoAll(V)(C) 等 11 个集合通信接口。 |
| 点对点通信算子 | Send、Recv、BatchSendRecv 等 3 个点对点通信接口。 |
| MC2 自定义算子 | HcclKfc\* 参数对象与 HcclCreateOpResCtx 等 9 个 MC2（Kernel Fusion Custom）自定义算子框架接口。 |

### 调用接口依赖的头文件和库文件说明

安装固件、驱动及 CANN 软件包后，编译、运行应用程序时才能引用到 HCCL 接口的头文件、库文件。

您需要根据实际使用的 HCCL 接口来 include 依赖的文件，各头文件的用途如下表所示。

HCCL 接口的头文件在 `${INSTALL_DIR}/include/hccl/` 目录下，库文件在 `${INSTALL_DIR}/lib64/` 目录下。`${INSTALL_DIR}` 请替换为 CANN 软件安装后文件存储路径下对应架构的子目录（`<arch>-linux/`，`<arch>` 为 `x86_64` 或 `aarch64`）。以 root 用户安装为例，默认存储路径为：`/usr/local/Ascend/ascend-toolkit/<version>/<arch>-linux/`。

> **须知：**
> 编译 HCCL 接口程序时，请按照 include 的头文件依赖对应的库文件，如果引用多余的 so 文件，可能导致版本功能异常或后续版本升级时存在兼容性问题。

**表 2** 头文件列表

| 定义接口的头文件 | 用途 | 对应的库文件 |
| --- | --- | --- |
| hccl/hccl.h | 用于定义 AllReduce、Broadcast、AllGather(V)、ReduceScatter(V)、Scatter、Reduce、AlltoAll(V)(C)、Send、Recv、BatchSendRecv 等 14 个集合通信与点对点通信算子接口。 | libhccl.so |
| hccl/hccl_mc2.h | 用于定义 MC2 自定义算子框架接口，包括 HcclKfc\* 参数对象分配/设置与 HcclCreateOpResCtx 通信资源上下文创建等 9 个接口。 | libhccl.so |

HCCL 以 `cann-hccl_<version>_linux-<arch>.run` 安装包形式发布，包含 `libhccl.so`、对外头文件与 `aicpu_hccl.tar.gz`（HCCL AICPU 算子包，含 Device 侧 `libscatter_aicpu_kernel.so`）。静态构建模式（`bash build.sh --static`）额外产出 `libhccl_static.a` 与 `cann-hccl-static_<version>_linux-<arch>.tar.gz`。自定义算子构建（`--custom_ops_path`）额外产出 `cann-hccl_custom_<name>_linux-<arch>.run`，属开发者定制产物，不在标准对外发布范围内。

## 版本与依赖

- **当前版本**：9.1.0（见 `version.cmake`）
- **构建依赖**：`hcomm >= 8.5`、`runtime >= 8.5`、`metadef >= 8.5`、`bisheng-compiler >= 8.5`、`asc-devkit >= 8.5`
- **运行依赖**：`hcomm >= 8.5`、`runtime >= 8.5`、`metadef >= 8.5`

> 当 `CANN_VERSION_NUM < 90000000` 时，HCCL 启用 `HCCL_CANN_COMPAT_850=ON` 前向兼容模式。

源码编译与安装流程详见 [构建指南](../build/build.md)；各算子函数原型、参数说明、数据类型支持与约束详见 [通信算子接口](./comm_op_interface/README.md)。
