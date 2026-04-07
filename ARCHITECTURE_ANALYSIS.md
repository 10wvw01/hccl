# HCCL 项目架构分析与设计文档

## 1. 项目概述

### 1.1 项目简介
HCCL（Huawei Collective Communication Library，昇腾集合通信库）是基于昇腾AI处理器的高性能集合通信库，为计算集群提供高性能、高可靠的通信方案。作为CANN（Compute Architecture for Neural Networks）的核心组件，HCCL对上支持多种AI框架，对下使能多款昇腾AI处理器之间的通信能力。

### 1.2 核心功能
- 提供单机、多机环境中的高性能集合通信和点对点通信
- 支持AllReduce、Broadcast、AllGather、ReduceScatter、AlltoAll等集合通信原语
- 支持Ring、Mesh、Recursive Halving-Doubling（RHD）等通信算法
- 支持HCCS、RoCE、PCIe等高速通信链路
- 支持单算子和图模式两种执行模式

## 2. 技术栈与依赖

### 2.1 核心依赖
- **CANN**: 华为昇腾计算架构，HCCL作为其核心组件
- **ACL (Ascend Computing Language)**: 昇腾计算语言，提供底层计算接口
- **HCOMM (Huawei Communication)**: 通信基础库，采用分层解耦设计
- **CMake**: 构建系统，支持跨平台构建

### 2.2 开发语言与标准
- **C++11/14**: 主要开发语言
- **C API**: 对外提供C语言接口，兼容多种编程语言
- **CUDA/ROCm 兼容接口**: 类似NCCL的API设计，便于迁移

### 2.3 目标架构
- **ARM64 (aarch64)**: 主要目标架构
- **昇腾AI处理器**: 华为自研AI加速器
- **Linux系统**: 主要支持的操作系统

## 3. 架构设计

### 3.1 整体架构分层

```
┌─────────────────────────────────────────────┐
│              AI框架层 (PyTorch, TensorFlow)  │
├─────────────────────────────────────────────┤
│            HCCL集合通信库 (用户API层)        │
├─────────────────────────────────────────────┤
│  算子实现层 (AllReduce, Broadcast等具体实现) │
├─────────────────────────────────────────────┤
│ 执行器/选择器/模板层 (算法编排与资源管理)     │
├─────────────────────────────────────────────┤
│      HCOMM通信基础库 (控制面/数据面分离)     │
├─────────────────────────────────────────────┤
│      昇腾硬件抽象层 (ACL/设备驱动)           │
└─────────────────────────────────────────────┘
```

### 3.2 核心模块设计

#### 3.2.1 公共模块 (`src/common/`)
- **日志模块**: `log.h/log.cc` - 统一日志系统
- **参数检查**: `param_check.h/cc` - 输入参数验证
- **类型定义**: `hccl_common.h` - 公共类型和常量
- **适配器层**: `adapter_acl.h/cc` - ACL接口适配
- **兼容性处理**: `compat.cc`, `device_compat.cc`
- **HCOMM动态加载**: `hcomm_dlsym/` - 通信库动态加载机制

#### 3.2.2 算子模块 (`src/ops/`)
采用插件化设计，每个算子独立实现：
- **AllReduce**: `all_reduce/` - 全局归约操作
- **Broadcast**: `broadcast/` - 广播操作
- **AllGather**: `all_gather/` - 全局收集操作
- **ReduceScatter**: `reduce_scatter/` - 归约分散操作
- **AlltoAll**: `all_to_all_v/` - 全交换操作
- **Send/Recv**: `send/`, `recv/` - 点对点通信
- **BatchSendRecv**: `batch_send_recv/` - 批量发送接收

每个算子内部遵循统一结构：
```
operator/
├── operator_op.cc/h      # 算子入口
├── executor/             # 执行器实现
│   ├── sequence_executor.cc/h
│   ├── parallel_executor.cc/h
│   └── concurrent_executor.cc/h
├── selector/            # 算法选择器
│   └── auto_selector.cc/h
└── template/            # 算法模板
    ├── aicpu/           # AI CPU模板
    ├── aiv/             # AI Vector模板
    └── ccu/             # 通信控制单元模板
```

#### 3.2.3 算子公共组件 (`src/ops/op_common/`)
提供可复用的基础组件：

1. **执行器框架** (`executor/`)
   - `executor_base.h/cc` - 执行器基类
   - `executor_v2_base.h/cc` - V2执行器基类
   - `channel/` - 通信通道管理
   - `registry/` - 执行器注册机制

2. **选择器框架** (`selector/`)
   - `auto_selector_base.h/cc` - 自动选择器基类
   - `execute_selector.h/cc` - 执行选择器
   - `selector_registry.h/cc` - 选择器注册

3. **模板框架** (`template/`)
   - `alg_template_base.h/cc` - 算法模板基类
   - `alg_v2_template_base.h/cc` - V2模板基类
   - 各硬件后端实现：
     - `aicpu/` - AI CPU后端
     - `aiv/` - AI Vector后端
     - `ccu/` - 通信控制单元后端
     - `dpu/` - 数据并行单元后端

4. **拓扑管理** (`topo/`)
   - `topo.h/cc` - 拓扑基类
   - `topo_host.h/cc` - 主机拓扑
   - `topo_match_*.h/cc` - 多种拓扑匹配算法

### 3.3 设计模式应用

#### 3.3.1 策略模式 (Strategy Pattern)
- **算法选择器**: 根据输入参数自动选择最优通信算法
- **执行器选择**: 根据硬件拓扑选择序列、并行或并发执行策略

#### 3.3.2 模板方法模式 (Template Method Pattern)
- **算法模板**: 定义通信算法的骨架，子类实现具体步骤
- **执行流程**: 统一的任务调度和执行流程

#### 3.3.3 工厂模式 (Factory Pattern)
- **注册机制**: 动态注册和创建执行器、选择器、模板
- **插件化**: 支持第三方算法扩展

#### 3.3.4 适配器模式 (Adapter Pattern)
- **硬件适配**: 统一不同硬件后端的接口
- **协议适配**: 支持多种通信协议（HCCS、RoCE、PCIe）

### 3.4 通信流程

#### 3.4.1 算子执行流程
```
1. 参数验证 → 2. 算法选择 → 3. 资源分配 → 4. 拓扑匹配 → 5. 任务编排 → 6. 内核执行 → 7. 结果返回
```

#### 3.4.2 多级通信优化
- **层级化通信**: 支持0-4级算法层级，适应不同规模集群
- **拓扑感知**: 根据硬件拓扑优化通信路径
- **流水线执行**: 支持并发和并行执行，最大化硬件利用率

## 4. 关键设计决策

### 4.1 分层解耦设计
- **控制面与数据面分离**: 借鉴SDN思想，控制面负责调度，数据面负责传输
- **硬件抽象层**: 统一不同昇腾硬件的接口
- **插件化架构**: 支持算法和硬件的灵活扩展

### 4.2 性能优化策略
1. **算法自适应**: 根据数据大小、rank数量自动选择最优算法
2. **拓扑优化**: 利用硬件拓扑信息优化通信路径
3. **流水线并行**: 支持计算与通信重叠
4. **零拷贝技术**: 减少内存拷贝开销

### 4.3 可扩展性设计
- **多后端支持**: AICPU、AIV、CCU、DPU等多种计算单元
- **协议扩展**: 支持HCCS、RoCE、PCIe等多种通信协议
- **算法扩展**: 插件式算法注册机制

### 4.4 可靠性设计
- **错误处理**: 统一的错误码和异常处理机制
- **容错通信**: 支持通信失败重试和降级处理
- **资源管理**: 完善的资源申请、使用和释放机制

## 5. 测试与验证体系

### 5.1 测试架构
- **单元测试** (`test/ut/`): 模块级功能验证
- **系统测试** (`test/st/`): 端到端功能验证
- **算法验证** (`test/st/algorithm/`): 通信算法正确性验证

### 5.2 模拟测试
- **通信模拟**: `test/st/algorithm/utils/src/hccl_proxy/` - 通信层模拟
- **拓扑模拟**: `topo_model/` - 硬件拓扑模拟
- **验证器**: `hccl_verifier/` - 通信语义验证

## 6. 构建与部署

### 6.1 构建系统
- **CMake跨平台构建**: 支持本地和交叉编译
- **模块化编译**: 每个算子独立编译，支持选择性构建
- **第三方库管理**: 统一的第三方库集成机制

### 6.2 部署模式
- **动态库部署**: `libhccl.so` 动态链接库
- **静态库选项**: 支持静态链接
- **自定义算子**: 支持用户自定义算子扩展

## 7. 术语表

### 7.1 核心概念
- **HCCL**: Huawei Collective Communication Library，华为集合通信库
- **CANN**: Compute Architecture for Neural Networks，华为昇腾计算架构
- **HCOMM**: Huawei Communication，华为通信基础库
- **集合通信**: Collective Communication，多进程/多设备间的协同通信
- **通信原语**: Communication Primitives，基础的通信操作（如AllReduce、Broadcast等）

### 7.2 技术术语
- **Rank**: 进程或设备的唯一标识符
- **通信域**: Communication Domain，参与通信的一组rank集合
- **拓扑**: Topology，设备间的连接关系和通信路径
- **算法层级**: Algorithm Hierarchy，多级通信优化策略
- **执行器**: Executor，负责算法执行的组件
- **选择器**: Selector，负责算法选择的组件
- **模板**: Template，算法实现的标准化模板

### 7.3 硬件相关
- **昇腾AI处理器**: Ascend AI Processor，华为自研AI加速器
- **AICPU**: AI CPU，AI计算核心
- **AIV**: AI Vector，AI向量计算单元
- **CCU**: Communication Control Unit，通信控制单元
- **DPU**: Data Parallel Unit，数据并行单元
- **HCCS**: Huawei Cache Coherent System，华为缓存一致系统
- **RoCE**: RDMA over Converged Ethernet，融合以太网上的RDMA

### 7.4 算法相关
- **Ring算法**: 环形通信算法
- **Mesh算法**: 网格通信算法
- **RHD**: Recursive Halving-Doubling，递归减半加倍算法
- **NHR**: Non-Hierarchical Ring，非分层环形算法

## 8. 演进与扩展

### 8.1 架构演进方向
1. **云原生支持**: 容器化和Kubernetes集成
2. **异构计算**: 支持CPU、GPU、NPU混合计算
3. **智能调度**: AI驱动的通信调度优化
4. **安全通信**: 端到端加密通信支持

### 8.2 生态集成
- **AI框架集成**: PyTorch、TensorFlow、MindSpore等
- **调度系统集成**: Kubernetes、Slurm等
- **监控系统集成**: Prometheus、Grafana等

## 9. 总结

HCCL作为昇腾生态的核心通信组件，采用了先进的分层解耦和插件化架构设计，具有以下特点：

1. **高性能**: 支持多种优化算法和硬件加速
2. **高可靠**: 完善的错误处理和容错机制
3. **易扩展**: 插件化架构支持快速功能扩展
4. **易集成**: 标准API设计与主流AI框架兼容
5. **可维护**: 清晰的模块划分和设计模式应用

该架构为大规模AI训练提供了可靠、高效的通信基础，是昇腾计算生态的重要基石。