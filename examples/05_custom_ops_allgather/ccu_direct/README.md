# 自定义通信算子 - AllGather（直接编译版）

## 样例介绍

本样例展示如何基于 HCCL 通信编程接口开发 AllGather 通信算子，并将 host 侧接口与 CCU kernel 编排代码直接编译进测试程序。包含以下功能点：

1. 基于 CCU_SCHED 通信引擎实现 AllGather 集合通信算子
2. 不生成 `libhccl_custom_allgather.so`
3. `main.cc` 编译时直接包含 `op_host` 与 `op_kernel_ccu` 源文件

## 目录结构

```text
├── CMakeLists.txt                      # CMake 直接编译样例配置文件
├── op_host/
│   ├── allgather.cc                    # HcclAllGatherCustom 算子实现源文件
│   ├── utils.cc                        # 工具模块（通道获取、线程获取、Kernel注册）
│   └── utils.h                         # 工具模块头文件
├── op_kernel_ccu/
│   ├── ccu_kernel.cc                   # CCU Kernel 实现逻辑
│   ├── ccu_kernel.h                    # CCU Kernel 头文件
│   ├── exec_op.cc                      # CCU 算子编排逻辑
│   └── exec_op.h                       # CCU 算子编排头文件
├── inc/
│   ├── hccl_custom_allgather.h         # 自定义 allgather 算子接口头文件
│   ├── common.h                        # 公共类型头文件
│   └── log.h                           # 日志宏定义
└── testcase/
    ├── main.cc                         # 样例实现源文件
    └── Makefile                        # 旧版 Makefile 编译配置文件
```

## 一、环境准备

### 1. 环境要求

本样例支持以下昇腾产品：

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>

### 2. 安装 CANN Toolkit 开发套件包

参考 [昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)，安装最新版本 CANN Toolkit 开发套件包。

### 3. 配置环境变量

按需选择合适的命令使环境变量生效。
    
```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装，${install_path}表示CANN-Toolkit包实际安装路径
# source ${install_path}/cann/set_env.sh
```

## 二、编译执行样例

### 1. 编译样例

在 `examples/05_custom_ops_allgather/ccu_direct` 代码目录下执行如下命令：

```bash
# 编译样例
cmake -S . -B build
cmake --build build -j
```

`CMakeLists.txt` 会直接编译以下源文件并链接生成 `custom_allgather_ccu`：

```text
testcase/main.cc
op_host/allgather.cc
op_host/utils.cc
op_kernel_ccu/exec_op.cc
op_kernel_ccu/ccu_kernel.cc
```

链接依赖来自 CANN/HCCL 环境：

```text
hcomm
c_sec
ascendcl
acl_rt
```

### 2. 执行样例

在 `examples/05_custom_ops_allgather/ccu_direct` 代码目录下执行如下命令：
 	 
```bash
# 运行样例
cmake --build build --target run_custom_allgather_ccu

# 或直接执行样例二进制
./build/custom_allgather_ccu
```

### 3. 样例结果示例

所有节点的输入数据初始化为该节点的 DeviceId。运行成功后，终端将输出类似以下的日志信息（以 2 卡运行为例）：

```text
Found 2 NPU device(s) available
rankId: 1, input: [ 1 ]
rankId: 0, input: [ 0 ]
rankId: 0, output: [ 0 1 ]
rankId: 1, output: [ 0 1 ]
```
