# pre-commit工具使用指导

## 概述

pre-commit是一个Git Hooks框架，用于在 `git commit` 时自动运行代码检查和格式化工具。本项目已配置以下检查：

| Hook             | 功能             | 说明                             |
| ---------------- | ---------------- | -------------------------------- |
| **clang-format** | C/C++ 代码格式化 | 自动格式化代码，保持风格一致     |
| **OAT Check**    | 开源合规检查     | 检测许可证头、禁止二进制文件提交 |

## 环境要求

- **Git**: 2.0+
- **Python**: 3.8+
- **pip**: 用于安装pre-commit及OAT Python包

首次运行检查时需要访问网络。pre-commit会根据仓库配置准备clang-format 16.0.0，OAT检查脚本会在缺少依赖时自动安装`oat-py>=1.0.1`。

## 安装步骤

### 1. 安装pre-commit

```bash
# 方式一: 使用pip
pip install pre-commit

# 方式二: 使用系统包管理器 (Ubuntu/Debian)
sudo apt install pre-commit
```

### 2. 项目路径下安装Git Hooks

```bash
# 进入代码仓根目录
cd /path/to/hccl
pre-commit install
```

安装成功后会显示：

```text
pre-commit installed at .git/hooks/pre-commit
```

## 使用方法

### 自动检查（推荐）

每次执行 `git commit` 时，pre-commit会自动运行检查：

```bash
git add .
git commit -m "your commit message"
```

输出示例：

```text
clang-format.............................................................Passed
OAT Compliance Check.....................................................Passed
```

### 手动运行检查

```bash
# 运行所有检查
pre-commit run

# 运行特定类型检查
pre-commit run clang-format
pre-commit run oat-check

# 检查所有文件（不限于暂存区）
pre-commit run --all-files
```

### 跳过检查（紧急情况）

```bash
git commit --no-verify -m "emergency fix"
```

> **注意**: 仅在紧急情况下使用，正常开发流程应保证检查通过。

## 检查项说明

### 1. clang-format

pre-commit根据仓库配置使用clang-format 16.0.0自动格式化C/C++代码，并遵循项目根目录下的[.clang-format](../../../.clang-format)配置。

### 2. OAT Compliance Check

OAT (Open Source Audit Tool) 检查开源合规性：

| 检查项         | 说明                           |
| -------------- | ------------------------------ |
| 许可证头检查   | 确保源文件包含CANN License头 |
| 二进制文件检查 | 禁止提交二进制文件             |
| 归档文件检查   | 禁止提交zip/tar等归档文件    |

当前OAT检查脚本使用Python实现，要求Python 3.7或以上版本。首次运行时会自动：

1. 检测本地是否已安装`oat-py`；
2. 缺少依赖时通过pip安装`oat-py>=1.0.1`；
3. 对暂存文件（或手动指定的文件）执行增量合规扫描；
4. 将检查摘要写入`oat_reports/result.txt`。

## 常见问题

### Q1: 首次提交时OAT检查很慢

**原因**: 首次运行可能需要通过网络安装`oat-py`及其Python依赖。

**解决**: 这是正常现象，后续检查会复用已安装的Python包，速度会更快。若安装失败，请检查pip配置和网络连接后重试。

## 相关文档

- [pre-commit官方文档](https://pre-commit.com/)
- [clang-format配置](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [oat-py Python包](https://pypi.org/project/oat-py/)
- [代码仓集成pre-commit指导](https://gitcode.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md)
