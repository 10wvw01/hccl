# pre-commit Usage Guide

## Overview

`pre-commit` is a Git hooks framework that automatically runs code check and formatting tools during `git commit`. The following checks have been configured for this project.

 | Hook             | Function            | Description                            |
| ---------------- | ---------------- | -------------------------------- |
| **clang-format** | C/C++ code formatting| Automatically formats code to maintain consistent style.    |
| **OAT Check**    | Open-source compliance check    | Checks license headers and prohibits binary file commits.|

## Environment Requirements

- **Git**: 2.0+
- **Python**: 3.8+
- **pip**: Used to install pre-commit and the OAT Python package

Network access is required when the checks run for the first time. Based on the repository configuration, pre-commit prepares clang-format 16.0.0, and the OAT check script automatically installs `oat-py>=1.0.1` if the dependency is missing.

## Installation Procedure

### 1. Install `pre-commit`

```bash
# Method 1: Using pip
pip install pre-commit

# Method 2: Using system package manager (Ubuntu or Debian)
sudo apt install pre-commit
```

### 2. Install Git Hooks in the Project Directory

```bash
# Go to the repository root directory
cd /path/to/hccl
pre-commit install
```

After the installation, the following information is displayed:

```text
pre-commit installed at .git/hooks/pre-commit
```

## Usage

### Automatic Check (Recommended)

Pre-commit automatically runs checks each time you execute `git commit`:

```bash
git add .
git commit -m "your commit message"
```

Output:

```text
clang-format.............................................................Passed
OAT Compliance Check.....................................................Passed
```

### Manual Check

```bash
# Run all checks
pre-commit run

# Run specific type checks
pre-commit run clang-format
pre-commit run oat-check

# Check all files (not limited to the staging area)
pre-commit run --all-files
```

### Skipping Checks (Emergency)

```bash
git commit --no-verify -m "emergency fix"
```

> **Note**: Use this only in emergencies. During normal development, ensure that all checks pass.

## Check Item Description

### 1. clang-format

Based on the repository configuration, pre-commit uses clang-format 16.0.0 to automatically format C and C++ code according to the [.clang-format](../../../.clang-format) file in the project root directory.

### 2. OAT Compliance Check

OAT (Open Source Audit Tool) checks open-source compliance:

| Check Item | Description |
| -------------- | ------------------------------ |
| License header check | Ensures that source files contain a CANN License header |
| Binary file check | Prevents binary file submissions |
| Archive file check | Prevents submission of archive files such as zip and tar |

The current OAT check script is implemented in Python and requires Python 3.7 or later. During the first run, it automatically performs the following actions:

1. Checks whether `oat-py` is installed.
2. Installs `oat-py>=1.0.1` through pip if the dependency is missing.
3. Runs an incremental compliance scan on staged files or manually specified files.
4. Writes the check summary to `oat_reports/result.txt`.

## Common Issues

### Q1: The OAT check is slow during the first commit

**Cause**: The first run may need to install `oat-py` and its Python dependencies over the network.

**Solution**: This is expected. Subsequent checks reuse the installed Python packages and are faster. If the installation fails, check the pip configuration and network connection, and then retry.

## Related Documents

- [Pre-commit official documentation](https://pre-commit.com/)
- [Clang-format configuration](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [oat-py Python package](https://pypi.org/project/oat-py/)
- [Repository pre-commit integration guide (Chinese)](https://gitcode.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md)
