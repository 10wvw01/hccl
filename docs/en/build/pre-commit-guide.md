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
- **clang-format**: 14.0+ (code formatting tool)
- **Java**: 17+ (OAT dependency, which can be automatically installed)
- **Maven**: 3.6+ (OAT dependency, which can be automatically installed)

## Installation Procedure

### 1. Install `pre-commit`

```bash
# Method 1: Use pip
pip install pre-commit

# Method 2: Use the system package manager (Ubuntu/Debian)
sudo apt install pre-commit
```

### 2. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt install clang-format openjdk-17-jre maven

# macOS
brew install clang-format openjdk@17 maven
```

### 3. Install Git Hooks in the Project Path

```bash
# Go to the root directory of the code repository
cd /path/to/hccl
pre-commit install
```

After the installation, the following information is displayed:

```
pre-commit installed at .git/hooks/pre-commit
```

## Usage

### Automatic Check (Recommended)

Each time you run `git commit`, `pre-commit` automatically performs checks.

```bash
git add .
git commit -m "your commit message"
```

Output:

```
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

### Skip Check (in Emergency)

```bash
git commit --no-verify -m "emergency fix"
```

> Note: Use this option only in emergencies. In normal development, ensure that all checks pass.

## Check Items

### 1. clang-format

Automatically formats C/C++ code according to the [.clang-format](../../.clang-format) configuration in the project root directory.

### 2. OAT Compliance Check

Open-source compliance check conducted by Open Source Audit Tool (OAT)

| Check Item        | Description                          |
| -------------- | ------------------------------ |
| License header check  | Ensure that the source file contains the CANN license header.|
| Binary file check| Binary file commits are not allowed.            |
| Archived file check  | Archived file commits (such as .zip and .tar files) are not allowed.   |

The OAT script automatically performs the following operations upon the first run:

1. Check/Install Java 17.
2. Check/Install Maven.
3. Clone and build `tools_oat` (about 1 to 2 minutes).

## FAQ

### Q1: Why is OAT check slow on the first commit?

Cause: OAT needs to be cloned and built on the first run.

Solution: This is normal. Subsequent commits will use the cached JAR, which will be much faster.

## References

- [pre-commit official documentation](https://pre-commit.com/)
- [clang-format configuration](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [OAT](https://gitcode.com/openharmony-sig/tools_oat)
- [Guide to integrating pre-commit into the code repository](https://gitcode.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md)
