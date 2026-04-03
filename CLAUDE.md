# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Startup Checklist

At the beginning of each session, read these key files to understand the current development context:

### Core Infrastructure
- [src/common/utils.h](src/common/utils.h): OMNI data structures (OpType, OmniSliceInfo, OmniSendRecvInfo, XmlInfo) used across all OMNI implementations

### AlltoAllV Operator Development
- [src/ops/all_to_all_v/executor/ins_omni_sole_executor.cc](src/ops/all_to_all_v/executor/ins_omni_sole_executor.cc): OMNI executor implementation with XML parsing and resource calculation
- [src/ops/all_to_all_v/executor/ins_omni_sole_executor.h](src/ops/all_to_all_v/executor/ins_omni_sole_executor.h): OMNI executor interface and registration

### AICPU Backend Templates
- [src/ops/all_to_all_v/template/aicpu/ins_temp_all_to_all_v_omni.cc](src/ops/all_to_all_v/template/aicpu/ins_temp_all_to_all_v_omni.cc): AICPU OMNI template with complete signal handling (15 OpType variants including local copy, send/recv with read/write/reduce, and group operations with loop-based alternatives)
- [src/ops/all_to_all_v/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc](src/ops/all_to_all_v/template/aicpu/ins_temp_all_to_all_v_mesh_1D.h): AICPU Mesh 1D template for AlltoAllV

### CCU Backend Templates
- [src/ops/all_to_all_v/template/ccu/ccu_temp_omni.cc](src/ops/all_to_all_v/template/ccu/ccu_temp_omni.h): CCU OMNI template with channel partitioning and DIE management
- [src/ops/all_to_all_v/template/ccu/ccu_temp_all_to_all_mesh2die.cc](src/ops/all_to_all_v/template/ccu/ccu_temp_all_to_all_mesh2die.h): CCU Mesh 2Die template for AlltoAll

### CCU Kernel Implementations
- [src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_omni.cc](src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_omni.h): CCU OMNI kernel with full signal type support (15 OpType variants, including native GroupBroadcast and GroupReduce operations)
- [src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_all_to_all_v_mesh2die.cc](src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_all_to_all_v_mesh2die.h): CCU AlltoAllV Mesh 2Die kernel implementation

## Build Commands

### Basic Build
The main build script is `build.sh` in the root directory:

```bash
# Compile host package only
bash build.sh --pkg

# Full build (host + device packages)
bash build.sh --pkg --full

# Build with custom third-party libraries path
bash build.sh --cann_3rd_lib_path={your_3rd_party_path}
```

### Testing
```bash
# Run unit tests (LLT)
bash build.sh --ut

# Run system tests
bash build.sh --st

# Run specific test targets
bash build.sh --open_hccl_test
bash build.sh --executor_hccl_test
bash build.sh --executor_reduce_hccl_test
bash build.sh --executor_pipeline_hccl_test
```

### Custom Operations
```bash
# Build custom operators
bash build.sh --custom_ops_path=<CUSTOM_OPS_PATH> --ops=<OPS> --vendor=<VENDOR>

# Build only AICPU kernel
bash build.sh --aicpu

# Build for ARM64/aarch64
bash build.sh --build_aarch
```

### Development Builds
```bash
# Debug build with ASAN
bash build.sh --asan --build-type=Debug

# Release build
bash build.sh --build-type=Release
```

### Output Location
- Built packages: `./build_out/cann-hccl_<version>_linux-<arch>.run`
- Intermediate build files: `./build/` and `./build_device/`
- Installed output: Default path is `/usr/local/Ascend/latest` for root users or `~/Ascend/latest` for non-root users

## Architecture Overview

### Project Context
HCCL (Huawei Collective Communication Library) is a high-performance collective communication library for Ascend AI processors. It's part of CANN (Compute Architecture for Neural Networks) and provides communication primitives for AI training clusters.

### Key Dependencies
- **CANN Toolkit**: Required for development (provides ACL, HCOMM libraries)
- **HCOMM**: Communication base library (control plane/data plane separation)
- **ACL**: Ascend Computing Language for low-level device operations
- **CMake 3.16+**: Build system with cross-compilation support

### Directory Structure
```
src/
├── common/              # Common utilities (logging, error handling, type definitions)
│   └── hcomm_dlsym/     # HCOMM dynamic loading mechanism
├── ops/                 # Communication operator implementations
│   ├── all_reduce/      # AllReduce operator
│   ├── broadcast/       # Broadcast operator  
│   ├── all_gather/      # AllGather operator
│   ├── reduce_scatter/  # ReduceScatter operator
│   ├── all_to_all_v/    # AlltoAll variants
│   ├── send/, recv/     # Point-to-point communication
│   └── op_common/       # Shared operator components
│       ├── executor/    # Execution framework (sequence, parallel, concurrent)
│       ├── selector/    # Algorithm selection logic
│       ├── template/    # Algorithm templates (AICPU, AIV, CCU, DPU backends)
│       └── topo/        # Topology discovery and matching
├── include/             # Public API headers (hccl.h, hccl_mc2.h)
└── interface_graph_mode/ # Graph mode execution interface
```

### Core Design Patterns

#### Plugin Architecture
Each communication operator follows a plugin structure:
- **Operator Entry**: `operator_op.cc/h` - Main operator implementation
- **Executor Layer**: Manages execution strategy (sequence, parallel, concurrent)
- **Selector Layer**: Chooses optimal algorithm based on parameters
- **Template Layer**: Hardware-specific algorithm implementations

#### Hardware Abstraction
Multiple backend support through template system:
- **AICPU**: AI CPU backend for host-side execution
- **AIV**: AI Vector backend for vectorized operations  
- **CCU**: Communication Control Unit backend
- **DPU**: Data Parallel Unit backend

#### Execution Flow
1. Parameter validation → 2. Algorithm selection → 3. Resource allocation → 4. Topology matching → 5. Task scheduling → 6. Kernel execution → 7. Result return

### OMNI Feature Status

OMNI (Optimized Multi-Node Interface) is an advanced communication pattern that provides fine-grained control over collective operations through XML-defined signal sequences.

#### Key Data Structures
- `OpType` enum (15 variants): Defines all supported operation types (see [utils.h:25-42](src/common/utils.h#L25-L42))
- `OmniSendRecvInfo`: Signal information structure with source/destination slice details
- `XmlInfo`: XML configuration containing signal sequences and resource information

#### AICPU Implementation
- **Complete Support**: All 15 OpType variants implemented via `Handle*()` functions (see [ins_temp_all_to_all_v_omni.cc:120-191](src/ops/all_to_all_v/template/aicpu/ins_temp_all_to_all_v_omni.cc#L120-L191))
- **XML Integration**: XML information passed via `CalcRes()` to template layer
- **Group Operations**: `GroupBroadcast` and `GroupReduce` implemented using loop-based point-to-point alternatives (see lines 995-998 and 1062-1066)
- **Channel Mapping**: `rankId2Channel_` lookup table for rank-to-channel resolution
- **Slice Address Calculation**: `base address + sliceIdx * processSize_` based on sliceType (0=input, 1=output, 2=cclbuf)

#### CCU Implementation
- **Native Support**: Full OpType support including native `GroupBroadcast` and `GroupReduce` operations
- **Kernel-Level Implementation**: Complete signal handling in `CcuKernelOmni::DoRepeatOmni()`
- **Reduce Loop Support**: Specialized reduce loop infrastructure for efficient reduction operations

#### Development Guidelines
1. **XML Signal Processing**: Implement `DoRepeatOmni()` with switch-case for all OpType variants
2. **Channel Management**: Build and maintain `rankId2Channel_` mapping for rank-to-channel resolution
3. **Error Handling**: Validate channel existence, slice information, and address validity
4. **Backend Compatibility**: Ensure consistent behavior across AICPU, CCU, and AIV backends
5. **Performance Considerations**: Loop-based alternatives in AICPU may have different performance characteristics than native CCU implementations

### Key Source Files

#### Common Infrastructure
- [src/common/log.cc](src/common/log.cc): Unified logging system
- [src/common/adapter_acl.cc](src/common/adapter_acl.cc): ACL interface adapter
- [src/common/device_compat.cc](src/common/device_compat.cc): Device compatibility layer

#### Operator Framework
- [src/ops/op_common/executor/executor_base.cc](src/ops/op_common/executor/executor_base.cc): Base executor class
- [src/ops/op_common/selector/auto_selector_base.cc](src/ops/op_common/selector/auto_selector_base.cc): Algorithm selector base
- [src/ops/op_common/template/alg_template_base.cc](src/ops/op_common/template/alg_template_base.cc): Algorithm template base

#### Topology Management
- [src/ops/op_common/topo/topo.cc](src/ops/op_common/topo/topo.cc): Topology base class
- [src/ops/op_common/topo/topo_match_1d.cc](src/ops/op_common/topo/topo_match_1d.cc): 1D topology matching

## Development Workflow

### Adding a New Operator
1. Create directory under `src/ops/` with operator name
2. Implement operator entry in `operator_op.cc/h`
3. Add executor implementations in `executor/` subdirectory
4. Add algorithm templates in `template/` subdirectory (AICPU/AIV/CCU/DPU)
5. Update `src/CMakeLists.txt` to include new source files
6. Add unit tests in `test/ut/`

### Cross-Compilation Support
The build system supports cross-compilation for ARM64 (aarch64) Ascend hardware:
- Set `AARCH_MODE=ON` in CMake
- Requires cross-compilation toolchain from CANN Toolkit
- Target architecture is hardcoded to aarch64-linux-gnu

### Custom Operator Development
Custom operators can be built using the `--custom_ops_path`, `--ops`, and `--vendor` flags. The build system creates separate host and device packages for custom operators.

## Testing Strategy

### Unit Tests (LLT)
- Located in `test/ut/`
- Focus on individual module functionality
- Built with `-DENABLE_UT=ON`
- Executed via `bash build.sh --ut`

### System Tests
- Located in `test/st/`
- End-to-end communication validation
- Includes algorithm correctness verification in `test/st/algorithm/`
- Uses simulation framework in `test/st/algorithm/utils/src/`

### Test Infrastructure
- **HCLL Proxy**: Communication layer simulation
- **Topology Model**: Hardware topology simulation  
- **Verifier**: Communication semantic validation

## Important Configuration Notes

### Environment Variables
Must source CANN environment before building:
```bash
source /usr/local/Ascend/cann/set_env.sh  # Default root installation
# OR
source ${install_path}/cann/set_env.sh    # Custom installation
```

### Third-Party Dependencies
Automatically downloaded during build unless specified via `--cann_3rd_lib_path`:
- makeself 2.5.0 (packaging)
- googletest 1.14.0 (testing)

### Signing and Security
- Device packages require signing for secure loading
- Use `--enable-sign` and `--sign-script` for custom signing
- Community builds may need to disable signature verification on test hardware

## Common Tasks

### Debugging Build Issues
1. Check CANN installation path is correct
2. Verify environment variables are sourced
3. Check third-party libraries are available
4. Use `--build-type=Debug` for debug symbols

### Performance Optimization
- Algorithm selection happens at runtime based on data size and topology
- Multiple algorithm implementations (Ring, Mesh, RHD) available
- Topology-aware communication path optimization

### Adding New Algorithm
1. Create template class inheriting from `AlgTemplateBase`
2. Implement hardware-specific versions (AICPU, AIV, CCU, DPU)
3. Register with template registry
4. Update selector logic to consider new algorithm

## Version Compatibility

- Source code tags correspond to CANN release versions
- Master branch may have version mismatch with released CANN
- Check [release repository](https://gitcode.com/cann/release-management) for version mapping