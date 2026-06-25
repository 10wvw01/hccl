# HCCL

## 🔥Latest News

- [2025/11/30] The HCCL project is officially open‑sourced.

## 🚀Overview

The Huawei Collective Communication Library (HCCL)​ is a high‑performance collective communication library designed for AI Processors. It delivers high‑performance, high‑reliability communication solutions for compute clusters and provides the following core capabilities:

- Provides high‑performance collective communication​ and point‑to‑point communication​ in both single‑node and multi‑node environments.
- Supports collective communication primitives such as AllReduce, Broadcast, AllGather, ReduceScatter, and AlltoAll.
- Supports communication algorithms including Ring, Mesh, and Recursive Halving‑Doubling (RHD).
- Supports high‑speed communication links such as HCCS, RoCE, and PCIe.
- Supports both single operator ​and graph  execution models.

HCCL is a core component of CANN. It supports multiple AI frameworks and enables communication between multiple AI processors. The following figure shows its software architecture.

<!-- <img src="" alt="hccl-architecture" style="width: 65%;  height:65%;"/> -->

HCCL consists of the HCCL and Huawei Communication (HCOMM) libraries.

- HCCL: includes built-in and extended communication operators, and provides external communication operator APIs.
- [HCOMM](https://gitcode.com/cann/hcomm): adopts a layered and decoupled design, dividing communication capabilities into a control plane​ and a data plane.

## Directory Structure

The key directories of this project are as follows:

```text
│── src                         # HCCL operator source code directory
|    ├── common                 # Common logic, including type definitions and log modules
|    └── ops                    # HCCL operator implementation
|        ├── all_gather         # AllGather operator implementation
|        ├── all_gather_v       # AllGatherV operator implementation
|        ├── all_reduce         # AllReduce operator implementation
|        ├── all_to_all_v       # AlltoAll, AlltoAllV, and AlltoAllVC operator implementation
|        ├── batch_send_recv    # BatchSendRecv operator implementation
|        ├── broadcast          # Broadcast operator implementation
|        ├── op_common          # Common operator components
|        │   ├── executor       # Executor
|        │   ├── selector       # Algorithm selector
|        │   ├── template       # Algorithm template
|        │   └── topo           # Communicator topology acquisition and conversion
|        ├── recv               # Recv operator implementation
|        ├── reduce             # Reduce operator implementation
|        ├── reduce_scatter     # ReduceScatter operator implementation
|        ├── reduce_scatter_v   # ReduceScatterV operator implementation
|        ├── scatter            # Scatter operator implementation
|        └── send               # Send operator implementation
├── include                     # HCCL external header file
├── test                         # Test code directory 
|   ├── ut                      # Unit test code directory
|   └── st                      # System test code directory
├── docs                        # Documentation directory
├── examples                    # Sample code directory
└── build.sh                    # Build script
```

## 📝Version Mapping

This project source code is released alongside CANN software versions. For details about the mapping between CANN software versions and project tags, see the release notes in [Release Management](https://gitcode.com/cann/release-management).
To ensure smooth source code customization, select the matching CANN version and GitCode tag source code. Using the master branch may cause version mismatch.

## 📌 Quick Start

To quickly build and experience this project, visit the following quick guide.

- Build from Source Code: Learn how to compile and install the project and perform basic tests and verification.
- [Sample running](./examples/README_en.md): Follow the detailed sample code and step-by-step instructions for quick experience.

## 📖Learning Resources

HCCL provides user guides, communication operator development guides, technical articles, and training videos. For details, see [HCCL references](./docs/README.md).

## 📝 Related Information

- [Contributions](CONTRIBUTING.md)
- [Security Statement](SECURITY.md)
- [Licenses](LICENSE)
