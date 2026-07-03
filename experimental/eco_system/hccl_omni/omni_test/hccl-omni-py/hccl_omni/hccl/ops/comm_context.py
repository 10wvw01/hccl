# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------

"""
CommContextManager — bridge between torch.distributed groups and HcclOmniRun.

Provides two paths:
  1. From torch.distributed group: auto-detects comm name, resolves HcclComm
  2. From direct HcclComm handle: bypasses torch.distributed entirely (COMM-03)

Backend auto-detection maps SoC name to communication backend
(Ascend910B → KFC, Ascend950 → Channel).
"""

from typing import Dict, Optional, Union, TypeAlias

import torch
import torch_npu

from ..op_builder.builder import OpBuilder
from ..op_builder.builder import AS_LIBRARY


class _CommContextOpBuilder(OpBuilder):
    """JIT-compiles the comm_context C++ extension on first load."""

    def __init__(self):
        super(_CommContextOpBuilder, self).__init__("comm_context")

    def sources(self):
        return ['csrc/comm_context.cpp']

    def schema(self) -> None:
        # CommContextManager is not a TORCH_LIBRARY operator — no schema needed
        return None

    def register_meta(self):
        pass


_comm_context_op_builder = _CommContextOpBuilder()
_op_module = _comm_context_op_builder.load()
CommContextManager: TypeAlias = _op_module.CommContextManager


# Default backend mapping: SoC substring → backend mode
DEFAULT_BACKEND = {
    "Ascend910B":   "kfc",
    "Ascend910_93": "kfc",
    "Ascend950":    "channel",
}


def create_comm_context(
    group: torch.distributed.distributed_c10d.ProcessGroup,
    backend: Optional[Union[str, Dict[str, str]]] = None,
) -> CommContextManager:
    """
    Create a CommContextManager from a torch.distributed process group.

    Automatically extracts the HCCL communicator name from the group,
    detects the correct backend from SoC name, and resolves the HcclComm handle.

    Args:
        group: A torch.distributed process group using HCCL backend.
        backend: Backend specification. String ("kfc"/"channel") or dict
                 mapping SoC substrings to backend names. Default auto-detects
                 from Ascend chip type.

    Returns:
        CommContextManager instance with resolved HcclComm handle.
    """
    if backend is None:
        backend = DEFAULT_BACKEND

    rank_id = torch.distributed.get_rank(group)
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
    world_size = torch.distributed.get_world_size(group)

    print(f'{group_name=}')

    return CommContextManager(group_name, world_size, rank_id, backend)
