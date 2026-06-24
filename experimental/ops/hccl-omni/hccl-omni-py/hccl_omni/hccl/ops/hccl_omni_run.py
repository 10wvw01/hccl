# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------

from typing import Optional

import torch
from torch.library import impl

from ..op_builder.builder import OpBuilder
from ..op_builder.builder import AS_LIBRARY


# =========================================================================
# OpBuilder: schema + Meta registration + JIT compilation
# =========================================================================

class _HcclOmniRunOpBuilder(OpBuilder):
    """
    OpBuilder for torch.ops.npu.hccl_omni_run().

    Parameter order matches the C++ HcclOmniRun API:
      sendBuf, recvBuf, sendType, recvType, xmlPath, opParam, ..., comm, stream
    """

    def __init__(self):
        super(_HcclOmniRunOpBuilder, self).__init__("hccl_omni_run")

    def sources(self):
        return ['csrc/hccl_omni_run.cpp']

    def schema(self) -> str:
        # Parameter order matches C++ HcclOmniRun:
        #   sendBuf, recvBuf, sendType, recvType, xmlPath, opParam, ..., comm, stream
        return (
            "hccl_omni_run(Tensor send_buf, Tensor recv_buf, "
            "int send_type, int recv_type, "
            "str xml_path, Tensor? op_param, "
            "int comm_handle, "
            "bool synchronize=False, int world_size=-1) -> Tensor"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def hccl_omni_run_meta(
            send_buf: torch.Tensor,
            recv_buf: torch.Tensor,
            send_type: int,
            recv_type: int,
            xml_path: str,
            op_param: torch.Tensor | None,
            comm_handle: int,
            synchronize: bool = False,
            world_size: int = -1,
        ):
            return torch.empty_like(recv_buf)


# Instantiate and load (triggers JIT compilation on first import)
_hccl_omni_run_op_builder = _HcclOmniRunOpBuilder()
_op_module = _hccl_omni_run_op_builder.load()


# =========================================================================
# PrivateUse1 backend — dispatches to JIT-compiled C++ module
# =========================================================================

@impl(AS_LIBRARY, _hccl_omni_run_op_builder.name, "PrivateUse1")
def _hccl_omni_run(
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    send_type: int,
    recv_type: int,
    xml_path: str,
    op_param: torch.Tensor | None,
    comm_handle: int,
    synchronize: bool = False,
    world_size: int = -1,
):
    return _op_module.hccl_omni_run(
        send_buf, recv_buf,
        send_type, recv_type,
        xml_path or "",
        op_param if op_param is not None else None,
        comm_handle,
        synchronize, world_size,
    )


# =========================================================================
# Public API — parameter order matches C++ HcclOmniRun
# =========================================================================

def hccl_omni_run(
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    send_type: int,
    recv_type: int,
    xml_path: str,
    op_param: torch.Tensor | None,
    comm_handle: int,
    *,
    synchronize: bool = False,
    world_size: int = -1,
) -> torch.Tensor:
    """
    HCCL OmniRun custom collective communication operator.

    Parameter order matches the C++ HcclOmniRun API:
      sendBuf, recvBuf, sendType, recvType, xmlPath, opParam, comm, stream

    Counts/displs are read from the op_param blob by the C++ implementation
    (OpParamBlob layout: 8-byte header + 4 x int64[128] arrays = 4104 bytes).

    Args:
        send_buf: Input tensor on NPU device, must be contiguous.
        recv_buf: Output tensor on NPU device, must be pre-allocated and contiguous.
        send_type: HcclDataType enum value (or -1 to infer from send_buf.dtype).
        recv_type: HcclDataType enum value (or -1 to infer from recv_buf.dtype).
        xml_path: XML algorithm config path. Pass "" for HcclAlltoAllV fallback.
        op_param: Operator parameter blob (required). Must be a uint8 CPU tensor
            with >= 4104 bytes. Use serialize_op_param() to create it.
        comm_handle: Opaque HcclComm handle (from CommContextManager or direct init).

    Keyword Args:
        synchronize: If True, calls aclrtSynchronizeStream after HCCL execution.
        world_size: Number of ranks. Used for validation.

    Returns:
        The recv_buf tensor, populated with received data after HCCL execution.
    """

    return torch.ops.npu.hccl_omni_run(
        send_buf, recv_buf,
        send_type, recv_type,
        xml_path or "",
        op_param if op_param is not None else None,
        comm_handle,
        synchronize, world_size,
    )
