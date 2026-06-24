# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------

import hashlib
import os
import tempfile
from abc import ABC, abstractmethod
from typing import List, Union

import torch
import torch.distributed as dist
import torch_npu
from torch.utils.cpp_extension import load
from torch.library import Library

ASCEND_HOME_PATH = "ASCEND_HOME_PATH"
AS_LIBRARY = Library("npu", "FRAGMENT")
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))


class OpBuilder(ABC):
    """
    Base class for building HCCL ops with TORCH_LIBRARY registration.

    Adapted from mega_moe OpBuilder pattern. Handles JIT compilation of C++
    extensions that call HCCL C APIs, schema registration via Library.define(),
    and backend registration via torch.library.impl.
    """
    _loaded_ops = {}

    def __init__(self, name: str):
        self.name = name
        self._cann_path = self.get_cann_path()
        self._package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.register_schema(self.schema())
        self.register_meta()

    def get_cann_path(self):
        if ASCEND_HOME_PATH in os.environ and os.path.exists(os.environ[ASCEND_HOME_PATH]):
            return os.environ[ASCEND_HOME_PATH]
        return None

    def get_absolute_paths(self, paths: List[str]):
        return [os.path.join(self._package_path, path) for path in paths]

    def register_schema(self, op_schema: Union[str, List[str], None]):
        if op_schema is None:
            return
        if isinstance(op_schema, str):
            op_schema = [op_schema]
        for schema in op_schema:
            AS_LIBRARY.define(schema)

    @abstractmethod
    def sources(self) -> List[str]:
        ...

    @abstractmethod
    def schema(self) -> str:
        ...

    @abstractmethod
    def register_meta(self):
        ...

    def include_paths(self):
        return [
            os.path.join(self._cann_path, 'include'),
            os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include')
        ]

    def cxx_args(self):
        args = [
            '-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack',
            '-fPIC', '-pie', '-s', '-fvisibility=hidden',
            '-D_FORTIFY_SOURCE=2', '-O2', '-w',
        ]
        # PYTH-06: enforce _GLIBCXX_USE_CXX11_ABI=0 for PyTorch wheel ABI compatibility
        if torch._C._GLIBCXX_USE_CXX11_ABI:
            args.append('-D_GLIBCXX_USE_CXX11_ABI=1')
        else:
            args.append('-D_GLIBCXX_USE_CXX11_ABI=0')
        return args

    def extra_ldflags(self):
        return [
            '-L' + os.path.join(PYTORCH_NPU_INSTALL_PATH, 'lib'),
            '-ltorch_npu'
        ]

    def _source_hash(self) -> str:
        """Compute a short hash of all C++ source files for cache-busting."""
        h = hashlib.md5()
        for src in self.get_absolute_paths(self.sources()):
            with open(src, 'rb') as f:
                h.update(f.read())
        return h.hexdigest()[:8]

    @staticmethod
    def _get_rank_suffix() -> str:
        """Get a rank-based suffix for build directory isolation.

        Each rank compiles into its own sub-directory to avoid lock-file
        races between concurrent torchrun processes.  Uses global rank
        when torch.distributed is initialized; falls back to PID otherwise.
        """
        if dist.is_available() and dist.is_initialized():
            return f'_rank{dist.get_rank()}'
        return f'_pid{os.getpid()}'

    def _resolve_build_directory(self):
        """Determine the JIT build directory.

        Each rank gets its own build sub-directory (via rank suffix) so
        concurrent torchrun processes never race on lock files.

        JIT cache control via HCCL_OMNI_JIT_CACHE env var:
          HCCL_OMNI_JIT_CACHE=1  — use default torch_extensions directory
                                    (ninja incremental build, fast startup,
                                    but assumes single-process or externally
                                    coordinated compilation)
          Default (unset or 0)   — embed source hash + rank suffix in
                                    directory name: source change → new hash
                                    → fresh compile; same source → ninja
                                    reuses cached .so within that rank's dir
        """
        use_cache = os.environ.get('HCCL_OMNI_JIT_CACHE', '0') == '1'
        if use_cache:
            return None  # torch default path
        build_directory = os.path.join(
            tempfile.gettempdir(),
            f'torch_extensions_{self.name}_{self._source_hash()}{self._get_rank_suffix()}'
        )
        os.makedirs(build_directory, exist_ok=True)
        return build_directory

    def load(self, verbose: bool = True):
        if self.name in __class__._loaded_ops:
            return __class__._loaded_ops[self.name]

        build_directory = self._resolve_build_directory()

        op_module = load(
            name=self.name,
            sources=self.get_absolute_paths(self.sources()),
            extra_include_paths=self.get_absolute_paths(self.include_paths()),
            extra_cflags=self.cxx_args(),
            extra_ldflags=self.extra_ldflags(),
            verbose=verbose,
            build_directory=build_directory,
        )
        __class__._loaded_ops[self.name] = op_module
        return op_module
