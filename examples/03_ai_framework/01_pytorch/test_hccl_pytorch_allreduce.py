# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).with_name("hccl_pytorch_allreduce_test.py")


def _load_example():
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.float32 = object()
    torch.arange = Mock(return_value="tensor")
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu = types.SimpleNamespace(
        set_device=Mock(), device_count=Mock(return_value=1)
    )
    dist = types.ModuleType("torch.distributed")
    dist.init_process_group = Mock()
    dist.all_reduce = Mock()
    dist.destroy_process_group = Mock()
    dist.ReduceOp = types.SimpleNamespace(SUM="sum")
    multiprocessing = types.ModuleType("torch.multiprocessing")
    multiprocessing.spawn = Mock()
    fake_modules = {
        "torch": torch,
        "torch_npu": torch_npu,
        "torch.distributed": dist,
        "torch.multiprocessing": multiprocessing,
    }
    with patch.dict(sys.modules, fake_modules):
        spec = importlib.util.spec_from_file_location(
            "hccl_pytorch_allreduce_test_under_test", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, dist


class RunHcclTest(unittest.TestCase):
    def test_success_destroys_process_group(self):
        module, dist = _load_example()
        module.run_hccl(0, 1, "127.0.0.1", 50001)
        dist.all_reduce.assert_called_once_with("tensor", op="sum")
        dist.destroy_process_group.assert_called_once_with()

    def test_allreduce_failure_propagates_and_destroys_process_group(self):
        module, dist = _load_example()
        failure = RuntimeError("allreduce failed")
        dist.all_reduce.side_effect = failure
        with self.assertRaises(RuntimeError) as context:
            module.run_hccl(0, 1, "127.0.0.1", 50001)
        self.assertIs(context.exception, failure)
        dist.destroy_process_group.assert_called_once_with()

    def test_init_failure_propagates_without_destroy(self):
        module, dist = _load_example()
        failure = RuntimeError("init failed")
        dist.init_process_group.side_effect = failure
        with self.assertRaises(RuntimeError) as context:
            module.run_hccl(0, 1, "127.0.0.1", 50001)
        self.assertIs(context.exception, failure)
        dist.destroy_process_group.assert_not_called()


if __name__ == "__main__":
    unittest.main()
