# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pathlib
import unittest


class IsHostDpuErrorTest(unittest.TestCase):
    def test_layer_query_failure_cannot_be_converted_to_true(self):
        source = (pathlib.Path(__file__).parents[3] / "src/ops/op_common/op_common.cc").read_text(encoding="utf-8")
        function = source.split("bool IsHostDpu", 1)[1].split("bool IsBarrierHostDpu", 1)[0]

        self.assertIn("ret = HcclRankGraphGetLayers", function)
        self.assertNotIn("CHK_RET(HcclRankGraphGetLayers", function)
        query = function.index("ret = HcclRankGraphGetLayers")
        failure = function.index("if (ret != HCCL_SUCCESS)", query)
        self.assertLess(failure, function.index("return false;", failure))


if __name__ == "__main__":
    unittest.main()
