# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pathlib
import re
import unittest


class TopoMatchErrorPropagationSourceTest(unittest.TestCase):
    def test_rank_graph_queries_propagate_failures(self):
        source_root = pathlib.Path(__file__).parents[3] / "src/ops/op_common/topo"
        expected_checked_calls = {
            "topo_match_ubx_1d.cc": 1,
            "topo_match_ubx.cc": 1,
            "topo_match_3_level.cc": 1,
            "topo_match_multilevel.cc": 2,
            "topo_match_squeeze_2d.cc": 1,
            "topo_match_1d.cc": 1,
        }
        unchecked_call = re.compile(
            r"(?m)^\s*(?!CHK_RET\()HcclRankGraph(?:GetLinks|GetTopoTypeByLayer)\s*\("
        )
        checked_call = re.compile(
            r"(?m)^\s*CHK_RET\(HcclRankGraph(?:GetLinks|GetTopoTypeByLayer)\s*\("
        )

        for file_name, expected_count in expected_checked_calls.items():
            with self.subTest(file_name=file_name):
                source = (source_root / file_name).read_text(encoding="utf-8")
                self.assertEqual(unchecked_call.findall(source), [])
                self.assertEqual(len(checked_call.findall(source)), expected_count)


if __name__ == "__main__":
    unittest.main()
