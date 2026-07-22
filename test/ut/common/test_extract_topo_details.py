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


class ExtractTopoDetailsSourceTest(unittest.TestCase):
    def test_rank_query_result_is_checked_before_output_is_read(self):
        source = (
            pathlib.Path(__file__).parents[3]
            / "src/ops/op_common/topo/topo_host.cc"
        ).read_text(encoding="utf-8")
        function = source.split("HcclResult ExtractTopoDetails", 1)[1].split(
            "HcclResult Is2DieFullMesh", 1
        )[0]

        query = re.search(
            r"ret\s*=\s*HcclRankGraphGetRanksByTopoInst\([^;]+;", function
        )
        self.assertIsNotNone(query)
        result_check = function.find("CHK_PRT_RET(ret != HCCL_SUCCESS", query.end())
        output_read = function.find("for (uint32_t rankIdx", query.end())
        self.assertGreaterEqual(result_check, 0)
        self.assertGreater(output_read, result_check)


if __name__ == "__main__":
    unittest.main()
