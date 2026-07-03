from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from hcclgen.api import generate_hccl_xml


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_generate_hccl_xml_from_input_json(tmp_path) -> None:
    data = json.loads((EXAMPLES / "input_4x4_1dmesh_clos.json").read_text(encoding="utf-8"))
    data["output"] = {"hccl_xml": "out/generated.xml"}
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    artifact = generate_hccl_xml(
        input_path,
        time_limit=30.0,
        threads=4,
        quiet=True,
    )

    expected_output = tmp_path / "out" / "generated.xml"
    assert artifact.xml_path == expected_output
    assert artifact.result.completed
    assert artifact.result.status_name == "OPTIMAL"
    assert artifact.result.total_epochs == 3
    assert artifact.result.max_steps == 3
    assert artifact.result.step_times == [1, 2]
    assert expected_output.exists()
    assert ET.parse(expected_output).getroot().tag == "root"
