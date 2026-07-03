from __future__ import annotations

import xml.etree.ElementTree as ET
import json
import subprocess
import sys
from pathlib import Path

from hcclgen.global_milp import solve_scip_allgather
from hcclgen.hccl_xml import write_hccl_xml
from hcclgen.input_schema import build_topology_from_input, load_hcclgen_input

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_writes_pair_exchange_hccl_xml(tmp_path) -> None:
    topology = build_topology_from_input(
        load_hcclgen_input(EXAMPLES / "input_4x4_1dmesh_clos.json")
    )
    result = solve_scip_allgather(
        topology,
        chunk_factor=1,
        total_epochs=3,
        max_steps=4,
        time_unit="slowest",
        route_dag_mode="spdag",
        time_limit=30.0,
        threads=4,
        quiet=True,
    )

    output_path = tmp_path / "schedule.xml"
    write_hccl_xml(output_path, topology, result, chunks_per_rank=1)

    root = ET.parse(output_path).getroot()
    assert root.tag == "root"
    npus = root.findall("./NPU")
    assert len(npus) == 16

    rank0 = root.find("./NPU[@rankId='0']")
    assert rank0 is not None
    res_request = rank0.find("./instruction[@opCode='resRequest']")
    assert res_request is not None
    assert int(res_request.attrib["netLayerNum"]) >= 2

    send_recv_writes = root.findall(".//instruction[@opCode='SendRecvWrite']")
    assert send_recv_writes
    assert {item.attrib["netLayerId"] for item in send_recv_writes} >= {"0", "1"}

    for instruction in send_recv_writes:
        dst_slices = instruction.findall("./dstSlice")
        assert dst_slices
        for dst_slice in dst_slices:
            assert dst_slice.attrib["rankId"] == dst_slice.attrib["recvRankId"]

    local_copies = root.findall(".//instruction[@opCode='LocalCopy']")
    assert local_copies


def test_cli_uses_output_hccl_xml_from_input_json(tmp_path) -> None:
    source = EXAMPLES / "input_4x4_1dmesh_clos.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data["output"] = {"hccl_xml": "out/from_config.xml"}
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "hcclgen.global_milp",
            "--input-json",
            str(input_path),
            "--total-epochs",
            "3",
            "--time-limit",
            "30",
            "--threads",
            "4",
            "--quiet",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    output_path = tmp_path / "out" / "from_config.xml"
    assert output_path.exists()
    assert f"hccl_xml: {output_path}" in completed.stdout
    assert "used_steps: 2" in completed.stdout
    assert "'step_len': 3" in completed.stdout
    assert ET.parse(output_path).getroot().tag == "root"
