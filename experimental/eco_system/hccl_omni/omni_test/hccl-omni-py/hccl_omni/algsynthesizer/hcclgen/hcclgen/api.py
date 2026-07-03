from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hcclgen.global_milp import GlobalMilpSchedule, solve_scip_allgather
from hcclgen.hccl_xml import write_hccl_xml
from hcclgen.input_schema import HcclGenInput, build_topology_from_input, load_hcclgen_input
from hcclgen.topology import (
    GridTopology,
    infer_allgather_input_capacity_t_lower_bound,
)


@dataclass(frozen=True)
class HcclXmlArtifact:
    spec: HcclGenInput
    topology: GridTopology
    result: GlobalMilpSchedule
    xml_path: Path


def generate_hccl_xml(
    input_json: Path | str,
    *,
    chunks: int = 1,
    total_epochs: int | None = None,
    max_steps: int | None = None,
    time_unit: str = "slowest",
    route_dag_mode: str = "spdag",
    clos_prune_degree: int | Literal["all"] = 1,
    output_xml: Path | str | None = None,
    require_pair_exchange: bool = True,
    time_limit: float = 60.0,
    threads: int = 0,
    quiet: bool = True,
) -> HcclXmlArtifact:
    input_path = Path(input_json)
    spec = load_hcclgen_input(input_path)
    topology = build_topology_from_input(
        spec,
        clos_prune_degree=clos_prune_degree,
    )
    effective_total_epochs = total_epochs
    if effective_total_epochs is None:
        effective_total_epochs = infer_allgather_input_capacity_t_lower_bound(
            topology,
            chunk_factor=chunks,
        )
    effective_max_steps = effective_total_epochs if max_steps is None else max_steps

    result = solve_scip_allgather(
        topology,
        chunk_factor=chunks,
        total_epochs=effective_total_epochs,
        max_steps=effective_max_steps,
        time_unit=time_unit,
        route_dag_mode=route_dag_mode,
        require_pair_exchange=require_pair_exchange,
        time_limit=time_limit,
        threads=threads,
        quiet=quiet,
    )
    xml_path = resolve_output_path(
        input_json=input_path,
        explicit_output=output_xml,
        config_output=spec.output.hccl_xml,
    )
    if xml_path is None:
        raise ValueError("未指定 HCCL XML 输出路径，请设置 output.hccl_xml 或 output_xml")
    write_hccl_xml(
        xml_path,
        topology,
        result,
        chunks_per_rank=chunks,
    )
    return HcclXmlArtifact(
        spec=spec,
        topology=topology,
        result=result,
        xml_path=xml_path,
    )


def resolve_output_path(
    *,
    input_json: Path,
    explicit_output: Path | str | None,
    config_output: str | None,
) -> Path | None:
    if explicit_output is not None:
        return Path(explicit_output)
    if config_output is None:
        return None
    output_path = Path(config_output)
    if output_path.is_absolute():
        return output_path
    return input_json.parent / output_path
