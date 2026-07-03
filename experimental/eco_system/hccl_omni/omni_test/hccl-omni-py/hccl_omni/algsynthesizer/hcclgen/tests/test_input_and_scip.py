from __future__ import annotations

from pathlib import Path

from hcclgen.global_milp import solve_scip_allgather
from hcclgen.input_schema import load_hcclgen_input, build_topology_from_input
from hcclgen.topology import (
    build_topology_edges,
    infer_allgather_input_capacity_t_lower_bound,
)


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_4x4_input_default_pruning_and_t_lower_bound() -> None:
    spec = load_hcclgen_input(EXAMPLES / "input_4x4_1dmesh_clos.json")
    topology = build_topology_from_input(spec)
    edges = build_topology_edges(topology)

    assert topology.ranks == 16
    assert sum(edge.kind == "1dmesh" for edge in edges) == 48
    assert sum(edge.kind == "clos" for edge in edges) == 48
    assert {edge.netlayer for edge in edges if edge.kind == "1dmesh"} == {0}
    assert {edge.netlayer for edge in edges if edge.kind == "clos"} == {1}
    assert spec.output.hccl_xml == "../results/4x4_hccl.xml"
    assert infer_allgather_input_capacity_t_lower_bound(
        topology,
        chunk_factor=1,
    ) == 3


def test_15rank_input_default_pruning_and_t_lower_bound() -> None:
    spec = load_hcclgen_input(EXAMPLES / "input_15rank_split.json")
    topology = build_topology_from_input(spec)
    edges = build_topology_edges(topology)

    assert topology.ranks == 15
    assert sum(edge.kind == "1dmesh" for edge in edges) == 42
    assert sum(edge.kind == "clos" for edge in edges) == 48
    assert infer_allgather_input_capacity_t_lower_bound(
        topology,
        chunk_factor=1,
    ) == 3


def test_scip_full_milp_solves_4x4_at_t3() -> None:
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

    assert result.completed
    assert result.status_name == "OPTIMAL"
    assert result.step_times == [1, 2]
    assert_pair_exchange_balanced(result.sends)


def test_scip_full_milp_solves_15rank_at_t3() -> None:
    topology = build_topology_from_input(
        load_hcclgen_input(EXAMPLES / "input_15rank_split.json")
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

    assert result.completed
    assert result.status_name == "OPTIMAL"
    assert result.step_times == [1, 2]
    assert_pair_exchange_balanced(result.sends)


def assert_pair_exchange_balanced(sends) -> None:
    by_bucket: dict[tuple[int, int, int, int], int] = {}
    for send in sends:
        pair = tuple(sorted((send.src, send.dst)))
        direction = 0 if send.src == pair[0] else 1
        key = (send.step, send.netlayer, pair[0], pair[1])
        by_bucket[key] = by_bucket.get(key, 0) + (1 if direction == 0 else -1)

    assert all(balance == 0 for balance in by_bucket.values())
