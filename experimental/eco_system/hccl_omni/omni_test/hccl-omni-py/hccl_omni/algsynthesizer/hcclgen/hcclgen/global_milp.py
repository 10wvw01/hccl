from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from hcclgen.input_schema import build_topology_from_input, load_hcclgen_input
from hcclgen.topology import (
    GridTopology,
    ScheduledSend,
    build_resource_bandwidths,
    build_topology_edges,
    compute_allgather_bandwidths,
    compute_allowed_edges_for_origins,
    infer_allgather_input_capacity_t_lower_bound,
    time_unit_bandwidth,
)


@dataclass(frozen=True)
class GlobalMilpSchedule:
    completed: bool
    status: str
    status_name: str
    step_times: list[int]
    sends: list[ScheduledSend]
    holders_by_chunk: tuple[int, ...]
    total_chunk_time: float
    normalized_time: float
    physical_busbw: float
    build_seconds: float
    solve_seconds: float
    objective_value: float | None
    total_epochs: int
    max_steps: int
    used_steps: int
    variable_counts: dict[str, int]


def solve_scip_allgather(
    topology: GridTopology,
    *,
    chunk_factor: int,
    total_epochs: int,
    max_steps: int,
    time_unit: str,
    route_dag_mode: str,
    require_pair_exchange: bool = True,
    origins: Iterable[int] | None = None,
    time_limit: float = 60.0,
    threads: int = 0,
    quiet: bool = True,
) -> GlobalMilpSchedule:
    try:
        import pyscipopt as scip
    except ImportError as exc:
        raise RuntimeError("SCIP 求解需要安装 pyscipopt") from exc

    if chunk_factor < 1:
        raise ValueError("chunks 必须至少为 1")
    if total_epochs < 1:
        raise ValueError("total_epochs 必须至少为 1")
    if max_steps < 1:
        raise ValueError("max_steps 必须至少为 1")

    build_start = time.time()
    edges = build_topology_edges(topology)
    resource_bandwidths = build_resource_bandwidths(topology, edges)
    unit_bandwidth = time_unit_bandwidth(topology, time_unit)
    ranks = topology.ranks
    origin_list = tuple(range(ranks) if origins is None else origins)
    active_chunks = tuple(
        origin * chunk_factor + piece
        for origin in origin_list
        for piece in range(chunk_factor)
    )
    all_ranks_mask = (1 << ranks) - 1

    allowed_edges_by_origin = compute_allowed_edges_for_origins(
        edges,
        ranks=ranks,
        origins=origin_list,
        route_dag_mode=route_dag_mode,
    )

    model = scip.Model("hcclgen_scip_allgather")
    if quiet:
        model.hideOutput()
    model.setEmphasis(scip.SCIP_PARAMEMPHASIS.FEASIBILITY, quiet=True)
    model.setParam("limits/time", float(time_limit))
    if threads > 0:
        model.setParam("parallel/maxnthreads", int(threads))
        model.setParam("lp/threads", int(threads))

    steps = range(max_steps)
    buffer_steps = range(max_steps + 1)

    step_len = {
        step: model.addVar(lb=0, ub=total_epochs, vtype="I", name=f"step_len_{step}")
        for step in steps
    }
    step_used = {
        step: model.addVar(vtype="B", name=f"step_used_{step}")
        for step in steps
    }
    has = {
        (chunk, rank, step): model.addVar(
            vtype="B",
            name=f"has_{chunk}_{rank}_{step}",
        )
        for chunk in active_chunks
        for rank in range(ranks)
        for step in buffer_steps
    }

    send_keys: list[tuple[int, int, int]] = []
    for chunk in active_chunks:
        origin = chunk // chunk_factor
        for edge_id in allowed_edges_by_origin[origin]:
            edge = edges[edge_id]
            if edge.dst == origin:
                continue
            for step in steps:
                send_keys.append((chunk, edge_id, step))
    send = {
        key: model.addVar(vtype="B", name=f"send_{key[0]}_{key[1]}_{key[2]}")
        for key in send_keys
    }

    model.addCons(
        scip.quicksum(step_len[step] for step in steps) == total_epochs,
        name="fixed_total_epochs",
    )
    for step in steps:
        model.addCons(
            step_len[step] <= total_epochs * step_used[step],
            name=f"step_len_upper_{step}",
        )
        model.addCons(
            step_len[step] >= step_used[step],
            name=f"step_len_lower_{step}",
        )
        if step + 1 < max_steps:
            model.addCons(
                step_used[step] >= step_used[step + 1],
                name=f"step_used_prefix_{step}",
            )

    for chunk in active_chunks:
        origin = chunk // chunk_factor
        for rank in range(ranks):
            model.addCons(
                has[chunk, rank, 0] == (1 if rank == origin else 0),
                name=f"initial_has_{chunk}_{rank}",
            )

    incoming_by_chunk_rank_step: dict[tuple[int, int, int], list[object]] = defaultdict(list)
    for chunk, edge_id, step in send_keys:
        edge = edges[edge_id]
        send_var = send[chunk, edge_id, step]
        model.addCons(
            send_var <= has[chunk, edge.src, step],
            name=f"send_available_{chunk}_{edge_id}_{step}",
        )
        model.addCons(
            send_var <= 1 - has[chunk, edge.dst, step],
            name=f"send_not_duplicate_{chunk}_{edge_id}_{step}",
        )
        model.addCons(
            send_var <= step_used[step],
            name=f"send_step_used_{chunk}_{edge_id}_{step}",
        )
        incoming_by_chunk_rank_step[chunk, edge.dst, step].append(send_var)

    for chunk in active_chunks:
        for rank in range(ranks):
            for step in steps:
                incoming = scip.quicksum(
                    incoming_by_chunk_rank_step.get((chunk, rank, step), ())
                )
                model.addCons(
                    has[chunk, rank, step + 1]
                    == has[chunk, rank, step] + incoming,
                    name=f"has_update_{chunk}_{rank}_{step}",
                )

    for chunk in active_chunks:
        for rank in range(ranks):
            model.addCons(
                has[chunk, rank, max_steps] == 1,
                name=f"final_has_{chunk}_{rank}",
            )

    sends_by_resource_step: dict[tuple[str, int], list[object]] = defaultdict(list)
    sends_by_peer_netlayer_step: dict[tuple[int, int, int, int], list[object]] = defaultdict(list)
    for chunk, edge_id, step in send_keys:
        edge = edges[edge_id]
        for resource in edges[edge_id].resources:
            sends_by_resource_step[resource, step].append(send[chunk, edge_id, step])
        sends_by_peer_netlayer_step[
            edge.src,
            edge.dst,
            edge.netlayer,
            step,
        ].append(send[chunk, edge_id, step])

    for resource, bandwidth in sorted(resource_bandwidths.items()):
        for step in steps:
            capacity = (bandwidth / unit_bandwidth) * step_len[step]
            model.addCons(
                scip.quicksum(sends_by_resource_step.get((resource, step), ()))
                <= capacity,
                name=f"capacity_{resource}_{step}",
            )

    if require_pair_exchange:
        peer_keys = {
            (min(src, dst), max(src, dst), netlayer, step)
            for src, dst, netlayer, step in sends_by_peer_netlayer_step
        }
        for low, high, netlayer, step in sorted(peer_keys):
            low_to_high = scip.quicksum(
                sends_by_peer_netlayer_step.get((low, high, netlayer, step), ())
            )
            high_to_low = scip.quicksum(
                sends_by_peer_netlayer_step.get((high, low, netlayer, step), ())
            )
            model.addCons(
                low_to_high == high_to_low,
                name=f"pair_exchange_l{netlayer}_{low}_{high}_{step}",
            )

    model.setObjective(
        scip.quicksum(step_used[step] for step in steps),
        "minimize",
    )
    build_seconds = time.time() - build_start

    solve_start = time.time()
    model.optimize()
    solve_seconds = time.time() - solve_start

    status = str(model.getStatus())
    status_name = scip_status_name(status)
    sol_count = model.getNSols()
    variable_counts = {
        "has": len(has),
        "send": len(send),
        "step_len": len(step_len),
        "step_used": len(step_used),
    }
    if sol_count == 0:
        return GlobalMilpSchedule(
            completed=False,
            status=status,
            status_name=status_name,
            step_times=[],
            sends=[],
            holders_by_chunk=(),
            total_chunk_time=0.0,
            normalized_time=0.0,
            physical_busbw=0.0,
            build_seconds=build_seconds,
            solve_seconds=solve_seconds,
            objective_value=None,
            total_epochs=total_epochs,
            max_steps=max_steps,
            used_steps=0,
            variable_counts=variable_counts,
        )

    used_steps = [
        step
        for step in steps
        if model.getVal(step_used[step]) > 0.5
    ]
    step_times = [int(round(model.getVal(step_len[step]))) for step in used_steps]
    scheduled_sends: list[ScheduledSend] = []
    for chunk, edge_id, step in send_keys:
        if model.getVal(send[chunk, edge_id, step]) <= 0.5:
            continue
        edge = edges[edge_id]
        scheduled_sends.append(
            ScheduledSend(
                step=step,
                chunk=chunk,
                src=edge.src,
                dst=edge.dst,
                kind=edge.kind,
                netlayer=edge.netlayer,
            )
        )
    scheduled_sends.sort(key=lambda item: (item.step, item.chunk, item.src, item.dst))

    holders: list[int] = []
    for chunk in active_chunks:
        mask = 0
        for rank in range(ranks):
            if model.getVal(has[chunk, rank, max_steps]) > 0.5:
                mask |= 1 << rank
        holders.append(mask)
    completed = all(mask == all_ranks_mask for mask in holders)
    total_chunk_time = float(sum(step_times))
    normalized_time = total_chunk_time / (chunk_factor * unit_bandwidth)
    bandwidths = compute_allgather_bandwidths(topology, total_time=normalized_time)

    return GlobalMilpSchedule(
        completed=completed,
        status=status,
        status_name=status_name,
        step_times=step_times,
        sends=scheduled_sends,
        holders_by_chunk=tuple(holders),
        total_chunk_time=total_chunk_time,
        normalized_time=normalized_time,
        physical_busbw=bandwidths["physical_busbw"],
        build_seconds=build_seconds,
        solve_seconds=solve_seconds,
        objective_value=float(model.getObjVal()) if sol_count > 0 else None,
        total_epochs=total_epochs,
        max_steps=max_steps,
        used_steps=len(used_steps),
        variable_counts=variable_counts,
    )


def scip_status_name(status: str) -> str:
    normalized = status.lower()
    return {
        "optimal": "OPTIMAL",
        "infeasible": "INFEASIBLE",
        "timelimit": "TIME_LIMIT",
        "bestsollimit": "BEST_SOL_LIMIT",
    }.get(normalized, status.upper())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 SCIP 完整 MILP 合成 AllGather 调度。"
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("-c", "--chunks", type=int, default=1)
    parser.add_argument("--total-epochs", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="step 数上限；默认等于 total_epochs。",
    )
    parser.add_argument("--time-unit", choices=("slowest", "fastest"), default="slowest")
    parser.add_argument(
        "--route-dag-mode",
        choices=("unrestricted", "spdag", "near-spdag"),
        default="spdag",
    )
    parser.add_argument(
        "--clos-prune-degree",
        default="1",
        help="Clos 默认剪枝 degree；可设为 all 完整展开。",
    )
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--solution-json", type=Path, default=None)
    parser.add_argument("--hccl-xml", type=Path, default=None)
    parser.add_argument(
        "--no-pair-exchange",
        action="store_true",
        help="关闭 pair exchange 约束；默认开启，便于后续生成保守 HCCL XML。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clos_prune_degree = parse_degree(args.clos_prune_degree)
    spec = load_hcclgen_input(args.input_json)
    topology = build_topology_from_input(spec, clos_prune_degree=clos_prune_degree)
    total_epochs = args.total_epochs
    if total_epochs is None:
        total_epochs = infer_allgather_input_capacity_t_lower_bound(
            topology,
            chunk_factor=args.chunks,
        )
    max_steps = total_epochs if args.max_steps is None else args.max_steps
    result = solve_scip_allgather(
        topology,
        chunk_factor=args.chunks,
        total_epochs=total_epochs,
        max_steps=max_steps,
        time_unit=args.time_unit,
        route_dag_mode=args.route_dag_mode,
        require_pair_exchange=not args.no_pair_exchange,
        time_limit=args.time_limit,
        threads=args.threads,
        quiet=args.quiet,
    )
    print_schedule_summary(topology, result, total_epochs=total_epochs)
    if args.solution_json is not None:
        write_solution_json(args.solution_json, topology, result)
    from hcclgen.api import resolve_output_path

    hccl_xml_path = resolve_output_path(
        input_json=args.input_json,
        explicit_output=args.hccl_xml,
        config_output=spec.output.hccl_xml,
    )
    if hccl_xml_path is not None:
        from hcclgen.hccl_xml import write_hccl_xml

        write_hccl_xml(
            hccl_xml_path,
            topology,
            result,
            chunks_per_rank=args.chunks,
        )
        print(f"hccl_xml: {hccl_xml_path}")


def parse_degree(value: str) -> int | Literal["all"]:
    if value.strip().lower() == "all":
        return "all"
    degree = int(value)
    if degree <= 0:
        raise ValueError("--clos-prune-degree 必须为正整数或 all")
    return degree


def print_schedule_summary(
    topology: GridTopology,
    result: GlobalMilpSchedule,
    *,
    total_epochs: int,
) -> None:
    print(f"topology: {topology.name}")
    print(f"ranks: {topology.ranks}")
    print(f"total_epochs: {total_epochs}")
    print(f"status: {result.status_name}")
    print(f"completed: {result.completed}")
    print(f"used_steps: {result.used_steps}")
    print(f"step_times: {result.step_times}")
    print(f"physical_busbw: {result.physical_busbw:.6g}")
    print(f"build_seconds: {result.build_seconds:.3f}")
    print(f"solve_seconds: {result.solve_seconds:.3f}")
    print(f"variables: {result.variable_counts}")


def write_solution_json(
    path: Path,
    topology: GridTopology,
    result: GlobalMilpSchedule,
) -> None:
    document = {
        "topology": {
            "name": topology.name,
            "ranks": topology.ranks,
            "resource_mode": topology.resource_mode,
        },
        "result": {
            "completed": result.completed,
            "status": result.status_name,
            "step_times": result.step_times,
            "total_chunk_time": result.total_chunk_time,
            "physical_busbw": result.physical_busbw,
            "build_seconds": result.build_seconds,
            "solve_seconds": result.solve_seconds,
            "variable_counts": result.variable_counts,
        },
        "sends": [
            {
                "step": send.step,
                "chunk": send.chunk,
                "src": send.src,
                "dst": send.dst,
                "kind": send.kind,
                "netlayer": send.netlayer,
            }
            for send in result.sends
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
