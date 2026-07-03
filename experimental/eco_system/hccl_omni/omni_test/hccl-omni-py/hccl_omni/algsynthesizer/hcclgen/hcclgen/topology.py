from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    bandwidth: float
    kind: str
    netlayer: int
    resources: tuple[str, ...] = ()


@dataclass(frozen=True)
class GridTopology:
    name: str
    ranks: int
    local_bw_raw: float
    scaleout_bw_raw: float
    local_bw: float
    scaleout_bw: float
    reference_bw_raw: float
    edges: tuple[Edge, ...]
    resource_bandwidths: dict[str, float]
    resource_mode: str = "hcclgen-input"

    @property
    def ranks_per_server(self) -> int:
        return self.ranks

    @property
    def physical_reference_bw(self) -> float:
        return self.reference_bw_raw


@dataclass(frozen=True)
class ScheduledSend:
    step: int
    chunk: int
    src: int
    dst: int
    kind: str
    netlayer: int


def build_topology_edges(topology: GridTopology) -> list[Edge]:
    return list(topology.edges)


def build_resource_bandwidths(
    topology: GridTopology,
    edges: list[Edge],
) -> dict[str, float]:
    return dict(topology.resource_bandwidths)


def time_unit_bandwidth(topology: GridTopology, time_unit: str) -> float:
    bandwidths = [edge.bandwidth for edge in build_topology_edges(topology)]
    if not bandwidths:
        raise ValueError("拓扑中没有边，无法计算 time unit")
    if time_unit == "slowest":
        return float(min(bandwidths))
    if time_unit == "fastest":
        return float(max(bandwidths))
    raise ValueError(f"未知 time_unit: {time_unit}")


def compute_allowed_edges_for_origins(
    edges: list[Edge],
    *,
    ranks: int,
    origins: tuple[int, ...],
    route_dag_mode: str,
) -> tuple[tuple[int, ...], ...]:
    if route_dag_mode not in {"unrestricted", "spdag", "near-spdag"}:
        raise ValueError(f"未知 route_dag_mode: {route_dag_mode}")

    allowed_by_origin: list[tuple[int, ...]] = [() for _ in range(ranks)]
    if route_dag_mode == "unrestricted":
        all_edges = tuple(range(len(edges)))
        for origin in origins:
            allowed_by_origin[origin] = all_edges
        return tuple(allowed_by_origin)

    outgoing = {rank: [] for rank in range(ranks)}
    for edge_id, edge in enumerate(edges):
        outgoing[edge.src].append((edge.dst, edge_id))

    for origin in origins:
        distances = shortest_hop_distances(outgoing, origin)
        if len(distances) != ranks:
            missing = sorted(set(range(ranks)) - set(distances))
            raise ValueError(
                f"SP-DAG 剪枝失败: origin={origin} 无法到达 {len(missing)} 个 rank"
            )
        if route_dag_mode == "spdag":
            allowed = tuple(
                edge_id
                for edge_id, edge in enumerate(edges)
                if distances.get(edge.src, -1) + 1 == distances.get(edge.dst)
            )
        else:
            allowed = tuple(
                edge_id
                for edge_id, edge in enumerate(edges)
                if distances.get(edge.src, -1) <= distances.get(edge.dst, -1)
                <= distances.get(edge.src, -1) + 1
            )
        allowed_by_origin[origin] = allowed
    return tuple(allowed_by_origin)


def shortest_hop_distances(
    outgoing: dict[int, list[tuple[int, int]]],
    origin: int,
) -> dict[int, int]:
    distances = {origin: 0}
    queue: deque[int] = deque([origin])
    while queue:
        src = queue.popleft()
        for dst, _ in outgoing[src]:
            if dst in distances:
                continue
            distances[dst] = distances[src] + 1
            queue.append(dst)
    return distances


def compute_allgather_bandwidths(
    topology: GridTopology,
    *,
    total_time: float,
) -> dict[str, float]:
    if total_time <= 0:
        return {
            "normalized_algbw": 0.0,
            "normalized_busbw": 0.0,
            "physical_algbw": 0.0,
            "physical_busbw": 0.0,
            "physical_busbw_bound": physical_busbw_bound(topology),
        }

    ranks = topology.ranks
    normalized_algbw = ranks / total_time
    normalized_busbw = (
        normalized_algbw if ranks <= 1 else normalized_algbw * (ranks - 1) / ranks
    )
    return {
        "normalized_algbw": normalized_algbw,
        "normalized_busbw": normalized_busbw,
        "physical_algbw": normalized_algbw * topology.physical_reference_bw,
        "physical_busbw": normalized_busbw * topology.physical_reference_bw,
        "physical_busbw_bound": physical_busbw_bound(topology),
    }


def physical_busbw_bound(topology: GridTopology) -> float:
    outgoing_resources = {rank: set() for rank in range(topology.ranks)}
    for edge in build_topology_edges(topology):
        for resource in edge.resources:
            if is_source_owned_resource(resource, edge.src, edge.dst):
                outgoing_resources[edge.src].add(resource)

    outgoing_capacity = {
        rank: sum(topology.resource_bandwidths[resource] for resource in resources)
        for rank, resources in outgoing_resources.items()
    }
    if not outgoing_capacity:
        return 0.0
    return min(outgoing_capacity.values()) * topology.physical_reference_bw


def infer_allgather_input_capacity_t_lower_bound(
    topology: GridTopology,
    *,
    chunk_factor: int,
) -> int:
    if chunk_factor < 1:
        raise ValueError("chunk_factor 必须至少为 1")

    incoming_resources = {rank: set() for rank in range(topology.ranks)}
    for edge in build_topology_edges(topology):
        for resource in edge.resources:
            if is_destination_owned_resource(resource, edge.dst):
                incoming_resources[edge.dst].add(resource)

    required_receive = (topology.ranks - 1) * chunk_factor
    lower_bound = 0
    for rank, resources in incoming_resources.items():
        capacity = sum(topology.resource_bandwidths[resource] for resource in resources)
        if capacity <= 0:
            raise ValueError(f"rank {rank} 的输入容量为 0")
        lower_bound = max(lower_bound, ceil(required_receive / capacity))
    return lower_bound


def is_source_owned_resource(resource: str, src: int, dst: int) -> bool:
    if resource.endswith(f"_rank{src}_out"):
        return True
    if resource.endswith(f"_out:{src}"):
        return True
    if "_logical:" in resource:
        return False
    return resource.endswith(f":{src}:{dst}")


def is_destination_owned_resource(resource: str, dst: int) -> bool:
    if resource.endswith(f"_rank{dst}_in"):
        return True
    if resource.endswith(f"_in:{dst}"):
        return True
    if "_logical:" in resource:
        return False
    return resource.endswith(f":{dst}")
