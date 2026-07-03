from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from hcclgen.sketch import build_balanced_bmatching_pairs
from hcclgen.topology import Edge, GridTopology


SUPPORTED_GROUP_TYPES = {"1DMESH", "Clos"}


@dataclass(frozen=True)
class InputGroup:
    level: int
    group_id: int
    ranks: tuple[int, ...]
    type: str
    bandwidth: float
    netlayer: int


@dataclass(frozen=True)
class CollectiveSpec:
    type: str
    chunksize: int


@dataclass(frozen=True)
class OutputSpec:
    hccl_xml: str | None = None


@dataclass(frozen=True)
class HcclGenInput:
    name: str
    rank_base: int
    num_node: int
    total_rank_num: int
    num_rank_node: tuple[int, ...]
    groups_by_level: tuple[tuple[InputGroup, ...], ...]
    collective: CollectiveSpec
    output: OutputSpec
    schema_version: str | None = None


def load_hcclgen_input(path: Path | str) -> HcclGenInput:
    input_path = Path(path)
    return parse_hcclgen_input(json.loads(input_path.read_text(encoding="utf-8")))


def parse_hcclgen_input(data: Mapping[str, Any]) -> HcclGenInput:
    if not isinstance(data, Mapping):
        raise ValueError("输入 JSON 顶层必须是对象")

    name = require_str(data, "name")
    rank_base = require_int(data, "rank_base")
    num_node = require_int(data, "num_node")
    total_rank_num = require_int(data, "total_rank_num")
    num_rank_node = tuple(require_int_list(data, "num_rank_node"))
    schema_version = optional_str(data, "schema_version")

    collective_data = require_mapping(data, "collective")
    collective = CollectiveSpec(
        type=require_str(collective_data, "type"),
        chunksize=require_int(collective_data, "chunksize"),
    )
    output = parse_output(data.get("output"))
    groups_by_level = parse_cluster(require_mapping(data, "cluster"))

    spec = HcclGenInput(
        name=name,
        rank_base=rank_base,
        num_node=num_node,
        total_rank_num=total_rank_num,
        num_rank_node=num_rank_node,
        groups_by_level=groups_by_level,
        collective=collective,
        output=output,
        schema_version=schema_version,
    )
    validate_hcclgen_input(spec)
    return spec


def parse_output(raw_output: Any) -> OutputSpec:
    if raw_output is None:
        return OutputSpec()
    if not isinstance(raw_output, Mapping):
        raise ValueError("output 必须是对象")
    return OutputSpec(
        hccl_xml=optional_str(raw_output, "hccl_xml"),
    )


def parse_cluster(cluster: Mapping[str, Any]) -> tuple[tuple[InputGroup, ...], ...]:
    level_pattern = re.compile(r"level(\d+)$")
    parsed_levels: dict[int, tuple[InputGroup, ...]] = {}

    for level_name, level_data in cluster.items():
        match = level_pattern.fullmatch(level_name)
        if match is None:
            raise ValueError(f"非法 level 名称: {level_name}")
        level = int(match.group(1))
        if not isinstance(level_data, Mapping):
            raise ValueError(f"{level_name} 必须是对象")
        raw_groups = level_data.get("groups")
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError(f"{level_name}.groups 必须是非空数组")

        groups: list[InputGroup] = []
        for raw_group in raw_groups:
            if not isinstance(raw_group, Mapping):
                raise ValueError(f"{level_name}.groups 中每项都必须是对象")
            groups.append(
                InputGroup(
                    level=level,
                    group_id=require_int(raw_group, "group_id"),
                    ranks=tuple(require_int_list(raw_group, "ranks")),
                    type=require_str(raw_group, "type"),
                    bandwidth=float(require_number(raw_group, "bandwidth")),
                    netlayer=require_int(raw_group, "netlayer"),
                )
            )
        parsed_levels[level] = tuple(groups)

    expected = list(range(len(parsed_levels)))
    actual = sorted(parsed_levels)
    if actual != expected:
        raise ValueError(f"level 必须从 level0 开始连续递增，实际为 {actual}")
    return tuple(parsed_levels[level] for level in expected)


def validate_hcclgen_input(spec: HcclGenInput) -> None:
    if spec.rank_base != 0:
        raise ValueError("rank_base 必须为 0")
    if spec.num_node <= 0:
        raise ValueError("num_node 必须为正数")
    if spec.total_rank_num <= 0:
        raise ValueError("total_rank_num 必须为正数")
    if len(spec.num_rank_node) != spec.num_node:
        raise ValueError("num_rank_node 长度必须等于 num_node")
    if any(value <= 0 for value in spec.num_rank_node):
        raise ValueError("num_rank_node 中每项都必须为正数")
    if sum(spec.num_rank_node) != spec.total_rank_num:
        raise ValueError("sum(num_rank_node) 必须等于 total_rank_num")
    if spec.collective.type != "allgather":
        raise ValueError("当前只支持 collective.type = allgather")
    if spec.collective.chunksize <= 0:
        raise ValueError("collective.chunksize 必须为正数")

    valid_ranks = set(range(spec.total_rank_num))
    for level, groups in enumerate(spec.groups_by_level):
        seen_ranks: set[int] = set()
        seen_group_ids: set[int] = set()
        for group in groups:
            if group.level != level:
                raise ValueError(f"level{level} 内 group.level 不一致")
            if group.group_id in seen_group_ids:
                raise ValueError(f"level{level} 内 group_id {group.group_id} 重复")
            seen_group_ids.add(group.group_id)
            if not group.ranks:
                raise ValueError(f"level{level} group {group.group_id} ranks 不能为空")
            if group.type not in SUPPORTED_GROUP_TYPES:
                raise ValueError(
                    f"level{level} group {group.group_id} type 不支持: {group.type}"
                )
            if group.bandwidth <= 0:
                raise ValueError(
                    f"level{level} group {group.group_id} bandwidth 必须为正数"
                )
            if group.netlayer < 0:
                raise ValueError(
                    f"level{level} group {group.group_id} netlayer 必须为非负整数"
                )
            for rank in group.ranks:
                if rank not in valid_ranks:
                    raise ValueError(f"level{level} rank {rank} 超出合法范围")
                if rank in seen_ranks:
                    raise ValueError(f"level{level} rank {rank} 出现在多个 group 中")
                seen_ranks.add(rank)


def build_topology_from_input(
    spec: HcclGenInput,
    *,
    clos_prune_degree: int | Literal["all"] = 1,
) -> GridTopology:
    raw_bandwidths = [
        group.bandwidth
        for groups in spec.groups_by_level
        for group in groups
    ]
    reference_bw_raw = min(raw_bandwidths)
    local_bw_raw = next(
        (
            group.bandwidth
            for groups in spec.groups_by_level
            for group in groups
            if group.type == "1DMESH"
        ),
        reference_bw_raw,
    )
    scaleout_bw_raw = max(
        (
            group.bandwidth
            for groups in spec.groups_by_level
            for group in groups
            if group.type == "Clos"
        ),
        default=local_bw_raw,
    )

    edges: list[Edge] = []
    resource_bandwidths: dict[str, float] = {}

    def normalized(raw_bandwidth: float) -> float:
        return raw_bandwidth / reference_bw_raw

    for groups in spec.groups_by_level:
        for group in groups:
            bandwidth = normalized(group.bandwidth)
            if group.type == "1DMESH":
                add_1dmesh_edges(edges, resource_bandwidths, group, bandwidth)
            elif group.type == "Clos":
                allowed_pairs = clos_allowed_pairs(
                    spec,
                    group,
                    degree=clos_prune_degree,
                )
                add_clos_edges(
                    edges,
                    resource_bandwidths,
                    group,
                    bandwidth,
                    allowed_pairs=allowed_pairs,
                )
            else:
                raise ValueError(f"未知 group type: {group.type}")

    return GridTopology(
        name=spec.name,
        ranks=spec.total_rank_num,
        local_bw_raw=local_bw_raw,
        scaleout_bw_raw=scaleout_bw_raw,
        local_bw=local_bw_raw / reference_bw_raw,
        scaleout_bw=scaleout_bw_raw / reference_bw_raw,
        reference_bw_raw=reference_bw_raw,
        edges=tuple(edges),
        resource_bandwidths=resource_bandwidths,
    )


def add_1dmesh_edges(
    edges: list[Edge],
    resource_bandwidths: dict[str, float],
    group: InputGroup,
    bandwidth: float,
) -> None:
    for src in group.ranks:
        for dst in group.ranks:
            if src == dst:
                continue
            resource = f"l{group.level}_direct:{src}:{dst}"
            set_resource_bandwidth(resource_bandwidths, resource, bandwidth)
            edges.append(
                Edge(
                    src=src,
                    dst=dst,
                    bandwidth=bandwidth,
                    kind="1dmesh",
                    netlayer=group.netlayer,
                    resources=(resource,),
                )
            )


def add_clos_edges(
    edges: list[Edge],
    resource_bandwidths: dict[str, float],
    group: InputGroup,
    bandwidth: float,
    *,
    allowed_pairs: set[tuple[int, int]] | None,
) -> None:
    for rank in group.ranks:
        set_resource_bandwidth(
            resource_bandwidths,
            f"l{group.level}_rank{rank}_out",
            bandwidth,
        )
        set_resource_bandwidth(
            resource_bandwidths,
            f"l{group.level}_rank{rank}_in",
            bandwidth,
        )

    for src in group.ranks:
        for dst in group.ranks:
            if src == dst:
                continue
            if allowed_pairs is not None and (src, dst) not in allowed_pairs:
                continue
            edges.append(
                Edge(
                    src=src,
                    dst=dst,
                    bandwidth=bandwidth,
                    kind="clos",
                    netlayer=group.netlayer,
                    resources=(
                        f"l{group.level}_rank{src}_out",
                        f"l{group.level}_rank{dst}_in",
                    ),
                )
            )


def clos_allowed_pairs(
    spec: HcclGenInput,
    group: InputGroup,
    *,
    degree: int | Literal["all"],
) -> set[tuple[int, int]] | None:
    if degree == "all" or group.level == 0:
        return None

    clos_ranks = set(group.ranks)
    lower_groups: list[tuple[int, ...]] = []
    covered_ranks: set[int] = set()
    for lower_group in spec.groups_by_level[group.level - 1]:
        intersection = tuple(rank for rank in lower_group.ranks if rank in clos_ranks)
        if intersection:
            lower_groups.append(intersection)
            covered_ranks.update(intersection)

    missing = sorted(clos_ranks - covered_ranks)
    if missing:
        raise ValueError(
            f"level{group.level} Clos group {group.group_id} 中的 rank "
            f"没有出现在 level{group.level - 1}: {missing}"
        )
    if len(lower_groups) < 2:
        return set()

    return set(build_balanced_bmatching_pairs(lower_groups, degree=degree))


def set_resource_bandwidth(
    resource_bandwidths: dict[str, float],
    resource: str,
    bandwidth: float,
) -> None:
    old = resource_bandwidths.get(resource)
    if old is not None and abs(old - bandwidth) > 1e-12:
        raise ValueError(f"资源 {resource} 被赋予了不同带宽: {old} vs {bandwidth}")
    resource_bandwidths[resource] = bandwidth


def require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} 必须是对象")
    return value


def require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} 必须是非空字符串")
    return value


def optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} 必须是非空字符串")
    return value


def require_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} 必须是整数")
    return value


def require_number(data: Mapping[str, Any], key: str) -> int | float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} 必须是数字")
    return value


def require_int_list(data: Mapping[str, Any], key: str) -> list[int]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} 必须是非空整数数组")
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{key} 必须是非空整数数组")
    return value
