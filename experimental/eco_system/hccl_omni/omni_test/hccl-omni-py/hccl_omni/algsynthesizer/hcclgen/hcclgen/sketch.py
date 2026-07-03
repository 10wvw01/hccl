from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Literal


def build_balanced_bmatching_pairs(
    groups: Iterable[Iterable[int]],
    *,
    degree: int | Literal["all"] = 1,
) -> dict[tuple[int, int], tuple[int, ...]]:
    normalized = validate_rank_groups(groups)
    normalized_degree = normalize_degree(degree)
    if normalized_degree == "all":
        return build_all_cross_group_pairs(normalized)

    pair_to_lanes: dict[tuple[int, int], list[int]] = defaultdict(list)
    endpoint_loads: Counter[int] = Counter()
    for left_group_id, left_group in enumerate(normalized):
        for right_group_id in range(left_group_id + 1, len(normalized)):
            right_group = normalized[right_group_id]
            rounds = min(normalized_degree, min(len(left_group), len(right_group)))
            used_pairs: set[tuple[int, int]] = set()
            for round_id in range(rounds):
                selected = choose_balanced_round(
                    left_group,
                    right_group,
                    endpoint_loads=endpoint_loads,
                    used_pairs=used_pairs,
                    round_id=round_id,
                )
                for lane, left, right in selected:
                    if left == right:
                        continue
                    pair_to_lanes[(left, right)].append(round_id * len(selected) + lane)
                    pair_to_lanes[(right, left)].append(round_id * len(selected) + lane)
                    used_pairs.add((left, right))
                    endpoint_loads[left] += 1
                    endpoint_loads[right] += 1

    return {
        pair: tuple(lanes)
        for pair, lanes in sorted(pair_to_lanes.items())
    }


def validate_rank_groups(
    groups: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(group) for group in groups)
    if len(normalized) < 2:
        raise ValueError("至少需要两个 group 才能做 Clos 剪枝")
    if any(not group for group in normalized):
        raise ValueError("group 不能为空")

    seen: set[int] = set()
    duplicates: list[int] = []
    for group in normalized:
        for rank in group:
            if rank < 0:
                raise ValueError(f"rank 不能为负数: {rank}")
            if rank in seen:
                duplicates.append(rank)
            seen.add(rank)
    if duplicates:
        raise ValueError(f"rank 在多个 group 中重复出现: {sorted(duplicates)}")
    return normalized


def normalize_degree(degree: int | str) -> int | Literal["all"]:
    if isinstance(degree, str):
        if degree.strip().lower() == "all":
            return "all"
        degree = int(degree)
    if degree <= 0:
        raise ValueError("degree 必须为正整数或 all")
    return degree


def build_all_cross_group_pairs(
    groups: tuple[tuple[int, ...], ...],
) -> dict[tuple[int, int], tuple[int, ...]]:
    pair_to_lanes: dict[tuple[int, int], list[int]] = defaultdict(list)
    for left_group_id, left_group in enumerate(groups):
        for right_group_id in range(left_group_id + 1, len(groups)):
            right_group = groups[right_group_id]
            lane = 0
            for left in left_group:
                for right in right_group:
                    pair_to_lanes[(left, right)].append(lane)
                    pair_to_lanes[(right, left)].append(lane)
                    lane += 1
    return {
        pair: tuple(lanes)
        for pair, lanes in sorted(pair_to_lanes.items())
    }


def choose_balanced_round(
    left_group: tuple[int, ...],
    right_group: tuple[int, ...],
    *,
    endpoint_loads: Counter[int],
    used_pairs: set[tuple[int, int]],
    round_id: int,
) -> tuple[tuple[int, int, int], ...]:
    best_score: tuple[int, int, tuple[tuple[int, int], ...]] | None = None
    best_edges: tuple[tuple[int, int, int], ...] | None = None
    left_offsets = range(len(left_group)) if len(left_group) < len(right_group) else (0,)
    right_offsets = range(len(right_group)) if len(right_group) < len(left_group) else (0,)
    for left_offset in left_offsets:
        for right_offset in right_offsets:
            edges = build_balanced_round_edges(
                left_group,
                right_group,
                left_offset=left_offset,
                right_offset=right_offset + round_id,
            )
            pair_set = tuple(sorted((left, right) for _, left, right in edges))
            if any(pair in used_pairs for pair in pair_set):
                continue
            projected_loads = endpoint_loads.copy()
            for _, left, right in edges:
                projected_loads[left] += 1
                projected_loads[right] += 1
            active = set(endpoint_loads)
            active.update(left_group)
            active.update(right_group)
            score = (
                max(projected_loads[rank] for rank in active),
                sum(projected_loads[rank] * projected_loads[rank] for rank in active),
                pair_set,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_edges = edges
    if best_edges is None:
        return ()
    return best_edges


def build_balanced_round_edges(
    left_group: tuple[int, ...],
    right_group: tuple[int, ...],
    *,
    left_offset: int,
    right_offset: int,
) -> tuple[tuple[int, int, int], ...]:
    edge_count = max(len(left_group), len(right_group))
    return tuple(
        (
            lane,
            left_group[(lane + left_offset) % len(left_group)],
            right_group[(lane + right_offset) % len(right_group)],
        )
        for lane in range(edge_count)
    )
