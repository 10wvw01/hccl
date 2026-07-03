from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from hcclgen.topology import GridTopology, ScheduledSend, build_topology_edges


@dataclass(frozen=True)
class PairExchangeOp:
    step: int
    rank: int
    peer: int
    netlayer: int
    sends: tuple[ScheduledSend, ...]


def build_hccl_xml(
    topology: GridTopology,
    schedule,
    *,
    chunks_per_rank: int,
    link_proto: str = "HCCS",
    max_slices_per_op: int = 15,
) -> ET.Element:
    if chunks_per_rank <= 0:
        raise ValueError("chunks_per_rank 必须为正数")
    if max_slices_per_op <= 0:
        raise ValueError("max_slices_per_op 必须为正数")
    if not schedule.completed:
        raise ValueError("不能为未完成的调度生成 HCCL XML")

    ops = lower_pair_exchange_ops(schedule.sends)
    ops_by_rank_step: dict[tuple[int, int], list[PairExchangeOp]] = defaultdict(list)
    channels_by_rank: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for op in ops:
        ops_by_rank_step[op.rank, op.step].append(op)
        channels_by_rank[op.rank].add((op.netlayer, op.peer))

    thread_ids = assign_thread_ids(topology.ranks, channels_by_rank)
    slave_thread_num = max(thread_ids.values(), default=0)
    net_layer_num = infer_net_layer_num(topology, channels_by_rank.values())
    max_step = max((op.step for op in ops), default=-1)

    root = ET.Element("root")
    for rank in range(topology.ranks):
        npu = ET.SubElement(root, "NPU", {"rankId": str(rank)})
        add_res_request(
            npu,
            rank,
            sorted(channels_by_rank.get(rank, ())),
            slave_thread_num,
            net_layer_num,
            link_proto,
        )

        initial_copy_done = False
        for step in range(max_step + 1):
            step_ops = sorted(
                ops_by_rank_step.get((rank, step), ()),
                key=lambda op: (thread_ids[rank, op.netlayer, op.peer], op.netlayer, op.peer),
            )
            if not step_ops:
                continue

            active_threads = [
                thread_ids[rank, op.netlayer, op.peer]
                for op in step_ops
            ]
            add_sync(npu, "PreSyncInterThreads", sorted(set(active_threads)))
            if not initial_copy_done:
                add_initial_local_copy(
                    npu,
                    rank,
                    topology.ranks,
                    chunks_per_rank,
                    max_slices_per_op,
                )
                initial_copy_done = True
            for op in step_ops:
                add_send_recv_write_batches(
                    npu,
                    op,
                    thread_ids[rank, op.netlayer, op.peer],
                    topology.ranks,
                    chunks_per_rank,
                    link_proto,
                    max_slices_per_op,
                )
            add_sync(npu, "PostSyncInterThreads", sorted(set(active_threads)))

        if not initial_copy_done:
            add_initial_local_copy(
                npu,
                rank,
                topology.ranks,
                chunks_per_rank,
                max_slices_per_op,
            )
        add_final_local_copy(
            npu,
            rank,
            topology.ranks,
            chunks_per_rank,
            max_slices_per_op,
        )
    return root


def write_hccl_xml(
    path: Path | str,
    topology: GridTopology,
    schedule,
    *,
    chunks_per_rank: int,
    link_proto: str = "HCCS",
    max_slices_per_op: int = 15,
) -> None:
    root = build_hccl_xml(
        topology,
        schedule,
        chunks_per_rank=chunks_per_rank,
        link_proto=link_proto,
        max_slices_per_op=max_slices_per_op,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    )


def lower_pair_exchange_ops(sends: Iterable[ScheduledSend]) -> list[PairExchangeOp]:
    by_direction: dict[tuple[int, int, int, int], list[ScheduledSend]] = defaultdict(list)
    for send in sends:
        by_direction[send.step, send.netlayer, send.src, send.dst].append(send)

    pair_keys = {
        (step, netlayer, min(src, dst), max(src, dst))
        for step, netlayer, src, dst in by_direction
    }
    ops: list[PairExchangeOp] = []
    for step, netlayer, low, high in sorted(pair_keys):
        low_to_high = sorted(
            by_direction.get((step, netlayer, low, high), ()),
            key=lambda item: item.chunk,
        )
        high_to_low = sorted(
            by_direction.get((step, netlayer, high, low), ()),
            key=lambda item: item.chunk,
        )
        if len(low_to_high) != len(high_to_low):
            raise ValueError(
                "pair exchange 不配平: "
                f"step={step}, netlayer={netlayer}, pair=({low},{high}), "
                f"{low}->{high}={len(low_to_high)}, {high}->{low}={len(high_to_low)}"
            )
        if not low_to_high:
            continue
        ops.append(
            PairExchangeOp(
                step=step,
                rank=low,
                peer=high,
                netlayer=netlayer,
                sends=tuple(low_to_high),
            )
        )
        ops.append(
            PairExchangeOp(
                step=step,
                rank=high,
                peer=low,
                netlayer=netlayer,
                sends=tuple(high_to_low),
            )
        )
    return sorted(ops, key=lambda op: (op.rank, op.step, op.netlayer, op.peer))


def assign_thread_ids(
    ranks: int,
    channels_by_rank: dict[int, set[tuple[int, int]]],
) -> dict[tuple[int, int, int], int]:
    thread_ids: dict[tuple[int, int, int], int] = {}
    for rank in range(ranks):
        channels = sorted(channels_by_rank.get(rank, ()))
        for thread_idx, (netlayer, peer) in enumerate(channels, start=1):
            thread_ids[rank, netlayer, peer] = thread_idx
    return thread_ids


def infer_net_layer_num(
    topology: GridTopology,
    channel_groups: Iterable[set[tuple[int, int]]],
) -> int:
    max_netlayer = max(
        (edge.netlayer for edge in build_topology_edges(topology)),
        default=-1,
    )
    for channels in channel_groups:
        for netlayer, _peer in channels:
            max_netlayer = max(max_netlayer, netlayer)
    return max_netlayer + 1


def add_res_request(
    parent: ET.Element,
    rank: int,
    channels: list[tuple[int, int]],
    slave_thread_num: int,
    net_layer_num: int,
    link_proto: str,
) -> None:
    instruction = ET.SubElement(parent, "instruction", {
        "opCode": "resRequest",
        "slaveThreadNum": str(slave_thread_num),
        "notifyNumOnMainThread": str(slave_thread_num),
        "notifyNumPerThread": "1",
        "netLayerNum": str(net_layer_num),
        "chanCount": str(len(channels)),
    })
    for netlayer, peer in channels:
        ET.SubElement(instruction, "channel", {
            "netLayerId": str(netlayer),
            "localRank": str(rank),
            "remoteRank": str(peer),
            "linkProto": link_proto,
        })


def add_sync(parent: ET.Element, op_code: str, thread_ids: list[int]) -> None:
    instruction = ET.SubElement(parent, "instruction", {
        "opCode": op_code,
        "mainThreadIdx": "0",
        "subThreadNum": str(len(thread_ids)),
    })
    for thread_id in thread_ids:
        ET.SubElement(instruction, "subThread", {"subThreadId": str(thread_id)})


def add_send_recv_write_batches(
    parent: ET.Element,
    op: PairExchangeOp,
    thread_idx: int,
    rank_count: int,
    chunks_per_rank: int,
    link_proto: str,
    max_slices_per_op: int,
) -> None:
    for offset in range(0, len(op.sends), max_slices_per_op):
        batch = op.sends[offset:offset + max_slices_per_op]
        instruction = ET.SubElement(parent, "instruction", {
            "opCode": "SendRecvWrite",
            "sliceNum": str(rank_count * chunks_per_rank),
            "netLayerId": str(op.netlayer),
            "linkProto": link_proto,
            "threadIdx": str(thread_idx),
            "srcSliceNum": str(len(batch)),
            "dstSliceNum": str(len(batch)),
        })
        for send in batch:
            source_rank, chunk_idx = source_and_chunk_index(send.chunk, chunks_per_rank)
            slice_idx = flat_slice_index(source_rank, chunk_idx, chunks_per_rank)
            buffer_type = "INPUT" if source_rank == op.rank else "HCCL_BUFFER"
            ET.SubElement(instruction, "srcSlice", {
                "bufferType": buffer_type,
                "sliceIdx": str(chunk_idx if buffer_type == "INPUT" else slice_idx),
                "rankId": str(op.rank),
            })
        for send in batch:
            source_rank, chunk_idx = source_and_chunk_index(send.chunk, chunks_per_rank)
            ET.SubElement(instruction, "dstSlice", {
                "bufferType": "HCCL_BUFFER",
                "sliceIdx": str(flat_slice_index(source_rank, chunk_idx, chunks_per_rank)),
                "rankId": str(op.peer),
                "recvRankId": str(op.peer),
            })


def add_initial_local_copy(
    parent: ET.Element,
    rank: int,
    rank_count: int,
    chunks_per_rank: int,
    max_slices_per_op: int,
) -> None:
    copies = [
        (
            "INPUT",
            "OUTPUT",
            chunk_idx,
            flat_slice_index(rank, chunk_idx, chunks_per_rank),
        )
        for chunk_idx in range(chunks_per_rank)
    ]
    add_local_copy_batches(parent, rank, rank_count, chunks_per_rank, copies, max_slices_per_op)


def add_final_local_copy(
    parent: ET.Element,
    rank: int,
    rank_count: int,
    chunks_per_rank: int,
    max_slices_per_op: int,
) -> None:
    copies = []
    for source_rank in range(rank_count):
        if source_rank == rank:
            continue
        for chunk_idx in range(chunks_per_rank):
            slice_idx = flat_slice_index(source_rank, chunk_idx, chunks_per_rank)
            copies.append(("HCCL_BUFFER", "OUTPUT", slice_idx, slice_idx))
    add_local_copy_batches(parent, rank, rank_count, chunks_per_rank, copies, max_slices_per_op)


def add_local_copy_batches(
    parent: ET.Element,
    rank: int,
    rank_count: int,
    chunks_per_rank: int,
    copies: list[tuple[str, str, int, int]],
    max_slices_per_op: int,
) -> None:
    total_slices = rank_count * chunks_per_rank
    for offset in range(0, len(copies), max_slices_per_op):
        batch = copies[offset:offset + max_slices_per_op]
        instruction = ET.SubElement(parent, "instruction", {
            "opCode": "LocalCopy",
            "sliceNum": str(total_slices),
            "threadIdx": "0",
            "srcSliceNum": str(len(batch)),
            "dstSliceNum": str(len(batch)),
        })
        for src_buffer, _dst_buffer, src_slice_idx, _dst_slice_idx in batch:
            ET.SubElement(instruction, "srcSlice", {
                "bufferType": src_buffer,
                "sliceIdx": str(src_slice_idx),
                "rankId": str(rank),
            })
        for _src_buffer, dst_buffer, _src_slice_idx, dst_slice_idx in batch:
            ET.SubElement(instruction, "dstSlice", {
                "bufferType": dst_buffer,
                "sliceIdx": str(dst_slice_idx),
                "rankId": str(rank),
            })


def source_and_chunk_index(flat_chunk: int, chunks_per_rank: int) -> tuple[int, int]:
    return flat_chunk // chunks_per_rank, flat_chunk % chunks_per_rank


def flat_slice_index(source_rank: int, chunk_idx: int, chunks_per_rank: int) -> int:
    return source_rank * chunks_per_rank + chunk_idx
