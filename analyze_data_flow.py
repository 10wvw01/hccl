#!/usr/bin/env python3
import re
import sys
import os
import json
import math
from collections import defaultdict

RANK_SIZE = 16
POD_SIZE = 4
SUPER_POD_SIZE = 8
INPUT_BASE_OFFSET = 0x10000000000
BASE_STEP = 0x30000000000
CCL_BUFFER_OFFSET = 0x20000000000
OUTPUT_OFFSET = 0x10000000000

RANK_TO_NPU = {
    0: (0, 0, 0), 1: (0, 0, 1), 2: (0, 0, 2), 3: (0, 0, 3),
    4: (0, 1, 0), 5: (0, 1, 1), 6: (0, 1, 2), 7: (0, 1, 3),
    8: (1, 0, 0), 9: (1, 0, 1), 10: (1, 0, 2), 11: (1, 0, 3),
    12: (1, 1, 0), 13: (1, 1, 1), 14: (1, 1, 2), 15: (1, 1, 3),
}

PODS = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12, 13, 14, 15],
}

SUPER_PODS = {
    0: [0, 1, 2, 3, 4, 5, 6, 7],
    1: [8, 9, 10, 11, 12, 13, 14, 15],
}


def base_addr_to_rank(base_addr):
    if base_addr < INPUT_BASE_OFFSET:
        return -1
    rank_id = (base_addr - INPUT_BASE_OFFSET) // BASE_STEP
    if 0 <= rank_id < RANK_SIZE:
        remainder = (base_addr - INPUT_BASE_OFFSET) % BASE_STEP
        if remainder == 0:
            return rank_id, "input"
        elif remainder == OUTPUT_OFFSET:
            return rank_id, "output"
        elif remainder == CCL_BUFFER_OFFSET:
            return rank_id, "ccl"
    return -1, "unknown"


def parse_log(log_path):
    tid_to_rank = {}
    rank_input_base = {}
    rank_output_base = {}
    data_trans_ops = []
    step_infos = []

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'\[(\d+)\]\[HcclGetRankId\]\s+rankId:\s+(\d+)', line)
            if m:
                tid, rank = int(m.group(1)), int(m.group(2))
                if tid not in tid_to_rank:
                    tid_to_rank[tid] = rank

            m = re.search(
                r'\[(\d+)\]\[ConstructHcclDfxOpInfo\].*inputMemAddr\[(0x[0-9a-fA-F]+)\].*outputMemAddr\[(0x[0-9a-fA-F]+)\]',
                line,
            )
            if m:
                tid = int(m.group(1))
                in_addr = int(m.group(2), 16)
                out_addr = int(m.group(3), 16)
                rank = tid_to_rank.get(tid, -1)
                if rank >= 0:
                    rank_input_base[rank] = in_addr
                    rank_output_base[rank] = out_addr

            m = re.search(
                r'\[(\d+)\]\[AlgDataTransWrapper\]\[(\w+)\]\[(\w+)\]\s+'
                r'sliceIdx\[(\d+)\],\s+sliceNum\[(\d+)\],\s+'
                r'srcBase\[(0x[0-9a-fA-F]+)\],\s+srcOffset\[(\d+)\],\s+srcAddr\[(0x[0-9a-fA-F]+)\],\s+'
                r'srcSize\[(\d+)\],\s+srcCount\[(\d+)\],\s+'
                r'dstBase\[(0x[0-9a-fA-F]+)\],\s+dstOffset\[(\d+)\],\s+dstAddr\[(0x[0-9a-fA-F]+)\],\s+'
                r'dstSize\[(\d+)\],\s+dstCount\[(\d+)\],\s+len\[(\d+)\],\s+'
                r'dataType\[(\d+)\],\s+reduceOp\[(\d+)\]',
                line,
            )
            if m:
                tid = int(m.group(1))
                op_type = m.group(2)
                op_sub = m.group(3)
                slice_idx = int(m.group(4))
                slice_num = int(m.group(5))
                src_base = int(m.group(6), 16)
                src_offset = int(m.group(7))
                src_addr = int(m.group(8), 16)
                src_size = int(m.group(9))
                src_count = int(m.group(10))
                dst_base = int(m.group(11), 16)
                dst_offset = int(m.group(12))
                dst_addr = int(m.group(13), 16)
                dst_size = int(m.group(14))
                dst_count = int(m.group(15))
                length = int(m.group(16))
                data_type = int(m.group(17))
                reduce_op = int(m.group(18))

                rank = tid_to_rank.get(tid, -1)
                src_rank, src_type = base_addr_to_rank(src_base)
                dst_rank, dst_type = base_addr_to_rank(dst_base)

                data_trans_ops.append({
                    'tid': tid,
                    'rank': rank,
                    'op_type': op_type,
                    'op_sub': op_sub,
                    'slice_idx': slice_idx,
                    'slice_num': slice_num,
                    'src_base': src_base,
                    'src_offset': src_offset,
                    'src_addr': src_addr,
                    'src_size': src_size,
                    'src_count': src_count,
                    'dst_base': dst_base,
                    'dst_offset': dst_offset,
                    'dst_addr': dst_addr,
                    'dst_size': dst_size,
                    'dst_count': dst_count,
                    'len': length,
                    'data_type': data_type,
                    'reduce_op': reduce_op,
                    'src_rank': src_rank,
                    'src_type': src_type,
                    'dst_rank': dst_rank,
                    'dst_type': dst_type,
                })

            m = re.search(
                r'\[(\d+)\]\[InsTempReduceScatterNHR\]\[GetStepInfoList\]\s+i\[(\d+)\]\s+txSliceIdx\[(\d+)\]\s+rxSliceIdx\[(\d+)\]',
                line,
            )
            if m:
                tid = int(m.group(1))
                step_i = int(m.group(2))
                tx_slice = int(m.group(3))
                rx_slice = int(m.group(4))
                rank = tid_to_rank.get(tid, -1)
                step_infos.append({
                    'tid': tid,
                    'rank': rank,
                    'step_i': step_i,
                    'tx_slice': tx_slice,
                    'rx_slice': rx_slice,
                })

    return tid_to_rank, rank_input_base, rank_output_base, data_trans_ops, step_infos


def get_pod(rank):
    for pod_id, ranks in PODS.items():
        if rank in ranks:
            return pod_id
    return -1


def get_super_pod(rank):
    for sp_id, ranks in SUPER_PODS.items():
        if rank in ranks:
            return sp_id
    return -1


def classify_operation(op):
    src_rank = op['src_rank']
    dst_rank = op['dst_rank']
    op_type = op['op_type']

    if op_type == 'LocalCopy':
        return 'local_copy'
    elif op_type == 'LocalReduce':
        return 'local_reduce'
    elif op_type == 'SendRecvWrite':
        if src_rank >= 0 and dst_rank >= 0:
            src_pod = get_pod(src_rank)
            dst_pod = get_pod(dst_rank)
            if src_pod == dst_pod:
                return 'intra_pod_write'
            else:
                src_sp = get_super_pod(src_rank)
                dst_sp = get_super_pod(dst_rank)
                if src_sp == dst_sp:
                    return 'inter_pod_write'
                else:
                    return 'inter_super_pod_write'
        return 'unknown_write'
    elif op_type == 'SendRecvWriteReduce':
        if src_rank >= 0 and dst_rank >= 0:
            src_pod = get_pod(src_rank)
            dst_pod = get_pod(dst_rank)
            if src_pod == dst_pod:
                return 'intra_pod_write_reduce'
            else:
                src_sp = get_super_pod(src_rank)
                dst_sp = get_super_pod(dst_rank)
                if src_sp == dst_sp:
                    return 'inter_pod_write_reduce'
                else:
                    return 'inter_super_pod_write_reduce'
        return 'unknown_write_reduce'
    return 'unknown'


def generate_text_report(data_trans_ops, step_infos, tid_to_rank):
    print("=" * 100)
    print("  ReduceScatter 数据流动分析报告")
    print("  算法: InsReduceScatterSequenceMesh1DNHRNHR")
    print("  拓扑: 2×2×4 Mesh (2超节点 × 2Pod × 4Rank)")
    print("=" * 100)

    print("\n## 1. Rank 映射表")
    print("-" * 80)
    print(f"{'Rank':>6} | {'NPU Pos':>12} | {'Pod':>4} | {'SuperPod':>9} | {'Pod内Rank':>10}")
    print("-" * 80)
    for rank in range(RANK_SIZE):
        npu = RANK_TO_NPU[rank]
        pod = get_pod(rank)
        sp = get_super_pod(rank)
        pod_ranks = PODS[pod]
        pod_local_idx = pod_ranks.index(rank)
        print(f"{rank:>6} | {str(npu):>12} | {pod:>4} | {sp:>9} | {pod_local_idx:>10}")

    print("\n## 2. 操作分类统计")
    print("-" * 80)
    op_categories = defaultdict(list)
    for op in data_trans_ops:
        cat = classify_operation(op)
        op_categories[cat].append(op)

    for cat, ops in sorted(op_categories.items()):
        print(f"  {cat:35s}: {len(ops):>4} 次")

    print("\n## 3. 各Rank数据流动详情")
    print("-" * 80)
    for rank in range(RANK_SIZE):
        rank_ops = [op for op in data_trans_ops if op['rank'] == rank]
        if not rank_ops:
            continue
        print(f"\n  Rank {rank} (NPU {RANK_TO_NPU[rank]}, Pod {get_pod(rank)}, SuperPod {get_super_pod(rank)}):")
        print(f"  {'操作类型':>25} | {'Slice':>6} | {'Src':>20} | {'Dst':>20} | {'分类':>25}")
        print(f"  {'':>25} | {'':>6} | {'':>20} | {'':>20} | {'':>25}")
        for op in rank_ops:
            op_full = f"{op['op_type']}_{op['op_sub']}"
            src_desc = f"R{op['src_rank']}({op['src_type']})+{op['src_offset']}" if op['src_rank'] >= 0 else f"0x{op['src_base']:x}+{op['src_offset']}"
            dst_desc = f"R{op['dst_rank']}({op['dst_type']})+{op['dst_offset']}" if op['dst_rank'] >= 0 else f"0x{op['dst_base']:x}+{op['dst_offset']}"
            cat = classify_operation(op)
            print(f"  {op_full:>25} | {op['slice_idx']:>3}/{op['slice_num']:<2} | {src_desc:>20} | {dst_desc:>20} | {cat:>25}")

    print("\n## 4. 跨Rank数据传输汇总 (SendRecvWrite / SendRecvWriteReduce)")
    print("-" * 80)
    cross_rank_ops = [op for op in data_trans_ops if op['op_type'] in ('SendRecvWrite', 'SendRecvWriteReduce') and op['src_rank'] >= 0 and op['dst_rank'] >= 0 and op['src_rank'] != op['dst_rank']]

    flow_map = defaultdict(lambda: defaultdict(list))
    for op in cross_rank_ops:
        flow_map[op['src_rank']][op['dst_rank']].append(op)

    for src_rank in sorted(flow_map.keys()):
        for dst_rank in sorted(flow_map[src_rank].keys()):
            ops = flow_map[src_rank][dst_rank]
            op_types = set(o['op_type'] for o in ops)
            slices = [(o['slice_idx'], o['slice_num']) for o in ops]
            src_pod = get_pod(src_rank)
            dst_pod = get_pod(dst_rank)
            src_sp = get_super_pod(src_rank)
            dst_sp = get_super_pod(dst_rank)
            level = "Pod内" if src_pod == dst_pod else ("超节点内" if src_sp == dst_sp else "跨超节点")
            print(f"  Rank {src_rank:>2} --> Rank {dst_rank:>2} | {level:>6} | 操作: {', '.join(op_types)} | Slices: {slices}")

    print("\n## 5. 步骤信息 (NHR Step)")
    print("-" * 80)
    rank_steps = defaultdict(list)
    for si in step_infos:
        rank_steps[si['rank']].append(si)

    for rank in sorted(rank_steps.keys()):
        steps = rank_steps[rank]
        print(f"  Rank {rank:>2}: ", end="")
        for s in steps:
            print(f"step[{s['step_i']}] tx={s['tx_slice']} rx={s['rx_slice']}  ", end="")
        print()


def generate_mermaid_diagram(data_trans_ops, output_path):
    cross_rank_ops = [
        op for op in data_trans_ops
        if op['op_type'] in ('SendRecvWrite', 'SendRecvWriteReduce')
        and op['src_rank'] >= 0 and op['dst_rank'] >= 0
    ]

    flow_map = defaultdict(lambda: defaultdict(lambda: {'types': set(), 'slices': set()}))
    for op in cross_rank_ops:
        info = flow_map[op['src_rank']][op['dst_rank']]
        info['types'].add(op['op_type'])
        if op['src_rank'] != op['dst_rank']:
            info['slices'].add((op['slice_idx'], op['slice_num']))

    lines = []
    lines.append("graph TD")

    for pod_id, ranks in sorted(PODS.items()):
        sp_id = get_super_pod(ranks[0])
        lines.append(f"    subgraph SuperPod{sp_id}_Pod{pod_id}[\"SuperPod {sp_id} / Pod {pod_id}\"]")
        for rank in ranks:
            npu = RANK_TO_NPU[rank]
            lines.append(f"        R{rank}[\"Rank {rank}<br/>NPU {npu}\"]")
        lines.append("    end")

    edge_id = 0
    intra_edges = []
    inter_pod_edges = []
    inter_sp_edges = []
    for src_rank in sorted(flow_map.keys()):
        for dst_rank in sorted(flow_map[src_rank].keys()):
            info = flow_map[src_rank][dst_rank]
            if src_rank == dst_rank:
                continue
            types_str = ", ".join(sorted(info['types']))
            slices_str = ", ".join(f"s[{s[0]}/{s[1]}]" for s in sorted(info['slices']))

            src_pod = get_pod(src_rank)
            dst_pod = get_pod(dst_rank)
            src_sp = get_super_pod(src_rank)
            dst_sp = get_super_pod(dst_rank)

            label = f"{types_str}<br/>{slices_str}"
            lines.append(f"    R{src_rank} -->|\"{label}\"| R{dst_rank}")

            if src_pod == dst_pod:
                intra_edges.append(edge_id)
            elif src_sp == dst_sp:
                inter_pod_edges.append(edge_id)
            else:
                inter_sp_edges.append(edge_id)
            edge_id += 1

    lines.append("")
    if intra_edges:
        ids = ",".join(str(e) for e in intra_edges)
        lines.append(f"    linkStyle {ids} stroke:#4CAF50,stroke-width:2px")
    if inter_pod_edges:
        ids = ",".join(str(e) for e in inter_pod_edges)
        lines.append(f"    linkStyle {ids} stroke:#FF9800,stroke-width:2px,stroke-dasharray:5 3")
    if inter_sp_edges:
        ids = ",".join(str(e) for e in inter_sp_edges)
        lines.append(f"    linkStyle {ids} stroke:#F44336,stroke-width:3px,stroke-dasharray:3 3")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"\nMermaid 数据流图已保存到: {output_path}")


def generate_ascii_diagram(data_trans_ops, output_path):
    cross_rank_ops = [
        op for op in data_trans_ops
        if op['op_type'] in ('SendRecvWrite', 'SendRecvWriteReduce')
        and op['src_rank'] >= 0 and op['dst_rank'] >= 0
        and op['src_rank'] != op['dst_rank']
    ]

    flow_map = defaultdict(lambda: defaultdict(lambda: {'types': set(), 'slices': set(), 'count': 0}))
    for op in cross_rank_ops:
        info = flow_map[op['src_rank']][op['dst_rank']]
        info['types'].add(op['op_type'])
        info['slices'].add((op['slice_idx'], op['slice_num']))
        info['count'] += 1

    lines = []
    lines.append("=" * 120)
    lines.append("  ReduceScatter 数据流动示意图 (ASCII)")
    lines.append("  算法: InsReduceScatterSequenceMesh1DNHRNHR")
    lines.append("  拓扑: 2×2×4 Mesh")
    lines.append("=" * 120)

    lines.append("")
    lines.append("  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    lines.append("  │                                    ReduceScatter 数据流动示意图                                         │")
    lines.append("  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤")
    lines.append("  │                                                                                                       │")
    lines.append("  │  SuperPod 0                                                                                           │")
    lines.append("  │  ┌─── Pod 0 ───────────────────┐    ┌─── Pod 1 ───────────────────┐                                 │")
    lines.append("  │  │  R0    R1    R2    R3       │    │  R4    R5    R6    R7       │                                 │")
    lines.append("  │  │  [0,0,0] [0,0,1] [0,0,2] [0,0,3] │    │  [0,1,0] [0,1,1] [0,1,2] [0,1,3] │                                 │")
    lines.append("  │  └─────────────────────────────┘    └─────────────────────────────┘                                 │")
    lines.append("  │                                                                                                       │")
    lines.append("  │  SuperPod 1                                                                                           │")
    lines.append("  │  ┌─── Pod 2 ───────────────────┐    ┌─── Pod 3 ───────────────────┐                                 │")
    lines.append("  │  │  R8    R9    R10   R11      │    │  R12   R13   R14   R15      │                                 │")
    lines.append("  │  │  [1,0,0] [1,0,1] [1,0,2] [1,0,3] │    │  [1,1,0] [1,1,1] [1,1,2] [1,1,3] │                                 │")
    lines.append("  │  └─────────────────────────────┘    └─────────────────────────────┘                                 │")
    lines.append("  │                                                                                                       │")
    lines.append("  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘")

    lines.append("")
    lines.append("  数据传输详情:")
    lines.append("  " + "-" * 110)
    lines.append(f"  {'源Rank':>8} | {'目标Rank':>8} | {'层级':>10} | {'操作类型':>25} | {'数据切片':>30} | {'次数':>4}")
    lines.append("  " + "-" * 110)

    for src_rank in sorted(flow_map.keys()):
        for dst_rank in sorted(flow_map[src_rank].keys()):
            info = flow_map[src_rank][dst_rank]
            src_pod = get_pod(src_rank)
            dst_pod = get_pod(dst_rank)
            src_sp = get_super_pod(src_rank)
            dst_sp = get_super_pod(dst_rank)

            if src_pod == dst_pod:
                level = "Pod内"
            elif src_sp == dst_sp:
                level = "超节点内"
            else:
                level = "跨超节点"

            types_str = ", ".join(sorted(info['types']))
            slices_str = ", ".join(f"[{s[0]}/{s[1]}]" for s in sorted(info['slices']))
            if len(slices_str) > 30:
                slices_str = slices_str[:27] + "..."

            lines.append(f"  R{src_rank:>6} | R{dst_rank:>6} | {level:>10} | {types_str:>25} | {slices_str:>30} | {info['count']:>4}")

    lines.append("")
    lines.append("  图例:")
    lines.append("    Pod内传输  (绿色) - 同一Pod内Rank之间的数据传输 (Intra-Pod NHR)")
    lines.append("    超节点内传输 (橙色) - 同一SuperPod不同Pod之间的数据传输 (Inter-Pod NHR)")
    lines.append("    跨超节点传输 (红色) - 不同SuperPod之间的数据传输 (Inter-SuperPod NHR)")
    lines.append("")

    lines.append("  各Rank本地操作统计:")
    lines.append("  " + "-" * 60)
    local_ops = defaultdict(lambda: defaultdict(int))
    for op in data_trans_ops:
        if op['op_type'] in ('LocalCopy', 'LocalReduce'):
            local_ops[op['rank']][op['op_type']] += 1

    for rank in range(RANK_SIZE):
        if rank in local_ops:
            lc = local_ops[rank].get('LocalCopy', 0)
            lr = local_ops[rank].get('LocalReduce', 0)
            lines.append(f"    Rank {rank:>2}: LocalCopy={lc:>3}, LocalReduce={lr:>3}")

    lines.append("")
    lines.append("  ReduceScatter 三级流水线说明:")
    lines.append("  " + "-" * 80)
    lines.append("    Level 0 (Intra-Pod):        Pod内4个Rank之间执行NHR ReduceScatter")
    lines.append("                                每个Rank将数据分为4个Slice, 在Pod内进行Reduce+Scatter")
    lines.append("    Level 1 (Inter-Pod):         同一SuperPod内2个Pod之间执行NHR ReduceScatter")
    lines.append("                                Pod间交换部分归约结果, 进一步聚合")
    lines.append("    Level 2 (Inter-SuperPod):    2个SuperPod之间执行NHR ReduceScatter")
    lines.append("                                跨SuperPod交换最终归约结果")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"ASCII 数据流图已保存到: {output_path}")


def generate_per_rank_flow_diagram(data_trans_ops, output_path):
    lines = []
    lines.append("=" * 120)
    lines.append("  各Rank数据流动详细示意图")
    lines.append("=" * 120)

    for rank in range(RANK_SIZE):
        rank_ops = [op for op in data_trans_ops if op['rank'] == rank]
        if not rank_ops:
            continue

        pod = get_pod(rank)
        sp = get_super_pod(rank)
        npu = RANK_TO_NPU[rank]

        lines.append("")
        lines.append(f"  ═══ Rank {rank} (NPU {npu}, Pod {pod}, SuperPod {sp}) ═══")
        lines.append("")

        send_ops = [op for op in rank_ops if op['op_type'] == 'SendRecvWrite']
        send_reduce_ops = [op for op in rank_ops if op['op_type'] == 'SendRecvWriteReduce']
        local_copy_ops = [op for op in rank_ops if op['op_type'] == 'LocalCopy']
        local_reduce_ops = [op for op in rank_ops if op['op_type'] == 'LocalReduce']

        if send_ops:
            lines.append("    [SendRecvWrite] 跨Rank数据写入:")
            for op in send_ops:
                src_desc = f"R{op['src_rank']}({op['src_type']})+{op['src_offset']}" if op['src_rank'] >= 0 else f"0x{op['src_base']:x}+{op['src_offset']}"
                dst_desc = f"R{op['dst_rank']}({op['dst_type']})+{op['dst_offset']}" if op['dst_rank'] >= 0 else f"0x{op['dst_base']:x}+{op['dst_offset']}"
                lines.append(f"      {src_desc} ──WRITE──▶ {dst_desc}  [slice {op['slice_idx']}/{op['slice_num']}]")

        if send_reduce_ops:
            lines.append("    [SendRecvWriteReduce] 跨Rank数据写入+归约:")
            for op in send_reduce_ops:
                src_desc = f"R{op['src_rank']}({op['src_type']})+{op['src_offset']}" if op['src_rank'] >= 0 else f"0x{op['src_base']:x}+{op['src_offset']}"
                dst_desc = f"R{op['dst_rank']}({op['dst_type']})+{op['dst_offset']}" if op['dst_rank'] >= 0 else f"0x{op['dst_base']:x}+{op['dst_offset']}"
                reduce_desc = f"reduceOp={op['reduce_op']}" if op['reduce_op'] != 255 else ""
                lines.append(f"      {src_desc} ──WRITE_REDUCE──▶ {dst_desc}  [slice {op['slice_idx']}/{op['slice_num']}] {reduce_desc}")

        if local_copy_ops:
            lines.append("    [LocalCopy] 本地数据拷贝:")
            for op in local_copy_ops:
                src_desc = f"{op['src_type']}+{op['src_offset']}"
                dst_desc = f"{op['dst_type']}+{op['dst_offset']}"
                lines.append(f"      {src_desc} ──COPY──▶ {dst_desc}  [slice {op['slice_idx']}/{op['slice_num']}]")

        if local_reduce_ops:
            lines.append("    [LocalReduce] 本地数据归约:")
            for op in local_reduce_ops:
                src_desc = f"{op['src_type']}+{op['src_offset']}"
                dst_desc = f"{op['dst_type']}+{op['dst_offset']}"
                lines.append(f"      {src_desc} ──REDUCE──▶ {dst_desc}  [slice {op['slice_idx']}/{op['slice_num']}] reduceOp={op['reduce_op']}")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"各Rank数据流详情已保存到: {output_path}")


def generate_dot_diagram(data_trans_ops, output_path):
    cross_rank_ops = [
        op for op in data_trans_ops
        if op['op_type'] in ('SendRecvWrite', 'SendRecvWriteReduce')
        and op['src_rank'] >= 0 and op['dst_rank'] >= 0
        and op['src_rank'] != op['dst_rank']
    ]

    flow_map = defaultdict(lambda: defaultdict(lambda: {'types': set(), 'slices': set(), 'count': 0}))
    for op in cross_rank_ops:
        info = flow_map[op['src_rank']][op['dst_rank']]
        info['types'].add(op['op_type'])
        info['slices'].add((op['slice_idx'], op['slice_num']))
        info['count'] += 1

    lines = []
    lines.append('digraph ReduceScatterFlow {')
    lines.append('    rankdir=LR;')
    lines.append('    node [shape=box, style=filled, fillcolor=lightblue, fontname="monospace"];')
    lines.append('    edge [fontname="monospace", fontsize=9];')
    lines.append('')

    for sp_id in sorted(SUPER_PODS.keys()):
        lines.append(f'    subgraph cluster_sp{sp_id} {{')
        lines.append(f'        label="SuperPod {sp_id}";')
        lines.append(f'        style=filled;')
        lines.append(f'        fillcolor=lightyellow;')
        for pod_id in sorted(PODS.keys()):
            pod_ranks = PODS[pod_id]
            if get_super_pod(pod_ranks[0]) != sp_id:
                continue
            lines.append(f'        subgraph cluster_pod{pod_id} {{')
            lines.append(f'            label="Pod {pod_id}";')
            lines.append(f'            style=filled;')
            lines.append(f'            fillcolor=lightcyan;')
            for rank in pod_ranks:
                npu = RANK_TO_NPU[rank]
                lines.append(f'            R{rank} [label="Rank {rank}\\nNPU {npu}"];')
            lines.append('        }')
        lines.append('    }')

    lines.append('')

    for src_rank in sorted(flow_map.keys()):
        for dst_rank in sorted(flow_map[src_rank].keys()):
            info = flow_map[src_rank][dst_rank]
            src_pod = get_pod(src_rank)
            dst_pod = get_pod(dst_rank)
            src_sp = get_super_pod(src_rank)
            dst_sp = get_super_pod(dst_rank)

            if src_pod == dst_pod:
                color = "green"
                penwidth = "1.5"
            elif src_sp == dst_sp:
                color = "orange"
                penwidth = "2.0"
            else:
                color = "red"
                penwidth = "2.5"

            types_str = ", ".join(sorted(info['types']))
            slices_str = ", ".join(f"[{s[0]}/{s[1]}]" for s in sorted(info['slices']))
            label = f"{types_str}\\n{slices_str}"

            lines.append(f'    R{src_rank} -> R{dst_rank} [label="{label}", color={color}, penwidth={penwidth}];')

    lines.append('}')

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Graphviz DOT 数据流图已保存到: {output_path}")
    print(f"  可使用以下命令生成图片: dot -Tpng {output_path} -o data_flow.png")


def _svg_rank_pos(rank):
    CELL_W = 120
    CELL_H = 80
    PAD_X = 60
    PAD_Y = 80
    POD_GAP_X = 80
    SP_GAP_Y = 100
    pod_id = get_pod(rank)
    sp_id = get_super_pod(rank)
    pod_local_idx = PODS[pod_id].index(rank)
    col_in_pod = pod_local_idx % 4
    row_in_pod = pod_local_idx // 4
    pod_col = pod_id % 2
    pod_row = pod_id // 2
    x = PAD_X + pod_col * (4 * CELL_W + POD_GAP_X) + col_in_pod * CELL_W + CELL_W // 2
    y = PAD_Y + pod_row * (1 * CELL_H + SP_GAP_Y) + row_in_pod * CELL_H + CELL_H // 2 + 40
    return x, y


def _svg_draw_background(svg_parts, rank_pos_fn, highlight_level=None):
    CELL_W = 120
    CELL_H = 80
    PAD_X = 60
    PAD_Y = 80
    POD_GAP_X = 80
    SP_GAP_Y = 100

    sp_colors = ['#E8F5E9', '#E3F2FD']
    sp_colors_hi = ['#C8E6C9', '#BBDEFB']
    pod_colors = [['#C8E6C9', '#A5D6A7'], ['#BBDEFB', '#90CAF9']]
    pod_colors_hi = [['#A5D6A7', '#81C784'], ['#90CAF9', '#64B5F6']]

    for sp_id in sorted(SUPER_PODS.keys()):
        sp_ranks = SUPER_PODS[sp_id]
        xs = [rank_pos_fn(r)[0] for r in sp_ranks]
        ys = [rank_pos_fn(r)[1] for r in sp_ranks]
        rx = min(xs) - CELL_W // 2 - 15
        ry = min(ys) - CELL_H // 2 - 25
        rw = max(xs) - min(xs) + CELL_W + 30
        rh = max(ys) - min(ys) + CELL_H + 40
        fill = sp_colors_hi[sp_id] if highlight_level == 'superpod' else sp_colors[sp_id]
        svg_parts.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="12" fill="{fill}" stroke="#999" stroke-width="2" opacity="0.5"/>')
        svg_parts.append(f'<text x="{rx + 10}" y="{ry + 18}" font-size="14" font-weight="bold" fill="#777">SuperPod {sp_id}</text>')

    for pod_id in sorted(PODS.keys()):
        pod_ranks = PODS[pod_id]
        sp_id = get_super_pod(pod_id)
        pod_local = pod_id if sp_id == 0 else pod_id - 2
        xs = [rank_pos_fn(r)[0] for r in pod_ranks]
        ys = [rank_pos_fn(r)[1] for r in pod_ranks]
        rx = min(xs) - CELL_W // 2 - 8
        ry = min(ys) - CELL_H // 2 - 8
        rw = max(xs) - min(xs) + CELL_W + 16
        rh = max(ys) - min(ys) + CELL_H + 16
        fill = pod_colors_hi[sp_id][pod_local % 2] if highlight_level == 'pod' else pod_colors[sp_id][pod_local % 2]
        svg_parts.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="8" fill="{fill}" stroke="#aaa" stroke-width="1.5" opacity="0.7"/>')
        svg_parts.append(f'<text x="{rx + 8}" y="{ry + 16}" font-size="11" fill="#888">Pod {pod_id}</text>')


def _svg_draw_ranks(svg_parts, rank_pos_fn, active_ranks=None):
    for rank in range(RANK_SIZE):
        cx, cy = rank_pos_fn(rank)
        npu = RANK_TO_NPU[rank]
        if active_ranks is not None and rank not in active_ranks:
            svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="28" fill="#BDBDBD" stroke="#9E9E9E" stroke-width="1.5" opacity="0.4"/>')
            svg_parts.append(f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="11" font-weight="bold" fill="#999" opacity="0.5">R{rank}</text>')
        else:
            svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="28" fill="#42A5F5" stroke="#1565C0" stroke-width="2"/>')
            svg_parts.append(f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="11" font-weight="bold" fill="white">R{rank}</text>')
            svg_parts.append(f'<text x="{cx}" y="{cy + 12}" text-anchor="middle" font-size="8" fill="white">{npu}</text>')


def _svg_draw_arrow(svg_parts, x1, y1, x2, y2, color, width, alpha, dash='', label=None, label_color=None):
    dx = x2 - x1
    dy = y2 - y1
    dist = (dx**2 + dy**2)**0.5
    if dist == 0:
        return
    offset_x = dx / dist * 30
    offset_y = dy / dist * 30
    sx = x1 + offset_x
    sy = y1 + offset_y
    ex = x2 - offset_x
    ey = y2 - offset_y
    mid_x = (sx + ex) / 2
    mid_y = (sy + ey) / 2
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    svg_parts.append(f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="{color}" stroke-width="{width}" opacity="{alpha}"{dash_attr}/>')
    arrow_size = 8
    ax1 = ex - arrow_size * dx / dist + arrow_size * 0.5 * dy / dist
    ay1 = ey - arrow_size * dy / dist - arrow_size * 0.5 * dx / dist
    ax2 = ex - arrow_size * dx / dist - arrow_size * 0.5 * dy / dist
    ay2 = ey - arrow_size * dy / dist + arrow_size * 0.5 * dx / dist
    svg_parts.append(f'<polygon points="{ex},{ey} {ax1},{ay1} {ax2},{ay2}" fill="{color}" opacity="{alpha}"/>')
    if label:
        lc = label_color or color
        svg_parts.append(f'<text x="{mid_x}" y="{mid_y - 6}" text-anchor="middle" font-size="8" fill="{lc}">{label}</text>')


def generate_svg_diagram(data_trans_ops, output_path):
    cross_rank_ops = [
        op for op in data_trans_ops
        if op['op_type'] in ('SendRecvWrite', 'SendRecvWriteReduce')
        and op['src_rank'] >= 0 and op['dst_rank'] >= 0
        and op['src_rank'] != op['dst_rank']
    ]

    flow_map = defaultdict(lambda: defaultdict(lambda: {'types': set(), 'slices': set(), 'count': 0}))
    for op in cross_rank_ops:
        info = flow_map[op['src_rank']][op['dst_rank']]
        info['types'].add(op['op_type'])
        info['slices'].add((op['slice_idx'], op['slice_num']))
        info['count'] += 1

    CELL_W = 120
    CELL_H = 80
    PAD_X = 60
    PAD_Y = 80
    POD_GAP_X = 80
    SP_GAP_Y = 100

    total_w = PAD_X * 2 + 2 * (4 * CELL_W) + POD_GAP_X
    total_h = PAD_Y * 2 + 2 * (1 * CELL_H) + SP_GAP_Y + 80

    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">')
    svg_parts.append(f'<rect width="{total_w}" height="{total_h}" fill="white"/>')
    svg_parts.append(f'<text x="{total_w//2}" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">ReduceScatter Data Flow (All Levels)</text>')
    svg_parts.append(f'<text x="{total_w//2}" y="52" text-anchor="middle" font-size="13" fill="#666">Algorithm: InsReduceScatterSequenceMesh1DNHRNHR | Topology: 2x2x4 Mesh</text>')

    _svg_draw_background(svg_parts, _svg_rank_pos)
    _svg_draw_ranks(svg_parts, _svg_rank_pos)

    level_styles = {
        'Pod内': {'color': '#4CAF50', 'dash': '', 'width': '1.5', 'alpha': '0.5'},
        '超节点内': {'color': '#FF9800', 'dash': '8,4', 'width': '2', 'alpha': '0.6'},
        '跨超节点': {'color': '#F44336', 'dash': '4,4', 'width': '2.5', 'alpha': '0.7'},
    }

    drawn_edges = set()
    for src_rank in sorted(flow_map.keys()):
        for dst_rank in sorted(flow_map[src_rank].keys()):
            info = flow_map[src_rank][dst_rank]
            src_pod = get_pod(src_rank)
            dst_pod = get_pod(dst_rank)
            src_sp = get_super_pod(src_rank)
            dst_sp = get_super_pod(dst_rank)

            if src_pod == dst_pod:
                level = 'Pod内'
            elif src_sp == dst_sp:
                level = '超节点内'
            else:
                level = '跨超节点'

            style = level_styles[level]
            edge_key = (min(src_rank, dst_rank), max(src_rank, dst_rank), level)
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)

            x1, y1 = _svg_rank_pos(src_rank)
            x2, y2 = _svg_rank_pos(dst_rank)

            types_str = ", ".join(sorted(info['types']))
            slices_str = ", ".join(f"[{s[0]}/{s[1]}]" for s in sorted(info['slices']))
            label = f"{types_str} {slices_str}"

            _svg_draw_arrow(svg_parts, x1, y1, x2, y2,
                            style['color'], style['width'], style['alpha'],
                            style['dash'], label if level != 'Pod内' else None)

    legend_y = total_h - 40
    svg_parts.append(f'<rect x="{PAD_X}" y="{legend_y - 15}" width="600" height="30" rx="5" fill="white" stroke="#ddd" opacity="0.9"/>')
    svg_parts.append(f'<line x1="{PAD_X + 10}" y1="{legend_y}" x2="{PAD_X + 40}" y2="{legend_y}" stroke="#4CAF50" stroke-width="2"/>')
    svg_parts.append(f'<text x="{PAD_X + 45}" y="{legend_y + 4}" font-size="11" fill="#333">Intra-Pod (SendRecvWrite)</text>')
    svg_parts.append(f'<line x1="{PAD_X + 210}" y1="{legend_y}" x2="{PAD_X + 240}" y2="{legend_y}" stroke="#FF9800" stroke-width="2" stroke-dasharray="8,4"/>')
    svg_parts.append(f'<text x="{PAD_X + 245}" y="{legend_y + 4}" font-size="11" fill="#333">Inter-Pod (SendRecvWriteReduce)</text>')
    svg_parts.append(f'<line x1="{PAD_X + 450}" y1="{legend_y}" x2="{PAD_X + 480}" y2="{legend_y}" stroke="#F44336" stroke-width="2.5" stroke-dasharray="4,4"/>')
    svg_parts.append(f'<text x="{PAD_X + 485}" y="{legend_y + 4}" font-size="11" fill="#333">Inter-SuperPod (SendRecvWriteReduce)</text>')

    svg_parts.append('</svg>')
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_parts))
    print(f"SVG 数据流图(总览)已保存到: {output_path}")


def generate_svg_intra_pod(data_trans_ops, output_path):
    cross_rank_ops = [
        op for op in data_trans_ops
        if op['op_type'] in ('SendRecvWrite', 'SendRecvWriteReduce')
        and op['src_rank'] >= 0 and op['dst_rank'] >= 0
        and op['src_rank'] != op['dst_rank']
    ]

    pod_ops = defaultdict(list)
    for op in cross_rank_ops:
        src_pod = get_pod(op['src_rank'])
        dst_pod = get_pod(op['dst_rank'])
        if src_pod == dst_pod:
            pod_ops[src_pod].append(op)

    POD_W = 680
    POD_H = 620
    POD_PAD_X = 40
    POD_PAD_Y = 80
    POD_GAP_X = 40
    POD_GAP_Y = 40
    PODS_PER_ROW = 2

    total_rows = (len(PODS) + PODS_PER_ROW - 1) // PODS_PER_ROW
    total_w = POD_PAD_X * 2 + PODS_PER_ROW * POD_W + (PODS_PER_ROW - 1) * POD_GAP_X
    total_h = POD_PAD_Y + total_rows * (POD_H + POD_GAP_Y) + 20

    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">')
    svg_parts.append(f'<rect width="{total_w}" height="{total_h}" fill="white"/>')
    svg_parts.append(f'<text x="{total_w//2}" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#4CAF50">Level 0: Intra-Pod Data Flow</text>')
    svg_parts.append(f'<text x="{total_w//2}" y="55" text-anchor="middle" font-size="12" fill="#666">每个Pod独立展示 | 箭头=数据写入方向 | 标签=Slice[Idx/Num]</text>')

    def pod_origin(pod_id):
        col = pod_id % PODS_PER_ROW
        row = pod_id // PODS_PER_ROW
        x = POD_PAD_X + col * (POD_W + POD_GAP_X)
        y = POD_PAD_Y + row * (POD_H + POD_GAP_Y)
        return x, y

    for pod_id in sorted(PODS.keys()):
        pod_ranks = PODS[pod_id]
        sp_id = get_super_pod(pod_id)
        ox, oy = pod_origin(pod_id)
        n = len(pod_ranks)

        svg_parts.append(f'<rect x="{ox}" y="{oy}" width="{POD_W}" height="{POD_H}" rx="12" fill="#F1F8E9" stroke="#4CAF50" stroke-width="2"/>')
        svg_parts.append(f'<text x="{ox + POD_W//2}" y="{oy + 22}" text-anchor="middle" font-size="14" font-weight="bold" fill="#2E7D32">Pod {pod_id} (SuperPod {sp_id})</text>')

        cx_center = ox + 190
        cy_center = oy + 340
        radius = 130

        rank_angles = {}
        rank_centers = {}
        for i, rank in enumerate(pod_ranks):
            angle = -90 + i * (360 / n)
            rad = angle * 3.14159265 / 180
            rx = cx_center + radius * __import__('math').cos(rad)
            ry = cy_center + radius * __import__('math').sin(rad)
            rank_angles[rank] = angle
            rank_centers[rank] = (rx, ry)

        for rank in pod_ranks:
            rx, ry = rank_centers[rank]
            npu = RANK_TO_NPU[rank]
            svg_parts.append(f'<circle cx="{rx:.1f}" cy="{ry:.1f}" r="32" fill="#42A5F5" stroke="#1565C0" stroke-width="2.5"/>')
            svg_parts.append(f'<text x="{rx:.1f}" y="{ry - 6:.1f}" text-anchor="middle" font-size="13" font-weight="bold" fill="white">R{rank}</text>')
            svg_parts.append(f'<text x="{rx:.1f}" y="{ry + 10:.1f}" text-anchor="middle" font-size="8" fill="white">NPU{npu}</text>')

        pod_flow = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for op in pod_ops.get(pod_id, []):
            pod_flow[op['src_rank']][op['dst_rank']][(op['slice_idx'], op['slice_num'])] += 1

        drawn = set()
        for src_rank in sorted(pod_flow.keys()):
            for dst_rank in sorted(pod_flow[src_rank].keys()):
                edge_key = (min(src_rank, dst_rank), max(src_rank, dst_rank))
                if edge_key in drawn:
                    continue
                drawn.add(edge_key)

                x1, y1 = rank_centers[src_rank]
                x2, y2 = rank_centers[dst_rank]

                dx = x2 - x1
                dy = y2 - y1
                dist = (dx**2 + dy**2)**0.5
                if dist == 0:
                    continue

                ndx = dx / dist
                ndy = dy / dist
                nx = -ndy
                ny = ndx

                r = 34
                sx = x1 + ndx * r
                sy = y1 + ndy * r
                ex = x2 - ndx * r
                ey = y2 - ndy * r

                ctrl_offset = 35
                cpx = (sx + ex) / 2 + nx * ctrl_offset
                cpy = (sy + ey) / 2 + ny * ctrl_offset

                path_d = f"M {sx:.1f} {sy:.1f} Q {cpx:.1f} {cpy:.1f} {ex:.1f} {ey:.1f}"
                svg_parts.append(f'<path d="{path_d}" fill="none" stroke="#4CAF50" stroke-width="2" opacity="0.8"/>')

                t = 0.92
                t1 = 1 - t
                px = t1*t1*sx + 2*t1*t*cpx + t*t*ex
                py = t1*t1*sy + 2*t1*t*cpy + t*t*ey
                adx = ex - px
                ady = ey - py
                adist = (adx**2 + ady**2)**0.5
                if adist > 0:
                    adx /= adist
                    ady /= adist
                    al = 9
                    a1x = ex - al * adx + al * 0.4 * (-ady)
                    a1y = ey - al * ady + al * 0.4 * adx
                    a2x = ex - al * adx - al * 0.4 * (-ady)
                    a2y = ey - al * ady - al * 0.4 * adx
                    svg_parts.append(f'<polygon points="{ex:.1f},{ey:.1f} {a1x:.1f},{a1y:.1f} {a2x:.1f},{a2y:.1f}" fill="#4CAF50" opacity="0.8"/>')

                fwd_slices = sorted(pod_flow[src_rank][dst_rank].keys())
                fwd_str = ",".join(f"[{s[0]}/{s[1]}]" for s in fwd_slices)
                mid_x = (sx + ex) / 2 + nx * 14
                mid_y = (sy + ey) / 2 + ny * 14
                svg_parts.append(f'<text x="{mid_x:.1f}" y="{mid_y:.1f}" text-anchor="middle" font-size="9" fill="#1B5E20" font-weight="bold">R{src_rank}→R{dst_rank} {fwd_str}</text>')

                rev_slices = sorted(pod_flow[dst_rank].get(src_rank, {}).keys())
                if rev_slices:
                    rev_str = ",".join(f"[{s[0]}/{s[1]}]" for s in rev_slices)
                    rev_mid_x = (sx + ex) / 2 - nx * 14
                    rev_mid_y = (sy + ey) / 2 - ny * 14
                    svg_parts.append(f'<text x="{rev_mid_x:.1f}" y="{rev_mid_y:.1f}" text-anchor="middle" font-size="9" fill="#66BB6A">R{dst_rank}→R{src_rank} {rev_str}</text>')

        table_x = ox + 400
        table_y = oy + 50
        cell_w = 55
        cell_h = 28
        svg_parts.append(f'<text x="{table_x + n * cell_w // 2}" y="{table_y - 5}" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">数据流矩阵 (src→dst)</text>')

        svg_parts.append(f'<rect x="{table_x - 1}" y="{table_y - 1}" width="{(n + 1) * cell_w + 2}" height="{(n + 1) * cell_h + 2}" fill="none" stroke="#999" stroke-width="1"/>')
        svg_parts.append(f'<rect x="{table_x}" y="{table_y}" width="{cell_w}" height="{cell_h}" fill="#E0E0E0"/>')
        svg_parts.append(f'<text x="{table_x + cell_w // 2}" y="{table_y + cell_h // 2 + 4}" text-anchor="middle" font-size="9" fill="#666">src\\dst</text>')

        for j, dst in enumerate(pod_ranks):
            cx = table_x + (j + 1) * cell_w
            svg_parts.append(f'<rect x="{cx}" y="{table_y}" width="{cell_w}" height="{cell_h}" fill="#E3F2FD"/>')
            svg_parts.append(f'<text x="{cx + cell_w // 2}" y="{table_y + cell_h // 2 + 4}" text-anchor="middle" font-size="10" font-weight="bold" fill="#1565C0">R{dst}</text>')

        for i, src in enumerate(pod_ranks):
            cy = table_y + (i + 1) * cell_h
            svg_parts.append(f'<rect x="{table_x}" y="{cy}" width="{cell_w}" height="{cell_h}" fill="#E3F2FD"/>')
            svg_parts.append(f'<text x="{table_x + cell_w // 2}" y="{cy + cell_h // 2 + 4}" text-anchor="middle" font-size="10" font-weight="bold" fill="#1565C0">R{src}</text>')

            for j, dst in enumerate(pod_ranks):
                cx = table_x + (j + 1) * cell_w
                ry = cy
                if src == dst:
                    svg_parts.append(f'<rect x="{cx}" y="{ry}" width="{cell_w}" height="{cell_h}" fill="#F5F5F5"/>')
                    svg_parts.append(f'<text x="{cx + cell_w // 2}" y="{ry + cell_h // 2 + 3}" text-anchor="middle" font-size="8" fill="#999">-</text>')
                else:
                    slices = sorted(pod_flow[src][dst].keys())
                    if slices:
                        svg_parts.append(f'<rect x="{cx}" y="{ry}" width="{cell_w}" height="{cell_h}" fill="#C8E6C9"/>')
                        s_str = ",".join(f"{s[0]}/{s[1]}" for s in slices)
                        svg_parts.append(f'<text x="{cx + cell_w // 2}" y="{ry + cell_h // 2 + 3}" text-anchor="middle" font-size="8" fill="#1B5E20" font-weight="bold">{s_str}</text>')
                    else:
                        svg_parts.append(f'<rect x="{cx}" y="{ry}" width="{cell_w}" height="{cell_h}" fill="white"/>')

        for i in range(n + 2):
            ly = table_y + i * cell_h
            svg_parts.append(f'<line x1="{table_x}" y1="{ly}" x2="{table_x + (n + 1) * cell_w}" y2="{ly}" stroke="#CCC" stroke-width="0.5"/>')
        for j in range(n + 2):
            lx = table_x + j * cell_w
            svg_parts.append(f'<line x1="{lx}" y1="{table_y}" x2="{lx}" y2="{table_y + (n + 1) * cell_h}" stroke="#CCC" stroke-width="0.5"/>')

    svg_parts.append('</svg>')
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_parts))
    print(f"SVG 数据流图(Pod内-展开)已保存到: {output_path}")


def generate_svg_step_by_step(data_trans_ops, step_infos, output_path):
    import math

    cross_rank_ops = [
        op for op in data_trans_ops
        if op['op_type'] in ('SendRecvWrite', 'SendRecvWriteReduce')
        and op['src_rank'] >= 0 and op['dst_rank'] >= 0
        and op['src_rank'] != op['dst_rank']
    ]

    intra_ops = []
    inter_pod_ops = []
    inter_sp_ops = []
    for op in cross_rank_ops:
        src_pod = get_pod(op['src_rank'])
        dst_pod = get_pod(op['dst_rank'])
        src_sp = get_super_pod(op['src_rank'])
        dst_sp = get_super_pod(op['dst_rank'])
        if src_pod == dst_pod:
            intra_ops.append(op)
        elif src_sp == dst_sp:
            inter_pod_ops.append(op)
        else:
            inter_sp_ops.append(op)

    def compute_nhr_steps(rank_size):
        return int(math.log2(rank_size)) if rank_size > 1 else 0

    def compute_nhr_step_info(rank_size):
        steps = []
        n_steps = compute_nhr_steps(rank_size)
        for step in range(n_steps):
            delta = 1 << step
            step_data = {}
            for x in range(rank_size):
                send_to = (x + rank_size - delta) % rank_size
                recv_from = (x + delta) % rank_size
                tx_slice_idx = send_to
                rx_slice_idx = x
                n_slices = (rank_size - 1 + (1 << step)) // (1 << (step + 1))
                delta_slice = 1 << (step + 1)
                tx_slices = []
                rx_slices = []
                tx_idx = tx_slice_idx
                rx_idx = rx_slice_idx
                for _ in range(n_slices):
                    tx_slices.append(tx_idx)
                    rx_slices.append(rx_idx)
                    tx_idx = (tx_idx + rank_size - delta_slice) % rank_size
                    rx_idx = (rx_idx + rank_size - delta_slice) % rank_size
                step_data[x] = {
                    'send_to': send_to,
                    'recv_from': recv_from,
                    'tx_slices': tx_slices,
                    'rx_slices': rx_slices,
                }
            steps.append(step_data)
        return steps

    intra_nhr_steps = compute_nhr_step_info(POD_SIZE)
    inter_pod_nhr_steps = compute_nhr_step_info(2)
    inter_sp_nhr_steps = compute_nhr_step_info(2)

    def classify_op_by_step(ops, nhr_steps, group_fn):
        step_ops = defaultdict(list)
        other_ops = []
        for op in ops:
            src_group = group_fn(op['src_rank'])
            dst_group = group_fn(op['dst_rank'])
            local_idx = group_fn(op['src_rank'], local=True)
            matched = False
            for step_idx, step_data in enumerate(nhr_steps):
                if local_idx in step_data:
                    info = step_data[local_idx]
                    if info['send_to'] == dst_group:
                        step_ops[step_idx].append(op)
                        matched = True
                        break
            if not matched:
                other_ops.append(op)
        return step_ops, other_ops

    def pod_local_idx(rank):
        pod = get_pod(rank)
        return PODS[pod].index(rank)

    def sp_local_pod_idx(rank):
        sp = get_super_pod(rank)
        pod = get_pod(rank)
        return sp * 2 + pod

    intra_step_ops = defaultdict(list)
    for op in intra_ops:
        src_local = pod_local_idx(op['src_rank'])
        dst_local = pod_local_idx(op['dst_rank'])
        for step_idx, step_data in enumerate(intra_nhr_steps):
            if src_local in step_data:
                info = step_data[src_local]
                if info['send_to'] == dst_local:
                    intra_step_ops[step_idx].append(op)
                    break

    inter_pod_step_ops = defaultdict(list)
    for op in inter_pod_ops:
        src_pod = get_pod(op['src_rank'])
        dst_pod = get_pod(op['dst_rank'])
        sp = get_super_pod(op['src_rank'])
        pod_local_in_sp = src_pod - sp * 2
        dst_local_in_sp = dst_pod - sp * 2
        for step_idx, step_data in enumerate(inter_pod_nhr_steps):
            if pod_local_in_sp in step_data:
                info = step_data[pod_local_in_sp]
                if info['send_to'] == dst_local_in_sp:
                    inter_pod_step_ops[step_idx].append(op)
                    break

    inter_sp_step_ops = defaultdict(list)
    for op in inter_sp_ops:
        src_sp = get_super_pod(op['src_rank'])
        dst_sp = get_super_pod(op['dst_rank'])
        for step_idx, step_data in enumerate(inter_sp_nhr_steps):
            if src_sp in step_data:
                info = step_data[src_sp]
                if info['send_to'] == dst_sp:
                    inter_sp_step_ops[step_idx].append(op)
                    break

    CELL_W = 130
    CELL_H = 90
    PAD_X = 60
    PAD_Y = 100
    POD_GAP_X = 100
    SP_GAP_Y = 120

    def rank_pos(rank):
        pod_id = get_pod(rank)
        sp_id = get_super_pod(rank)
        pod_local_idx = PODS[pod_id].index(rank)
        col_in_pod = pod_local_idx % 4
        pod_col = pod_id % 2
        pod_row = pod_id // 2
        x = PAD_X + pod_col * (4 * CELL_W + POD_GAP_X) + col_in_pod * CELL_W + CELL_W // 2
        y = PAD_Y + pod_row * (1 * CELL_H + SP_GAP_Y) + CELL_H // 2 + 40
        return x, y

    total_w = PAD_X * 2 + 2 * (4 * CELL_W) + POD_GAP_X
    total_h = PAD_Y * 2 + 2 * (1 * CELL_H) + SP_GAP_Y + 140

    n_intra_steps = compute_nhr_steps(POD_SIZE)
    n_inter_pod_steps = compute_nhr_steps(2)
    n_inter_sp_steps = compute_nhr_steps(2)
    total_steps = n_intra_steps + n_inter_pod_steps + n_inter_sp_steps

    phase_names = []
    for i in range(n_intra_steps):
        phase_names.append(f"Level0-IntraPod Step{i}")
    for i in range(n_inter_pod_steps):
        phase_names.append(f"Level1-InterPod Step{i}")
    for i in range(n_inter_sp_steps):
        phase_names.append(f"Level2-InterSP Step{i}")

    all_step_ops = []
    for i in range(n_intra_steps):
        all_step_ops.append(intra_step_ops.get(i, []))
    for i in range(n_inter_pod_steps):
        all_step_ops.append(inter_pod_step_ops.get(i, []))
    for i in range(n_inter_sp_steps):
        all_step_ops.append(inter_sp_step_ops.get(i, []))

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">')
    svg.append('<style>')
    svg.append('  .rank-circle { transition: fill 0.3s, stroke 0.3s; }')
    svg.append('  .rank-label { pointer-events: none; }')
    svg.append('  .flow-arrow { transition: opacity 0.3s; }')
    svg.append('  .step-btn { cursor: pointer; }')
    svg.append('  .step-btn:hover rect { fill: #E3F2FD; }')
    svg.append('  .info-text { font-family: monospace; }')
    svg.append('</style>')

    svg.append(f'<rect width="{total_w}" height="{total_h}" fill="#FAFAFA"/>')

    svg.append(f'<text x="{total_w//2}" y="28" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">ReduceScatter 三级NHR数据流 - 逐步展示</text>')
    svg.append(f'<text x="{total_w//2}" y="48" text-anchor="middle" font-size="12" fill="#666">InsReduceScatterSequenceMesh1DNHRNHR | 2×2×4 Mesh | 点击步骤按钮切换</text>')

    sp_colors = ['#E8F5E9', '#E3F2FD']
    pod_colors = [['#C8E6C9', '#A5D6A7'], ['#BBDEFB', '#90CAF9']]

    for sp_id in sorted(SUPER_PODS.keys()):
        sp_ranks = SUPER_PODS[sp_id]
        xs = [rank_pos(r)[0] for r in sp_ranks]
        ys = [rank_pos(r)[1] for r in sp_ranks]
        rx = min(xs) - CELL_W // 2 - 15
        ry = min(ys) - CELL_H // 2 - 25
        rw = max(xs) - min(xs) + CELL_W + 30
        rh = max(ys) - min(ys) + CELL_H + 40
        svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="12" fill="{sp_colors[sp_id]}" stroke="#999" stroke-width="2" opacity="0.5"/>')
        svg.append(f'<text x="{rx + 10}" y="{ry + 18}" font-size="14" font-weight="bold" fill="#777">SuperPod {sp_id}</text>')

    for pod_id in sorted(PODS.keys()):
        pod_ranks = PODS[pod_id]
        sp_id = get_super_pod(pod_id)
        pod_local = pod_id if sp_id == 0 else pod_id - 2
        xs = [rank_pos(r)[0] for r in pod_ranks]
        ys = [rank_pos(r)[1] for r in pod_ranks]
        rx = min(xs) - CELL_W // 2 - 8
        ry = min(ys) - CELL_H // 2 - 8
        rw = max(xs) - min(xs) + CELL_W + 16
        rh = max(ys) - min(ys) + CELL_H + 16
        svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="8" fill="{pod_colors[sp_id][pod_local % 2]}" stroke="#aaa" stroke-width="1.5" opacity="0.7"/>')
        svg.append(f'<text x="{rx + 8}" y="{ry + 16}" font-size="11" fill="#888">Pod {pod_id}</text>')

    for rank in range(RANK_SIZE):
        cx, cy = rank_pos(rank)
        npu = RANK_TO_NPU[rank]
        svg.append(f'<circle class="rank-circle" id="rank_{rank}" cx="{cx}" cy="{cy}" r="32" fill="#42A5F5" stroke="#1565C0" stroke-width="2.5"/>')
        svg.append(f'<text class="rank-label" x="{cx}" y="{cy - 6}" text-anchor="middle" font-size="13" font-weight="bold" fill="white">R{rank}</text>')
        svg.append(f'<text class="rank-label" x="{cx}" y="{cy + 10}" text-anchor="middle" font-size="8" fill="white">NPU{npu}</text>')

    step_colors = ['#4CAF50', '#66BB6A', '#FF9800', '#F44336']
    step_bg_colors = ['#E8F5E9', '#F1F8E9', '#FFF3E0', '#FFEBEE']

    for step_idx in range(total_steps):
        ops = all_step_ops[step_idx]
        color = step_colors[step_idx % len(step_colors)]
        bg = step_bg_colors[step_idx % len(step_bg_colors)]

        for op in ops:
            x1, y1 = rank_pos(op['src_rank'])
            x2, y2 = rank_pos(op['dst_rank'])
            dx = x2 - x1
            dy = y2 - y1
            dist = (dx**2 + dy**2)**0.5
            if dist == 0:
                continue

            ndx = dx / dist
            ndy = dy / dist
            nx_dir = -ndy
            ny_dir = ndx

            r = 34
            sx = x1 + ndx * r
            sy = y1 + ndy * r
            ex = x2 - ndx * r
            ey = y2 - ndy * r

            src_pod = get_pod(op['src_rank'])
            dst_pod = get_pod(op['dst_rank'])
            ctrl_offset = 30 if src_pod == dst_pod else 50
            cpx = (sx + ex) / 2 + nx_dir * ctrl_offset
            cpy = (sy + ey) / 2 + ny_dir * ctrl_offset

            path_d = f"M {sx:.1f} {sy:.1f} Q {cpx:.1f} {cpy:.1f} {ex:.1f} {ey:.1f}"

            t = 0.92
            t1 = 1 - t
            px = t1*t1*sx + 2*t1*t*cpx + t*t*ex
            py = t1*t1*sy + 2*t1*t*cpy + t*t*ey
            adx = ex - px
            ady = ey - py
            adist = (adx**2 + ady**2)**0.5
            arrow_pts = ""
            if adist > 0:
                adx_n = adx / adist
                ady_n = ady / adist
                al = 10
                a1x = ex - al * adx_n + al * 0.4 * (-ady_n)
                a1y = ey - al * ady_n + al * 0.4 * adx_n
                a2x = ex - al * adx_n - al * 0.4 * (-ady_n)
                a2y = ey - al * ady_n - al * 0.4 * adx_n
                arrow_pts = f'{ex:.1f},{ey:.1f} {a1x:.1f},{a1y:.1f} {a2x:.1f},{a2y:.1f}'

            slice_label = f"S[{op['slice_idx']}/{op['slice_num']}]"
            op_label = "SRW" if op['op_type'] == 'SendRecvWrite' else "SRWR"
            label = f"{op_label} {slice_label}"

            mid_x = (sx + ex) / 2 + nx_dir * 18
            mid_y = (sy + ey) / 2 + ny_dir * 18

            svg.append(f'<g class="flow-arrow" id="arrow_{step_idx}_{op["src_rank"]}_{op["dst_rank"]}_{op["slice_idx"]}" style="display:none">')
            svg.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.5" opacity="0.85"/>')
            if arrow_pts:
                svg.append(f'<polygon points="{arrow_pts}" fill="{color}" opacity="0.85"/>')
            svg.append(f'<text x="{mid_x:.1f}" y="{mid_y:.1f}" text-anchor="middle" font-size="9" fill="{color}" font-weight="bold" paint-order="stroke" stroke="white" stroke-width="3">{label}</text>')
            svg.append('</g>')

    btn_y = total_h - 80
    btn_w = 160
    btn_h = 30
    btn_gap = 10
    total_btn_w = total_steps * btn_w + (total_steps - 1) * btn_gap
    btn_start_x = (total_w - total_btn_w) // 2

    svg.append(f'<rect x="{btn_start_x - 20}" y="{btn_y - 30}" width="{total_btn_w + 40}" height="{90}" rx="8" fill="white" stroke="#ddd" opacity="0.95"/>')
    svg.append(f'<text x="{total_w//2}" y="{btn_y - 10}" text-anchor="middle" font-size="11" fill="#666">点击切换步骤 (当前步骤高亮显示)</text>')

    for step_idx in range(total_steps):
        bx = btn_start_x + step_idx * (btn_w + btn_gap)
        color = step_colors[step_idx % len(step_colors)]
        bg = step_bg_colors[step_idx % len(step_bg_colors)]
        svg.append(f'<g class="step-btn" id="btn_{step_idx}" onclick="showStep({step_idx})">')
        svg.append(f'<rect x="{bx}" y="{btn_y}" width="{btn_w}" height="{btn_h}" rx="6" fill="{bg}" stroke="{color}" stroke-width="1.5" id="btn_rect_{step_idx}"/>')
        svg.append(f'<text x="{bx + btn_w//2}" y="{btn_y + btn_h//2 + 4}" text-anchor="middle" font-size="10" font-weight="bold" fill="{color}">{phase_names[step_idx]}</text>')
        svg.append('</g>')

    info_y = btn_y + btn_h + 12
    svg.append(f'<text id="step_info" x="{total_w//2}" y="{info_y}" text-anchor="middle" font-size="11" fill="#333" class="info-text">点击上方按钮查看各步骤数据流</text>')

    svg.append('<script type="text/javascript"><![CDATA[')
    svg.append('var totalSteps = ' + str(total_steps) + ';')
    svg.append('var currentStep = -1;')

    svg.append('function showStep(stepIdx) {')

    svg.append('  for (var s = 0; s < totalSteps; s++) {')
    svg.append('    var arrows = document.querySelectorAll(\'[id^="arrow_\' + s + \'_"]\');')
    svg.append('    for (var i = 0; i < arrows.length; i++) {')
    svg.append('      arrows[i].style.display = (s === stepIdx) ? "" : "none";')
    svg.append('    }')
    svg.append('    var btnRect = document.getElementById("btn_rect_" + s);')
    svg.append('    if (s === stepIdx) {')
    svg.append('      btnRect.setAttribute("stroke-width", "3");')
    svg.append('      btnRect.setAttribute("fill", btnRect.getAttribute("fill").replace("E8F5E9", "A5D6A7").replace("F1F8E9", "C5E1A5").replace("FFF3E0", "FFE0B2").replace("FFEBEE", "FFCDD2"));')
    svg.append('    } else {')
    svg.append('      btnRect.setAttribute("stroke-width", "1.5");')
    svg.append('      var fills = ["#E8F5E9", "#F1F8E9", "#FFF3E0", "#FFEBEE"];')
    svg.append('      btnRect.setAttribute("fill", fills[s % 4]);')
    svg.append('    }')
    svg.append('  }')

    svg.append('  for (var r = 0; r < ' + str(RANK_SIZE) + '; r++) {')
    svg.append('    var circle = document.getElementById("rank_" + r);')
    svg.append('    circle.setAttribute("fill", "#42A5F5");')
    svg.append('    circle.setAttribute("stroke", "#1565C0");')
    svg.append('    circle.setAttribute("stroke-width", "2.5");')
    svg.append('  }')

    svg.append('  var stepOps = ' + json.dumps([[{"src": op["src_rank"], "dst": op["dst_rank"], "slice": f"[{op['slice_idx']}/{op['slice_num']}]", "op": op["op_type"]} for op in all_step_ops[s]] for s in range(total_steps)]) + ';')

    svg.append('  if (stepIdx >= 0 && stepIdx < stepOps.length) {')
    svg.append('    var activeRanks = new Set();')
    svg.append('    var desc = [];')
    svg.append('    for (var i = 0; i < stepOps[stepIdx].length; i++) {')
    svg.append('      var op = stepOps[stepIdx][i];')
    svg.append('      activeRanks.add(op.src);')
    svg.append('      activeRanks.add(op.dst);')
    svg.append('      desc.push("R" + op.src + "→R" + op.dst + " " + op.slice + " " + op.op);')
    svg.append('    }')
    svg.append('    activeRanks.forEach(function(r) {')
    svg.append('      var circle = document.getElementById("rank_" + r);')
    svg.append('      circle.setAttribute("fill", "#FF7043");')
    svg.append('      circle.setAttribute("stroke", "#BF360C");')
    svg.append('      circle.setAttribute("stroke-width", "3.5");')
    svg.append('    });')
    svg.append('    var infoEl = document.getElementById("step_info");')
    svg.append('    var infoText = desc.length > 0 ? desc.slice(0, 8).join(" | ") : "无跨Rank数据传输";')
    svg.append('    if (desc.length > 8) infoText += " ... 共" + desc.length + "条";')
    svg.append('    infoEl.textContent = infoText;')
    svg.append('  }')
    svg.append('  currentStep = stepIdx;')
    svg.append('}')

    svg.append('var autoPlay = false;')
    svg.append('var autoTimer = null;')
    svg.append('function toggleAutoPlay() {')
    svg.append('  autoPlay = !autoPlay;')
    svg.append('  var btn = document.getElementById("auto_btn_text");')
    svg.append('  if (autoPlay) {')
    svg.append('    btn.textContent = "⏸ 暂停";')
    svg.append('    var next = (currentStep + 1) % totalSteps;')
    svg.append('    showStep(next);')
    svg.append('    autoTimer = setInterval(function() {')
    svg.append('      var next = (currentStep + 1) % totalSteps;')
    svg.append('      showStep(next);')
    svg.append('    }, 2000);')
    svg.append('  } else {')
    svg.append('    btn.textContent = "▶ 自动播放";')
    svg.append('    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }')
    svg.append('  }')
    svg.append('}')

    svg.append(']]></script>')

    auto_btn_x = btn_start_x + total_btn_w + 15
    svg.append(f'<g class="step-btn" onclick="toggleAutoPlay()">')
    svg.append(f'<rect x="{auto_btn_x}" y="{btn_y}" width="90" height="{btn_h}" rx="6" fill="#E8EAF6" stroke="#3F51B5" stroke-width="1.5"/>')
    svg.append(f'<text id="auto_btn_text" x="{auto_btn_x + 45}" y="{btn_y + btn_h//2 + 4}" text-anchor="middle" font-size="10" font-weight="bold" fill="#3F51B5">▶ 自动播放</text>')
    svg.append('</g>')

    svg.append('</svg>')

    with open(output_path, 'w') as f:
        f.write('\n'.join(svg))
    print(f"SVG 逐步数据流图已保存到: {output_path}")


def generate_svg_step_detail(data_trans_ops, step_infos, output_path):
    import math

    cross_rank_ops = [
        op for op in data_trans_ops
        if op['op_type'] in ('SendRecvWrite', 'SendRecvWriteReduce')
        and op['src_rank'] >= 0 and op['dst_rank'] >= 0
        and op['src_rank'] != op['dst_rank']
    ]

    def compute_nhr_step_info(rank_size):
        steps = []
        n_steps = int(math.log2(rank_size)) if rank_size > 1 else 0
        for step in range(n_steps):
            delta = 1 << step
            step_data = {}
            for x in range(rank_size):
                send_to = (x + rank_size - delta) % rank_size
                recv_from = (x + delta) % rank_size
                tx_slice_idx = send_to
                rx_slice_idx = x
                n_slices = (rank_size - 1 + (1 << step)) // (1 << (step + 1))
                delta_slice = 1 << (step + 1)
                tx_slices = []
                rx_slices = []
                tx_idx = tx_slice_idx
                rx_idx = rx_slice_idx
                for _ in range(n_slices):
                    tx_slices.append(tx_idx)
                    rx_slices.append(rx_idx)
                    tx_idx = (tx_idx + rank_size - delta_slice) % rank_size
                    rx_idx = (rx_idx + rank_size - delta_slice) % rank_size
                step_data[x] = {
                    'send_to': send_to,
                    'recv_from': recv_from,
                    'tx_slices': tx_slices,
                    'rx_slices': rx_slices,
                }
            steps.append(step_data)
        return steps

    intra_nhr_steps = compute_nhr_step_info(POD_SIZE)
    inter_pod_nhr_steps = compute_nhr_step_info(2)
    inter_sp_nhr_steps = compute_nhr_step_info(2)

    n_intra = len(intra_nhr_steps)
    n_inter_pod = len(inter_pod_nhr_steps)
    n_inter_sp = len(inter_sp_nhr_steps)
    total_phases = n_intra + n_inter_pod + n_inter_sp

    phase_configs = []
    for i in range(n_intra):
        phase_configs.append({
            'name': f'Level0 Intra-Pod Step {i}',
            'short': f'L0-S{i}',
            'nhr_steps': intra_nhr_steps,
            'rank_size': POD_SIZE,
            'color': '#4CAF50',
            'bg': '#E8F5E9',
            'level': 'intra',
        })
    for i in range(n_inter_pod):
        phase_configs.append({
            'name': f'Level1 Inter-Pod Step {i}',
            'short': f'L1-S{i}',
            'nhr_steps': inter_pod_nhr_steps,
            'rank_size': 2,
            'color': '#FF9800',
            'bg': '#FFF3E0',
            'level': 'inter_pod',
        })
    for i in range(n_inter_sp):
        phase_configs.append({
            'name': f'Level2 Inter-SuperPod Step {i}',
            'short': f'L2-S{i}',
            'nhr_steps': inter_sp_nhr_steps,
            'rank_size': 2,
            'color': '#F44336',
            'bg': '#FFEBEE',
            'level': 'inter_sp',
        })

    PAGE_W = 1400
    PAGE_H = 900
    RANK_R = 28

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="{PAGE_H}" viewBox="0 0 {PAGE_W} {PAGE_H}">')
    svg.append('<style>')
    svg.append('  .rank-node { cursor: pointer; }')
    svg.append('  .rank-node:hover circle { stroke-width: 4; }')
    svg.append('  .flow-line { transition: opacity 0.3s; }')
    svg.append('</style>')

    svg.append(f'<rect width="{PAGE_W}" height="{PAGE_H}" fill="#FAFAFA"/>')

    svg.append(f'<text x="{PAGE_W//2}" y="28" text-anchor="middle" font-size="20" font-weight="bold" fill="#333">ReduceScatter 三级NHR数据流 - 详细步骤视图</text>')
    svg.append(f'<text x="{PAGE_W//2}" y="50" text-anchor="middle" font-size="12" fill="#666">每页展示一个步骤的所有Rank通信关系 | 包含NHR算法的发送/接收切片详情</text>')

    nav_y = 62
    nav_btn_w = 100
    nav_gap = 6
    total_nav_w = total_phases * nav_btn_w + (total_phases - 1) * nav_gap
    nav_start_x = (PAGE_W - total_nav_w) // 2

    svg.append(f'<rect x="{nav_start_x - 10}" y="{nav_y - 5}" width="{total_nav_w + 20}" height="32" rx="6" fill="white" stroke="#ddd"/>')
    for pi, cfg in enumerate(phase_configs):
        bx = nav_start_x + pi * (nav_btn_w + nav_gap)
        svg.append(f'<g class="step-btn" onclick="showPhase({pi})">')
        svg.append(f'<rect id="nav_rect_{pi}" x="{bx}" y="{nav_y}" width="{nav_btn_w}" height="22" rx="4" fill="{cfg["bg"]}" stroke="{cfg["color"]}" stroke-width="1.5"/>')
        svg.append(f'<text x="{bx + nav_btn_w//2}" y="{nav_y + 15}" text-anchor="middle" font-size="10" font-weight="bold" fill="{cfg["color"]}">{cfg["short"]}</text>')
        svg.append('</g>')

    main_y = 105
    main_h = 500
    svg.append(f'<rect x="20" y="{main_y}" width="{PAGE_W - 40}" height="{main_h}" rx="10" fill="white" stroke="#ddd"/>')

    def draw_phase_ranks(svg, phase_idx, offset_x, offset_y, area_w, area_h):
        cfg = phase_configs[phase_idx]
        nhr_steps = cfg['nhr_steps']
        step_idx_in_level = phase_idx - ([0] * n_intra + [n_intra] * n_inter_pod + [n_intra + n_inter_pod] * n_inter_sp)[phase_idx]
        if phase_idx >= n_intra + n_inter_pod:
            step_idx_in_level = phase_idx - n_intra - n_inter_pod
        elif phase_idx >= n_intra:
            step_idx_in_level = phase_idx - n_intra

        if cfg['level'] == 'intra':
            sub_groups = PODS
            group_label_fn = lambda pid: f"Pod {pid}"
        elif cfg['level'] == 'inter_pod':
            sub_groups = SUPER_PODS
            group_label_fn = lambda sid: f"SuperPod {sid}"
        else:
            sub_groups = {0: list(range(RANK_SIZE))}
            group_label_fn = lambda gid: "All Ranks"

        n_groups = len(sub_groups)
        group_w = area_w // n_groups

        rank_positions = {}
        for gi, (gid, ranks) in enumerate(sorted(sub_groups.items())):
            gx = offset_x + gi * group_w + group_w // 2
            n_ranks = len(ranks)
            radius = min(group_w // 2 - 50, area_h // 2 - 50, 140)

            svg.append(f'<rect x="{gx - group_w//2 + 10}" y="{offset_y}" width="{group_w - 20}" height="{area_h}" rx="8" fill="{cfg["bg"]}" opacity="0.3"/>')
            svg.append(f'<text x="{gx}" y="{offset_y + 18}" text-anchor="middle" font-size="12" font-weight="bold" fill="{cfg["color"]}">{group_label_fn(gid)}</text>')

            center_y = offset_y + area_h // 2 + 10
            for ri, rank in enumerate(ranks):
                angle = -90 + ri * (360 / n_ranks)
                rad_a = angle * math.pi / 180
                rx = gx + radius * math.cos(rad_a)
                ry = center_y + radius * math.sin(rad_a)
                rank_positions[rank] = (rx, ry)

        return rank_positions

    for pi in range(total_phases):
        cfg = phase_configs[pi]
        svg.append(f'<g id="phase_{pi}" style="display:none">')
        svg.append(f'<text x="{PAGE_W//2}" y="{main_y + 25}" text-anchor="middle" font-size="16" font-weight="bold" fill="{cfg["color"]}">{cfg["name"]}</text>')

        rank_positions = draw_phase_ranks(svg, pi, 30, main_y + 35, PAGE_W - 60, main_h - 50)

        for rank, (rx, ry) in rank_positions.items():
            svg.append(f'<circle class="rank-node" cx="{rx:.1f}" cy="{ry:.1f}" r="{RANK_R}" fill="#42A5F5" stroke="#1565C0" stroke-width="2.5"/>')
            svg.append(f'<text x="{rx:.1f}" y="{ry - 5:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="white">R{rank}</text>')
            svg.append(f'<text x="{rx:.1f}" y="{ry + 9:.1f}" text-anchor="middle" font-size="7" fill="white">NPU{RANK_TO_NPU[rank]}</text>')

        step_idx_in_level = pi
        if pi >= n_intra + n_inter_pod:
            step_idx_in_level = pi - n_intra - n_inter_pod
        elif pi >= n_intra:
            step_idx_in_level = pi - n_intra

        nhr_steps = cfg['nhr_steps']
        if step_idx_in_level < len(nhr_steps):
            step_data = nhr_steps[step_idx_in_level]
            drawn_arrows = set()

            if cfg['level'] == 'intra':
                for pod_id, pod_ranks in sorted(PODS.items()):
                    for local_idx, rank in enumerate(pod_ranks):
                        if local_idx in step_data:
                            info = step_data[local_idx]
                            dst_local = info['send_to']
                            dst_rank = pod_ranks[dst_local]
                            if rank in rank_positions and dst_rank in rank_positions:
                                x1, y1 = rank_positions[rank]
                                x2, y2 = rank_positions[dst_rank]
                                edge_key = (min(rank, dst_rank), max(rank, dst_rank))
                                if edge_key not in drawn_arrows:
                                    drawn_arrows.add(edge_key)
                                    _draw_curved_arrow(svg, x1, y1, x2, y2, cfg['color'], RANK_R,
                                                       f"tx:{info['tx_slices']} rx:{info['rx_slices']}")

            elif cfg['level'] == 'inter_pod':
                for sp_id, sp_ranks in sorted(SUPER_PODS.items()):
                    pods_in_sp = [get_pod(r) for r in sp_ranks]
                    unique_pods = sorted(set(pods_in_sp))
                    for pod_local, pod_id in enumerate(unique_pods):
                        if pod_local in step_data:
                            info = step_data[pod_local]
                            dst_pod_local = info['send_to']
                            dst_pod_id = unique_pods[dst_pod_local]
                            src_rep = PODS[pod_id][0]
                            dst_rep = PODS[dst_pod_id][0]
                            if src_rep in rank_positions and dst_rep in rank_positions:
                                x1, y1 = rank_positions[src_rep]
                                x2, y2 = rank_positions[dst_rep]
                                _draw_curved_arrow(svg, x1, y1, x2, y2, cfg['color'], RANK_R,
                                                   f"Pod{pod_id}→Pod{dst_pod_id} tx:{info['tx_slices']}")

            elif cfg['level'] == 'inter_sp':
                for sp_local in range(2):
                    if sp_local in step_data:
                        info = step_data[sp_local]
                        dst_sp = info['send_to']
                        src_rep = SUPER_PODS[sp_local][0]
                        dst_rep = SUPER_PODS[dst_sp][0]
                        if src_rep in rank_positions and dst_rep in rank_positions:
                            x1, y1 = rank_positions[src_rep]
                            x2, y2 = rank_positions[dst_rep]
                            _draw_curved_arrow(svg, x1, y1, x2, y2, cfg['color'], RANK_R,
                                               f"SP{sp_local}→SP{dst_sp} tx:{info['tx_slices']}")

        detail_y = main_y + main_h + 15
        svg.append(f'<rect x="20" y="{detail_y}" width="{PAGE_W - 40}" height="{PAGE_H - detail_y - 10}" rx="8" fill="white" stroke="#ddd"/>')
        svg.append(f'<text x="40" y="{detail_y + 20}" font-size="12" font-weight="bold" fill="{cfg["color"]}">NHR算法步骤详情 ({cfg["name"]}):</text>')

        if step_idx_in_level < len(nhr_steps):
            step_data = nhr_steps[step_idx_in_level]
            ty = detail_y + 38
            col_w = (PAGE_W - 60) // min(len(step_data), 4)
            for xi, (local_idx, info) in enumerate(sorted(step_data.items())):
                col = xi % 4
                row = xi // 4
                tx = 40 + col * col_w
                tty = ty + row * 65
                svg.append(f'<rect x="{tx}" y="{tty - 12}" width="{col_w - 10}" height="58" rx="4" fill="{cfg["bg"]}" opacity="0.5"/>')
                svg.append(f'<text x="{tx + 5}" y="{tty + 2}" font-size="10" font-weight="bold" fill="#333">Rank[{local_idx}]</text>')
                svg.append(f'<text x="{tx + 5}" y="{tty + 16}" font-size="9" fill="#555">→ Rank[{info["send_to"]}] tx_slices={info["tx_slices"]}</text>')
                svg.append(f'<text x="{tx + 5}" y="{tty + 30}" font-size="9" fill="#555">← Rank[{info["recv_from"]}] rx_slices={info["rx_slices"]}</text>')
                svg.append(f'<text x="{tx + 5}" y="{tty + 44}" font-size="9" fill="#888">delta={1 << step_idx_in_level}, nSlices={len(info["tx_slices"])}</text>')

        svg.append('</g>')

    svg.append('<script type="text/javascript"><![CDATA[')
    svg.append(f'var totalPhases = {total_phases};')
    svg.append('var currentPhase = -1;')
    svg.append('function showPhase(idx) {')
    svg.append('  for (var i = 0; i < totalPhases; i++) {')
    svg.append('    var el = document.getElementById("phase_" + i);')
    svg.append('    el.style.display = (i === idx) ? "" : "none";')
    svg.append('    var navRect = document.getElementById("nav_rect_" + i);')
    svg.append('    navRect.setAttribute("stroke-width", (i === idx) ? "3" : "1.5");')
    svg.append('  }')
    svg.append('  currentPhase = idx;')
    svg.append('}')
    svg.append('showPhase(0);')
    svg.append(']]></script>')

    svg.append('</svg>')

    with open(output_path, 'w') as f:
        f.write('\n'.join(svg))
    print(f"SVG 详细步骤数据流图已保存到: {output_path}")


def _draw_curved_arrow(svg, x1, y1, x2, y2, color, r, label_text):
    dx = x2 - x1
    dy = y2 - y1
    dist = (dx**2 + dy**2)**0.5
    if dist == 0:
        return

    ndx = dx / dist
    ndy = dy / dist
    nx = -ndy
    ny = ndx

    sx = x1 + ndx * (r + 2)
    sy = y1 + ndy * (r + 2)
    ex = x2 - ndx * (r + 2)
    ey = y2 - ndy * (r + 2)

    ctrl_offset = min(40, dist * 0.2)
    cpx = (sx + ex) / 2 + nx * ctrl_offset
    cpy = (sy + ey) / 2 + ny * ctrl_offset

    path_d = f"M {sx:.1f} {sy:.1f} Q {cpx:.1f} {cpy:.1f} {ex:.1f} {ey:.1f}"
    svg.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.5" opacity="0.85"/>')

    t = 0.92
    t1 = 1 - t
    px = t1*t1*sx + 2*t1*t*cpx + t*t*ex
    py = t1*t1*sy + 2*t1*t*cpy + t*t*ey
    adx = ex - px
    ady = ey - py
    adist = (adx**2 + ady**2)**0.5
    if adist > 0:
        adx_n = adx / adist
        ady_n = ady / adist
        al = 10
        a1x = ex - al * adx_n + al * 0.4 * (-ady_n)
        a1y = ey - al * ady_n + al * 0.4 * adx_n
        a2x = ex - al * adx_n - al * 0.4 * (-ady_n)
        a2y = ey - al * ady_n - al * 0.4 * adx_n
        svg.append(f'<polygon points="{ex:.1f},{ey:.1f} {a1x:.1f},{a1y:.1f} {a2x:.1f},{a2y:.1f}" fill="{color}" opacity="0.85"/>')

    mid_x = (sx + ex) / 2 + nx * 16
    mid_y = (sy + ey) / 2 + ny * 16
    svg.append(f'<text x="{mid_x:.1f}" y="{mid_y:.1f}" text-anchor="middle" font-size="8" fill="{color}" font-weight="bold" paint-order="stroke" stroke="white" stroke-width="3">{label_text}</text>')


def generate_svg_inter_pod(data_trans_ops, output_path):
    cross_rank_ops = [
        op for op in data_trans_ops
        if op['op_type'] in ('SendRecvWrite', 'SendRecvWriteReduce')
        and op['src_rank'] >= 0 and op['dst_rank'] >= 0
        and op['src_rank'] != op['dst_rank']
    ]

    flow_map = defaultdict(lambda: defaultdict(lambda: {'types': set(), 'slices': set(), 'count': 0}))
    for op in cross_rank_ops:
        src_pod = get_pod(op['src_rank'])
        dst_pod = get_pod(op['dst_rank'])
        src_sp = get_super_pod(op['src_rank'])
        dst_sp = get_super_pod(op['dst_rank'])
        if src_pod == dst_pod or src_sp != dst_sp:
            continue
        info = flow_map[op['src_rank']][op['dst_rank']]
        info['types'].add(op['op_type'])
        info['slices'].add((op['slice_idx'], op['slice_num']))
        info['count'] += 1

    CELL_W = 120
    CELL_H = 80
    PAD_X = 60
    PAD_Y = 80
    POD_GAP_X = 80
    SP_GAP_Y = 100

    total_w = PAD_X * 2 + 2 * (4 * CELL_W) + POD_GAP_X
    total_h = PAD_Y * 2 + 2 * (1 * CELL_H) + SP_GAP_Y + 80

    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">')
    svg_parts.append(f'<rect width="{total_w}" height="{total_h}" fill="white"/>')
    svg_parts.append(f'<text x="{total_w//2}" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#FF9800">Level 1: Inter-Pod Data Flow</text>')
    svg_parts.append(f'<text x="{total_w//2}" y="52" text-anchor="middle" font-size="13" fill="#666">同一SuperPod内不同Pod之间 SendRecvWriteReduce | sliceNum=2</text>')

    _svg_draw_background(svg_parts, _svg_rank_pos, highlight_level='superpod')
    _svg_draw_ranks(svg_parts, _svg_rank_pos)

    drawn_edges = set()
    for src_rank in sorted(flow_map.keys()):
        for dst_rank in sorted(flow_map[src_rank].keys()):
            info = flow_map[src_rank][dst_rank]
            edge_key = (min(src_rank, dst_rank), max(src_rank, dst_rank))
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)

            x1, y1 = _svg_rank_pos(src_rank)
            x2, y2 = _svg_rank_pos(dst_rank)

            slices_str = ", ".join(f"[{s[0]}/{s[1]}]" for s in sorted(info['slices']))
            label = f"SRWR {slices_str}"

            _svg_draw_arrow(svg_parts, x1, y1, x2, y2,
                            '#FF9800', '2.2', '0.8', '8,4', label, '#E65100')

    legend_y = total_h - 40
    svg_parts.append(f'<rect x="{PAD_X}" y="{legend_y - 15}" width="450" height="30" rx="5" fill="white" stroke="#ddd" opacity="0.9"/>')
    svg_parts.append(f'<line x1="{PAD_X + 10}" y1="{legend_y}" x2="{PAD_X + 40}" y2="{legend_y}" stroke="#FF9800" stroke-width="2" stroke-dasharray="8,4"/>')
    svg_parts.append(f'<text x="{PAD_X + 45}" y="{legend_y + 4}" font-size="11" fill="#333">Inter-Pod SendRecvWriteReduce (sliceNum=2)</text>')

    svg_parts.append('</svg>')
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_parts))
    print(f"SVG 数据流图(Pod间)已保存到: {output_path}")


def generate_svg_inter_superpod(data_trans_ops, output_path):
    cross_rank_ops = [
        op for op in data_trans_ops
        if op['op_type'] in ('SendRecvWrite', 'SendRecvWriteReduce')
        and op['src_rank'] >= 0 and op['dst_rank'] >= 0
        and op['src_rank'] != op['dst_rank']
    ]

    flow_map = defaultdict(lambda: defaultdict(lambda: {'types': set(), 'slices': set(), 'count': 0}))
    for op in cross_rank_ops:
        src_sp = get_super_pod(op['src_rank'])
        dst_sp = get_super_pod(op['dst_rank'])
        if src_sp == dst_sp:
            continue
        info = flow_map[op['src_rank']][op['dst_rank']]
        info['types'].add(op['op_type'])
        info['slices'].add((op['slice_idx'], op['slice_num']))
        info['count'] += 1

    CELL_W = 120
    CELL_H = 80
    PAD_X = 60
    PAD_Y = 80
    POD_GAP_X = 80
    SP_GAP_Y = 100

    total_w = PAD_X * 2 + 2 * (4 * CELL_W) + POD_GAP_X
    total_h = PAD_Y * 2 + 2 * (1 * CELL_H) + SP_GAP_Y + 80

    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">')
    svg_parts.append(f'<rect width="{total_w}" height="{total_h}" fill="white"/>')
    svg_parts.append(f'<text x="{total_w//2}" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#F44336">Level 2: Inter-SuperPod Data Flow</text>')
    svg_parts.append(f'<text x="{total_w//2}" y="52" text-anchor="middle" font-size="13" fill="#666">跨SuperPod之间 SendRecvWriteReduce | sliceNum=1</text>')

    _svg_draw_background(svg_parts, _svg_rank_pos, highlight_level='superpod')
    _svg_draw_ranks(svg_parts, _svg_rank_pos)

    drawn_edges = set()
    for src_rank in sorted(flow_map.keys()):
        for dst_rank in sorted(flow_map[src_rank].keys()):
            info = flow_map[src_rank][dst_rank]
            edge_key = (min(src_rank, dst_rank), max(src_rank, dst_rank))
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)

            x1, y1 = _svg_rank_pos(src_rank)
            x2, y2 = _svg_rank_pos(dst_rank)

            slices_str = ", ".join(f"[{s[0]}/{s[1]}]" for s in sorted(info['slices']))
            label = f"SRWR {slices_str}"

            _svg_draw_arrow(svg_parts, x1, y1, x2, y2,
                            '#F44336', '2.5', '0.85', '4,4', label, '#B71C1C')

    legend_y = total_h - 40
    svg_parts.append(f'<rect x="{PAD_X}" y="{legend_y - 15}" width="450" height="30" rx="5" fill="white" stroke="#ddd" opacity="0.9"/>')
    svg_parts.append(f'<line x1="{PAD_X + 10}" y1="{legend_y}" x2="{PAD_X + 40}" y2="{legend_y}" stroke="#F44336" stroke-width="2.5" stroke-dasharray="4,4"/>')
    svg_parts.append(f'<text x="{PAD_X + 45}" y="{legend_y + 4}" font-size="11" fill="#333">Inter-SuperPod SendRecvWriteReduce (sliceNum=1)</text>')

    svg_parts.append('</svg>')
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_parts))
    print(f"SVG 数据流图(跨SuperPod)已保存到: {output_path}")


def main():
    if len(sys.argv) < 2:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log202606031748_inter2.log")
    else:
        log_path = sys.argv[1]

    if not os.path.exists(log_path):
        print(f"错误: 日志文件不存在: {log_path}")
        sys.exit(1)

    print(f"正在解析日志文件: {log_path}")
    tid_to_rank, rank_input_base, rank_output_base, data_trans_ops, step_infos = parse_log(log_path)

    print(f"  解析到 {len(tid_to_rank)} 个线程-Rank映射")
    print(f"  解析到 {len(rank_input_base)} 个Rank的内存地址映射")
    print(f"  解析到 {len(data_trans_ops)} 个数据传输操作")
    print(f"  解析到 {len(step_infos)} 个步骤信息")

    output_dir = os.path.dirname(os.path.abspath(__file__))

    generate_text_report(data_trans_ops, step_infos, tid_to_rank)

    ascii_path = os.path.join(output_dir, "data_flow_ascii.txt")
    generate_ascii_diagram(data_trans_ops, ascii_path)

    per_rank_path = os.path.join(output_dir, "data_flow_per_rank.txt")
    generate_per_rank_flow_diagram(data_trans_ops, per_rank_path)

    mermaid_path = os.path.join(output_dir, "data_flow.mermaid")
    generate_mermaid_diagram(data_trans_ops, mermaid_path)

    dot_path = os.path.join(output_dir, "data_flow.dot")
    generate_dot_diagram(data_trans_ops, dot_path)

    svg_path = os.path.join(output_dir, "data_flow.svg")
    generate_svg_diagram(data_trans_ops, svg_path)

    svg_intra_path = os.path.join(output_dir, "data_flow_intra_pod.svg")
    generate_svg_intra_pod(data_trans_ops, svg_intra_path)

    svg_inter_pod_path = os.path.join(output_dir, "data_flow_inter_pod.svg")
    generate_svg_inter_pod(data_trans_ops, svg_inter_pod_path)

    svg_inter_sp_path = os.path.join(output_dir, "data_flow_inter_superpod.svg")
    generate_svg_inter_superpod(data_trans_ops, svg_inter_sp_path)

    svg_step_path = os.path.join(output_dir, "data_flow_step_by_step.svg")
    generate_svg_step_by_step(data_trans_ops, step_infos, svg_step_path)

    svg_step_detail_path = os.path.join(output_dir, "data_flow_step_detail.svg")
    generate_svg_step_detail(data_trans_ops, step_infos, svg_step_detail_path)

    print("\n" + "=" * 80)
    print("  分析完成! 生成的文件:")
    print(f"    1. ASCII 数据流图:          {ascii_path}")
    print(f"    2. 各Rank数据流详情:        {per_rank_path}")
    print(f"    3. Mermaid 数据流图:        {mermaid_path}")
    print(f"    4. Graphviz DOT 图:         {dot_path}")
    print(f"    5. SVG 数据流图(总览):      {svg_path}")
    print(f"    6. SVG 数据流图(Pod内):     {svg_intra_path}")
    print(f"    7. SVG 数据流图(Pod间):     {svg_inter_pod_path}")
    print(f"    8. SVG 数据流图(跨超节点):  {svg_inter_sp_path}")
    print(f"    9. SVG 逐步数据流(交互式):  {svg_step_path}")
    print(f"   10. SVG 详细步骤视图:        {svg_step_detail_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
