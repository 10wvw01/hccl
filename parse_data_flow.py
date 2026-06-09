#!/usr/bin/env python3
import re
import sys
from collections import defaultdict

LOG_FILE = "/home/crx/code/hccl_three_sequence_rs/hccl_2039/log202606081040.log"

LAYER0_SIZE = 4
LAYER1_SIZE = 2
LAYER2_SIZE = 2
TOTAL_RANKS = LAYER0_SIZE * LAYER1_SIZE * LAYER2_SIZE

def rank_to_pos(rank):
    l0_idx = rank % LAYER0_SIZE
    l1_idx = (rank // LAYER0_SIZE) % LAYER1_SIZE
    l2_idx = rank // (LAYER0_SIZE * LAYER1_SIZE)
    return l0_idx, l1_idx, l2_idx

def pos_label(rank):
    l0, l1, l2 = rank_to_pos(rank)
    return f"R{rank}\n(S{l2}-P{l1}-D{l0})"

def parse_log(filepath):
    ranks = {}
    current_rank = None
    current_thread = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'rankId is : (\d+)', line)
            if m:
                current_rank = int(m.group(1))
                ranks[current_rank] = {'level0': [], 'level1': [], 'level2': [], 'output': []}
                continue

            if current_rank is None:
                continue

            m2 = re.match(r'rankIdx:(\d+),\s*threadIdx:(\d+),\s*\[(.+)\]', line)
            if not m2:
                continue

            rank = int(m2.group(1))
            tid = int(m2.group(2))
            action = m2.group(3)

            if rank != current_rank:
                continue

            if 'LocalCopy' in action and 'INPUT' in action and 'CCL' in action:
                m3 = re.search(r'srcSlice=DataSlice\[BufferType::INPUT, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\].*dstSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]', action)
                if m3:
                    ranks[current_rank]['level0'].append({
                        'type': 'LocalCopy',
                        'src_offset': int(m3.group(1), 16),
                        'src_size': int(m3.group(2), 16),
                        'dst_offset': int(m3.group(3), 16),
                        'dst_size': int(m3.group(4), 16),
                    })

            elif 'LocalReduce' in action:
                m3 = re.search(r'srcSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\].*dstSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]', action)
                if m3:
                    ranks[current_rank]['level0'].append({
                        'type': 'LocalReduce',
                        'src_offset': int(m3.group(1), 16),
                        'src_size': int(m3.group(2), 16),
                        'dst_offset': int(m3.group(3), 16),
                        'dst_size': int(m3.group(4), 16),
                    })

            elif 'WriteReduce' in action:
                m3 = re.search(r'remoteRank=(\d+).*localSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\].*remoteSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]', action)
                if m3:
                    remote = int(m3.group(1))
                    entry = {
                        'type': 'WriteReduce',
                        'remote_rank': remote,
                        'local_offset': int(m3.group(2), 16),
                        'local_size': int(m3.group(3), 16),
                        'remote_offset': int(m3.group(4), 16),
                        'remote_size': int(m3.group(5), 16),
                    }
                    _, l1_src, l2_src = rank_to_pos(current_rank)
                    _, l1_dst, l2_dst = rank_to_pos(remote)
                    if l2_src != l2_dst:
                        ranks[current_rank]['level2'].append(entry)
                    elif l1_src != l1_dst:
                        ranks[current_rank]['level1'].append(entry)
                    else:
                        ranks[current_rank]['level0'].append(entry)

            elif re.search(r'\[Write\]:', action) and 'INPUT' in action:
                m3 = re.search(r'remoteRank=(\d+).*localSlice=DataSlice\[BufferType::INPUT, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\].*remoteSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]', action)
                if m3:
                    remote = int(m3.group(1))
                    entry = {
                        'type': 'Write',
                        'remote_rank': remote,
                        'local_offset': int(m3.group(2), 16),
                        'local_size': int(m3.group(3), 16),
                        'remote_offset': int(m3.group(4), 16),
                        'remote_size': int(m3.group(5), 16),
                    }
                    _, l1_src, l2_src = rank_to_pos(current_rank)
                    _, l1_dst, l2_dst = rank_to_pos(remote)
                    if l2_src != l2_dst:
                        ranks[current_rank]['level2'].append(entry)
                    elif l1_src != l1_dst:
                        ranks[current_rank]['level1'].append(entry)
                    else:
                        ranks[current_rank]['level0'].append(entry)

            elif 'LocalCopy' in action and 'CCL' in action and 'OUTPUT' in action:
                m3 = re.search(r'srcSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\].*dstSlice=DataSlice\[BufferType::OUTPUT, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]', action)
                if m3:
                    ranks[current_rank]['output'].append({
                        'type': 'Output',
                        'src_offset': int(m3.group(1), 16),
                        'src_size': int(m3.group(2), 16),
                        'dst_offset': int(m3.group(3), 16),
                        'dst_size': int(m3.group(4), 16),
                    })

    return ranks

def fmt_size(sz):
    if sz >= 0x100000:
        return f"{sz/0x100000:.1f}MB"
    elif sz >= 0x400:
        return f"{sz/0x400:.1f}KB"
    else:
        return f"{sz}B"

def fmt_offset(off):
    return f"0x{off:X}"

def generate_mermaid(ranks):
    lines = []
    lines.append("graph TD")

    lines.append("")
    lines.append("subgraph Legend")
    lines.append("    L0[\"Level0: Intra-Server (ZAxisDetour)\"]:::level0")
    lines.append("    L1[\"Level1: Inter-Server (NHR)\"]:::level1")
    lines.append("    L2[\"Level2: Inter-Pod (NHR)\"]:::level2")
    lines.append("    IN[\"INPUT Buffer\"]:::input")
    lines.append("    CC[\"CCL Buffer\"]:::ccl")
    lines.append("    OU[\"OUTPUT Buffer\"]:::output")
    lines.append("end")
    lines.append("")

    for l2 in range(LAYER2_SIZE):
        lines.append(f"subgraph Pod{l2}[\"Pod {l2} (SuperPod {l2})\"]")
        for l1 in range(LAYER1_SIZE):
            lines.append(f"  subgraph Server{l2}_{l1}[\"Server {l2}-{l1}\"]")
            for l0 in range(LAYER0_SIZE):
                rank = l2 * LAYER1_SIZE * LAYER0_SIZE + l1 * LAYER0_SIZE + l0
                lines.append(f"    R{rank}[\"{pos_label(rank)}\"]:::ranknode")
            lines.append(f"  end")
        lines.append(f"end")
        lines.append("")

    lines.append("")
    lines.append("%% ========== Level0: Intra-Server Data Flow ==========")
    for rank in range(TOTAL_RANKS):
        if rank not in ranks:
            continue
        l0_idx, l1_idx, l2_idx = rank_to_pos(rank)
        for entry in ranks[rank]['level0']:
            if entry['type'] == 'LocalCopy':
                lines.append(f"R{rank}_IN[\"R{rank} INPUT\\n{fmt_offset(entry['src_offset'])} {fmt_size(entry['src_size'])}\"]:::input --> R{rank}_CCL[\"R{rank} CCL\\n{fmt_offset(entry['dst_offset'])} {fmt_size(entry['dst_size'])}\"]:::ccl")
                lines.append(f"R{rank}_IN -.-> R{rank}")
                lines.append(f"R{rank}_CCL -.-> R{rank}")
            elif entry['type'] == 'LocalReduce':
                lines.append(f"R{rank}_CCL2[\"R{rank} CCL\\n{fmt_offset(entry['src_offset'])} {fmt_size(entry['src_size'])}\"]:::ccl -->|Reduce| R{rank}_CCLD[\"R{rank} CCL\\n{fmt_offset(entry['dst_offset'])} {fmt_size(entry['dst_size'])}\"]:::ccl")
                lines.append(f"R{rank}_CCL2 -.-> R{rank}")
                lines.append(f"R{rank}_CCLD -.-> R{rank}")

    lines.append("")
    lines.append("%% ========== Level1: Inter-Server WriteReduce ==========")
    for rank in range(TOTAL_RANKS):
        if rank not in ranks:
            continue
        for entry in ranks[rank]['level1']:
            remote = entry['remote_rank']
            if entry['type'] == 'WriteReduce':
                lines.append(f"R{rank}[\"{pos_label(rank)}\"] -->|\"L1 WriteReduce\\nCCL {fmt_offset(entry['local_offset'])} {fmt_size(entry['local_size'])} → R{remote} CCL {fmt_offset(entry['remote_offset'])}\"| R{remote}[\"{pos_label(remote)}\"]:::level1edge")

    lines.append("")
    lines.append("%% ========== Level2: Inter-Pod WriteReduce ==========")
    for rank in range(TOTAL_RANKS):
        if rank not in ranks:
            continue
        for entry in ranks[rank]['level2']:
            remote = entry['remote_rank']
            if entry['type'] == 'WriteReduce':
                lines.append(f"R{rank}[\"{pos_label(rank)}\"] ==>|\"L2 WriteReduce\\nCCL {fmt_offset(entry['local_offset'])} {fmt_size(entry['local_size'])} → R{remote} CCL {fmt_offset(entry['remote_offset'])}\"| R{remote}[\"{pos_label(remote)}\"]:::level2edge")

    lines.append("")
    lines.append("%% ========== Level0 Scatter: Write INPUT→CCL ==========")
    for rank in range(TOTAL_RANKS):
        if rank not in ranks:
            continue
        for entry in ranks[rank]['level0']:
            if entry['type'] == 'Write':
                remote = entry['remote_rank']
                lines.append(f"R{rank}[\"{pos_label(rank)}\"] -.->|\"L0 Write\\nINPUT {fmt_offset(entry['local_offset'])} {fmt_size(entry['local_size'])} → R{remote} CCL {fmt_offset(entry['remote_offset'])}\"| R{remote}[\"{pos_label(remote)}\"]:::level0edge")

    lines.append("")
    lines.append("%% ========== Output ==========")
    for rank in range(TOTAL_RANKS):
        if rank not in ranks:
            continue
        for entry in ranks[rank]['output']:
            lines.append(f"R{rank}[\"{pos_label(rank)}\"] -->|\"Output\\nCCL {fmt_offset(entry['src_offset'])} {fmt_size(entry['src_size'])} → OUTPUT {fmt_offset(entry['dst_offset'])}\"| R{rank}_OUT[\"R{rank} OUTPUT\"]:::output")

    lines.append("")
    lines.append("classDef level0 fill:#4CAF50,stroke:#2E7D32,color:white")
    lines.append("classDef level1 fill:#2196F3,stroke:#1565C0,color:white")
    lines.append("classDef level2 fill:#FF5722,stroke:#D84315,color:white")
    lines.append("classDef input fill:#FFF9C4,stroke:#F9A825,color:#333")
    lines.append("classDef ccl fill:#E1BEE7,stroke:#7B1FA2,color:#333")
    lines.append("classDef output fill:#C8E6C9,stroke:#2E7D32,color:#333")
    lines.append("classDef ranknode fill:#E3F2FD,stroke:#1565C0,color:#333")
    lines.append("classDef level0edge stroke:#4CAF50,stroke-width:2px")
    lines.append("classDef level1edge stroke:#2196F3,stroke-width:3px")
    lines.append("classDef level2edge stroke:#FF5722,stroke-width:4px")

    return "\n".join(lines)


def generate_detailed_mermaid_per_level(ranks):
    results = {}

    for level_name, level_idx in [("level0", 0), ("level1", 1), ("level2", 2)]:
        lines = []
        lines.append("graph TD")
        lines.append("")

        if level_idx == 0:
            lines.append("subgraph title[\"Level 0: Intra-Server (ZAxisDetour) — INPUT→CCL Copy + LocalReduce\"]")
            lines.append("end")
            lines.append("")
            for l2 in range(LAYER2_SIZE):
                for l1 in range(LAYER1_SIZE):
                    lines.append(f"subgraph Server{l2}_{l1}[\"Server {l2}-{l1}\"]")
                    for l0 in range(LAYER0_SIZE):
                        rank = l2 * LAYER1_SIZE * LAYER0_SIZE + l1 * LAYER0_SIZE + l0
                        lines.append(f"  R{rank}[\"{pos_label(rank)}\"]:::ranknode")
                    lines.append("end")
                lines.append("")

            for rank in range(TOTAL_RANKS):
                if rank not in ranks:
                    continue
                step = 0
                for entry in ranks[rank]['level0']:
                    if entry['type'] == 'LocalCopy':
                        lines.append(f"  R{rank}_IN{step}[\"R{rank} INPUT\\n{fmt_offset(entry['src_offset'])} {fmt_size(entry['src_size'])}\"]:::input -->|Copy| R{rank}_CCL{step}[\"R{rank} CCL\\n{fmt_offset(entry['dst_offset'])} {fmt_size(entry['dst_size'])}\"]:::ccl")
                        step += 1
                    elif entry['type'] == 'LocalReduce':
                        lines.append(f"  R{rank}_CCLS{step}[\"R{rank} CCL\\n{fmt_offset(entry['src_offset'])} {fmt_size(entry['src_size'])}\"]:::ccl -->|Reduce| R{rank}_CCLD{step}[\"R{rank} CCL\\n{fmt_offset(entry['dst_offset'])} {fmt_size(entry['dst_size'])}\"]:::ccl")
                        step += 1
                    elif entry['type'] == 'Write':
                        remote = entry['remote_rank']
                        lines.append(f"  R{rank}[\"{pos_label(rank)}\"] -.->|\"Write INPUT→CCL\\n{fmt_size(entry['local_size'])} → R{remote}\"| R{remote}[\"{pos_label(remote)}\"]")

            for rank in range(TOTAL_RANKS):
                if rank not in ranks:
                    continue
                for entry in ranks[rank]['output']:
                    lines.append(f"  R{rank}[\"{pos_label(rank)}\"] -->|\"Output {fmt_size(entry['src_size'])}\"| R{rank}_OUT[\"R{rank} OUTPUT\"]:::output")

        elif level_idx == 1:
            lines.append("subgraph title[\"Level 1: Inter-Server (NHR) — WriteReduce CCL→CCL\"]")
            lines.append("end")
            lines.append("")
            for l2 in range(LAYER2_SIZE):
                lines.append(f"subgraph Pod{l2}[\"Pod {l2}\"]")
                for l1 in range(LAYER1_SIZE):
                    lines.append(f"  subgraph Server{l2}_{l1}[\"Server {l2}-{l1}\"]")
                    for l0 in range(LAYER0_SIZE):
                        rank = l2 * LAYER1_SIZE * LAYER0_SIZE + l1 * LAYER0_SIZE + l0
                        lines.append(f"    R{rank}[\"{pos_label(rank)}\"]:::ranknode")
                    lines.append("  end")
                lines.append("end")
                lines.append("")

            for rank in range(TOTAL_RANKS):
                if rank not in ranks:
                    continue
                for entry in ranks[rank]['level1']:
                    remote = entry['remote_rank']
                    if entry['type'] == 'WriteReduce':
                        lines.append(f"  R{rank}[\"{pos_label(rank)}\"] -->|\"WriteReduce\\nCCL {fmt_offset(entry['local_offset'])} {fmt_size(entry['local_size'])}\\n→ R{remote} CCL {fmt_offset(entry['remote_offset'])}\"| R{remote}[\"{pos_label(remote)}\"]:::level1edge")

        elif level_idx == 2:
            lines.append("subgraph title[\"Level 2: Inter-Pod (NHR) — WriteReduce CCL→CCL\"]")
            lines.append("end")
            lines.append("")
            for l2 in range(LAYER2_SIZE):
                lines.append(f"subgraph Pod{l2}[\"Pod {l2}\"]")
                for l1 in range(LAYER1_SIZE):
                    lines.append(f"  subgraph Server{l2}_{l1}[\"Server {l2}-{l1}\"]")
                    for l0 in range(LAYER0_SIZE):
                        rank = l2 * LAYER1_SIZE * LAYER0_SIZE + l1 * LAYER0_SIZE + l0
                        lines.append(f"    R{rank}[\"{pos_label(rank)}\"]:::ranknode")
                    lines.append("  end")
                lines.append("end")
                lines.append("")

            for rank in range(TOTAL_RANKS):
                if rank not in ranks:
                    continue
                for entry in ranks[rank]['level2']:
                    remote = entry['remote_rank']
                    if entry['type'] == 'WriteReduce':
                        lines.append(f"  R{rank}[\"{pos_label(rank)}\"] ==>|\"WriteReduce\\nCCL {fmt_offset(entry['local_offset'])} {fmt_size(entry['local_size'])}\\n→ R{remote} CCL {fmt_offset(entry['remote_offset'])}\"| R{remote}[\"{pos_label(remote)}\"]:::level2edge")

        lines.append("")
        lines.append("classDef level0 fill:#4CAF50,stroke:#2E7D32,color:white")
        lines.append("classDef level1 fill:#2196F3,stroke:#1565C0,color:white")
        lines.append("classDef level2 fill:#FF5722,stroke:#D84315,color:white")
        lines.append("classDef input fill:#FFF9C4,stroke:#F9A825,color:#333")
        lines.append("classDef ccl fill:#E1BEE7,stroke:#7B1FA2,color:#333")
        lines.append("classDef output fill:#C8E6C9,stroke:#2E7D32,color:#333")
        lines.append("classDef ranknode fill:#E3F2FD,stroke:#1565C0,color:#333")
        lines.append("classDef level1edge stroke:#2196F3,stroke-width:3px")
        lines.append("classDef level2edge stroke:#FF5722,stroke-width:4px")

        results[level_name] = "\n".join(lines)

    return results


def generate_text_summary(ranks):
    lines = []
    lines.append("=" * 80)
    lines.append("ReduceScatter 3-Level Data Flow Summary (4x2x2 = 16 Ranks)")
    lines.append("Topology: Layer0=4 (intra-server), Layer1=2 (inter-server), Layer2=2 (inter-pod)")
    lines.append("=" * 80)

    for rank in range(TOTAL_RANKS):
        if rank not in ranks:
            continue
        l0, l1, l2 = rank_to_pos(rank)
        lines.append("")
        lines.append(f"--- Rank {rank} (Pod{l2}-Server{l1}-Device{l0}) ---")

        l0_entries = ranks[rank]['level0']
        l1_entries = ranks[rank]['level1']
        l2_entries = ranks[rank]['level2']
        out_entries = ranks[rank]['output']

        lines.append(f"  [Level0 - Intra-Server] {len(l0_entries)} operations:")
        for e in l0_entries:
            if e['type'] == 'LocalCopy':
                lines.append(f"    INPUT[{fmt_offset(e['src_offset'])}:{fmt_size(e['src_size'])}] --Copy--> CCL[{fmt_offset(e['dst_offset'])}:{fmt_size(e['dst_size'])}]")
            elif e['type'] == 'LocalReduce':
                lines.append(f"    CCL[{fmt_offset(e['src_offset'])}:{fmt_size(e['src_size'])}] --Reduce--> CCL[{fmt_offset(e['dst_offset'])}:{fmt_size(e['dst_size'])}]")
            elif e['type'] == 'Write':
                lines.append(f"    INPUT[{fmt_offset(e['local_offset'])}:{fmt_size(e['local_size'])}] --Write--> R{e['remote_rank']} CCL[{fmt_offset(e['remote_offset'])}:{fmt_size(e['remote_size'])}]")

        lines.append(f"  [Level1 - Inter-Server] {len(l1_entries)} operations:")
        for e in l1_entries:
            if e['type'] == 'WriteReduce':
                lines.append(f"    CCL[{fmt_offset(e['local_offset'])}:{fmt_size(e['local_size'])}] --WriteReduce--> R{e['remote_rank']} CCL[{fmt_offset(e['remote_offset'])}:{fmt_size(e['remote_size'])}]")
            elif e['type'] == 'Write':
                lines.append(f"    INPUT[{fmt_offset(e['local_offset'])}:{fmt_size(e['local_size'])}] --Write--> R{e['remote_rank']} CCL[{fmt_offset(e['remote_offset'])}:{fmt_size(e['remote_size'])}]")

        lines.append(f"  [Level2 - Inter-Pod] {len(l2_entries)} operations:")
        for e in l2_entries:
            if e['type'] == 'WriteReduce':
                lines.append(f"    CCL[{fmt_offset(e['local_offset'])}:{fmt_size(e['local_size'])}] --WriteReduce--> R{e['remote_rank']} CCL[{fmt_offset(e['remote_offset'])}:{fmt_size(e['remote_size'])}]")
            elif e['type'] == 'Write':
                lines.append(f"    INPUT[{fmt_offset(e['local_offset'])}:{fmt_size(e['local_size'])}] --Write--> R{e['remote_rank']} CCL[{fmt_offset(e['remote_offset'])}:{fmt_size(e['remote_size'])}]")

        lines.append(f"  [Output] {len(out_entries)} operations:")
        for e in out_entries:
            lines.append(f"    CCL[{fmt_offset(e['src_offset'])}:{fmt_size(e['src_size'])}] --Output--> OUTPUT[{fmt_offset(e['dst_offset'])}:{fmt_size(e['dst_size'])}]")

    lines.append("")
    lines.append("=" * 80)
    lines.append("Data Flow Summary by Level:")
    lines.append("=" * 80)

    for level_name, level_label in [("level0", "Level0 (Intra-Server)"), ("level1", "Level1 (Inter-Server)"), ("level2", "Level2 (Inter-Pod)")]:
        lines.append(f"\n{level_label}:")
        edges = defaultdict(list)
        for rank in range(TOTAL_RANKS):
            if rank not in ranks:
                continue
            for e in ranks[rank][level_name]:
                if e['type'] in ('WriteReduce', 'Write'):
                    key = (rank, e['remote_rank'])
                    edges[key].append(f"{fmt_size(e['local_size'])}")
        if edges:
            for (src, dst), sizes in sorted(edges.items()):
                lines.append(f"  R{src} → R{dst}: {', '.join(sizes)}")
        else:
            lines.append("  (no remote operations, all local)")

    return "\n".join(lines)


def main():
    print(f"Parsing log: {LOG_FILE}")
    ranks = parse_log(LOG_FILE)
    print(f"Parsed {len(ranks)} ranks")

    for rank in sorted(ranks.keys()):
        l0 = len(ranks[rank]['level0'])
        l1 = len(ranks[rank]['level1'])
        l2 = len(ranks[rank]['level2'])
        out = len(ranks[rank]['output'])
        print(f"  Rank {rank}: Level0={l0} ops, Level1={l1} ops, Level2={l2} ops, Output={out} ops")

    out_dir = "/home/crx/code/hccl_three_sequence_rs/hccl_2039"

    text_summary = generate_text_summary(ranks)
    with open(f"{out_dir}/data_flow_summary.txt", 'w') as f:
        f.write(text_summary)
    print(f"\nWrote data_flow_summary.txt")

    per_level = generate_detailed_mermaid_per_level(ranks)
    for level_name, content in per_level.items():
        fname = f"{out_dir}/data_flow_{level_name}.mermaid"
        with open(fname, 'w') as f:
            f.write(content)
        print(f"Wrote {fname}")

    full_mermaid = generate_mermaid(ranks)
    with open(f"{out_dir}/data_flow_full.mermaid", 'w') as f:
        f.write(full_mermaid)
    print(f"Wrote data_flow_full.mermaid")

    print("\nDone! View .mermaid files with VS Code Mermaid preview or https://mermaid.live")

if __name__ == "__main__":
    main()
