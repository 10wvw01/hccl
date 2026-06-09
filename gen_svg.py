#!/usr/bin/env python3
import re
import html as html_mod
import sys
import os

OWNER_COLORS = [
    ("#C8E6C9", "#2E7D32"),
    ("#BBDEFB", "#1565C0"),
    ("#FFE0B2", "#E65100"),
    ("#F8BBD0", "#AD1457"),
]


def infer_topology(ranks):
    total = len(ranks)
    write_pairs = set()
    for rank in ranks:
        for e in ranks[rank].get('write', []):
            remote = e['remote_rank']
            if rank != remote:
                write_pairs.add((min(rank, remote), max(rank, remote)))

    l0 = total
    for d in range(2, total + 1):
        if total % d == 0:
            l0 = d
            break

    if write_pairs:
        best_l0 = l0
        for candidate_l0 in range(2, total + 1):
            if total % candidate_l0 != 0:
                continue
            all_intra = all(s // candidate_l0 == d // candidate_l0 for s, d in write_pairs)
            if all_intra:
                best_l0 = candidate_l0
                break
        l0 = best_l0

    l1 = total // l0
    l2 = 1

    wr_pairs = set()
    for rank in ranks:
        for e in ranks[rank].get('write_reduce_l1', []):
            remote = e['remote_rank']
            if rank != remote:
                wr_pairs.add((min(rank, remote), max(rank, remote)))

    if wr_pairs:
        wr_distances = sorted(set(abs(s - d) for s, d in wr_pairs))
        if len(wr_distances) >= 2:
            min_dist = wr_distances[0]
            l1 = min_dist // l0
            if l1 < 2:
                l1 = 2
            l2 = total // (l0 * l1)
        elif len(wr_distances) == 1:
            min_dist = wr_distances[0]
            l1 = total // l0
            l2 = 1
        else:
            l1 = total // l0
            l2 = 1

    return l0, l1, l2


def _rank_to_pos(rank, l0, l1, l2):
    return rank % l0, (rank // l0) % l1, rank // (l0 * l1)


def build_offset_owner_map(ranks, l0_size):
    offset_owner = {}
    for rank in ranks:
        for e in ranks[rank].get('localcopy', []):
            off = e['dst_offset']
            owner = rank % l0_size
            if off not in offset_owner:
                offset_owner[off] = owner
    for rank in ranks:
        for e in ranks[rank].get('localreduce', []):
            for key in ('src_offset', 'dst_offset'):
                off = e[key]
                if off not in offset_owner:
                    offset_owner[off] = None
        for e in ranks[rank].get('write', []):
            off = e['remote_offset']
            if off not in offset_owner:
                offset_owner[off] = None
            if e.get('local_buf', 'INPUT') == 'CCL':
                off = e['local_offset']
                if off not in offset_owner:
                    offset_owner[off] = None
        for e in ranks[rank].get('read', []):
            if e.get('local_buf', 'CCL') == 'CCL':
                off = e['local_offset']
                if off not in offset_owner:
                    offset_owner[off] = None
            off = e['remote_offset']
            if off not in offset_owner:
                offset_owner[off] = None
        for e in ranks[rank].get('write_reduce_l1', []):
            for key in ('local_offset', 'remote_offset'):
                off = e[key]
                if off not in offset_owner:
                    offset_owner[off] = None
        for e in ranks[rank].get('write_reduce_l2', []):
            for key in ('local_offset', 'remote_offset'):
                off = e[key]
                if off not in offset_owner:
                    offset_owner[off] = None
        for e in ranks[rank].get('output', []):
            off = e['src_offset']
            if off not in offset_owner:
                offset_owner[off] = None
    owned_offsets = sorted(off for off, ow in offset_owner.items() if ow is not None)
    for off in sorted(offset_owner.keys()):
        if offset_owner[off] is not None:
            continue
        best = None
        for owned in owned_offsets:
            if owned <= off:
                best = owned
            else:
                break
        if best is not None:
            offset_owner[off] = offset_owner[best]
        else:
            offset_owner[off] = 0
    return offset_owner


def build_input_owner_map(ranks, l0_size):
    offset_owner = {}
    for rank in ranks:
        for e in ranks[rank].get('localcopy', []):
            off = e['src_offset']
            owner = rank % l0_size
            if off not in offset_owner:
                offset_owner[off] = owner
        for e in ranks[rank].get('write', []):
            if e.get('local_buf', 'INPUT') == 'INPUT':
                off = e['local_offset']
                if off not in offset_owner:
                    offset_owner[off] = rank % l0_size
        for e in ranks[rank].get('read', []):
            if e.get('local_buf', 'INPUT') == 'INPUT':
                off = e['local_offset']
                if off not in offset_owner:
                    offset_owner[off] = rank % l0_size
    return offset_owner


def collect_ccl_offsets(ranks):
    ccl_offsets = set()
    for rank in ranks:
        for e in ranks[rank].get('localcopy', []):
            ccl_offsets.add(e['dst_offset'])
        for e in ranks[rank].get('localreduce', []):
            ccl_offsets.add(e['src_offset'])
            ccl_offsets.add(e['dst_offset'])
        for e in ranks[rank].get('write', []):
            if e.get('local_buf', 'INPUT') == 'CCL':
                ccl_offsets.add(e['local_offset'])
            ccl_offsets.add(e['remote_offset'])
        for e in ranks[rank].get('write_reduce_l1', []):
            ccl_offsets.add(e['local_offset'])
            ccl_offsets.add(e['remote_offset'])
        for e in ranks[rank].get('write_reduce_l2', []):
            ccl_offsets.add(e['local_offset'])
            ccl_offsets.add(e['remote_offset'])
        for e in ranks[rank].get('read', []):
            if e.get('local_buf', 'CCL') == 'CCL':
                ccl_offsets.add(e['local_offset'])
            ccl_offsets.add(e['remote_offset'])
        for e in ranks[rank].get('output', []):
            ccl_offsets.add(e['src_offset'])
    return sorted(ccl_offsets)


def collect_input_offsets(ranks):
    input_offsets = set()
    for rank in ranks:
        for e in ranks[rank].get('localcopy', []):
            input_offsets.add(e['src_offset'])
        for e in ranks[rank].get('write', []):
            if e.get('local_buf', 'INPUT') == 'INPUT':
                input_offsets.add(e['local_offset'])
        for e in ranks[rank].get('read', []):
            if e.get('local_buf', 'INPUT') == 'INPUT':
                input_offsets.add(e['local_offset'])
    return sorted(input_offsets)


def collect_level_ccl_offsets(ranks):
    ccl_offsets = set()
    for rank in ranks:
        for e in ranks[rank].get('write_reduce_l1', []):
            ccl_offsets.add(e['local_offset'])
            ccl_offsets.add(e['remote_offset'])
        for e in ranks[rank].get('write_reduce_l2', []):
            ccl_offsets.add(e['local_offset'])
            ccl_offsets.add(e['remote_offset'])
        for e in ranks[rank].get('write', []):
            if e.get('local_buf', 'INPUT') == 'CCL':
                ccl_offsets.add(e['local_offset'])
            if e.get('remote_buf', 'CCL') == 'CCL':
                ccl_offsets.add(e['remote_offset'])
        for e in ranks[rank].get('read', []):
            if e.get('local_buf', 'CCL') == 'CCL':
                ccl_offsets.add(e['local_offset'])
            if e.get('remote_buf', 'CCL') == 'CCL':
                ccl_offsets.add(e['remote_offset'])
        for e in ranks[rank].get('output', []):
            ccl_offsets.add(e['src_offset'])
        for e in ranks[rank].get('localreduce', []):
            ccl_offsets.add(e['src_offset'])
            ccl_offsets.add(e['dst_offset'])
    return sorted(ccl_offsets)


def build_level_ccl_owner_map(ranks, l0_size):
    offset_owner = {}
    for rank in ranks:
        for e in ranks[rank].get('write_reduce_l1', []):
            off = e['local_offset']
            owner = rank % l0_size
            if off not in offset_owner:
                offset_owner[off] = owner
            off = e['remote_offset']
            remote_owner = e['remote_rank'] % l0_size
            if off not in offset_owner:
                offset_owner[off] = remote_owner
        for e in ranks[rank].get('write_reduce_l2', []):
            off = e['local_offset']
            owner = rank % l0_size
            if off not in offset_owner:
                offset_owner[off] = owner
            off = e['remote_offset']
            remote_owner = e['remote_rank'] % l0_size
            if off not in offset_owner:
                offset_owner[off] = remote_owner
        for e in ranks[rank].get('write', []):
            if e.get('local_buf', 'INPUT') == 'CCL':
                off = e['local_offset']
                owner = rank % l0_size
                if off not in offset_owner:
                    offset_owner[off] = owner
            if e.get('remote_buf', 'CCL') == 'CCL':
                off = e['remote_offset']
                remote_owner = e['remote_rank'] % l0_size
                if off not in offset_owner:
                    offset_owner[off] = remote_owner
        for e in ranks[rank].get('read', []):
            if e.get('local_buf', 'CCL') == 'CCL':
                off = e['local_offset']
                owner = rank % l0_size
                if off not in offset_owner:
                    offset_owner[off] = owner
            if e.get('remote_buf', 'CCL') == 'CCL':
                off = e['remote_offset']
                remote_owner = e['remote_rank'] % l0_size
                if off not in offset_owner:
                    offset_owner[off] = remote_owner
    for rank in ranks:
        for e in ranks[rank].get('localreduce', []):
            for key in ('src_offset', 'dst_offset'):
                off = e[key]
                if off not in offset_owner:
                    offset_owner[off] = None
        for e in ranks[rank].get('output', []):
            off = e['src_offset']
            if off not in offset_owner:
                offset_owner[off] = None
    owned_offsets = sorted(off for off, ow in offset_owner.items() if ow is not None)
    for off in sorted(offset_owner.keys()):
        if offset_owner[off] is not None:
            continue
        best = None
        for owned in owned_offsets:
            if owned <= off:
                best = owned
            else:
                break
        if best is not None:
            offset_owner[off] = offset_owner[best]
        else:
            offset_owner[off] = 0
    return offset_owner


def owner_colors(owner, l0_size):
    return OWNER_COLORS[owner % len(OWNER_COLORS)]


def fmt_off(off):
    return f"{off}"


def parse_log(filepath):
    ranks = {}
    current_rank = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'rankId is : (\d+)', line)
            if m:
                current_rank = int(m.group(1))
                ranks[current_rank] = {
                    'localcopy': [],
                    'localreduce': [],
                    'write': [],
                    'write_reduce_l1': [],
                    'write_reduce_l2': [],
                    'read': [],
                    'output': [],
                }
                continue
            if current_rank is None:
                continue
            m2 = re.match(r'rankIdx:(\d+),\s*threadIdx:(\d+),\s*\[(.+)\]', line)
            if not m2:
                continue
            rank = int(m2.group(1))
            action = m2.group(3)
            if rank != current_rank:
                continue

            if 'LocalCopy' in action and 'INPUT' in action and 'CCL' in action and 'OUTPUT' not in action:
                m3 = re.search(
                    r'srcSlice=DataSlice\[BufferType::INPUT, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]'
                    r'.*dstSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]',
                    action)
                if m3:
                    ranks[current_rank]['localcopy'].append({
                        'type': 'LocalCopy',
                        'src_offset': int(m3.group(1), 16),
                        'src_size': int(m3.group(2), 16),
                        'dst_offset': int(m3.group(3), 16),
                        'dst_size': int(m3.group(4), 16),
                    })

            elif 'LocalReduce' in action:
                m3 = re.search(
                    r'srcSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]'
                    r'.*dstSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]',
                    action)
                if m3:
                    ranks[current_rank]['localreduce'].append({
                        'type': 'LocalReduce',
                        'src_offset': int(m3.group(1), 16),
                        'src_size': int(m3.group(2), 16),
                        'dst_offset': int(m3.group(3), 16),
                        'dst_size': int(m3.group(4), 16),
                    })

            elif 'WriteReduce' in action:
                m3 = re.search(
                    r'remoteRank=(\d+)'
                    r'.*localSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]'
                    r'.*remoteSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]',
                    action)
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
                    ranks[current_rank]['write_reduce_l1'].append(entry)

            elif 'Write]:' in action:
                m3 = re.search(
                    r'remoteRank=(\d+)'
                    r'.*localSlice=DataSlice\[BufferType::(\w+), offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]'
                    r'.*remoteSlice=DataSlice\[BufferType::(\w+), offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]',
                    action)
                if m3:
                    remote = int(m3.group(1))
                    local_buf = m3.group(2)
                    ranks[current_rank]['write'].append({
                        'type': 'Write',
                        'remote_rank': remote,
                        'local_buf': local_buf,
                        'local_offset': int(m3.group(3), 16),
                        'local_size': int(m3.group(4), 16),
                        'remote_buf': m3.group(5),
                        'remote_offset': int(m3.group(6), 16),
                        'remote_size': int(m3.group(7), 16),
                    })

            elif 'LocalCopy' in action and 'CCL' in action and 'OUTPUT' in action:
                m3 = re.search(
                    r'srcSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]'
                    r'.*dstSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]'
                    r'.*dstSlice=DataSlice\[BufferType::OUTPUT, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]',
                    action)
                if not m3:
                    m3 = re.search(
                        r'srcSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]'
                        r'.*dstSlice=DataSlice\[BufferType::OUTPUT, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]',
                        action)
                if m3:
                    ranks[current_rank]['output'].append({
                        'type': 'Output',
                        'src_offset': int(m3.group(1), 16),
                        'src_size': int(m3.group(2), 16),
                        'dst_offset': int(m3.group(3), 16),
                        'dst_size': int(m3.group(4), 16),
                    })

            elif 'Read]:' in action:
                m3 = re.search(
                    r'remoteRank=(\d+)'
                    r'.*localSlice=DataSlice\[BufferType::(\w+), offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]'
                    r'.*remoteSlice=DataSlice\[BufferType::(\w+), offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]',
                    action)
                if m3:
                    remote = int(m3.group(1))
                    local_buf = m3.group(2)
                    ranks[current_rank]['read'].append({
                        'type': 'Read',
                        'remote_rank': remote,
                        'local_buf': local_buf,
                        'local_offset': int(m3.group(3), 16),
                        'local_size': int(m3.group(4), 16),
                        'remote_buf': m3.group(5),
                        'remote_offset': int(m3.group(6), 16),
                        'remote_size': int(m3.group(7), 16),
                    })

    l0, l1, l2 = infer_topology(ranks)
    l1_entries = []
    l2_entries = []
    for rank in ranks:
        for e in ranks[rank]['write_reduce_l1']:
            _, s1, s2 = _rank_to_pos(rank, l0, l1, l2)
            _, d1, d2 = _rank_to_pos(e['remote_rank'], l0, l1, l2)
            if s2 != d2:
                l2_entries.append((rank, e))
            elif s1 != d1:
                l1_entries.append((rank, e))
    ranks_out = {}
    for rank in ranks:
        ranks_out[rank] = {
            'localcopy': ranks[rank]['localcopy'],
            'localreduce': ranks[rank]['localreduce'],
            'write': ranks[rank]['write'],
            'write_reduce_l1': [],
            'write_reduce_l2': [],
            'read': ranks[rank]['read'],
            'output': ranks[rank]['output'],
        }
    for rank, e in l1_entries:
        ranks_out[rank]['write_reduce_l1'].append(e)
    for rank, e in l2_entries:
        ranks_out[rank]['write_reduce_l2'].append(e)

    return ranks_out, l0, l1, l2


def esc(t):
    return html_mod.escape(str(t))


def rect(x, y, w, h, fill, stroke="#333", rx=2, sw=1):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" rx="{rx}" stroke-width="{sw}"/>\n'


def text(x, y, t, sz=10, fill="#333", anchor="middle", bold=False):
    fw = ' font-weight="bold"' if bold else ''
    return f'<text x="{x}" y="{y}" font-size="{sz}" fill="{fill}" text-anchor="{anchor}" font-family="Consolas,Arial,sans-serif"{fw}>{esc(t)}</text>\n'


def arrow_line(x1, y1, x2, y2, color, w=2, dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ''
    dx = x2 - x1
    dy = y2 - y1
    ln = (dx * dx + dy * dy) ** 0.5
    if ln > 0:
        ux, uy = dx / ln, dy / ln
    else:
        ux, uy = 0, 1
    px1 = x2 - ux * 5 + uy * 3
    py1 = y2 - uy * 5 - ux * 3
    px2 = x2 - ux * 5 - uy * 3
    py2 = y2 - uy * 5 + ux * 3
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}"{da}/>\n'
            f'<polygon points="{x2},{y2} {px1},{py1} {px2},{py2}" fill="{color}"/>\n')


def curved_arrow(x1, y1, x2, y2, color, w=1.5, curve=15, dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ''
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2 - curve
    dx = x2 - x1
    dy = y2 - y1
    ln = (dx * dx + dy * dy) ** 0.5
    if ln > 0:
        ux, uy = dx / ln, dy / ln
    else:
        ux, uy = 1, 0
    t = 0.95
    tx = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * mx + t ** 2 * x2
    ty = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * my + t ** 2 * y2
    tdx = tx - x2
    tdy = ty - y2
    tln = (tdx * tdx + tdy * tdy) ** 0.5
    if tln > 0:
        nux, nuy = tdx / tln, tdy / tln
    else:
        nux, nuy = -ux, -uy
    px1 = x2 + nux * 6 + nuy * 3
    py1 = y2 + nuy * 6 - nux * 3
    px2 = x2 + nux * 6 - nuy * 3
    py2 = y2 + nuy * 6 + nux * 3
    return (f'<path d="M{x1},{y1} Q{mx},{my} {x2},{y2}" fill="none" stroke="{color}" stroke-width="{w}"{da}/>\n'
            f'<polygon points="{x2},{y2} {px1},{py1} {px2},{py2}" fill="{color}"/>\n')


def bezier_arrow(x1, y1, cx1, cy1, cx2, cy2, x2, y2, color, w=2, dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ''
    dx = x2 - cx2
    dy = y2 - cy2
    ln = (dx * dx + dy * dy) ** 0.5
    if ln > 0:
        ux, uy = dx / ln, dy / ln
    else:
        ux, uy = 0, 1
    px1 = x2 - ux * 5 + uy * 3
    py1 = y2 - uy * 5 - ux * 3
    px2 = x2 - ux * 5 - uy * 3
    py2 = y2 - uy * 5 + ux * 3
    return (f'<path d="M{x1},{y1} C{cx1},{cy1} {cx2},{cy2} {x2},{y2}" fill="none" stroke="{color}" stroke-width="{w}"{da}/>\n'
            f'<polygon points="{x2},{y2} {px1},{py1} {px2},{py2}" fill="{color}"/>\n')


def draw_buffer_row(s, bar_x0, label_w, chunk_w, bar_h, y, label, label_color, offsets, l0_size, offset_owner=None, chunks_fill_stroke=None):
    s += text(bar_x0 + label_w // 2, y + bar_h // 2 + 6, label, 13, label_color, bold=True)
    num_chunks = len(offsets)
    s += rect(bar_x0 + label_w, y, chunk_w * num_chunks, bar_h, "#F5F5F5", "#9E9E9E", 2, 1)
    for i in range(num_chunks):
        cx = bar_x0 + label_w + i * chunk_w
        off = offsets[i]
        if chunks_fill_stroke:
            fill_c, stroke_c = chunks_fill_stroke[i]
        else:
            owner = offset_owner.get(off, i % l0_size) if offset_owner else i % l0_size
            fill_c, stroke_c = owner_colors(owner, l0_size)
        s += rect(cx, y, chunk_w, bar_h, fill_c, stroke_c, 0, 1)
        s += text(cx + chunk_w / 2, y + bar_h / 2 + 4, fmt_off(off), 8, stroke_c, bold=True)
    return s


def gen_level0_svg(ranks, l0_size, l1_size, l2_size, ccl_offsets, input_offsets, ccl_owner, input_owner, include_output=False):
    num_ccl = len(ccl_offsets)
    num_input = len(input_offsets)
    total_ranks = len(ranks)
    W = 2600
    bar_x0 = 140
    bar_total_w = W - 180
    bar_h = 36
    label_w = 110
    data_w = bar_total_w - label_w
    ccl_chunk_w = data_w / num_ccl
    input_chunk_w = data_w / num_input

    num_servers = l1_size * l2_size
    row_gap = 18
    rank_gap = 8
    srv_gap = 24
    rank_block_h = (2 if not include_output else 3) * bar_h + (2 if not include_output else 3) * row_gap + 20
    srv_h = l0_size * (rank_block_h + rank_gap) + 50
    H = 80 + num_servers * (srv_h + srv_gap) + 30

    LC_COLOR = "#2E7D32"
    LR_COLOR = "#7B1FA2"
    WR_COLOR = "#D32F2F"
    OUT_COLOR = "#00695C"

    ccl_off_to_idx = {off: i for i, off in enumerate(ccl_offsets)}
    input_off_to_idx = {off: i for i, off in enumerate(input_offsets)}

    s = f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
    s += rect(0, 0, W, H, "white", "white")
    s += text(W // 2, 22, "Level 0: Server内数据流 — INPUT(源) / CCL(源+目的)", 18, "#1a237e", bold=True)

    legend_y = 42
    legends = [
        (LC_COLOR, "━━", "LocalCopy: INPUT→CCL"),
        (LR_COLOR, "╌╌", "LocalReduce: CCL→CCL"),
        (WR_COLOR, "╌╌", "Write: INPUT/CCL→远端CCL"),
        ("#1976D2", "╌╌", "Read: 远端CCL→本地CCL/INPUT"),
    ]
    lx = 80
    for color, sym, desc in legends:
        s += text(lx, legend_y, f"{sym} {desc}", 11, color, "left", bold=True)
        lx += 300

    rank_layout = {}

    srv_idx = 0
    for l2 in range(l2_size):
        for l1 in range(l1_size):
            sy = 65 + srv_idx * (srv_h + srv_gap)
            s += rect(15, sy, W - 30, srv_h, "#E8F5E9", "#2E7D32", 6, 2)
            s += text(70, sy + 18, f"Pod{l2} Server{l1}", 14, "#1a237e", bold=True)

            for l0 in range(l0_size):
                rank = l2 * l1_size * l0_size + l1 * l0_size + l0
                if rank not in ranks:
                    continue
                ry = sy + 35 + l0 * (rank_block_h + rank_gap)

                s += rect(30, ry, W - 60, rank_block_h, "#FAFAFA", "#A5D6A7", 3, 1)
                s += text(70, ry + 14, f"R{rank}", 13, "#1a237e", bold=True)

                input_y = ry + 22
                ccl_y = input_y + bar_h + row_gap
                output_y = ccl_y + bar_h + row_gap if include_output else None

                s = draw_buffer_row(s, bar_x0, label_w, input_chunk_w, bar_h, input_y, "INPUT[源]", "#F57F17", input_offsets, l0_size, input_owner)
                s = draw_buffer_row(s, bar_x0, label_w, ccl_chunk_w, bar_h, ccl_y, "CCL[源+目的]", "#4A148C", ccl_offsets, l0_size, ccl_owner)

                for e in ranks[rank]['localcopy']:
                    src_start = e['src_offset']
                    src_end = src_start + e['src_size']
                    for off in input_offsets:
                        if off >= src_start and off < src_end:
                            idx = input_off_to_idx.get(off)
                            if idx is not None:
                                _, sc = owner_colors(input_owner.get(off, idx % l0_size), l0_size)
                                s += rect(bar_x0 + label_w + idx * input_chunk_w, input_y, input_chunk_w, bar_h, "#C8E6C9", sc, 0, 2)
                    dst_start = e['dst_offset']
                    dst_end = dst_start + e['dst_size']
                    for off in ccl_offsets:
                        if off >= dst_start and off < dst_end:
                            idx = ccl_off_to_idx.get(off)
                            if idx is not None:
                                _, sc = owner_colors(ccl_owner.get(off, idx % l0_size), l0_size)
                                s += rect(bar_x0 + label_w + idx * ccl_chunk_w, ccl_y, ccl_chunk_w, bar_h, "#C8E6C9", sc, 0, 2)

                for e in ranks[rank]['localreduce']:
                    src_start = e['src_offset']
                    src_end = src_start + e['src_size']
                    for off in ccl_offsets:
                        if off >= src_start and off < src_end:
                            idx = ccl_off_to_idx.get(off)
                            if idx is not None:
                                _, sc = owner_colors(ccl_owner.get(off, idx % l0_size), l0_size)
                                s += rect(bar_x0 + label_w + idx * ccl_chunk_w, ccl_y, ccl_chunk_w, bar_h, "#FFCC80", sc, 0, 2)
                    dst_start = e['dst_offset']
                    dst_end = dst_start + e['dst_size']
                    for off in ccl_offsets:
                        if off >= dst_start and off < dst_end:
                            idx = ccl_off_to_idx.get(off)
                            if idx is not None:
                                _, sc = owner_colors(ccl_owner.get(off, idx % l0_size), l0_size)
                                s += rect(bar_x0 + label_w + idx * ccl_chunk_w, ccl_y, ccl_chunk_w, bar_h, "#E1BEE7", sc, 0, 2)

                for e in ranks[rank]['write']:
                    src_start = e['local_offset']
                    src_end = src_start + e['local_size']
                    if e.get('local_buf', 'INPUT') == 'INPUT':
                        for off in input_offsets:
                            if off >= src_start and off < src_end:
                                idx = input_off_to_idx.get(off)
                                if idx is not None:
                                    _, sc = owner_colors(input_owner.get(off, idx % l0_size), l0_size)
                                    s += rect(bar_x0 + label_w + idx * input_chunk_w, input_y, input_chunk_w, bar_h, "#FFCC80", sc, 0, 2)
                    else:
                        for off in ccl_offsets:
                            if off >= src_start and off < src_end:
                                idx = ccl_off_to_idx.get(off)
                                if idx is not None:
                                    _, sc = owner_colors(ccl_owner.get(off, idx % l0_size), l0_size)
                                    s += rect(bar_x0 + label_w + idx * ccl_chunk_w, ccl_y, ccl_chunk_w, bar_h, "#FFCC80", sc, 0, 2)

                for e in ranks[rank]['read']:
                    local_start = e['local_offset']
                    local_end = local_start + e['local_size']
                    if e.get('local_buf', 'CCL') == 'CCL':
                        for off in ccl_offsets:
                            if off >= local_start and off < local_end:
                                idx = ccl_off_to_idx.get(off)
                                if idx is not None:
                                    _, sc = owner_colors(ccl_owner.get(off, idx % l0_size), l0_size)
                                    s += rect(bar_x0 + label_w + idx * ccl_chunk_w, ccl_y, ccl_chunk_w, bar_h, "#B3E5FC", sc, 0, 2)
                    else:
                        for off in input_offsets:
                            if off >= local_start and off < local_end:
                                idx = input_off_to_idx.get(off)
                                if idx is not None:
                                    _, sc = owner_colors(input_owner.get(off, idx % l0_size), l0_size)
                                    s += rect(bar_x0 + label_w + idx * input_chunk_w, input_y, input_chunk_w, bar_h, "#B3E5FC", sc, 0, 2)

                if include_output:
                    s = draw_buffer_row(s, bar_x0, label_w, ccl_chunk_w, bar_h, output_y, "OUTPUT[目的]", "#00695C", ccl_offsets, l0_size, ccl_owner)
                    for e in ranks[rank]['output']:
                        src_idx = ccl_off_to_idx.get(e['src_offset'])
                        if src_idx is None:
                            continue
                        _, sc = owner_colors(ccl_owner.get(e['src_offset'], src_idx % l0_size), l0_size)
                        s += rect(bar_x0 + label_w + src_idx * ccl_chunk_w, output_y, ccl_chunk_w, bar_h, "#B2DFDB", sc, 0, 2)

                rank_layout[rank] = {
                    'input_y': input_y,
                    'ccl_y': ccl_y,
                    'output_y': output_y,
                    'ry': ry,
                }

            srv_idx += 1

    for rank in range(total_ranks):
        if rank not in ranks or rank not in rank_layout:
            continue
        lay = rank_layout[rank]
        input_y = lay['input_y']
        ccl_y = lay['ccl_y']

        for e in ranks[rank]['localcopy']:
            src_idx = input_off_to_idx.get(e['src_offset'])
            dst_idx = ccl_off_to_idx.get(e['dst_offset'])
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * input_chunk_w + input_chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * ccl_chunk_w + ccl_chunk_w / 2
            s += arrow_line(sx, input_y + bar_h, dx, ccl_y, LC_COLOR, 2.5)

        for e in ranks[rank]['localreduce']:
            src_idx = ccl_off_to_idx.get(e['src_offset'])
            dst_idx = ccl_off_to_idx.get(e['dst_offset'])
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * ccl_chunk_w + ccl_chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * ccl_chunk_w + ccl_chunk_w / 2
            s += curved_arrow(sx, ccl_y + 2, dx, ccl_y + 2, LR_COLOR, 1.8, curve=20, dash="4,2")

    READ_COLOR = "#1976D2"

    write_by_pair = {}
    for rank in range(total_ranks):
        if rank not in ranks:
            continue
        for e in ranks[rank]['write']:
            remote = e['remote_rank']
            key = (rank, remote)
            if key not in write_by_pair:
                write_by_pair[key] = []
            write_by_pair[key].append(e)

    pair_idx = 0
    for (src_rank, dst_rank), entries in sorted(write_by_pair.items()):
        if src_rank not in rank_layout or dst_rank not in rank_layout:
            continue
        _, l1s, l2s = _rank_to_pos(src_rank, l0_size, l1_size, l2_size)
        _, l1d, l2d = _rank_to_pos(dst_rank, l0_size, l1_size, l2_size)
        if l1s != l1d or l2s != l2d:
            continue

        src_lay = rank_layout[src_rank]
        dst_lay = rank_layout[dst_rank]

        src_dev = src_rank % l0_size
        dst_dev = dst_rank % l0_size

        right_x = bar_x0 + label_w + num_input * input_chunk_w + 8
        bend_offset = 15 + pair_idx * 18

        for e in entries:
            if e.get('local_buf', 'INPUT') == 'INPUT':
                src_idx = input_off_to_idx.get(e['local_offset'])
                sx = bar_x0 + label_w + src_idx * input_chunk_w + input_chunk_w / 2 if src_idx is not None else None
                y1 = src_lay['input_y'] + bar_h
            else:
                src_idx = ccl_off_to_idx.get(e['local_offset'])
                sx = bar_x0 + label_w + src_idx * ccl_chunk_w + ccl_chunk_w / 2 if src_idx is not None else None
                y1 = src_lay['ccl_y'] + bar_h
            dst_idx = ccl_off_to_idx.get(e['remote_offset'])
            dx = bar_x0 + label_w + dst_idx * ccl_chunk_w + ccl_chunk_w / 2 if dst_idx is not None else None
            y2 = dst_lay['ccl_y']
            if sx is None or dx is None:
                continue
            s += bezier_arrow(
                sx, y1,
                sx + bend_offset, (y1 + y2) / 2,
                dx + bend_offset, (y1 + y2) / 2,
                dx, y2,
                WR_COLOR, 1.5, dash="4,2"
            )

        s += text(right_x + bend_offset + 2, (src_lay['input_y'] + dst_lay['ccl_y'] + bar_h) / 2,
                  f"R{src_dev}→R{dst_dev} W×{len(entries)}", 8, WR_COLOR, "left", bold=True)

        pair_idx += 1

    read_by_pair = {}
    for rank in range(total_ranks):
        if rank not in ranks:
            continue
        for e in ranks[rank]['read']:
            remote = e['remote_rank']
            key = (remote, rank)
            if key not in read_by_pair:
                read_by_pair[key] = []
            read_by_pair[key].append(e)

    for (src_rank, dst_rank), entries in sorted(read_by_pair.items()):
        if src_rank not in rank_layout or dst_rank not in rank_layout:
            continue
        _, l1s, l2s = _rank_to_pos(src_rank, l0_size, l1_size, l2_size)
        _, l1d, l2d = _rank_to_pos(dst_rank, l0_size, l1_size, l2_size)
        if l1s != l1d or l2s != l2d:
            continue

        src_lay = rank_layout[src_rank]
        dst_lay = rank_layout[dst_rank]

        src_dev = src_rank % l0_size
        dst_dev = dst_rank % l0_size

        right_x = bar_x0 + label_w + num_input * input_chunk_w + 8
        bend_offset = 15 + pair_idx * 18

        for e in entries:
            src_idx = ccl_off_to_idx.get(e['remote_offset'])
            sx = bar_x0 + label_w + src_idx * ccl_chunk_w + ccl_chunk_w / 2 if src_idx is not None else None
            y1 = src_lay['ccl_y'] + bar_h
            if e.get('local_buf', 'CCL') == 'CCL':
                dst_idx = ccl_off_to_idx.get(e['local_offset'])
                dx = bar_x0 + label_w + dst_idx * ccl_chunk_w + ccl_chunk_w / 2 if dst_idx is not None else None
                y2 = dst_lay['ccl_y']
            else:
                dst_idx = input_off_to_idx.get(e['local_offset'])
                dx = bar_x0 + label_w + dst_idx * input_chunk_w + input_chunk_w / 2 if dst_idx is not None else None
                y2 = dst_lay['input_y']
            if sx is None or dx is None:
                continue
            s += bezier_arrow(
                sx, y1,
                sx + bend_offset, (y1 + y2) / 2,
                dx + bend_offset, (y1 + y2) / 2,
                dx, y2,
                READ_COLOR, 1.5, dash="4,2"
            )

        s += text(right_x + bend_offset + 2, (src_lay['ccl_y'] + dst_lay['ccl_y'] + bar_h) / 2,
                  f"R{src_dev}→R{dst_dev} Rd×{len(entries)}", 8, READ_COLOR, "left", bold=True)

        pair_idx += 1

    if include_output:
        for rank in range(total_ranks):
            if rank not in ranks or rank not in rank_layout:
                continue
            lay = rank_layout[rank]
            ccl_y = lay['ccl_y']
            output_y = lay['output_y']
            if output_y is None:
                continue
            for e in ranks[rank]['output']:
                src_idx = ccl_off_to_idx.get(e['src_offset'])
                if src_idx is None:
                    continue
                sx = bar_x0 + label_w + src_idx * ccl_chunk_w + ccl_chunk_w / 2
                dx = bar_x0 + label_w + src_idx * ccl_chunk_w + ccl_chunk_w / 2
                bend = 20
                s += bezier_arrow(sx, ccl_y + bar_h, sx + bend, ccl_y + bar_h, dx + bend, output_y, dx, output_y, OUT_COLOR, 2.5)

    return s + "</svg>"


def gen_level1_svg(ranks, l0_size, l1_size, l2_size, ccl_offsets, ccl_owner, include_output=False):
    num_chunks = len(ccl_offsets)
    total_ranks = len(ranks)
    W = 2600
    bar_x0 = 140
    bar_total_w = W - 180
    bar_h = 36
    label_w = 110
    data_w = bar_total_w - label_w
    chunk_w = data_w / num_chunks

    OUT_COLOR = "#00695C"
    num_servers = l1_size * l2_size
    row_gap = 18
    rank_gap = 8
    srv_gap = 24
    rank_block_h = (1 if not include_output else 2) * bar_h + (1 if not include_output else 2) * row_gap + 20
    srv_h = l0_size * (rank_block_h + rank_gap) + 50
    H = 80 + num_servers * (srv_h + srv_gap) + 30

    off_to_idx = {off: i for i, off in enumerate(ccl_offsets)}

    title = "Level 1: Server间数据流(同Pod) — CCL[源+目的]"
    if include_output:
        title = "Level 1: Server间数据流(同Pod) + Output — CCL[源+目的] / OUTPUT[目的]"

    s = f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
    s += rect(0, 0, W, H, "white", "white")
    s += text(W // 2, 22, title, 18, "#0D47A1", bold=True)
    legend_text = "━━ WriteReduce: CCL→远端CCL (跨Server同Pod) | 橙色高亮=源块"
    if include_output:
        legend_text += " | ━━ Output: CCL→OUTPUT"
    s += text(W // 2, 42, legend_text, 11, "#0D47A1")

    rank_layout = {}

    srv_idx = 0
    for l2 in range(l2_size):
        for l1 in range(l1_size):
            sy = 65 + srv_idx * (srv_h + srv_gap)
            s += rect(15, sy, W - 30, srv_h, "#E3F2FD", "#1565C0", 6, 2)
            s += text(70, sy + 18, f"Pod{l2} Server{l1}", 14, "#0D47A1", bold=True)

            for l0 in range(l0_size):
                rank = l2 * l1_size * l0_size + l1 * l0_size + l0
                if rank not in ranks:
                    continue
                ry = sy + 35 + l0 * (rank_block_h + rank_gap)

                s += rect(30, ry, W - 60, rank_block_h, "#FAFAFA", "#90CAF9", 3, 1)
                s += text(70, ry + 14, f"R{rank}", 13, "#0D47A1", bold=True)

                ccl_y = ry + 22
                output_y = ccl_y + bar_h + row_gap if include_output else None

                s = draw_buffer_row(s, bar_x0, label_w, chunk_w, bar_h, ccl_y, "CCL[源+目的]", "#4A148C", ccl_offsets, l0_size, ccl_owner)
                if include_output:
                    s = draw_buffer_row(s, bar_x0, label_w, chunk_w, bar_h, output_y, "OUTPUT[目的]", "#00695C", ccl_offsets, l0_size, ccl_owner)
                    for e in ranks[rank]['output']:
                        src_idx = off_to_idx.get(e['src_offset'])
                        if src_idx is None:
                            continue
                        _, sc = owner_colors(ccl_owner.get(e['src_offset'], src_idx % l0_size), l0_size)
                        s += rect(bar_x0 + label_w + src_idx * chunk_w, output_y, chunk_w, bar_h, "#B2DFDB", sc, 0, 2)

                for e in ranks[rank]['write_reduce_l1']:
                    src_idx = off_to_idx.get(e['local_offset'])
                    if src_idx is None:
                        continue
                    _, sc = owner_colors(ccl_owner.get(e['local_offset'], src_idx % l0_size), l0_size)
                    s += rect(bar_x0 + label_w + src_idx * chunk_w, ccl_y, chunk_w, bar_h, "#FFCC80", sc, 0, 2)

                for e in ranks[rank]['write']:
                    if e.get('local_buf', 'INPUT') == 'CCL':
                        src_idx = off_to_idx.get(e['local_offset'])
                        if src_idx is not None:
                            _, sc = owner_colors(ccl_owner.get(e['local_offset'], src_idx % l0_size), l0_size)
                            s += rect(bar_x0 + label_w + src_idx * chunk_w, ccl_y, chunk_w, bar_h, "#FFCC80", sc, 0, 2)
                    if e.get('remote_buf', 'CCL') == 'CCL':
                        dst_idx = off_to_idx.get(e['remote_offset'])
                        if dst_idx is not None:
                            _, sc = owner_colors(ccl_owner.get(e['remote_offset'], dst_idx % l0_size), l0_size)
                            s += rect(bar_x0 + label_w + dst_idx * chunk_w, ccl_y, chunk_w, bar_h, "#FFE0B2", sc, 0, 2)

                for e in ranks[rank]['read']:
                    if e.get('local_buf', 'CCL') == 'CCL':
                        dst_idx = off_to_idx.get(e['local_offset'])
                        if dst_idx is not None:
                            _, sc = owner_colors(ccl_owner.get(e['local_offset'], dst_idx % l0_size), l0_size)
                            s += rect(bar_x0 + label_w + dst_idx * chunk_w, ccl_y, chunk_w, bar_h, "#B3E5FC", sc, 0, 2)
                    if e.get('remote_buf', 'CCL') == 'CCL':
                        src_idx = off_to_idx.get(e['remote_offset'])
                        if src_idx is not None:
                            _, sc = owner_colors(ccl_owner.get(e['remote_offset'], src_idx % l0_size), l0_size)
                            s += rect(bar_x0 + label_w + src_idx * chunk_w, ccl_y, chunk_w, bar_h, "#BBDEFB", sc, 0, 2)

                rank_layout[rank] = {
                    'ccl_y': ccl_y,
                    'output_y': output_y,
                    'ry': ry,
                }

            srv_idx += 1

    wr_by_pair = {}
    for rank in range(total_ranks):
        if rank not in ranks:
            continue
        for e in ranks[rank]['write_reduce_l1']:
            remote = e['remote_rank']
            key = (rank, remote)
            if key not in wr_by_pair:
                wr_by_pair[key] = []
            wr_by_pair[key].append(e)

    pair_idx = 0
    for (src_rank, dst_rank), entries in sorted(wr_by_pair.items()):
        if src_rank not in rank_layout or dst_rank not in rank_layout:
            continue
        src_ccl_y = rank_layout[src_rank]['ccl_y']
        dst_ccl_y = rank_layout[dst_rank]['ccl_y']
        bend_offset = 15 + pair_idx * 16

        for e in entries:
            src_idx = off_to_idx.get(e['local_offset'])
            dst_idx = off_to_idx.get(e['remote_offset'])
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * chunk_w + chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * chunk_w + chunk_w / 2
            _, sc = owner_colors(ccl_owner.get(e['local_offset'], src_idx % l0_size), l0_size)
            y1 = src_ccl_y + bar_h
            y2 = dst_ccl_y
            s += bezier_arrow(
                sx, y1,
                sx + bend_offset, (y1 + y2) / 2,
                dx + bend_offset, (y1 + y2) / 2,
                dx, y2,
                sc, 2, dash="6,3"
            )

        right_x = bar_x0 + label_w + num_chunks * chunk_w + 8
        s += text(right_x + bend_offset + 2, (src_ccl_y + dst_ccl_y + bar_h) / 2,
                  f"R{src_rank}→R{dst_rank} WR×{len(entries)}", 8, "#1565C0", "left", bold=True)
        pair_idx += 1

    READ_COLOR = "#1976D2"
    write_ccl_by_pair = {}
    for rank in range(total_ranks):
        if rank not in ranks:
            continue
        for e in ranks[rank]['write']:
            if e.get('local_buf', 'INPUT') == 'CCL':
                remote = e['remote_rank']
                key = (rank, remote)
                if key not in write_ccl_by_pair:
                    write_ccl_by_pair[key] = []
                write_ccl_by_pair[key].append(e)

    for (src_rank, dst_rank), entries in sorted(write_ccl_by_pair.items()):
        if src_rank not in rank_layout or dst_rank not in rank_layout:
            continue
        _, l1s, l2s = _rank_to_pos(src_rank, l0_size, l1_size, l2_size)
        _, l1d, l2d = _rank_to_pos(dst_rank, l0_size, l1_size, l2_size)
        if l1s != l1d or l2s != l2d:
            continue
        src_ccl_y = rank_layout[src_rank]['ccl_y']
        dst_ccl_y = rank_layout[dst_rank]['ccl_y']
        bend_offset = 15 + pair_idx * 16

        for e in entries:
            src_idx = off_to_idx.get(e['local_offset'])
            dst_idx = off_to_idx.get(e['remote_offset'])
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * chunk_w + chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * chunk_w + chunk_w / 2
            y1 = src_ccl_y + bar_h
            y2 = dst_ccl_y
            s += bezier_arrow(
                sx, y1,
                sx + bend_offset, (y1 + y2) / 2,
                dx + bend_offset, (y1 + y2) / 2,
                dx, y2,
                "#D32F2F", 1.5, dash="4,2"
            )

        right_x = bar_x0 + label_w + num_chunks * chunk_w + 8
        s += text(right_x + bend_offset + 2, (src_ccl_y + dst_ccl_y + bar_h) / 2,
                  f"R{src_rank}→R{dst_rank} W×{len(entries)}", 8, "#D32F2F", "left", bold=True)
        pair_idx += 1

    read_by_pair = {}
    for rank in range(total_ranks):
        if rank not in ranks:
            continue
        for e in ranks[rank]['read']:
            remote = e['remote_rank']
            key = (remote, rank)
            if key not in read_by_pair:
                read_by_pair[key] = []
            read_by_pair[key].append(e)

    for (src_rank, dst_rank), entries in sorted(read_by_pair.items()):
        if src_rank not in rank_layout or dst_rank not in rank_layout:
            continue
        _, l1s, l2s = _rank_to_pos(src_rank, l0_size, l1_size, l2_size)
        _, l1d, l2d = _rank_to_pos(dst_rank, l0_size, l1_size, l2_size)
        if l1s != l1d or l2s != l2d:
            continue
        src_ccl_y = rank_layout[src_rank]['ccl_y']
        dst_ccl_y = rank_layout[dst_rank]['ccl_y']
        bend_offset = 15 + pair_idx * 16

        for e in entries:
            src_idx = off_to_idx.get(e['remote_offset'])
            dst_idx = off_to_idx.get(e['local_offset']) if e.get('local_buf', 'CCL') == 'CCL' else None
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * chunk_w + chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * chunk_w + chunk_w / 2
            y1 = src_ccl_y + bar_h
            y2 = dst_ccl_y
            s += bezier_arrow(
                sx, y1,
                sx + bend_offset, (y1 + y2) / 2,
                dx + bend_offset, (y1 + y2) / 2,
                dx, y2,
                READ_COLOR, 1.5, dash="4,2"
            )

        right_x = bar_x0 + label_w + num_chunks * chunk_w + 8
        s += text(right_x + bend_offset + 2, (src_ccl_y + dst_ccl_y + bar_h) / 2,
                  f"R{src_rank}→R{dst_rank} Rd×{len(entries)}", 8, READ_COLOR, "left", bold=True)
        pair_idx += 1

    if include_output:
        for rank in range(total_ranks):
            if rank not in ranks or rank not in rank_layout:
                continue
            lay = rank_layout[rank]
            ccl_y = lay['ccl_y']
            output_y = lay.get('output_y')
            if output_y is None:
                continue
            for e in ranks[rank]['output']:
                src_idx = off_to_idx.get(e['src_offset'])
                if src_idx is None:
                    continue
                sx = bar_x0 + label_w + src_idx * chunk_w + chunk_w / 2
                dx = bar_x0 + label_w + src_idx * chunk_w + chunk_w / 2
                bend = 20
                s += bezier_arrow(sx, ccl_y + bar_h, sx + bend, ccl_y + bar_h, dx + bend, output_y, dx, output_y, OUT_COLOR, 2.5)

    return s + "</svg>"


def gen_level2_svg(ranks, l0_size, l1_size, l2_size, ccl_offsets, ccl_owner, include_output=False):
    num_chunks = len(ccl_offsets)
    total_ranks = len(ranks)
    W = 2600
    bar_x0 = 140
    bar_total_w = W - 180
    bar_h = 36
    label_w = 110
    data_w = bar_total_w - label_w
    chunk_w = data_w / num_chunks

    num_pods = l2_size
    row_gap = 18
    rank_gap = 8
    pod_gap = 24
    ranks_per_pod = l1_size * l0_size
    rank_block_h = (1 if not include_output else 2) * bar_h + (1 if not include_output else 2) * row_gap + 20
    pod_h = ranks_per_pod * (rank_block_h + rank_gap) + 50
    H = 80 + num_pods * (pod_h + pod_gap) + 30

    WR_COLOR = "#BF360C"
    OUT_COLOR = "#00695C"

    off_to_idx = {off: i for i, off in enumerate(ccl_offsets)}

    title = "Level 2: Pod间数据流 — CCL[源+目的]"
    if include_output:
        title = "Level 2: Pod间数据流 + Output — CCL[源+目的] / OUTPUT[目的]"

    s = f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
    s += rect(0, 0, W, H, "white", "white")
    s += text(W // 2, 22, title, 18, "#BF360C", bold=True)

    legend_y = 42
    legends = [
        (WR_COLOR, "━━", "WriteReduce: CCL→远端CCL (跨Pod)"),
    ]
    if include_output:
        legends.append((OUT_COLOR, "━━", "Output: CCL→OUTPUT (最终输出)"))
    lx = 80
    for color, sym, desc in legends:
        s += text(lx, legend_y, f"{sym} {desc}", 11, color, "left", bold=True)
        lx += 350

    rank_layout = {}

    for l2 in range(l2_size):
        py = 65 + l2 * (pod_h + pod_gap)
        s += rect(15, py, W - 30, pod_h, "#FBE9E7", "#D84315", 6, 2)
        s += text(70, py + 18, f"Pod{l2}", 14, "#BF360C", bold=True)

        ri = 0
        for l1 in range(l1_size):
            for l0 in range(l0_size):
                rank = l2 * l1_size * l0_size + l1 * l0_size + l0
                if rank not in ranks:
                    continue
                ry = py + 35 + ri * (rank_block_h + rank_gap)

                s += rect(30, ry, W - 60, rank_block_h, "#FAFAFA", "#FFAB91", 3, 1)
                s += text(70, ry + 14, f"R{rank}", 13, "#BF360C", bold=True)

                ccl_y = ry + 22
                output_y = ccl_y + bar_h + row_gap if include_output else None

                s = draw_buffer_row(s, bar_x0, label_w, chunk_w, bar_h, ccl_y, "CCL[源+目的]", "#4A148C", ccl_offsets, l0_size, ccl_owner)
                if include_output:
                    s = draw_buffer_row(s, bar_x0, label_w, chunk_w, bar_h, output_y, "OUTPUT[目的]", "#00695C", ccl_offsets, l0_size, ccl_owner)
                    for e in ranks[rank]['output']:
                        src_idx = off_to_idx.get(e['src_offset'])
                        if src_idx is None:
                            continue
                        _, sc = owner_colors(ccl_owner.get(e['src_offset'], src_idx % l0_size), l0_size)
                        s += rect(bar_x0 + label_w + src_idx * chunk_w, output_y, chunk_w, bar_h, "#B2DFDB", sc, 0, 2)

                for e in ranks[rank]['write_reduce_l2']:
                    src_idx = off_to_idx.get(e['local_offset'])
                    if src_idx is None:
                        continue
                    _, sc = owner_colors(ccl_owner.get(e['local_offset'], src_idx % l0_size), l0_size)
                    s += rect(bar_x0 + label_w + src_idx * chunk_w, ccl_y, chunk_w, bar_h, "#FFCC80", sc, 0, 2)

                for e in ranks[rank]['write']:
                    if e.get('local_buf', 'INPUT') == 'CCL':
                        src_idx = off_to_idx.get(e['local_offset'])
                        if src_idx is not None:
                            _, sc = owner_colors(ccl_owner.get(e['local_offset'], src_idx % l0_size), l0_size)
                            s += rect(bar_x0 + label_w + src_idx * chunk_w, ccl_y, chunk_w, bar_h, "#FFCC80", sc, 0, 2)
                    if e.get('remote_buf', 'CCL') == 'CCL':
                        dst_idx = off_to_idx.get(e['remote_offset'])
                        if dst_idx is not None:
                            _, sc = owner_colors(ccl_owner.get(e['remote_offset'], dst_idx % l0_size), l0_size)
                            s += rect(bar_x0 + label_w + dst_idx * chunk_w, ccl_y, chunk_w, bar_h, "#FFE0B2", sc, 0, 2)

                for e in ranks[rank]['read']:
                    if e.get('local_buf', 'CCL') == 'CCL':
                        dst_idx = off_to_idx.get(e['local_offset'])
                        if dst_idx is not None:
                            _, sc = owner_colors(ccl_owner.get(e['local_offset'], dst_idx % l0_size), l0_size)
                            s += rect(bar_x0 + label_w + dst_idx * chunk_w, ccl_y, chunk_w, bar_h, "#B3E5FC", sc, 0, 2)
                    if e.get('remote_buf', 'CCL') == 'CCL':
                        src_idx = off_to_idx.get(e['remote_offset'])
                        if src_idx is not None:
                            _, sc = owner_colors(ccl_owner.get(e['remote_offset'], src_idx % l0_size), l0_size)
                            s += rect(bar_x0 + label_w + src_idx * chunk_w, ccl_y, chunk_w, bar_h, "#BBDEFB", sc, 0, 2)

                rank_layout[rank] = {
                    'ccl_y': ccl_y,
                    'output_y': output_y,
                    'ry': ry,
                }

                ri += 1

    wr_by_pair = {}
    for rank in range(total_ranks):
        if rank not in ranks:
            continue
        for e in ranks[rank]['write_reduce_l2']:
            remote = e['remote_rank']
            key = (rank, remote)
            if key not in wr_by_pair:
                wr_by_pair[key] = []
            wr_by_pair[key].append(e)

    pair_idx = 0
    for (src_rank, dst_rank), entries in sorted(wr_by_pair.items()):
        if src_rank not in rank_layout or dst_rank not in rank_layout:
            continue
        src_ccl_y = rank_layout[src_rank]['ccl_y']
        dst_ccl_y = rank_layout[dst_rank]['ccl_y']
        bend_offset = 15 + pair_idx * 16

        for e in entries:
            src_idx = off_to_idx.get(e['local_offset'])
            dst_idx = off_to_idx.get(e['remote_offset'])
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * chunk_w + chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * chunk_w + chunk_w / 2
            _, sc = owner_colors(ccl_owner.get(e['local_offset'], src_idx % l0_size), l0_size)
            y1 = src_ccl_y + bar_h
            y2 = dst_ccl_y
            s += bezier_arrow(
                sx, y1,
                sx + bend_offset, (y1 + y2) / 2,
                dx + bend_offset, (y1 + y2) / 2,
                dx, y2,
                sc, 2, dash="6,3"
            )

        right_x = bar_x0 + label_w + num_chunks * chunk_w + 8
        s += text(right_x + bend_offset + 2, (src_ccl_y + dst_ccl_y + bar_h) / 2,
                  f"R{src_rank}→R{dst_rank} WR×{len(entries)}", 8, "#BF360C", "left", bold=True)
        pair_idx += 1

    READ_COLOR = "#1976D2"
    write_ccl_by_pair = {}
    for rank in range(total_ranks):
        if rank not in ranks:
            continue
        for e in ranks[rank]['write']:
            if e.get('local_buf', 'INPUT') == 'CCL':
                remote = e['remote_rank']
                key = (rank, remote)
                if key not in write_ccl_by_pair:
                    write_ccl_by_pair[key] = []
                write_ccl_by_pair[key].append(e)

    for (src_rank, dst_rank), entries in sorted(write_ccl_by_pair.items()):
        if src_rank not in rank_layout or dst_rank not in rank_layout:
            continue
        _, l1s, l2s = _rank_to_pos(src_rank, l0_size, l1_size, l2_size)
        _, l1d, l2d = _rank_to_pos(dst_rank, l0_size, l1_size, l2_size)
        if l2s != l2d:
            continue
        src_ccl_y = rank_layout[src_rank]['ccl_y']
        dst_ccl_y = rank_layout[dst_rank]['ccl_y']
        bend_offset = 15 + pair_idx * 16

        for e in entries:
            src_idx = off_to_idx.get(e['local_offset'])
            dst_idx = off_to_idx.get(e['remote_offset'])
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * chunk_w + chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * chunk_w + chunk_w / 2
            y1 = src_ccl_y + bar_h
            y2 = dst_ccl_y
            s += bezier_arrow(
                sx, y1,
                sx + bend_offset, (y1 + y2) / 2,
                dx + bend_offset, (y1 + y2) / 2,
                dx, y2,
                "#D32F2F", 1.5, dash="4,2"
            )

        right_x = bar_x0 + label_w + num_chunks * chunk_w + 8
        s += text(right_x + bend_offset + 2, (src_ccl_y + dst_ccl_y + bar_h) / 2,
                  f"R{src_rank}→R{dst_rank} W×{len(entries)}", 8, "#D32F2F", "left", bold=True)
        pair_idx += 1

    read_by_pair = {}
    for rank in range(total_ranks):
        if rank not in ranks:
            continue
        for e in ranks[rank]['read']:
            remote = e['remote_rank']
            key = (remote, rank)
            if key not in read_by_pair:
                read_by_pair[key] = []
            read_by_pair[key].append(e)

    for (src_rank, dst_rank), entries in sorted(read_by_pair.items()):
        if src_rank not in rank_layout or dst_rank not in rank_layout:
            continue
        _, l1s, l2s = _rank_to_pos(src_rank, l0_size, l1_size, l2_size)
        _, l1d, l2d = _rank_to_pos(dst_rank, l0_size, l1_size, l2_size)
        if l2s != l2d:
            continue
        src_ccl_y = rank_layout[src_rank]['ccl_y']
        dst_ccl_y = rank_layout[dst_rank]['ccl_y']
        bend_offset = 15 + pair_idx * 16

        for e in entries:
            src_idx = off_to_idx.get(e['remote_offset'])
            dst_idx = off_to_idx.get(e['local_offset']) if e.get('local_buf', 'CCL') == 'CCL' else None
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * chunk_w + chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * chunk_w + chunk_w / 2
            y1 = src_ccl_y + bar_h
            y2 = dst_ccl_y
            s += bezier_arrow(
                sx, y1,
                sx + bend_offset, (y1 + y2) / 2,
                dx + bend_offset, (y1 + y2) / 2,
                dx, y2,
                READ_COLOR, 1.5, dash="4,2"
            )

        right_x = bar_x0 + label_w + num_chunks * chunk_w + 8
        s += text(right_x + bend_offset + 2, (src_ccl_y + dst_ccl_y + bar_h) / 2,
                  f"R{src_rank}→R{dst_rank} Rd×{len(entries)}", 8, READ_COLOR, "left", bold=True)
        pair_idx += 1

    if include_output:
        for rank in range(total_ranks):
            if rank not in ranks or rank not in rank_layout:
                continue
            lay = rank_layout[rank]
            ccl_y = lay['ccl_y']
            output_y = lay.get('output_y')
            if output_y is None:
                continue
            for e in ranks[rank]['output']:
                src_idx = off_to_idx.get(e['src_offset'])
                if src_idx is None:
                    continue
                sx = bar_x0 + label_w + src_idx * chunk_w + chunk_w / 2
                dx = bar_x0 + label_w + src_idx * chunk_w + chunk_w / 2
                bend = 20
                s += bezier_arrow(sx, ccl_y + bar_h, sx + bend, ccl_y + bar_h, dx + bend, output_y, dx, output_y, OUT_COLOR, 2.5)

    return s + "</svg>"


def gen_summary_svg(ranks, l0_size, l1_size, l2_size, ccl_offsets, input_offsets, ccl_owner, input_owner, lv_ccl_offsets, lv_ccl_owner):
    num_ccl = len(ccl_offsets)
    num_input = len(input_offsets)
    num_lv_ccl = len(lv_ccl_offsets)
    total_ranks = len(ranks)
    W = 3200
    bar_x0 = 140
    bar_total_w = W - 200
    bar_h = 28
    label_w = 100
    data_w = bar_total_w - label_w
    input_chunk_w = data_w / num_input
    ccl_chunk_w = data_w / num_ccl
    lv_ccl_chunk_w = data_w / num_lv_ccl if num_lv_ccl > 0 else data_w

    row_gap = 12
    rank_gap = 6
    srv_gap = 20
    pod_gap = 24
    rank_block_h = 3 * bar_h + 3 * row_gap + 16

    num_pods = l2_size
    ranks_per_pod = l1_size * l0_size
    srv_h = l0_size * (rank_block_h + rank_gap) + 40
    pod_h = l1_size * (srv_h + srv_gap) + 20
    H = 80 + num_pods * (pod_h + pod_gap) + 30

    LC_COLOR = "#2E7D32"
    LR_COLOR = "#7B1FA2"
    WR_COLOR = "#D32F2F"
    WR_L1_COLOR = "#1565C0"
    WR_L2_COLOR = "#BF360C"
    OUT_COLOR = "#00695C"

    ccl_off_to_idx = {off: i for i, off in enumerate(ccl_offsets)}
    input_off_to_idx = {off: i for i, off in enumerate(input_offsets)}
    lv_ccl_off_to_idx = {off: i for i, off in enumerate(lv_ccl_offsets)}

    s = f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
    s += rect(0, 0, W, H, "white", "white")
    s += text(W // 2, 22, "数据流汇总图 — INPUT / CCL / OUTPUT", 18, "#1a237e", bold=True)

    legend_y = 42
    legends = [
        (LC_COLOR, "━━", "LocalCopy"),
        (LR_COLOR, "╌╌", "LocalReduce"),
        (WR_COLOR, "╌╌", "Write(同Server)"),
        ("#1976D2", "╌╌", "Read(远端→本地)"),
        (WR_L1_COLOR, "━━", "WriteReduce L1(跨Server)"),
        (WR_L2_COLOR, "━━", "WriteReduce L2(跨Pod)"),
        (OUT_COLOR, "━━", "Output"),
    ]
    lx = 40
    for color, sym, desc in legends:
        s += text(lx, legend_y, f"{sym} {desc}", 10, color, "left", bold=True)
        lx += 200

    rank_layout = {}

    for l2i in range(l2_size):
        py = 65 + l2i * (pod_h + pod_gap)
        s += rect(15, py, W - 30, pod_h, "#FFF3E0", "#E65100", 6, 2)
        s += text(70, py + 18, f"Pod{l2i}", 14, "#BF360C", bold=True)

        for l1i in range(l1_size):
            sy = py + 30 + l1i * (srv_h + srv_gap)
            s += rect(25, sy, W - 50, srv_h, "#E8F5E9", "#2E7D32", 4, 2)
            s += text(70, sy + 16, f"Server{l1i}", 12, "#1a237e", bold=True)

            for l0i in range(l0_size):
                rank = l2i * l1_size * l0_size + l1i * l0_size + l0i
                if rank not in ranks:
                    continue
                ry = sy + 30 + l0i * (rank_block_h + rank_gap)

                s += rect(35, ry, W - 70, rank_block_h, "#FAFAFA", "#A5D6A7", 3, 1)
                s += text(70, ry + 12, f"R{rank}", 11, "#1a237e", bold=True)

                input_y = ry + 18
                ccl_y = input_y + bar_h + row_gap
                output_y = ccl_y + bar_h + row_gap

                s = draw_buffer_row(s, bar_x0, label_w, input_chunk_w, bar_h, input_y, "INPUT", "#F57F17", input_offsets, l0_size, input_owner)
                s = draw_buffer_row(s, bar_x0, label_w, ccl_chunk_w, bar_h, ccl_y, "CCL", "#4A148C", ccl_offsets, l0_size, ccl_owner)
                if num_lv_ccl > 0:
                    s = draw_buffer_row(s, bar_x0, label_w, lv_ccl_chunk_w, bar_h, output_y, "OUTPUT", "#00695C", lv_ccl_offsets, l0_size, lv_ccl_owner)

                for e in ranks[rank]['localcopy']:
                    src_start = e['src_offset']
                    src_end = src_start + e['src_size']
                    for off in input_offsets:
                        if off >= src_start and off < src_end:
                            idx = input_off_to_idx.get(off)
                            if idx is not None:
                                _, sc = owner_colors(input_owner.get(off, idx % l0_size), l0_size)
                                s += rect(bar_x0 + label_w + idx * input_chunk_w, input_y, input_chunk_w, bar_h, "#C8E6C9", sc, 0, 2)
                    dst_start = e['dst_offset']
                    dst_end = dst_start + e['dst_size']
                    for off in ccl_offsets:
                        if off >= dst_start and off < dst_end:
                            idx = ccl_off_to_idx.get(off)
                            if idx is not None:
                                _, sc = owner_colors(ccl_owner.get(off, idx % l0_size), l0_size)
                                s += rect(bar_x0 + label_w + idx * ccl_chunk_w, ccl_y, ccl_chunk_w, bar_h, "#C8E6C9", sc, 0, 2)

                for e in ranks[rank]['localreduce']:
                    src_start = e['src_offset']
                    src_end = src_start + e['src_size']
                    for off in ccl_offsets:
                        if off >= src_start and off < src_end:
                            idx = ccl_off_to_idx.get(off)
                            if idx is not None:
                                _, sc = owner_colors(ccl_owner.get(off, idx % l0_size), l0_size)
                                s += rect(bar_x0 + label_w + idx * ccl_chunk_w, ccl_y, ccl_chunk_w, bar_h, "#FFCC80", sc, 0, 2)
                    dst_start = e['dst_offset']
                    dst_end = dst_start + e['dst_size']
                    for off in ccl_offsets:
                        if off >= dst_start and off < dst_end:
                            idx = ccl_off_to_idx.get(off)
                            if idx is not None:
                                _, sc = owner_colors(ccl_owner.get(off, idx % l0_size), l0_size)
                                s += rect(bar_x0 + label_w + idx * ccl_chunk_w, ccl_y, ccl_chunk_w, bar_h, "#E1BEE7", sc, 0, 2)

                for e in ranks[rank]['write']:
                    src_start = e['local_offset']
                    src_end = src_start + e['local_size']
                    if e.get('local_buf', 'INPUT') == 'INPUT':
                        for off in input_offsets:
                            if off >= src_start and off < src_end:
                                idx = input_off_to_idx.get(off)
                                if idx is not None:
                                    _, sc = owner_colors(input_owner.get(off, idx % l0_size), l0_size)
                                    s += rect(bar_x0 + label_w + idx * input_chunk_w, input_y, input_chunk_w, bar_h, "#FFCC80", sc, 0, 2)
                    else:
                        for off in ccl_offsets:
                            if off >= src_start and off < src_end:
                                idx = ccl_off_to_idx.get(off)
                                if idx is not None:
                                    _, sc = owner_colors(ccl_owner.get(off, idx % l0_size), l0_size)
                                    s += rect(bar_x0 + label_w + idx * ccl_chunk_w, ccl_y, ccl_chunk_w, bar_h, "#FFCC80", sc, 0, 2)

                for e in ranks[rank]['read']:
                    local_start = e['local_offset']
                    local_end = local_start + e['local_size']
                    if e.get('local_buf', 'CCL') == 'CCL':
                        for off in ccl_offsets:
                            if off >= local_start and off < local_end:
                                idx = ccl_off_to_idx.get(off)
                                if idx is not None:
                                    _, sc = owner_colors(ccl_owner.get(off, idx % l0_size), l0_size)
                                    s += rect(bar_x0 + label_w + idx * ccl_chunk_w, ccl_y, ccl_chunk_w, bar_h, "#B3E5FC", sc, 0, 2)
                    else:
                        for off in input_offsets:
                            if off >= local_start and off < local_end:
                                idx = input_off_to_idx.get(off)
                                if idx is not None:
                                    _, sc = owner_colors(input_owner.get(off, idx % l0_size), l0_size)
                                    s += rect(bar_x0 + label_w + idx * input_chunk_w, input_y, input_chunk_w, bar_h, "#B3E5FC", sc, 0, 2)

                for e in ranks[rank]['output']:
                    src_start = e['src_offset']
                    src_end = src_start + e['src_size']
                    for off in lv_ccl_offsets:
                        if off >= src_start and off < src_end:
                            idx = lv_ccl_off_to_idx.get(off)
                            if idx is not None:
                                _, sc = owner_colors(lv_ccl_owner.get(off, idx % l0_size), l0_size)
                                s += rect(bar_x0 + label_w + idx * lv_ccl_chunk_w, output_y, lv_ccl_chunk_w, bar_h, "#B2DFDB", sc, 0, 2)

                for e in ranks[rank]['write_reduce_l1'] + ranks[rank]['write_reduce_l2']:
                    src_start = e['local_offset']
                    src_end = src_start + e['local_size']
                    for off in ccl_offsets:
                        if off >= src_start and off < src_end:
                            idx = ccl_off_to_idx.get(off)
                            if idx is not None:
                                _, sc = owner_colors(ccl_owner.get(off, idx % l0_size), l0_size)
                                s += rect(bar_x0 + label_w + idx * ccl_chunk_w, ccl_y, ccl_chunk_w, bar_h, "#FFCC80", sc, 0, 2)

                rank_layout[rank] = {
                    'input_y': input_y,
                    'ccl_y': ccl_y,
                    'output_y': output_y,
                    'ry': ry,
                }

    for rank in range(total_ranks):
        if rank not in ranks or rank not in rank_layout:
            continue
        lay = rank_layout[rank]
        input_y = lay['input_y']
        ccl_y = lay['ccl_y']
        output_y = lay['output_y']

        for e in ranks[rank]['localcopy']:
            src_idx = input_off_to_idx.get(e['src_offset'])
            dst_idx = ccl_off_to_idx.get(e['dst_offset'])
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * input_chunk_w + input_chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * ccl_chunk_w + ccl_chunk_w / 2
            s += arrow_line(sx, input_y + bar_h, dx, ccl_y, LC_COLOR, 2)

        for e in ranks[rank]['localreduce']:
            src_idx = ccl_off_to_idx.get(e['src_offset'])
            dst_idx = ccl_off_to_idx.get(e['dst_offset'])
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * ccl_chunk_w + ccl_chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * ccl_chunk_w + ccl_chunk_w / 2
            s += curved_arrow(sx, ccl_y + 2, dx, ccl_y + 2, LR_COLOR, 1.5, curve=15, dash="4,2")

    write_by_pair = {}
    for rank in range(total_ranks):
        if rank not in ranks:
            continue
        for e in ranks[rank]['write']:
            remote = e['remote_rank']
            key = (rank, remote)
            if key not in write_by_pair:
                write_by_pair[key] = []
            write_by_pair[key].append(e)

    pair_idx = 0
    for (src_rank, dst_rank), entries in sorted(write_by_pair.items()):
        if src_rank not in rank_layout or dst_rank not in rank_layout:
            continue
        _, l1s, l2s = _rank_to_pos(src_rank, l0_size, l1_size, l2_size)
        _, l1d, l2d = _rank_to_pos(dst_rank, l0_size, l1_size, l2_size)
        if l1s != l1d or l2s != l2d:
            continue

        src_lay = rank_layout[src_rank]
        dst_lay = rank_layout[dst_rank]
        bend_offset = 12 + pair_idx * 14

        for e in entries:
            if e.get('local_buf', 'INPUT') == 'INPUT':
                src_idx = input_off_to_idx.get(e['local_offset'])
                sx = bar_x0 + label_w + src_idx * input_chunk_w + input_chunk_w / 2 if src_idx is not None else None
                y1 = src_lay['input_y'] + bar_h
            else:
                src_idx = ccl_off_to_idx.get(e['local_offset'])
                sx = bar_x0 + label_w + src_idx * ccl_chunk_w + ccl_chunk_w / 2 if src_idx is not None else None
                y1 = src_lay['ccl_y'] + bar_h
            dst_idx = ccl_off_to_idx.get(e['remote_offset'])
            dx = bar_x0 + label_w + dst_idx * ccl_chunk_w + ccl_chunk_w / 2 if dst_idx is not None else None
            y2 = dst_lay['ccl_y']
            if sx is None or dx is None:
                continue
            s += bezier_arrow(
                sx, y1,
                sx + bend_offset, (y1 + y2) / 2,
                dx + bend_offset, (y1 + y2) / 2,
                dx, y2,
                WR_COLOR, 1.2, dash="4,2"
            )

        right_x = bar_x0 + label_w + num_input * input_chunk_w + 8
        s += text(right_x + bend_offset + 2, (src_lay['input_y'] + dst_lay['ccl_y'] + bar_h) / 2,
                  f"R{src_rank % l0_size}→R{dst_rank % l0_size} W×{len(entries)}", 7, WR_COLOR, "left", bold=True)
        pair_idx += 1

    READ_COLOR = "#1976D2"
    read_by_pair = {}
    for rank in range(total_ranks):
        if rank not in ranks:
            continue
        for e in ranks[rank]['read']:
            remote = e['remote_rank']
            key = (remote, rank)
            if key not in read_by_pair:
                read_by_pair[key] = []
            read_by_pair[key].append(e)

    for (src_rank, dst_rank), entries in sorted(read_by_pair.items()):
        if src_rank not in rank_layout or dst_rank not in rank_layout:
            continue
        _, l1s, l2s = _rank_to_pos(src_rank, l0_size, l1_size, l2_size)
        _, l1d, l2d = _rank_to_pos(dst_rank, l0_size, l1_size, l2_size)
        if l1s != l1d or l2s != l2d:
            continue

        src_lay = rank_layout[src_rank]
        dst_lay = rank_layout[dst_rank]
        bend_offset = 12 + pair_idx * 14

        for e in entries:
            src_idx = ccl_off_to_idx.get(e['remote_offset'])
            sx = bar_x0 + label_w + src_idx * ccl_chunk_w + ccl_chunk_w / 2 if src_idx is not None else None
            y1 = src_lay['ccl_y'] + bar_h
            if e.get('local_buf', 'CCL') == 'CCL':
                dst_idx = ccl_off_to_idx.get(e['local_offset'])
                dx = bar_x0 + label_w + dst_idx * ccl_chunk_w + ccl_chunk_w / 2 if dst_idx is not None else None
                y2 = dst_lay['ccl_y']
            else:
                dst_idx = input_off_to_idx.get(e['local_offset'])
                dx = bar_x0 + label_w + dst_idx * input_chunk_w + input_chunk_w / 2 if dst_idx is not None else None
                y2 = dst_lay['input_y']
            if sx is None or dx is None:
                continue
            s += bezier_arrow(
                sx, y1,
                sx + bend_offset, (y1 + y2) / 2,
                dx + bend_offset, (y1 + y2) / 2,
                dx, y2,
                READ_COLOR, 1.2, dash="4,2"
            )

        right_x = bar_x0 + label_w + num_input * input_chunk_w + 8
        s += text(right_x + bend_offset + 2, (src_lay['ccl_y'] + dst_lay['ccl_y'] + bar_h) / 2,
                  f"R{src_rank % l0_size}→R{dst_rank % l0_size} Rd×{len(entries)}", 7, READ_COLOR, "left", bold=True)
        pair_idx += 1

    for rank in range(total_ranks):
        if rank not in ranks or rank not in rank_layout:
            continue
        lay = rank_layout[rank]
        ccl_y = lay['ccl_y']
        output_y = lay['output_y']

        for e in ranks[rank]['write_reduce_l1']:
            remote = e['remote_rank']
            if remote not in rank_layout:
                continue
            src_idx = ccl_off_to_idx.get(e['local_offset'])
            dst_idx = ccl_off_to_idx.get(e['remote_offset'])
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * ccl_chunk_w + ccl_chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * ccl_chunk_w + ccl_chunk_w / 2
            dst_ccl_y = rank_layout[remote]['ccl_y']
            s += arrow_line(sx, ccl_y + bar_h, dx, dst_ccl_y, WR_L1_COLOR, 2)

        for e in ranks[rank]['write_reduce_l2']:
            remote = e['remote_rank']
            if remote not in rank_layout:
                continue
            src_idx = ccl_off_to_idx.get(e['local_offset'])
            dst_idx = ccl_off_to_idx.get(e['remote_offset'])
            if src_idx is None or dst_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * ccl_chunk_w + ccl_chunk_w / 2
            dx = bar_x0 + label_w + dst_idx * ccl_chunk_w + ccl_chunk_w / 2
            dst_ccl_y = rank_layout[remote]['ccl_y']
            s += bezier_arrow(
                sx, ccl_y + bar_h,
                sx + 30, ccl_y + bar_h + 20,
                dx + 30, dst_ccl_y - 20,
                dx, dst_ccl_y,
                WR_L2_COLOR, 2
            )

        for e in ranks[rank]['output']:
            src_idx = ccl_off_to_idx.get(e['src_offset'])
            if src_idx is None:
                continue
            sx = bar_x0 + label_w + src_idx * ccl_chunk_w + ccl_chunk_w / 2
            dst_idx = lv_ccl_off_to_idx.get(e['src_offset'])
            if dst_idx is None:
                dx = sx
            else:
                dx = bar_x0 + label_w + dst_idx * lv_ccl_chunk_w + lv_ccl_chunk_w / 2
            bend = 15
            s += bezier_arrow(sx, ccl_y + bar_h, sx + bend, ccl_y + bar_h, dx + bend, output_y, dx, output_y, OUT_COLOR, 2)

    return s + "</svg>"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate HCCL data flow SVG diagrams from log files')
    parser.add_argument('log_file', help='Path to the HCCL log file')
    parser.add_argument('-o', '--output-dir', help='Output directory for SVG files (default: same as log file)')
    parser.add_argument('--l0', type=int, help='Override L0 (devices per server) topology')
    parser.add_argument('--l1', type=int, help='Override L1 (servers per pod) topology')
    parser.add_argument('--l2', type=int, help='Override L2 (number of pods) topology')
    parser.add_argument('--levels', type=str, default='summary',
                        help='Comma-separated levels to generate (e.g. "0,1,2,summary"). Default: summary')
    args = parser.parse_args()

    log_file = args.log_file
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(log_file))

    ranks, l0, l1, l2 = parse_log(log_file)
    if args.l0 or args.l1 or args.l2:
        if args.l0:
            l0 = args.l0
        if args.l1:
            l1 = args.l1
        if args.l2:
            l2 = args.l2
        all_wr = []
        for rank in ranks:
            all_wr.extend((rank, e) for e in ranks[rank]['write_reduce_l1'])
            all_wr.extend((rank, e) for e in ranks[rank]['write_reduce_l2'])
        for rank in ranks:
            ranks[rank]['write_reduce_l1'] = []
            ranks[rank]['write_reduce_l2'] = []
        for rank, e in all_wr:
            _, s1, s2 = _rank_to_pos(rank, l0, l1, l2)
            _, d1, d2 = _rank_to_pos(e['remote_rank'], l0, l1, l2)
            if s2 != d2:
                ranks[rank]['write_reduce_l2'].append(e)
            elif s1 != d1:
                ranks[rank]['write_reduce_l1'].append(e)
            else:
                ranks[rank]['write_reduce_l1'].append(e)
    if l0 * l1 * l2 != len(ranks):
        print(f"WARNING: L0*L1*L2={l0*l1*l2} != total_ranks={len(ranks)}, topology may be incorrect")

    ccl_offsets = collect_ccl_offsets(ranks)
    input_offsets = collect_input_offsets(ranks)
    ccl_owner = build_offset_owner_map(ranks, l0)
    input_owner = build_input_owner_map(ranks, l0)
    total_ranks = len(ranks)

    print(f"Parsed {total_ranks} ranks, topology: L0={l0} L1={l1} L2={l2}")
    print(f"CCL offsets ({len(ccl_offsets)}): {ccl_offsets}")
    print(f"INPUT offsets ({len(input_offsets)}): {input_offsets}")

    for rank in sorted(ranks.keys()):
        r = ranks[rank]
        print(f"  R{rank}: LocalCopy={len(r['localcopy'])} LocalReduce={len(r['localreduce'])} "
              f"Write={len(r['write'])} Read={len(r['read'])} WR_L1={len(r['write_reduce_l1'])} WR_L2={len(r['write_reduce_l2'])} "
              f"Out={len(r['output'])}")

    levels = set(l.strip() for l in args.levels.split(','))

    lv_ccl_offsets = collect_level_ccl_offsets(ranks)
    lv_ccl_owner = build_level_ccl_owner_map(ranks, l0)
    print(f"Level1/2 CCL offsets ({len(lv_ccl_offsets)}): {lv_ccl_offsets}")

    if '0' in levels:
        content = gen_level0_svg(ranks, l0, l1, l2, ccl_offsets, input_offsets, ccl_owner, input_owner, include_output=(l2 == 1 and l1 == 1))
        path = f"{out_dir}/data_flow_level0.svg"
        with open(path, 'w') as f:
            f.write(content)
        print(f"Wrote {path}")

    if '1' in levels:
        content = gen_level1_svg(ranks, l0, l1, l2, lv_ccl_offsets, lv_ccl_owner, include_output=(l2 == 1 and l1 > 1))
        path = f"{out_dir}/data_flow_level1.svg"
        with open(path, 'w') as f:
            f.write(content)
        print(f"Wrote {path}")

    if '2' in levels:
        content = gen_level2_svg(ranks, l0, l1, l2, lv_ccl_offsets, lv_ccl_owner, include_output=(l2 > 1))
        path = f"{out_dir}/data_flow_level2.svg"
        with open(path, 'w') as f:
            f.write(content)
        print(f"Wrote {path}")

    if 'summary' in levels:
        content = gen_summary_svg(ranks, l0, l1, l2, ccl_offsets, input_offsets, ccl_owner, input_owner, lv_ccl_offsets, lv_ccl_owner)
        path = f"{out_dir}/data_flow_summary.svg"
        with open(path, 'w') as f:
            f.write(content)
        print(f"Wrote {path}")

    print("\nDone! Open .svg files in browser or VS Code preview.")


if __name__ == "__main__":
    main()
