"""Jamie's tracking-correction rules and the review list built on them (copied from Motion).

Rule (Jamie, 2026-10-25): cells generally do not appear or disappear, so a cell keeps one number
for the whole video. Where a number ends, the number that starts nearest to it carries on its name
(up to 4 frames later, within 2.0 body radii, areas within 3.0x, pieces under
60 px ignored); each chain keeps the number it holds longest; no new numbers; masks never
change. Looser "unsure" joins (up to 6 frames, 3.0 radii) are listed for review and
applied only when approved.

Copied verbatim (function bodies) from Motion/manual_editing/trajectory_review, so the rule here
and the one Jamie approved there are the same code:
  gap_link.py       sha256 b630283470fd267f...  frame_stats tracks link chains name_chains plan solve
  review_data.py    sha256 1b9240258b70da92...  join_info loose_links build names conflicts frame_lut
                                                   index_frames apply_ops op_frames compose row_runs
  continuity_round.py sha256 7e0744c8e31dfa54... patch_from
  Motion/code/manual_pixel_edits.py sha256 9b74e10635a99d8d... identity_patch_state_sha256
Only cross-module names were changed (g.plan -> plan, g.MAX_GAP -> MAX_GAP). Parity with Motion is
checked in tests/test_correction_rules.py.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_GAP = 4         # frames between an end and a start (2 h at 30 min/frame)
MAX_RADII = 2.0     # start centroid within this many body radii of the end centroid
MIN_AREA = 60       # pieces smaller than this are noise, never linked
AREA_RATIO = 3.0   # end and start areas within this ratio
JUMP = 2.0          # a number moving further than this (body radii), with no overlap, starts a new track
JUMP_PX = 15.0     # ... and further than this many pixels
SAME_NAME = 0.5     # the same number continuing after an absence is preferred by this much
CHECK_RADII, CHECK_GAP = 1.5, 3   # applied joins this far or late are flagged to look at first
LOOSE_GAP, LOOSE_RADII = 6, 3.0   # unsure joins


def frame_stats(labels):
    """Per frame: {name: (area, cy, cx)} and per consecutive frame pair: names whose masks overlap."""
    T, H, W = labels.shape
    yy, xx = np.mgrid[:H, :W]
    stats, overlap = [], []
    for f in range(T):
        p = labels[f].ravel()
        n = int(p.max())+1
        area = np.bincount(p, minlength=n)
        sy = np.bincount(p, yy.ravel(), n); sx = np.bincount(p, xx.ravel(), n)
        stats.append({int(k): (int(area[k]), sy[k]/area[k], sx[k]/area[k]) for k in np.flatnonzero(area) if k})
        if f:
            same = labels[f-1] == labels[f]
            overlap.append(set(np.unique(labels[f][same]).tolist()) - {0})
    return stats, overlap


def tracks(stats, overlap, skip):
    """Cut each number into tracks at absences and jumps."""
    out, open_ = [], {}
    for f, s in enumerate(stats):
        for name, (a, cy, cx) in s.items():
            if name in skip:
                continue
            t = open_.get(name)
            if t is not None and t['last'] == f-1:
                pa, py, px = stats[f-1][name]
                r = math.sqrt(max(pa, a)/math.pi)
                if name in overlap[f-1] or math.hypot(cy-py, cx-px) <= max(JUMP_PX, JUMP*r):
                    t['last'] = f
                    continue
            t = dict(id=len(out), name=name, first=f, last=f)
            out.append(t); open_[name] = t
    return out


def link(trk, stats, T, cut=frozenset()):
    """One-to-one pairs end -> start by nearest centroid in body radii."""
    ends = [t for t in trk if t['last'] < T-1]
    starts = [t for t in trk if t['first'] > 0]
    big = 1e6
    cost = np.full((len(ends), len(starts)), big)
    for i, e in enumerate(ends):
        ea, ey, ex = stats[e['last']][e['name']]
        for j, s in enumerate(starts):
            gap = s['first']-e['last']
            if not 1 <= gap <= MAX_GAP or (e['id'], s['id']) in cut:
                continue
            sa, sy, sx = stats[s['first']][s['name']]
            if min(ea, sa) < MIN_AREA or max(ea, sa) > AREA_RATIO*min(ea, sa):
                continue
            r = math.sqrt(ea/math.pi)
            d = math.hypot(sy-ey, sx-ex)
            if d <= MAX_RADII*r:
                cost[i, j] = d/r + 0.5*(gap-1) - SAME_NAME*(s['name'] == e['name'])
    rows, cols = linear_sum_assignment(cost) if cost.size else ([], [])
    return {ends[i]['id']: (starts[j]['id'], cost[i, j]) for i, j in zip(rows, cols) if cost[i, j] < big}


def chains(trk, links):
    nxt = {a: b for a, (b, _) in links.items()}
    heads = set(range(len(trk))) - set(nxt.values())
    out = []
    for h in sorted(heads):
        c = [h]
        while c[-1] in nxt:
            c.append(nxt[c[-1]])
        out.append(c)
    return out


def name_chains(trk, links, forbidden=frozenset()):
    """Each chain keeps the number it carries for most frames; longer chains choose first.
    A number is free when no other track that will keep it holds it in the chain's frames
    (a track in a chain preferring another number is expected to give it up). A chain with no
    free number is split at its weakest link. Returns None plus the clash when a lone track
    finds its own number taken."""
    span = lambda k: trk[k]['last']-trk[k]['first']+1
    def ranked(c):
        weight = collections.Counter()
        for k in c:
            weight[trk[k]['name']] += span(k)
        return [n for n, _ in weight.most_common() if (tuple(c), n) not in forbidden]
    queue = chains(trk, links)
    keeps = {k for c in queue for k in c if ranked(c)[:1] == [trk[k]['name']]}
    pending = {(t['name'], f): t['id'] for t in trk if t['id'] in keeps for f in range(t['first'], t['last']+1)}
    taken, named, dropped = {}, [], 0
    while queue:
        queue.sort(key=lambda c: sum(span(k) for k in c))
        c = queue.pop()
        mine = set(c)
        frames = [f for k in c for f in range(trk[k]['first'], trk[k]['last']+1)]
        for name in ranked(c):
            if all((name, f) not in taken and pending.get((name, f), c[0]) in mine for f in frames):
                break
        else:
            if len(c) == 1:
                k = c[0]
                return None, next(taken[(trk[k]['name'], f)] for f in range(trk[k]['first'], trk[k]['last']+1)
                                  if (trk[k]['name'], f) in taken)
            i = max(range(len(c)-1), key=lambda i: links[c[i]][1])
            queue += [c[:i+1], c[i+1:]]
            dropped += 1
            continue
        for k in c:
            for f in range(trk[k]['first'], trk[k]['last']+1):
                pending.pop((trk[k]['name'], f), None)
        taken.update(((name, f), (tuple(c), name)) for f in frames)
        named.append((c, name))
    return named, dropped


def plan(labels, skip=frozenset()):
    """Tracks, links and named chains; shared by solve and the correction reviewer."""
    T = len(labels)
    stats, overlap = frame_stats(labels)
    trk = tracks(stats, overlap, skip)
    links = link(trk, stats, T)
    forbidden = set()
    while True:
        named, dropped = name_chains(trk, links, frozenset(forbidden))
        if named is not None:
            break
        forbidden.add(dropped)  # that chain may not take the number a lone track needs
    return stats, trk, links, named, dropped, forbidden


def solve(labels, skip=frozenset()):
    T = len(labels)
    stats, trk, links, named, dropped, forbidden = plan(labels, skip)
    current = labels.copy()
    renamed = 0
    for c, name in named:
        for k in c:
            t = trk[k]
            if t['name'] != name:
                for f in range(t['first'], t['last']+1):
                    current[f][labels[f] == t['name']] = name
                renamed += 1
    mid_starts = sum(1 for c, _ in named if trk[c[0]]['first'] > 0)
    mid_ends = sum(1 for c, _ in named if trk[c[-1]]['last'] < T-1)
    return current, dict(tracks=len(trk), links=len(links), tracks_renamed=renamed, dropped_links=dropped, refused_names=len(forbidden),
                         chains=len(named), chains_starting_mid_video=mid_starts, chains_ending_mid_video=mid_ends,
                         link_list=[dict(end=trk[a]['name'], end_frame=trk[a]['last'], start=trk[b]['name'],
                                         start_frame=trk[b]['first'], cost=round(float(c), 3))
                                    for a, (b, c) in links.items()])


def join_info(stats, trk, a, b):
    """Gap, distance in body radii and area ratio between the end of track a and the start of b."""
    ta, tb = trk[a], trk[b]
    ea, ey, ex = stats[ta['last']][ta['name']]
    sa, sy, sx = stats[tb['first']][tb['name']]
    r = math.sqrt(ea/math.pi)
    return dict(frame=tb['first'], gap=tb['first']-ta['last'], radii=round(math.hypot(sy-ey, sx-ex)/r, 2),
                area_ratio=round(max(ea, sa)/min(ea, sa), 2), centre=[round(sy, 1), round(sx, 1)],
                radius=round(math.sqrt(max(ea, sa)/math.pi), 1))


def loose_links(trk, stats, T):
    global MAX_GAP, MAX_RADII
    saved = MAX_GAP, MAX_RADII
    MAX_GAP, MAX_RADII = LOOSE_GAP, LOOSE_RADII
    try:
        return link(trk, stats, T)
    finally:
        MAX_GAP, MAX_RADII = saved


def build(labels, skip=frozenset()):
    """Tracks, the gap_link names and the review items for one label stack."""
    T = len(labels)
    stats, trk, links, named, _, _ = plan(labels, skip)
    final = [t['name'] for t in trk]
    chain_of, members = {}, []
    for ci, (c, name) in enumerate(named):
        members.append(c)
        for k in c:
            final[k] = name
            chain_of[k] = ci
    occ = {(final[t['id']], f): t['id'] for t in trk for f in range(t['first'], t['last']+1)}
    renamed = [t['id'] for t in trk if final[t['id']] != t['name']]
    # Undoing one rename alone would clash with the track now holding its old number: group them.
    parent = {k: k for k in renamed}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    for k in renamed:
        t = trk[k]
        for f in range(t['first'], t['last']+1):
            h = occ.get((t['name'], f))
            if h is not None and h != k and h in parent:
                parent[find(k)] = find(h)
    groups = collections.defaultdict(list)
    for k in renamed:
        groups[find(k)].append(k)
    items = []
    for ks in groups.values():
        ks.sort(key=lambda k: trk[k]['first'])
        joins = []
        for k in ks:
            c = members[chain_of[k]]
            i = c.index(k)
            joins.append(join_info(stats, trk, c[i-1], k) if i else join_info(stats, trk, k, c[1]))
        j = min(joins, key=lambda x: x['frame'])
        items.append(dict(kind='applied', renames=[[k, trk[k]['name'], final[k]] for k in ks],
                          check=any(x['radii'] > CHECK_RADII or x['gap'] >= CHECK_GAP for x in joins), **j))
    # Unsure: looser end -> start pairs between a chain's last track and another chain's first,
    # where the end's number is free over the whole later chain.
    for a, (b, _) in loose_links(trk, stats, T).items():
        ca, cb = chain_of[a], chain_of[b]
        if ca == cb or members[ca][-1] != a or members[cb][0] != b or final[a] == final[b]:
            continue
        name, mine = final[a], set(members[cb])
        if any(occ.get((name, f), k) not in mine for k in members[cb] for f in range(trk[k]['first'], trk[k]['last']+1)):
            continue
        items.append(dict(kind='unsure', renames=[[k, final[k], name] for k in members[cb]], check=False,
                          **join_info(stats, trk, a, b)))
    items.sort(key=lambda it: (it['frame'], it['renames'][0][0]))
    for it in items:
        it['key'] = ('A' if it['kind'] == 'applied' else 'U') + '-' + '-'.join(
            f"{trk[k]['first']}.{trk[k]['name']}" for k, _, _ in it['renames'])
    return dict(T=T, tracks=[[t['name'], t['first'], t['last']] for t in trk], final=final, items=items)


def names(data, decisions):
    """Number of every track under the decisions ({key: 'deny'|'approve'})."""
    out = [n for n, _, _ in data['tracks']]
    for it in data['items']:
        d = decisions.get(it['key'])
        if (it['kind'] == 'applied' and d != 'deny') or (it['kind'] == 'unsure' and d == 'approve'):
            for k, _, new in it['renames']:
                out[k] = new
    return out


def conflicts(data, decisions):
    """Item keys whose renames would put one number on two tracks in one frame."""
    nm = names(data, decisions)
    owner, clash = {}, set()
    for k, (_, first, last) in enumerate(data['tracks']):
        for f in range(first, last+1):
            h = owner.setdefault((nm[k], f), k)
            if h != k:
                clash.update((h, k))
    if not clash:
        return []
    return [it['key'] for it in data['items'] if any(k in clash for k, _, _ in it['renames'])]


def frame_lut(data, nm, f, size):
    """Lookup table mapping original numbers to decided numbers in frame f."""
    lut = np.arange(size, dtype=np.uint16)
    for k in data['by_frame'][f]:
        lut[data['tracks'][k][0]] = nm[k]
    return lut


def index_frames(data):
    """Tracks whose number can change, listed per frame (for fast per-frame relabelling)."""
    movable = {k for it in data['items'] for k, _, _ in it['renames']}
    data['by_frame'] = [[] for _ in range(data['T'])]
    for k in movable:
        _, first, last = data['tracks'][k]
        for f in range(first, last+1):
            data['by_frame'][f].append(k)
    return data


def _run_index(runs, shape, y0, x0):
    """Flat indices of the pixels in runs [y, x, length] (full-frame coordinates) inside a frame or crop of
    this shape whose corner is (y0, x0); runs outside are clipped as apply_ops always did."""
    h, w = shape
    r = np.asarray(runs, dtype=np.int64).reshape(-1, 3)
    y, xa, xb = r[:, 0] - y0, np.maximum(r[:, 1] - x0, 0), np.minimum(r[:, 1] - x0 + r[:, 2], w)
    ok = (y >= 0) & (y < h) & (xa < xb)
    y, xa, n = y[ok], xa[ok], (xb - xa)[ok]
    if not len(n):
        return np.empty(0, np.int64)
    start = np.repeat(y * w + xa - np.r_[0, np.cumsum(n)[:-1]], n)
    return start + np.arange(int(n.sum()))


def apply_ops(lab, f, ops, y0=0, x0=0, splits=True):
    """Jamie's manual edits that cover frame f, in order, on one frame (or a crop whose corner is y0, x0).

    Edits name cells by their number after the decisions and the edits before them:
      swap a<->b, renumber from->to (also merges into an existing number), delete cell,
      split / piece: the listed pixels of host move to child (pixel runs [y, x, length] per frame;
      a piece is one clicked part in one frame, and child 0 deletes it);
      keep: one cell followed over frames, moves {frame: [[host, child, runs], ...]} (note_actions.py);
      paint (not in Motion, 2026-11-17): the listed pixels become number 'to' (0 erases); with 'only' set,
      just the pixels that carried that number (0 = background, so a drawn or grown outline never takes
      another cell's pixels).
    """
    out = lab
    for op in ops:
        if not op['first'] <= f <= op['last']:
            continue
        if out is lab:
            out = lab.copy()
        kind = op['op']
        if kind == 'swap':
            a, b = out == op['a'], out == op['b']
            out[a], out[b] = op['b'], op['a']
        elif kind == 'renumber':
            out[out == op['from']] = op['to']
        elif kind == 'delete':
            out[out == op['cell']] = 0
        elif kind in ('split', 'piece', 'paint') and splits:
            runs = op['runs'].get(str(f))
            if not runs:
                continue
            idx = _run_index(runs, out.shape, y0, x0)
            if not out.flags.c_contiguous:
                out = np.ascontiguousarray(out)
            flat = out.reshape(-1)
            if kind == 'paint':
                only = op.get('only')
                flat[idx if only is None else idx[flat[idx] == only]] = op['to']
            else:
                flat[idx[flat[idx] == op['host']]] = op['child']
        elif kind == 'keep' and splits:
            h, w = out.shape
            before, moved = out.copy(), []   # every move reads the frame as it was, then all are written
            for host, child, runs in op['moves'].get(str(f), ()):
                for y, x, n in runs:
                    y, xa, xb = y-y0, max(x-x0, 0), min(x-x0+n, w)
                    if 0 <= y < h and xa < xb:
                        hit = np.flatnonzero(before[y, xa:xb] == host)
                        moved.append((y, hit+xa, child))
            for y, xs, child in moved:
                out[y, xs] = child
    return out


def op_frames(ops):
    return {f for op in ops for f in range(op['first'], op['last']+1)}


# Not in Motion (added 2026-11-11): edits name cells by number and are applied after the decisions, so a decision
# made after an edit could renumber the edited cell and leave the edit with nothing to act on (Jamie's slice 8:
# four edits voided by approving unsure joins afterwards). follow_decisions rewrites the edits so they keep
# acting on the same cells.
_NUMBER_FIELDS = {'swap': ('a', 'b'), 'renumber': ('from', 'to'), 'delete': ('cell',),
                  'split': ('host', 'child'), 'piece': ('host', 'child'), 'paint': ('only', 'to')}
_TARGETS = ('to', 'child')   # numbers an edit gives, which need not be in the frame


def follow_decisions(data, old, new, ops):
    """Edits rewritten from decisions OLD to NEW so each still acts on the cells it was made on.

    In every frame an edit covers, a number a decision renames there is replaced by its new number; a number the
    edit gives (renumber 'to', split/piece 'child') that is not in the frame follows its tracks wherever they all
    get one new number. An edit whose numbers change part-way is cut into frame ranges. The first rewrite keeps
    the edit as it was made in 'as_made'. Returns (ops, number of edits rewritten).
    """
    nm_old, nm_new = names(data, old), names(data, new)
    if nm_old == nm_new:
        return list(ops), 0
    moved = collections.defaultdict(set)
    for k, n in enumerate(nm_old):
        moved[n].add(nm_new[k])
    anywhere = {n: next(iter(s)) for n, s in moved.items() if len(s) == 1 and n not in s}

    def mapping(f):
        return {nm_old[k]: nm_new[k] for k in data['by_frame'][f] if nm_old[k] != nm_new[k]}

    def follow(here, n, target):
        if not n:
            return n
        if n in here:
            return here[n]
        return anywhere.get(n, n) if target and n not in set(here.values()) else n

    out, changed = [], 0
    for op in ops:
        kind = op['op']
        made = {k: v for k, v in op.items() if k not in ('runs', 'moves', 'as_made')}
        if kind == 'keep':   # moves are listed frame by frame already: rewrite each frame in place
            moves = {}
            for f, rows in op.get('moves', {}).items():
                here = mapping(int(f)) if int(f) < len(data['by_frame']) else {}
                moves[f] = [[follow(here, h, False), follow(here, c, True), r] for h, c, r in rows]
            if moves != op.get('moves', {}):
                changed += 1
                out.append(dict(op, moves=moves, as_made=op.get('as_made', made)))
            else:
                out.append(op)
            continue
        fields = _NUMBER_FIELDS.get(kind, ())
        segments = []
        for f in range(op['first'], op['last'] + 1):
            here = mapping(f) if f < len(data['by_frame']) else {}
            key = tuple(follow(here, op.get(name), name in _TARGETS) for name in fields)
            if segments and segments[-1][2] == key:
                segments[-1][1] = f
            else:
                segments.append([f, f, key])
        if all(key == tuple(op.get(name) for name in fields) for _, _, key in segments):
            out.append(op)
            continue
        changed += 1
        for a, b, key in segments:
            part = dict(op, first=a, last=b, as_made=op.get('as_made', made))
            part.update(zip(fields, key))
            if kind in ('split', 'piece', 'paint'):
                part['runs'] = {f: r for f, r in op.get('runs', {}).items() if a <= int(f) <= b}
            out.append(part)
    return out, changed


def compose(labels, data, decisions, ops=()):
    nm = names(data, decisions)
    out = labels.copy()
    size = int(labels.max())+1
    for f in range(len(labels)):
        if data['by_frame'][f]:
            out[f] = frame_lut(data, nm, f, size)[labels[f]]
    for f in sorted(op_frames(ops)):
        if f < len(out):
            out[f] = apply_ops(out[f], f, ops)
    return out


def row_runs(mask, y0=0, x0=0):
    """[y, x, length] runs of a boolean mask, in full-frame coordinates."""
    runs = []
    for y in np.flatnonzero(mask.any(1)):
        d = np.diff(np.concatenate([[0], mask[y].astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            runs.append([int(y+y0), int(a+x0), int(b-a)])
    return runs


def patch_from(original, current):
    frames = []
    for f in range(len(original)):
        diff = original[f] != current[f]
        if not diff.any():
            continue
        runs = []
        for y in np.flatnonzero(diff.any(1)):
            xs = np.flatnonzero(diff[y])
            start = xs[0]
            for k in range(1, len(xs)+1):
                end = k == len(xs) or xs[k] != xs[k-1]+1 or \
                    (original[f, y, xs[k]], current[f, y, xs[k]]) != (original[f, y, xs[k-1]], current[f, y, xs[k-1]])
                if end:
                    x = xs[k-1]
                    runs.append([int(y), int(start), int(x-start+1), int(original[f, y, x]), int(current[f, y, x])])
                    if k < len(xs):
                        start = xs[k]
        frames.append(dict(imagej_frame=f+1, runs=runs))
    return dict(shape=list(original.shape), frames=frames,
                parent_state_sha256=identity_patch_state_sha256(original, frames))


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"identity patch {name} must be an integer >= {minimum}")
    if value > int(np.iinfo(np.uint64).max):
        raise ValueError(f"identity patch {name} exceeds uint64 capacity")
    return value


def _frames(frames: Any, shape: tuple[int, ...]) -> list[dict[str, Any]]:
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError("identity patch requires a positive TYX shape")
    if not isinstance(frames, list) or not frames:
        raise ValueError("identity patch frames must be a non-empty list")
    result = []
    seen = set()
    for entry in frames:
        if not isinstance(entry, dict) or set(entry) != {"imagej_frame", "runs"}:
            raise ValueError("identity patch frame requires imagej_frame and runs only")
        frame = _integer(entry["imagej_frame"], "imagej_frame", 1)
        if frame > shape[0] or frame in seen:
            raise ValueError("identity patch frame is out of bounds or duplicated")
        seen.add(frame)
        if not isinstance(entry["runs"], list) or not entry["runs"]:
            raise ValueError("identity patch runs must be a non-empty list")
        runs = []
        for row in entry["runs"]:
            if not isinstance(row, list) or len(row) != 5:
                raise ValueError("identity patch run must be [y, x, length, source, target]")
            y, x, length, source, target = [
                _integer(value, name, 1 if name == "length" else 0)
                for value, name in zip(row, ("y", "x", "length", "source", "target"))]
            if y >= shape[1] or x >= shape[2] or length > shape[2] - x:
                raise ValueError("identity patch run is out of bounds")
            runs.append([y, x, length, source, target])
        runs.sort()
        for previous, current in zip(runs, runs[1:]):
            if previous[0] == current[0] and previous[1] + previous[2] > current[1]:
                raise ValueError("identity patch runs overlap")
        result.append({"imagej_frame": frame, "runs": runs})
    return sorted(result, key=lambda entry: entry["imagej_frame"])


def identity_patch_state_sha256(labels: np.ndarray, frames: list[dict]) -> str:
    """Fingerprint shape, selection, and its current values, independent of dtype.

    Changes outside the explicit selection do not invalidate this fingerprint.
    The selected expected sources are checked separately when applying the edit.
    """
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("identity patch requires integer TYX labels")
    normalized = _frames(frames, labels.shape)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(labels.shape), "frames": normalized},
                             sort_keys=True, separators=(",", ":")).encode("ascii"))
    for entry in normalized:
        plane = labels[entry["imagej_frame"] - 1]
        for y, x, length, _source, _target in entry["runs"]:
            values = plane[y, x:x + length]
            if np.any(values < 0):
                raise ValueError("identity patch selected labels must be non-negative")
            digest.update(np.asarray(values, dtype="<u8").tobytes())
    return digest.hexdigest()

