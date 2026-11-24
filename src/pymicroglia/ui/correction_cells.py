"""Whole-recording views for the correction reviewer (Jamie 2026-11-17).

  current_stack   the labels as they are now (approved joins, decisions and edits), with per-frame cell summaries;
                  recomputed only for frames whose labels changed since the last state
  timeline        every cell's frames (a Gantt chart) with events worth a look, flagged by the tracking rule's own
                  measures (correction_rules: body radii, area ratio, smallest piece): starts and ends inside the
                  recording, gaps, jumps without overlap, sudden size changes, one number on separate pieces
  follow_box      one fixed crop that keeps the followed cells in view over a frame range
  tower           the package's tracking tower (figure_tables.tracking_tower) for the followed cells

Nothing here writes files; the reviewer's edits stay in edits.json.
"""
from __future__ import annotations

import base64
import collections
import io
import json
import math
import threading

import numpy as np
from scipy import ndimage as ndi

from ..tracking import correction_rules as rd

_CACHE: dict[str, dict] = {}
_LOCKS: dict[str, threading.Lock] = collections.defaultdict(threading.Lock)
GOLDEN = .61803398875
TOWER_SIDE, TOWER_FRAMES = 420, 60   # tower display only: enlarged crop side (px) and frames used for track lines


def identity_rgb(n: int) -> tuple[int, int, int]:
    """The tracking tower's identity colour (figure_tables.tracking_tower.identity_colours) as 0-255 RGB."""
    from ..figure_tables.tracking_tower import identity_colours
    return tuple(int(round(v * 255)) for v in identity_colours([int(n)])[int(n)])


def colour_lut(size: int) -> np.ndarray:
    """identity_rgb for numbers 0..size-1 at once (0 = grey)."""
    from matplotlib.colors import hsv_to_rgb
    i = np.arange(max(1, size))
    hsv = np.stack([(i * GOLDEN) % 1, .50 + .14 * (i % 3), .88 + .05 * (i % 2)], 1)
    lut = np.round(hsv_to_rgb(hsv) * 255).astype(np.uint8)
    lut[0] = (120, 120, 120)
    return lut


def _state(rec) -> str:
    from .correction_review import _decisions, _rev
    return json.dumps(_decisions(rec.folder), sort_keys=True) + _rev(rec.folder)


_GRID: dict = {}
BUILD_THREADS = 4
_TOWERS: dict = {}       # (folder, state, request) -> result (last few)
_DISPLAY: dict = {}      # folder -> ((used frames, preset), shown, limits, recipe)
_TRACES: dict = {}       # folder -> (state, result)
_TOWER_LOCKS: dict = collections.defaultdict(threading.Lock)


def _grid(shape):
    """Flattened row and column numbers of a frame shape (made once)."""
    got = _GRID.get(shape)
    if got is None:
        yy, xx = np.indices(shape)
        got = _GRID[shape] = (yy.ravel().astype(np.float64), xx.ravel().astype(np.float64))
    return got


def _frame_summary(lab) -> dict:
    """{number: (area, cy, cx, y0, y1, x0, x1, pieces)} for one frame; pieces counts separate parts of at least
    the rule's smallest linkable size (rd.MIN_AREA)."""
    from skimage.measure import label as parts_of
    flat = lab.ravel()
    n = int(flat.max()) + 1
    if n == 1:
        return {}
    area = np.bincount(flat, minlength=n)
    yy, xx = _grid(lab.shape)
    sy, sx = np.bincount(flat, yy, n), np.bincount(flat, xx, n)
    parts = parts_of(lab, background=0, connectivity=2)
    part_area = np.bincount(parts.ravel())
    owner = np.zeros(len(part_area), np.int64)
    owner[parts.ravel()] = flat
    big = (part_area >= rd.MIN_AREA) & (np.arange(len(part_area)) > 0)
    pieces = np.bincount(owner[big], minlength=n)
    boxes = ndi.find_objects(lab)
    out = {}
    for k in np.flatnonzero(area):
        if k == 0:
            continue
        s = boxes[k - 1]
        out[int(k)] = (int(area[k]), float(sy[k] / area[k]), float(sx[k] / area[k]),
                       s[0].start, s[0].stop, s[1].start, s[1].stop, int(pieces[k]))
    return out


def _number_summary(lab, n):
    """_frame_summary's entry for one number (None when absent), computed round its box only."""
    mask = lab == n
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if not len(rows):
        return None
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    crop = mask[y0:y1, x0:x1]
    yy, xx = np.nonzero(crop)
    parts, k = ndi.label(crop, structure=np.ones((3, 3), bool))
    pieces = int((np.bincount(parts.ravel())[1:] >= rd.MIN_AREA).sum())
    return (int(len(yy)), float(yy.mean() + y0), float(xx.mean() + x0), y0, y1, x0, x1, pieces)


def _update_summary(summary, before, after):
    """A frame's summary after an edit: only the numbers whose pixels changed are measured again."""
    diff = before != after
    if not diff.any():
        return summary
    out = dict(summary)
    for n in (set(np.unique(before[diff]).tolist()) | set(np.unique(after[diff]).tolist())) - {0}:
        entry = _number_summary(after, int(n))
        if entry is None:
            out.pop(int(n), None)
        else:
            out[int(n)] = entry
    return out


def _compose_frame(rec, nm, ops, f):
    """One frame of the current labels, as rd.compose makes it."""
    data, labels = rec.data, rec.labels
    lab = rd.frame_lut(data, nm, f, int(labels.max()) + 1)[labels[f]] if data['by_frame'][f] else labels[f].copy()
    return rd.apply_ops(lab, f, ops)


def current_stack(rec) -> dict:
    """{'stack': T x H x W current labels, 'per': [frame summaries], 'state': str} (cached per review state).

    After an edit is added only that edit is applied, to the frames it covers; after an undo only the undone
    edit's frames are recomposed (Jamie 2026-11-17: the app got slower as edits grew; a full rebuild of 241
    frames x 219 edits took 1.6 s after every edit). Any other change (a decision) rebuilds everything."""
    from .correction_review import _decisions, _ops
    key = str(rec.folder)
    with _LOCKS[key]:
        state = _state(rec)
        old = _CACHE.get(key)
        if old is not None and old["rec"] is rec and old["state"] == state:
            return old
        decisions, ops = _decisions(rec.folder), _ops(rec.folder)
        changed = None
        if old is not None and old["rec"] is rec and old["decisions"] == decisions:
            before = old["ops"]
            if len(ops) >= len(before) and ops[:len(before)] == before:        # edits added
                stack, added, per = old["stack"], ops[len(before):], list(old["per"])
                changed = sorted(f for f in rd.op_frames(added) if f < len(stack))
                for f in changed:
                    was = stack[f].copy()
                    stack[f] = rd.apply_ops(stack[f], f, added)
                    per[f] = _update_summary(per[f], was, stack[f])
            elif len(ops) < len(before) and before[:len(ops)] == ops:          # edits undone
                stack, nm, per = old["stack"], rd.names(rec.data, decisions), list(old["per"])
                changed = sorted(f for f in rd.op_frames(before[len(ops):]) if f < len(stack))
                for f in changed:
                    was = stack[f].copy()
                    stack[f] = _compose_frame(rec, nm, ops, f)
                    per[f] = _update_summary(per[f], was, stack[f])
        if changed is None:
            # every frame composed as rd.compose does, on several threads (speed pass 2026-11-18)
            from concurrent.futures import ThreadPoolExecutor
            nm, stack = rd.names(rec.data, decisions), np.empty_like(rec.labels)
            same = old is not None and old["rec"] is rec

            def one(f):
                stack[f] = _compose_frame(rec, nm, ops, f)
                return old["per"][f] if same and np.array_equal(old["stack"][f], stack[f]) else _frame_summary(stack[f])
            with ThreadPoolExecutor(BUILD_THREADS) as pool:
                per = list(pool.map(one, range(len(stack))))
        entry = {"rec": rec, "state": state, "stack": stack, "per": per, "timeline": None,
                 "decisions": decisions, "ops": ops}
        _CACHE[key] = entry
        return entry


def cached_frames(rec):
    """frame(f) from the whole-recording labels when they have been built for this recording (brought up to
    date incrementally), else None so the caller composes frames itself."""
    old = _CACHE.get(str(rec.folder))
    if old is None or old["rec"] is not rec:
        return None
    stack = current_stack(rec)["stack"]

    def frame(f):   # read-only: callers must copy before changing a frame
        view = stack[f].view()
        view.flags.writeable = False
        return view
    return frame


def _runs(frames):
    runs = []
    for f in frames:
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    return runs


def _overlap(stack, n, a, b, ra, rb):
    """Whether number n shares a pixel in frames a and b (only the overlap of their two boxes can)."""
    y0, y1, x0, x1 = max(ra[3], rb[3]), min(ra[4], rb[4]), max(ra[5], rb[5]), min(ra[6], rb[6])
    return y0 < y1 and x0 < x1 and bool(((stack[a, y0:y1, x0:x1] == n) & (stack[b, y0:y1, x0:x1] == n)).any())


def _events(n, frames, per, stack, T):
    """Events for one number: [frame, kind, text]."""
    ev = []
    runs = _runs(frames)
    if frames[0] > 0:
        ev.append([frames[0], "start", f"{n} starts at frame {frames[0]}"])
    if frames[-1] < T - 1:
        ev.append([frames[-1], "end", f"{n} ends at frame {frames[-1]}"])
    for (_, b), (c, _) in zip(runs, runs[1:]):
        ev.append([c, "gap", f"{n} missing frames {b + 1}-{c - 1}, back at {c}"])
    piece_frames = []
    for a, b in zip(frames, frames[1:]):
        if b != a + 1:
            continue
        ea, ya, xa = per[a][n][:3]
        eb, yb, xb = per[b][n][:3]
        ratio = max(ea, eb) / max(1, min(ea, eb))
        if ratio >= rd.AREA_RATIO:
            ev.append([b, "size", f"{n} {'grows' if eb > ea else 'shrinks'} x{ratio:.1f} at frame {b}"])
        radii = math.hypot(yb - ya, xb - xa) / math.sqrt(ea / math.pi)
        if radii > rd.JUMP and not _overlap(stack, n, a, b, per[a][n], per[b][n]):
            ev.append([b, "jump", f"{n} jumps {radii:.1f} body radii with no overlap at frame {b}"])
    for f in frames:
        if per[f][n][7] >= 2:
            piece_frames.append(f)
    for a, b in _runs(piece_frames):
        ev.append([a, "pieces", f"{n} is on {per[a][n][7]} separate pieces, frames {a}-{b}"])
    return sorted(ev)


def timeline(rec) -> dict:
    """Every cell's frames and flagged events, plus where joins, edits and questions sit."""
    from .correction_review import _ops, _read
    entry = current_stack(rec)
    if entry["timeline"] is not None:
        return entry["timeline"]
    per, stack, T = entry["per"], entry["stack"], len(entry["stack"])
    present = collections.defaultdict(list)
    for f, summary in enumerate(per):
        for n in summary:
            present[n].append(f)
    cells = [{"n": n, "runs": _runs(fs), "events": _events(n, fs, per, stack, T)} for n, fs in sorted(present.items())]
    edits = []
    for i, op in enumerate(_ops(rec.folder)):
        nums = sorted({int(op[k]) for k in ("a", "b", "from", "to", "cell", "host", "child", "only") if op.get(k)})
        edits.append({"i": i, "op": op["op"], "first": op["first"], "last": op["last"], "cells": nums})
    notes = _read(rec.folder / "notes.json", [])
    out = {"T": T, "cells": cells, "edits": edits,
           "joins": [{"key": it["key"], "frame": it["frame"], "kind": it["kind"],
                      "cells": sorted({n for _, _, n in it["renames"]} | {o for _, o, _ in it["renames"]})}
                     for it in rec.data["items"]],
           "questions": [{"frame": n["frame"], "cell": n.get("cell"), "text": n["text"]} for n in notes
                         if n.get("tag") == "Question" and not n.get("answer")],
           "counts": dict(collections.Counter(e[1] for c in cells for e in c["events"]))}
    entry["timeline"] = out
    return out


def follow_box(rec, cells, lo=None, hi=None):
    """(lo, hi, (y0, y1, x0, x1), (first, last)): the frames shown (the asked ones, else the whole recording, so
    the place a cell appears or disappears can be seen before and after - Jamie 2026-11-17), one crop holding the
    followed cells over the shown frames they are in (over all their frames if none), padded by their largest body
    radius on each side, and the first and last frame any followed cell is in."""
    entry = current_stack(rec)
    per, T = entry["per"], len(entry["per"])
    seen = [f for f in range(T) if any(c in per[f] for c in cells)]
    if not seen:
        raise ValueError(f"Cell{'s' if len(cells) > 1 else ''} {', '.join(map(str, cells))} not in the recording now")
    lo = 0 if lo is None else max(0, int(lo))
    hi = T - 1 if hi is None else min(T - 1, int(hi))
    if lo > hi:
        lo, hi = hi, lo
    rows = [per[f][c] for f in range(lo, hi + 1) for c in cells if c in per[f]] or \
        [per[f][c] for f in seen for c in cells if c in per[f]]
    H, W = entry["stack"].shape[1:]
    pad = int(math.ceil(max(math.sqrt(r[0] / math.pi) for r in rows))) * 2
    y0, y1 = max(0, min(r[3] for r in rows) - pad), min(H, max(r[4] for r in rows) + pad)
    x0, x1 = max(0, min(r[5] for r in rows) - pad), min(W, max(r[6] for r in rows) + pad)
    return lo, hi, (int(y0), int(y1), int(x0), int(x1)), (seen[0], seen[-1])


def prewarm(rec) -> None:
    """In the background: the current labels, the timeline, the plotting library, the tower of every cell and the trace grid,
    so the first Timeline and Tower clicks after opening a recording find them made (speed pass 2026-11-18)."""
    def run():
        try:
            timeline(rec)
            tower(rec, [])
            from .correction_outputs import _traces
            traces_for(rec, _traces)
        except Exception:   # the button works it out itself and reports any problem
            pass
    threading.Thread(target=run, daemon=True, name="correction-prewarm").start()


_SIGNAL_TABLE: dict = {}  # folder -> (state, frames used, cadence) and the per-cell signal table of the tower
_LAST_TOWER: dict = {}    # folder -> the last tower asked for (cells, lo, hi, style, preset, planes)
_TRACE_LOCKS = collections.defaultdict(threading.Lock)
_REFRESH: dict = {}       # folder -> number of the latest edit's refresh (older ones stop)


def refresh_after_edit(rec) -> None:
    """In the background after an edit (speed pass 2026-11-20: the tower took ~5 s and the trace grid ~2 s to
    rebuild on the first click after every edit): the current labels and timeline, the last tower opened and
    the trace grid if it was opened, so those buttons find them made. A newer edit stops an older refresh
    between steps; a click while a step runs waits for that step instead of doing it twice."""
    key = str(rec.folder)
    gen = _REFRESH[key] = _REFRESH.get(key, 0) + 1

    def run():
        import time
        time.sleep(0.4)   # the page's own reload after the edit goes first
        steps = [lambda: timeline(rec)]
        if key in _LAST_TOWER:
            cells, lo, hi, style, preset, planes = _LAST_TOWER[key]
            steps.append(lambda: tower(rec, list(cells), lo, hi, style=style, preset=preset, planes=planes))
        if key in _TRACES:
            from .correction_outputs import _traces
            steps.append(lambda: traces_for(rec, _traces))
        try:
            for step in steps:
                if _REFRESH.get(key) != gen:
                    return
                step()
        except Exception:   # the button works it out itself and reports any problem
            pass
    threading.Thread(target=run, daemon=True, name="correction-refresh").start()


def _display(rec, used, preset, raw):
    """tracking_tower.display_stack of these frames, remembered for the last frames and preset asked (the raw
    frames never change, so only a new frame range or preset filters again)."""
    from ..figure_tables import tracking_tower as tt
    key = (tuple(used), preset)
    got = _DISPLAY.get(str(rec.folder))
    if got is None or got[0] != key:
        shown, limits, recipe = tt.display_stack(raw, preset)
        got = _DISPLAY[str(rec.folder)] = (key, shown, limits, recipe)
    return got[1], got[2], got[3]


def tower(rec, cells, lo=None, hi=None, *, style="line", preset="cd68", planes=8) -> dict:
    """As _tower, remembered per review state and request: reopening it without an edit in between is instant."""
    _LAST_TOWER[str(rec.folder)] = (tuple(cells), lo, hi, style, preset, planes)
    with _TOWER_LOCKS[str(rec.folder)]:   # a click during the background warm-up waits for it, not a second copy
        key = (str(rec.folder), _state(rec), tuple(cells), lo, hi, style, preset, planes)
        if key not in _TOWERS:
            result = _tower(rec, cells, lo, hi, style=style, preset=preset, planes=planes)
            _TOWERS[key] = result
            while len(_TOWERS) > 6:
                _TOWERS.pop(next(iter(_TOWERS)))
        return _TOWERS[key]


def traces_for(rec, make):
    """make(rec), remembered per review state (the trace grid)."""
    with _TRACE_LOCKS[str(rec.folder)]:   # a click during a background rebuild waits for it
        state = _state(rec)
        got = _TRACES.get(str(rec.folder))
        if got is None or got[0] != state:
            got = _TRACES[str(rec.folder)] = (state, make(rec))
        return got[1]


def _tower(rec, cells, lo=None, hi=None, *, style="line", preset="cd68", planes=8) -> dict:
    """The package tracking tower of the whole field and every cell over frames lo..hi (default the whole
    recording), as a PNG data URL; followed cells (if any) are emphasised, the rest faded (Jamie 2026-11-17:
    'the full tower with all cells and all lines')."""
    from matplotlib.figure import Figure
    from ..figure_tables import tracking_tower as tt
    from ..measure.cell_signal import analyse as cell_signal
    from ..visualisation.panels.tracking_tower import tower as draw_tower
    stack = current_stack(rec)["stack"]
    T, H, W = stack.shape
    lo = 0 if lo is None else max(0, int(lo))
    hi = T - 1 if hi is None else min(T - 1, int(hi))
    if hi - lo < 1:
        raise ValueError("The tower needs at least two frames")
    y0, y1, x0, x1 = 0, H, 0, W
    # The tower's layout (arrows, label offsets) is sized for a full field: a small crop is enlarged to about
    # TOWER_SIDE pixels, and the track lines use at most TOWER_FRAMES evenly spaced frames (always the last).
    step = max(1, math.ceil((hi - lo + 1) / TOWER_FRAMES))
    used = list(range(lo, hi + 1, step))
    if used[-1] != hi:
        used.append(hi)
    lab = stack[used][:, y0:y1, x0:x1].copy()
    raw = np.stack([rec.raw_frame(f)[y0:y1, x0:x1] for f in used]).astype(np.float32)
    cadence = (rec.interval_min or 30) / 60
    hours = (np.array(used) - lo) * cadence
    shown, limits, recipe = _display(rec, used, preset, raw)
    sig_key = (str(rec.folder), _state(rec), tuple(used), cadence * step)   # the same for every emphasis
    got = _SIGNAL_TABLE.get(str(rec.folder))
    if got is None or got[0] != sig_key:
        got = _SIGNAL_TABLE[str(rec.folder)] = (sig_key, cell_signal(raw, lab, source_frame_offset=0,
                                                                     frame_interval_h=cadence * step))
    table = got[1]
    k = max(1, round(TOWER_SIDE / max(lab.shape[1:])))
    lab, shown = (np.repeat(np.repeat(a, k, 1), k, 2) for a in (lab, shown))
    settings = tt.settings(snapshot_interval_h=max(cadence, hours[-1] / max(1, planes - 1)),
                           reference_duration_h=None, figure_width_in=7., time_label_size=13., axis_label_size=12.)
    data = tt.analyse(lab, shown, display_range=limits, frame_hours=hours, measurements=table, style=style,
                      tower=settings, lut=recipe["lut"], display_gamma=recipe["gamma"],
                      emphasis=set(cells) if cells else None)
    figure = Figure()
    draw_tower(figure, data)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=90, facecolor="white", bbox_inches="tight")
    frames = [used[int(f)] for f in data["checks"]["source_frames"]]
    return {"image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
            "frames": frames, "cells": list(cells), "lo": lo, "hi": hi, "box": [y0, y1, x0, x1],
            "identities": [int(i) for i in data["checks"]["identities"]],
            "colours": {int(c): "#%02x%02x%02x" % identity_rgb(c) for c in cells},
            "snapshot_hours": round(settings.snapshot_interval_h, 2), "display": preset}
