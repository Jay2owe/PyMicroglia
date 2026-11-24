"""Correction reviewer: every name join beside Before / After / Raw / Changes (Motion's reviewer, in the workbench).

Port of Motion's correction reviewer (Motion/manual_editing/trajectory_review/correction_reviewer.py, Jamie
2026-10-26). The pictures are made by the same code: the crop and frames of each join (item_window), the
per-join contrast, the numbers on every cell, the whole-recording map and the panels (bundle, panels) are
copied unchanged; only where a recording's files come from differs. The edits are the same too: swap, give
number, delete, piece only, draw split (Motion's split engine) and keep split / whole recording.

A *review set* is one recording:

    <root>/agent_review/<stem>/review.json     the joins (Jamie's rule, pymicroglia.tracking.correction_rules)
                               decisions.json  {item key: "approve" | "deny"}
                               notes.json      [{item, frame, cell, panel, y, x, text, time, tag}]
                               edits.json      [{op, first, last, ...}]  applied after the decisions
                               exports.json    every export and the Motion session it made

Applied joins stay unless denied; unsure joins apply only when approved. Export writes the result as one new
Motion editor session through the Motion service (apply_identity_patch). Motion is never imported.
"""
from __future__ import annotations

import base64
import collections
import datetime as _dt
import hashlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request as UrlRequest, urlopen

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

from ..tracking import correction_edit_tools as et
from ..tracking import correction_note_actions as na
from ..tracking import correction_rules as rd

SCHEMA = "pymicroglia.correction-review.v1"
FOLDER = "agent_review"
# ---- as in correction_reviewer.py
CYAN, MAGENTA, GREEN, RED, ORANGE = (0, 220, 255), (255, 70, 220), (60, 230, 90), (255, 40, 40), (255, 150, 0)
GREY = (120, 120, 120)
PURPLE = (190, 120, 255)  # pixels changed by Jamie's manual edits
OVERVIEW = 300  # px, whole-recording map
PANELS = ("Before", "After", "Raw", "Changes")
CHANNELS = {'Red': ('red_registered.tif', 'Original red channel'),
            'Green': ('green_registered.tif', 'Original green channel'),
            'Phase': ('phase_registered.tif', 'Phase contrast (transmitted light)')}
QUICK_NOTES = (("Swap", "Two numbers swap cells"), ("Same cell", "Two numbers are one cell"),
               ("Split", "One cell split over several numbers, or one number over several cells"),
               ("Not a cell", "No cell here: delete this number"),
               ("Missed join", "The number stops and the cell comes back under another number"),
               ("Wrong number", "This cell should carry another number"))
NOTE_KINDS = tuple(label for label, _ in QUICK_NOTES)
EDIT_OPS = ("swap", "renumber", "delete", "piece", "split", "keep", "paint")
_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(2)


# ----------------------------------------------------------------------------- drawing (copied unchanged)
def edges(lab):
    e = np.zeros(lab.shape, bool)
    e[1:] |= lab[1:] != lab[:-1]; e[:-1] |= lab[:-1] != lab[1:]
    e[:, 1:] |= lab[:, 1:] != lab[:, :-1]; e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    return e & (lab > 0)


def thick(mask):
    out = mask.copy()
    out[1:] |= mask[:-1]; out[:-1] |= mask[1:]; out[:, 1:] |= mask[:, :-1]; out[:, :-1] |= mask[:, 1:]
    return out


def centroids(lab):
    """{label: (y, x)} for every label in a crop."""
    flat = lab.ravel()
    n = int(flat.max())+1
    cnt = np.bincount(flat, minlength=n)
    yy, xx = np.indices(lab.shape)
    sy = np.bincount(flat, yy.ravel(), n); sx = np.bincount(flat, xx.ravel(), n)
    return {int(k): (sy[k]/cnt[k], sx[k]/cnt[k]) for k in np.flatnonzero(cnt) if k}


def up(a, k):
    return np.repeat(np.repeat(a, k, 0), k, 1)


QUESTION_BEFORE, QUESTION_AFTER = 8, 30   # frames shown around a question's frame


def place_key(frame, y, x) -> str:
    """The view of one assistant question: its frame and the clicked pixel (full-frame coordinates)."""
    return f"place:{int(frame)}:{int(y)}:{int(x)}"


def _place(rec, key):
    """A question's view as a pseudo-join: no renames, centred on the asked cell, frames around it."""
    try:
        f, y, x = (int(v) for v in key.split(":")[1:4])
    except ValueError:
        raise ValueError("Unknown question view") from None
    T = rec.data['T']
    if not (0 <= f < T):
        raise ValueError("Question frame is outside the recording")
    cell = int(current_frame_fn(rec)(f)[y, x])
    return {"key": key, "kind": "question", "check": False, "frame": f, "centre": (y, x), "radius": 22,
            "renames": [], "cells": [cell] if cell else [],
            "lo": max(0, f-QUESTION_BEFORE), "hi": min(T-1, f+QUESTION_AFTER)}


def follow_key(cells, lo=None, hi=None) -> str:
    """The view that follows cells (their numbers now) over frames lo..hi (blank: every frame they are in)."""
    return f"follow:{'' if lo is None else int(lo)}:{'' if hi is None else int(hi)}:" + ",".join(
        str(int(c)) for c in sorted(set(cells)))


def _follow(rec, key):
    """Followed cells as a pseudo-join: one fixed crop holding them all, over their frames (Jamie 2026-11-17)."""
    from .correction_cells import follow_box
    try:
        _, lo, hi, listed = key.split(":", 3)
        cells = sorted({int(c) for c in listed.split(",") if c.strip()})
        lo, hi = (int(lo) if lo else None), (int(hi) if hi else None)
    except ValueError:
        raise ValueError("Unknown follow view") from None
    if not cells:
        raise ValueError("Right-click a cell to follow it")
    lo, hi, box, present = follow_box(rec, cells, lo, hi)
    return {"key": key, "kind": "follow", "check": False, "frame": present[0], "renames": [], "cells": cells,
            "lo": lo, "hi": hi, "box": box, "present": present,
            "centre": ((box[0]+box[1])//2, (box[2]+box[3])//2), "radius": 0}


def item_window(rec, it):
    if it.get('kind') in ('question', 'follow'):
        return it['lo'], it['hi']
    T = rec.data['T']
    j = it['frame']
    last = max(rec.data['tracks'][k][2] for k, _, _ in it['renames'])
    lo, hi = max(0, j-6), min(T-1, max(j+6, min(last, j+30)+2))
    return lo, hi


def _contrast(frames):
    """Display range of the raw frames (finite pixels)."""
    v = np.concatenate([r[np.isfinite(r)] for r in frames])
    lo_v, hi_v = np.percentile(v, [1, 99.5])
    return lo_v, max(hi_v, lo_v+1e-6)


def channel_range(rec, name, frames, box):
    """Contrast of one extra channel over the item's frames (a sample of them for long views)."""
    y0, y1, x0, x1 = box
    v = np.concatenate([rec.channel_frame(name, f)[y0:y1, x0:x1].astype(np.float32).ravel() for f in frames])
    lo_v, hi_v = np.percentile(v, [0.5, 99.5])
    return lo_v, max(hi_v, lo_v+1e-6)


def channel_crop(rec, name, f, box, rng):
    y0, y1, x0, x1 = box
    c = rec.channel_frame(name, f)[y0:y1, x0:x1].astype(np.float32)
    gamma = 1.0 if name == 'Phase' else .7
    return (255*np.clip((c-rng[0])/(rng[1]-rng[0]), 0, 1)**gamma).astype(np.uint8)


SAMPLE = 40   # frames used for a view's contrast (all of them in a join's window, which is shorter)


class _Frames:
    """A view's frames, drawn when first asked for: a follow view can cover the whole recording, so only the
    most recent frames of a long view are kept."""

    def __init__(self, make, lo, hi):
        self.make, self.lo, self.hi = make, lo, hi
        self.keep = hi - lo + 1 if hi - lo < 60 else 48
        self.done = collections.OrderedDict()
        self.lock = threading.Lock()

    def __contains__(self, f):
        return isinstance(f, (int, np.integer)) and self.lo <= f <= self.hi

    def __getitem__(self, f):
        if f not in self:
            raise KeyError(f)
        with self.lock:
            if f in self.done:
                self.done.move_to_end(f)
                return self.done[f]
        fr = self.make(f)
        with self.lock:
            self.done[f] = fr
            while len(self.done) > self.keep:
                self.done.popitem(last=False)
        return fr


def _box(rec, it):
    """The crop of an item: its own box (follow views) or a square around the join, as the reviewer drew it."""
    if it.get('box'):
        return tuple(int(v) for v in it['box'])
    H, W = rec.labels.shape[1:]
    half = int(min(110, max(40, 4*it['radius'])))
    cy, cx = it['centre']
    y0 = int(min(max(cy-half, 0), max(H-2*half, 0))); x0 = int(min(max(cx-half, 0), max(W-2*half, 0)))
    return y0, min(H, y0+2*half), x0, min(W, x0+2*half)


def bundle(rec, it, decisions, channels=(), ops=(), rev=0):
    """Crops for every frame of one item: before/after/raw/changes, any extra channels, the overview.

    Before is the current numbers without this join (every other decision and Jamie's edits applied), After the
    current numbers with it. Departure from Motion's reviewer, at Jamie's request (2026-11-11): its Before showed
    the source numbers, so after a swap the picture kept the old number while a click reported the new one, and
    a deleted cell stayed drawn but could not be clicked. Frames are drawn when first shown (2026-11-17), so a
    follow view can span the whole recording; contrast comes from up to SAMPLE frames spread over the item."""
    data = rec.data
    lo, hi = item_window(rec, it)
    forced = dict(decisions); forced[it['key']] = 'approve'
    nm_after = rd.names(data, forced)
    without = dict(decisions); without[it['key']] = 'deny'
    nm_before = rd.names(data, without)
    nm_now = rd.names(data, decisions)
    targets = {new for _, _, new in it['renames']} | set(it.get('cells', ()))
    olds = {data['tracks'][k][0] for k, _, _ in it['renames']}
    y0, y1, x0, x1 = box = _box(rec, it)
    sample = sorted({int(round(v)) for v in np.linspace(lo, hi, min(SAMPLE, hi-lo+1))})
    lo_v, hi_v = _contrast([rec.raw_frame(f)[y0:y1, x0:x1] for f in sample])
    scale = rec.ov_scale
    channels = tuple(c for c in channels if c in rec.channels)
    ranges = {c: channel_range(rec, c, sample, box) for c in channels}
    seen = set()
    for f in (() if it.get('kind') == 'follow' else range(lo, hi+1)):   # follow views depend on every decision
        present = set(np.unique(rec.labels[f, y0:y1, x0:x1]).tolist())
        seen.update(k for k in data['by_frame'][f] if data['tracks'][k][0] in present)

    def make(f):
        r = rec.raw_frame(f)[y0:y1, x0:x1]
        g = (255*np.clip((np.nan_to_num(r)-lo_v)/(hi_v-lo_v), 0, 1)**.7).astype(np.uint8)
        src = rec.labels[f, y0:y1, x0:x1]
        relabel = (lambda nm: rd.frame_lut(data, nm, f, rec.size)[src]) if data['by_frame'][f] else (lambda nm: src)
        before, after, now = relabel(nm_before), relabel(nm_after), relabel(nm_now)
        if ops:
            before = rd.apply_ops(before, f, ops, y0, x0)
            after = rd.apply_ops(after, f, ops, y0, x0)
            edited_now = rd.apply_ops(now, f, ops, y0, x0)
            edited, now = edited_now != now, edited_now
        else:
            edited = np.zeros(src.shape, bool)
        item_px = np.zeros(src.shape, bool)
        active = []
        for k, _, _ in it['renames']:
            name, first, last = data['tracks'][k]
            if first <= f <= last:
                item_px |= src == name
                active.append(name)
        changed_now = (src != now) & ~item_px & ~edited
        def nums(lab, strong, colour):
            return [(y, x, str(n), colour if n in strong else (200, 200, 200)) for n, (y, x) in centroids(lab).items()]
        small = rec.labels[f][rec.ov_y[:, None], rec.ov_x[None, :]]
        small_now = rd.frame_lut(data, nm_now, f, rec.size)[small] if data['by_frame'][f] else small
        if ops:  # splits are pixel lists, too fine for this map
            small_now = rd.apply_ops(small_now, f, ops, splits=False)
        ov = np.zeros(small.shape+(3,), np.uint8)
        ov[small > 0] = (70, 70, 70)
        ov[small != small_now] = ORANGE
        if active:
            ov[np.isin(small, active)] = RED
        return dict(grey=g, before=before, after=after, item=item_px, other=changed_now,
                    nb=nums(before, targets | olds, (255, 255, 255)), na=nums(after, targets, (160, 255, 170)),
                    overview=ov, ch={c: channel_crop(rec, c, f, box, ranges[c]) for c in channels}, now=now,
                    edited=edited)

    if it.get('kind') == 'follow':   # its crop follows every decision, so it depends on all of them
        deps = {x['key']: decisions.get(x['key']) for x in data['items']}
    else:   # decisions this picture depends on: items renaming any track visible in it
        deps = {key: decisions.get(key) for k in seen | {k for k, _, _ in it['renames']} for key in rec.items_of[k]}
    return dict(key=it['key'], kind=it.get('kind', 'join'), lo=lo, hi=hi,
                box=(y0*scale, x0*scale, y1*scale, x1*scale), frames=_Frames(make, lo, hi), targets=list(targets),
                origin=(y0, x0), channels=channels, rev=rev, deps=deps)


def panels(fr, targets, k, colours=None, every=None, current=False):
    """Before / After (or one Current panel) / Raw / Changes at integer zoom k, outlines one display pixel wide
    (two when highlighted). colours {number: rgb}: followed cells, drawn thick in their own colour on every
    outline panel; every: a colour per number (colour_lut) for all outlines instead of grey (2026-11-17)."""
    grey = np.repeat(up(fr['grey'], k)[..., None], 3, 2)
    item = up(fr['item'], k)
    colours = colours or {}
    out = {}

    def follow(im, lab, width):
        for n, col in colours.items():
            m = lab == n
            if m.any():
                e = edges(np.where(m, lab, 0))
                im[thick(e) if width > 1 else e] = col

    views = ((('Current', up(fr['before'], k)),) if current else
             (('Before', up(fr['before'], k)), ('After', up(fr['after'], k))))
    for name, lab in views:
        im = grey.copy()
        e = edges(lab)
        im[e] = GREY if every is None else every[np.minimum(lab[e], len(every)-1)]
        tgt = np.isin(lab, targets)
        if name == 'Before':
            im[thick(edges(np.where(tgt & ~item, lab, 0)))] = CYAN
            im[thick(edges(np.where(item, lab, 0)))] = MAGENTA
        elif name == 'After':
            im[thick(edges(np.where(tgt, lab, 0)))] = GREEN
        else:
            im[thick(edges(np.where(tgt & ~np.isin(lab, list(colours)), lab, 0)))] = CYAN
        follow(im, lab, 2)
        out[name] = im
    out['Raw'] = grey
    ch = (grey*0.45).astype(np.uint8)
    ch[up(fr['other'], k)] = ORANGE
    changed = up(fr['item'] & (fr['before'] != fr['after']), k)
    ch[changed] = RED
    ch[up(fr['edited'], k)] = PURPLE
    if fr['ch']:
        # Extra channels carry the Before colours as thin outlines so the signal underneath stays visible.
        lab = up(fr['before'], k)
        tgt = np.isin(lab, targets)
        for name, g in fr['ch'].items():
            im = np.repeat(up(g, k)[..., None], 3, 2)
            im[edges(np.where(tgt & ~item, lab, 0))] = CYAN
            im[edges(np.where(item, lab, 0))] = MAGENTA
            follow(im, lab, 1)
            out[name] = im
    out['Changes'] = ch
    return out


# ----------------------------------------------------------------------------- one recording
def _resolve(base: Path, entry: dict) -> Path:
    path = Path(entry["path"])
    return path if path.is_absolute() or entry.get("path_kind") == "external_absolute" else base / path


def source_info(folder: str | Path) -> dict:
    """Labels, raw stack and frame interval of a Motion editing bundle or saved session folder."""
    folder = Path(folder)
    bundle_file, session_file = folder / "motion-editing-bundle.json", folder / "motion-editing-session.json"
    if bundle_file.is_file():
        manifest = json.loads(bundle_file.read_text(encoding="utf-8"))
        labels = _resolve(folder, manifest["assets"]["canonical_labels"])
        labels_sha = manifest["assets"]["canonical_labels"].get("sha256")
        kind = "bundle"
    elif session_file.is_file():
        session = json.loads(session_file.read_text(encoding="utf-8"))
        labels = _resolve(folder, session["outputs"]["curated_labels"])
        labels_sha = session["outputs"]["curated_labels"].get("sha256")
        bundle_file = _resolve(folder, session["parent_bundle"])
        manifest = json.loads(Path(bundle_file).read_text(encoding="utf-8"))
        kind = "session"
    else:
        raise ValueError(f"{folder} is not a Motion editing bundle or saved session")
    raw_entry = manifest["assets"]["registered_raw"]
    return {"folder": str(folder), "kind": kind, "stem": manifest.get("stem") or folder.name,
            "labels": str(labels), "labels_sha256": labels_sha,
            "raw": str(_resolve(Path(bundle_file).parent, raw_entry)),
            "raw_frames": raw_entry.get("label_frame_to_source"),
            "frame_interval_min": (manifest.get("time") or {}).get("frame_interval_min")}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Recording:
    """Labels, raw movie and review list of one recording (as correction_reviewer.Recording)."""

    def __init__(self, folder: Path):
        review = json.loads((folder / "review.json").read_text(encoding="utf-8"))
        src = review["source"]
        self.folder, self.stem = folder, review["stem"]
        labels_path = Path(src["labels"])
        if _sha(labels_path) != src["labels_sha256"]:
            raise ValueError("The source labels changed since this review set was made; make it again")
        self.labels = tifffile.imread(labels_path)
        self.data = rd.index_frames(review)
        self.raw = tifffile.TiffFile(src["raw"])
        self.pages = self.raw.series[0].pages
        frames = src.get("raw_frames") or list(range(len(self.labels)))
        self.display = {k: int(v) for k, v in enumerate(frames)}
        interval = src.get("frame_interval_min")
        if interval is None:
            try:
                interval = source_info(src["folder"]).get("frame_interval_min")
            except (OSError, ValueError, KeyError):
                interval = None
        self.interval_min = interval
        self.hours = {k: (f"{k*interval/60:g}" if interval else "?") for k in self.display}
        self.size = int(self.labels.max())+1
        H, W = self.labels.shape[1:]
        self.ov_scale = OVERVIEW/max(H, W)
        self.ov_y = np.minimum((np.arange(int(H*self.ov_scale))+.5)/self.ov_scale, H-1).astype(int)
        self.ov_x = np.minimum((np.arange(int(W*self.ov_scale))+.5)/self.ov_scale, W-1).astype(int)
        self.items_of = collections.defaultdict(list)
        for it in self.data['items']:
            for k, _, _ in it['renames']:
                self.items_of[k].append(it['key'])
        self.lock = threading.Lock()
        self.raw_cache = collections.OrderedDict()
        self.channels = {}          # Red / Green / Phase stacks: none recorded for these bundles
        self.chan_files, self.chan_cache = {}, {}
        self.bundles = {}           # item key -> bundle
        self.pending = {}
        self.composed = collections.OrderedDict()

    def raw_frame(self, f):
        with self.lock:
            if f in self.raw_cache:
                self.raw_cache.move_to_end(f)
                return self.raw_cache[f]
            a = self.pages[self.display[f]].asarray().astype(np.float32)
            self.raw_cache[f] = a
            if len(self.raw_cache) > 120:
                self.raw_cache.popitem(last=False)
            return a

    def channel_frame(self, name, f):
        with self.lock:
            cache = self.chan_cache[name]
            if f in cache:
                cache.move_to_end(f)
                return cache[f]
            t = self.chan_files.get(name)
            if t is None:
                t = self.chan_files[name] = tifffile.TiffFile(self.channels[name])
                if t.series[0].shape != self.raw.series[0].shape:
                    raise ValueError(f'{self.channels[name].name} is not on the raw movie canvas')
            a = cache[f] = t.series[0].pages[self.display[f]].asarray()
            if len(cache) > 60:
                cache.popitem(last=False)
            return a


_RECORDINGS: dict[tuple[str, float], Recording] = {}
PREWARM = False   # the Workbench sets this: a recording opened in the reviewer builds its timeline and tower in the background


def _review_dir(root: str | Path, stem: str) -> Path:
    if not stem or any(c in stem for c in "/\\:") or stem.startswith("."):
        raise ValueError("Choose a review set by its recording name")
    return Path(root) / FOLDER / stem


def _read(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _write(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    # edits.json is written compact with orjson (speed pass 2026-11-19: Python's encoder took 0.66 s per edit at
    # 14 MB; indenting it took 9.5 s, 2026-11-18); the other files are small and stay readable.
    edits = path.name == "edits.json"
    if edits:
        import orjson
        tmp.write_bytes(orjson.dumps(value, default=int, option=orjson.OPT_SERIALIZE_NUMPY))
    else:
        tmp.write_text(json.dumps(value, default=int, indent=1), encoding="utf-8")
    for wait in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, None):
        try:   # Windows refuses the swap while another request or the background warm-up is reading the file
            os.replace(tmp, path)
            break
        except PermissionError:
            if wait is None:
                raise
            time.sleep(wait)
    if edits:   # remember what was written, so the next read does not parse 14 MB again
        _OPS_CACHE[str(path)] = (_stat_key(path), [dict(op) for op in value])
        _LIGHT.pop(str(path), None)


def _rec(root, stem) -> Recording:
    folder = _review_dir(root, stem)
    key = (str(folder.resolve()), (folder / "review.json").stat().st_mtime)
    with _LOCK:
        rec = _RECORDINGS.get(key)
        if rec is None:
            for old in [k for k in _RECORDINGS if k[0] == key[0]]:
                _RECORDINGS.pop(old).raw.close()
            rec = _RECORDINGS[key] = Recording(folder)
            if PREWARM:
                from .correction_cells import prewarm
                prewarm(rec)
                warm_model(rec)
                from ..tracking.correction_grow import DEFAULT_LEVEL
                warm_signal(rec, DEFAULT_LEVEL)   # the first Seed / Grow / preview at the default level finds it made
    return rec


def _state(root, stem):
    """(folder, data, labels), for the page assistant."""
    rec = _rec(root, stem)
    return rec.folder, rec.data, rec.labels


def _decisions(folder):
    return _read(folder / "decisions.json", {})


_OPS_CACHE: dict = {}


def _stat_key(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size, st.st_ino)   # each save is a new file (os.replace): new st_ino


_LIGHT: dict = {}


def _edit_list(folder):
    """The edits without their pixel runs: all the page shows (history, counts). Speed pass 2026-11-19: every
    reload sent the whole 14 MB edits.json to the browser."""
    path = folder / "edits.json"
    key = _stat_key(path)
    got = _LIGHT.get(str(path))
    if got is None or got[0] != key:
        got = _LIGHT[str(path)] = (key, [{k: v for k, v in op.items() if k not in ("runs", "moves")}
                                         for op in _ops(folder, copy=False)])
    return got[1]


def _ops(folder, copy=True):
    """The edits, parsed once per change of edits.json (speed pass 2026-11-18: 13 MB took 0.3 s to read on every
    request). Each call gets its own list of its own edit dicts, so callers may change them."""
    path = folder / "edits.json"
    key = _stat_key(path)
    if key is None:
        return []
    got = _OPS_CACHE.get(str(path))
    if got is None or got[0] != key:
        import orjson
        got = _OPS_CACHE[str(path)] = (key, orjson.loads(path.read_bytes()))
    return [dict(op) for op in got[1]] if copy else list(got[1])


def create_review(root: str | Path, source: str | Path, *, agent_session: str | Path | None = None,
                  note: str = "") -> dict:
    """Run Jamie's rule on a source session and save the review list (never changes the source)."""
    info = source_info(source)
    labels = tifffile.imread(info["labels"])
    actual = _sha(Path(info["labels"]))
    if info["labels_sha256"] and actual != info["labels_sha256"]:
        raise ValueError(f"{info['labels']} does not match its manifest fingerprint")
    data = rd.build(labels)
    folder = _review_dir(root, info["stem"])
    folder.mkdir(parents=True, exist_ok=True)
    review = {"schema": SCHEMA, "stem": info["stem"], "created_utc": datetime.now(timezone.utc).isoformat(),
              "source": {**info, "labels_sha256": actual},
              "agent_session": str(agent_session) if agent_session else None,
              "rules": {"max_gap": rd.MAX_GAP, "max_radii": rd.MAX_RADII, "min_area": rd.MIN_AREA,
                        "area_ratio": rd.AREA_RATIO, "loose_gap": rd.LOOSE_GAP, "loose_radii": rd.LOOSE_RADII},
              "shape": list(labels.shape), "note": note, **data}
    _write(folder / "review.json", review)
    for name, empty in (("decisions.json", {}), ("notes.json", []), ("edits.json", []), ("exports.json", [])):
        if not (folder / name).exists():
            _write(folder / name, empty)
    return summary(root, info["stem"])


def list_reviews(root: str | Path) -> list[dict]:
    base = Path(root) / FOLDER
    if not base.is_dir():
        return []
    return [summary(root, p.name) for p in sorted(base.iterdir()) if (p / "review.json").is_file()]


def summary(root, stem) -> dict:
    folder = _review_dir(root, stem)
    review = _read(folder / "review.json", None)
    if review is None:
        raise ValueError(f"No review set named {stem}")
    decisions, items = _decisions(folder), review["items"]
    return {"stem": stem, "source": review["source"]["folder"], "frames": review["T"],
            "applied": sum(it["kind"] == "applied" for it in items),
            "unsure": sum(it["kind"] == "unsure" for it in items),
            "decided": sum(it["key"] in decisions for it in items),
            "notes": len(_read(folder / "notes.json", [])), "edits": len(_ops(folder)),
            "exports": _read(folder / "exports.json", [])}


def stamp(root, stem) -> float:
    """Latest change to a review set's saved files (the page reloads when an agent edits them)."""
    folder = _review_dir(root, stem)
    return max((folder / n).stat().st_mtime for n in ("review.json", "decisions.json", "notes.json",
                                                       "edits.json", "exports.json") if (folder / n).exists())


def load(root, stem) -> dict:
    """Everything the page lists: one row per join (as the reviewer's table), decisions, notes, edits."""
    rec = _rec(root, stem)
    data, folder = rec.data, rec.folder
    rows = []
    for it in data["items"]:
        spans = [data["tracks"][k][1:] for k, _, _ in it["renames"]]
        rows.append({**{k: it[k] for k in ("key", "kind", "check", "frame", "radii", "gap", "area_ratio")},
                     "old": sorted({o for _, o, _ in it["renames"]}), "new": sorted({n for _, _, n in it["renames"]}),
                     "first": min(a for a, _ in spans), "last": max(b for _, b in spans)})
    notes = _read(folder / "notes.json", [])
    questions, current = [], None
    for n in notes:   # each open question with a clicked cell gets its own row, opening at that cell and frame
        if n.get("tag") == "Question" and not n.get("answer") and n.get("y") is not None:
            f = int(n["frame"])
            current = current or current_frame_fn(rec)
            cell = int(current(f)[int(n["y"]), int(n["x"])]) or n.get("cell")   # the number there now
            questions.append({"key": place_key(f, n["y"], n["x"]), "kind": "question", "check": False, "frame": f,
                              "radii": "", "gap": "", "area_ratio": "", "old": [cell] if cell else [],
                              "new": [], "first": max(0, f-QUESTION_BEFORE),
                              "last": min(data["T"]-1, f+QUESTION_AFTER), "text": n["text"]})
    rows = questions + rows
    decisions = _decisions(folder)
    review = _read(folder / "review.json", {})
    return {"stem": stem, "frames": data["T"], "shape": data["shape"], "items": rows, "hours": rec.hours,
            "source": review.get("source", {}).get("folder"), "agent_session": review.get("agent_session"),
            "rules": review.get("rules"), "decisions": decisions, "notes": notes,
            "edits": _edit_list(folder), "exports": _read(folder / "exports.json", []),
            "note_kinds": [{"label": a, "tip": b} for a, b in QUICK_NOTES], "channels": sorted(rec.channels),
            "conflicts": rd.conflicts(data, decisions)}


def _check_edit(edit: dict, frames: int) -> dict:
    op = edit.get("op")
    if op not in EDIT_OPS:
        raise ValueError(f"Edit must be one of {', '.join(EDIT_OPS)}")
    first, last = int(edit["first"]), int(edit["last"])
    if not 0 <= first <= last < frames:
        raise ValueError("Edit frames are outside the recording")
    return json.loads(json.dumps(edit, default=int))   # plain numbers only, as the reviewer saves them


def save(root, stem, *, decisions=None, notes=None, edits=None) -> dict:
    """Replace the saved decisions, notes and/or edits (atomic; nothing else is written)."""
    rec = _rec(root, stem)
    with _LOCK:
        keys = {it["key"] for it in rec.data["items"]}
        if decisions is not None:
            bad = [k for k, v in decisions.items() if k not in keys or v not in ("approve", "deny")]
            if bad:
                raise ValueError(f"Unknown decisions: {bad[:3]}")
            if edits is None:   # edits keep acting on the cells they were made on (rd.follow_decisions)
                followed, changed = rd.follow_decisions(rec.data, _decisions(rec.folder), decisions, _ops(rec.folder))
                if changed:
                    _write(rec.folder / "edits.json", followed)
            _write(rec.folder / "decisions.json", decisions)
        if notes is not None:
            _write(rec.folder / "notes.json", list(notes))
        if edits is not None:
            _write(rec.folder / "edits.json", [_check_edit(e, rec.data["T"]) for e in edits])
    return load(root, stem)


# ----------------------------------------------------------------------------- decisions (as Reviewer.decide)
def decide(root, stem, key: str, value: str | None) -> dict:
    rec = _rec(root, stem)
    mine = _decisions(rec.folder)
    if key not in {it["key"] for it in rec.data["items"]}:
        raise ValueError("Unknown join")
    trial = dict(mine)
    if value is None:
        trial.pop(key, None)
    else:
        trial[key] = value
    new = set(rd.conflicts(rec.data, trial)) - set(rd.conflicts(rec.data, mine))
    if new:
        others = [str(i+1) for i, x in enumerate(rec.data["items"]) if x["key"] in new and x["key"] != key]
        return {"ok": False, "message": f"Not possible: two cells would share a number with correction(s) "
                                        f"{', '.join(others[:5])} - decide those first"}
    save(root, stem, decisions=trial)
    _refresh(rec)
    return {"ok": True, "previous": mine.get(key), "decisions": trial, "edits": _edit_list(rec.folder)}


# ----------------------------------------------------------------------------- pictures
FIELD = "~field:"   # key suffix: the reviewer's zoomed-out or moved crop, full-frame y0,y1,x0,x1 (Jamie 2026-11-17)


def base_key(key):
    """The item key without a field suffix (what decisions, edits and notes are stored under)."""
    return key.partition(FIELD)[0] if key else key


def _item(rec, key):
    key, _, field = key.partition(FIELD)
    if key.startswith("place:"):
        it = _place(rec, key)
    elif key.startswith("follow:"):
        it = _follow(rec, key)
    else:
        it = next((x for x in rec.data["items"] if x["key"] == key), None)
        if it is None:
            raise ValueError("Unknown join")
    if field:
        H, W = rec.labels.shape[1:]
        try:
            y0, y1, x0, x1 = (int(v) for v in field.split(","))
        except ValueError:
            raise ValueError("Unknown field of view") from None
        y0, y1, x0, x1 = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
        if y1 - y0 < 8 or x1 - x0 < 8:
            raise ValueError("Field of view too small")
        it = dict(it, box=(y0, y1, x0, x1))
    return it


_REV_CACHE: dict = {}


def _rev(folder) -> str:
    """sha1 of edits.json, worked out once per change of the file (same value as before)."""
    path = folder / "edits.json"
    key = _stat_key(path)
    got = _REV_CACHE.get(str(path))
    if got is None or got[0] != key:
        got = _REV_CACHE[str(path)] = (key, hashlib.sha1(path.read_bytes() if path.exists() else b"").hexdigest())
    return got[1]


def _valid(b, decisions, rev, channels=None):
    return (b["rev"] == rev and all(decisions.get(k) == v for k, v in b["deps"].items())
            and (channels is None or tuple(b["channels"]) == channels))


def _bundle(rec, key, prefetch=(), channels=None):
    """This join's pictures (made once, redone when a decision it depends on, an edit or the shown
    channels change). channels=None takes whatever is cached (for clicks)."""
    decisions, rev = _decisions(rec.folder), _rev(rec.folder)
    if channels is not None:
        channels = tuple(c for c in CHANNELS if c in channels and c in rec.channels)
    b = rec.bundles.get(key)
    if b is None or not _valid(b, decisions, rev, channels):
        fut = rec.pending.pop(key, None)
        b = fut.result() if fut is not None and not fut.cancelled() else None
        if b is None or not _valid(b, decisions, rev, channels):
            b = bundle(rec, _item(rec, key), decisions, channels or (), _ops(rec.folder), rev)
        rec.bundles[key] = b
        if len(rec.bundles) > 16:
            rec.bundles.pop(next(iter(rec.bundles)))
    for other in prefetch:   # the next joins, drawn in the background as the reviewer does
        known = rec.bundles.get(other)
        if other not in rec.pending and not (known and _valid(known, decisions, rev, channels)):
            rec.pending[other] = _POOL.submit(bundle, rec, _item(rec, other), dict(decisions), channels or (),
                                              _ops(rec.folder), rev)
    return b


def item_view(root, stem, key: str) -> dict:
    rec = _rec(root, stem)
    it = _item(rec, key)
    lo, hi = item_window(rec, it)
    return {"lo": lo, "hi": hi, "kind": it.get("kind", "join"), "cells": list(it.get("cells", ())),
            "box": list(_box(rec, it)), "present": list(it.get("present", (lo, hi)))}


def _png(a) -> str:
    buffer = io.BytesIO()
    Image.fromarray(a).save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def frame_view(root, stem, key: str, frame: int, k: int, prefetch=(), channels=(), follow=(), every=False) -> dict:
    """One frame of a join: the panels at zoom k, the map, and the numbers to write on each panel.

    follow: numbers drawn (outline and number) in their own colour, the tracking tower's identity colours;
    every: every outline and number in its own colour instead of grey."""
    from .correction_cells import colour_lut, identity_rgb
    rec = _rec(root, stem)
    b = _bundle(rec, key, prefetch, tuple(channels))
    if frame not in b["frames"]:
        raise ValueError("Frame is outside this join's frames")
    fr = b["frames"][frame]
    k = int(min(max(k, 1), 12))
    follow = tuple(sorted({int(n) for n in follow if int(n) > 0}))
    ck = (key, frame, k, b["rev"], json.dumps(b["deps"], sort_keys=True), tuple(b["channels"]), follow, bool(every))
    out = rec.composed.get(ck)
    if out is None:
        colours = {n: identity_rgb(n) for n in follow}
        lut = colour_lut(int(max(fr["before"].max(), fr["after"].max())) + 1) if every else None
        imgs = panels(fr, b["targets"], k, colours, lut, current=b["kind"] in ("question", "follow"))
        def nums(rows):
            def colour(t, col):
                n = int(t)
                if n in colours:
                    return colours[n]
                if lut is not None and col == (200, 200, 200):
                    return tuple(int(v) for v in lut[min(n, len(lut)-1)])
                return col
            return [[float(y), float(x), t, '#%02x%02x%02x' % colour(t, col)] for y, x, t, col in rows]
        out = {"panels": {n: _png(a) for n, a in imgs.items()}, "overview": _png(fr["overview"]),
               "nb": nums(fr["nb"]), "na": nums(fr["na"]), "box": list(b["box"]), "origin": list(b["origin"]),
               "crop": list(fr["grey"].shape), "k": k, "lo": b["lo"], "hi": b["hi"], "names": list(imgs),
               "kind": b["kind"], "colours": {str(n): '#%02x%02x%02x' % c for n, c in colours.items()},
               "shape": list(rec.labels.shape[1:]), "scale": rec.ov_scale}
        rec.composed[ck] = out
        if len(rec.composed) > 80:
            rec.composed.popitem(last=False)
    return out


# ----------------------------------------------------------------------------- picking and edits
def current_frame_fn(rec):
    """Full current labels of a frame: approved session, decisions, then edits (read from the whole-recording
    labels when they are built, else composed per frame)."""
    from .correction_cells import cached_frames
    fast = cached_frames(rec)
    if fast is not None:
        return fast
    nm = rd.names(rec.data, _decisions(rec.folder))
    ops, data, labels, size = _ops(rec.folder), rec.data, rec.labels, rec.size
    def frame(f):
        lab = rd.frame_lut(data, nm, f, size)[labels[f]] if data['by_frame'][f] else labels[f]
        return rd.apply_ops(lab, f, ops)
    return frame


def pick(root, stem, key: str, frame: int, y: int, x: int, panel: str) -> dict:
    """What a click hits: the number the cell carries now, the number in the clicked panel, the end of its run."""
    rec = _rec(root, stem)
    b = _bundle(rec, key)
    fr = b["frames"][frame]
    y0, x0 = b["origin"]
    cy, cx = y - y0, x - x0
    if not (0 <= cy < fr["before"].shape[0] and 0 <= cx < fr["before"].shape[1]):
        raise ValueError("Outside the picture")
    now = int(fr["now"][cy, cx])
    lab = fr["after"] if panel == "After" else fr["before"]
    to = et.run_end(current_frame_fn(rec), now, frame, rec.data["T"] - 1) if now else frame
    return {"frame": frame, "y": y, "x": x, "cell": int(lab[cy, cx]) or None, "panel": panel, "now": now or None,
            "to": to}


_SIGNAL = {}
_SIG_LOCK = threading.Lock()
SIGNAL_BLOCK = 24     # frames worked out together when one is missing
SIGNAL_THREADS = 8    # the filters release the GIL, so threads run them side by side
SIGNAL_KEEP = 600     # remembered frames (about 140 MB at 466 x 504): two levels over a whole recording


def _signal_at(rec, frame, level):
    """correction_grow.signal for a frame and level, remembered for the last few asked (hovering repeats them)."""
    from ..tracking import correction_grow as cg
    key = (str(rec.folder), int(frame), round(float(level), 4))
    if key not in _SIGNAL:   # missing: work out a block of frames around it at once, in parallel (speed pass)
        T = rec.data['T']
        block = [f for f in range(max(0, int(frame) - SIGNAL_BLOCK // 2), min(T, int(frame) + SIGNAL_BLOCK))
                 if (key[0], f, key[2]) not in _SIGNAL]
        raws = {f: rec.raw_frame(f) for f in block}   # read in order (one file handle)
        with ThreadPoolExecutor(SIGNAL_THREADS) as pool:
            got = dict(zip(block, pool.map(lambda f: cg.signal(raws[f], float(level)), block)))
        with _SIG_LOCK:
            for f, m in got.items():
                _SIGNAL[(key[0], f, key[2])] = m
            while len(_SIGNAL) > SIGNAL_KEEP:
                _SIGNAL.pop(next(iter(_SIGNAL)))
    return _SIGNAL[key]


_WARMING = set()


def warm_signal(rec, level) -> None:
    """Work out the signal of every frame at this level in the background (speed pass 2026-11-18: the first
    seed at a new level spent ~6 s on it), so a seed or grow started afterwards finds it ready. Started by the
    seed / grow preview; at most one per recording and level."""
    tag = (str(rec.folder), round(float(level), 4))
    if tag in _WARMING:
        return
    _WARMING.add(tag)

    def run():
        try:
            for f in range(0, rec.data['T'], SIGNAL_BLOCK):
                if (tag[0], f, tag[1]) not in _SIGNAL:
                    _signal_at(rec, f, level)
        except Exception:   # a closed or replaced recording: the next seed works it out itself
            pass
        finally:
            _WARMING.discard(tag)
    threading.Thread(target=run, daemon=True, name="signal-warm").start()


_MODEL: dict = {}             # folder -> outline model label stack (tracking.correction_model)
_MODEL_LOCKS: dict = {}
MODEL_FILE = "model_outlines.npz"


def model_stack(rec):
    """The LV200 outline model's pieces for every frame of the recording, worked out once per recording (about
    3.5 minutes for 241 frames) and kept beside the review as model_outlines.npz (reused while the model weights are the same)."""
    import os
    from ..learned_mask import lv200
    from ..tracking.correction_model import predict_labels
    key = str(rec.folder)
    lock = _MODEL_LOCKS.setdefault(key, threading.Lock())
    with lock:
        if key in _MODEL:
            return _MODEL[key]
        path, weights, T = rec.folder / MODEL_FILE, str(lv200.weights()), rec.data['T']
        labels = None
        if path.exists():
            try:
                with np.load(path) as d:
                    if str(d["weights"]) == weights and d["labels"].shape[0] == T:
                        labels = d["labels"]
            except Exception:
                labels = None
        if labels is None:
            labels = predict_labels(np.stack([rec.raw_frame(f) for f in range(T)]))
            tmp = rec.folder / (MODEL_FILE + ".tmp")
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, labels=labels, weights=np.array(weights))
            os.replace(tmp, path)
        _MODEL[key] = labels
        return labels


def warm_model(rec) -> None:
    """Work out the model's pieces in the background when a recording opens, so Model grow is instant."""
    def run():
        try:
            model_stack(rec)
        except Exception:   # no weights or no torch here: Model grow reports it when used
            pass
    threading.Thread(target=run, daemon=True, name="model-warm").start()


def grow_preview(root, stem, frame: int, y: int, x: int, level=None, kind="seed", box=None) -> dict:
    """What Seed or Grow would take at (y, x) in this frame at this grow level, without saving anything
    (Jamie 2026-11-17: "preview the pixels that would be included at that grow level"). 'take' is the region
    (seed: the free signal at the spot; grow: the free signal touching the cell there), 'free' is all other
    unassigned signal at this level; both as [y, x, length] runs inside box (y0, y1, x0, x1)."""
    from ..tracking import correction_grow as cg
    rec = _rec(root, stem)
    warm_signal(rec, level if level not in (None, "") else cg.DEFAULT_LEVEL)
    level = float(level) if level not in (None, "") else cg.DEFAULT_LEVEL
    lab = current_frame_fn(rec)(int(frame))
    fg = _signal_at(rec, frame, level)
    H, W = lab.shape
    y0, y1, x0, x1 = box or (0, H, 0, W)
    y0, x0, y1, x1 = max(0, int(y0)), max(0, int(x0)), min(H, int(y1)), min(W, int(x1))
    now = int(lab[y, x]) if 0 <= y < H and 0 <= x < W else 0
    if kind == "grow":
        take = cg.grow_cell(lab, fg, now) if now else np.zeros(lab.shape, bool)
        msg = (f"Grow {now}: +{int(take.sum())} px at level {level:g}" if now else "Hover over a cell to see what Grow adds")
    else:
        take = cg.seed_region(lab, fg, y, x) if not now else np.zeros(lab.shape, bool)
        msg = (f"That is cell {now}: Seed needs free signal" if now else
               f"Seed: {int(take.sum())} px at level {level:g}" if take.any() else f"No free signal here at level {level:g}")
    free = fg & (lab == 0) & ~take
    crop = (slice(y0, y1), slice(x0, x1))
    return {"frame": int(frame), "level": level, "kind": kind, "cell": now or None, "count": int(take.sum()),
            "take": rd.row_runs(take[crop], y0, x0), "free": rd.row_runs(free[crop], y0, x0), "message": msg}


def split_preview(root, stem, frame: int, points, flip=False, number=None, box=None) -> dict:
    """Which side of a drawn split line keeps the number and which is cut off, in this frame (nothing saved)."""
    rec = _rec(root, stem)
    lab = current_frame_fn(rec)(int(frame))
    points = [(int(y), int(x)) for y, x in points]
    host, part, (y0, y1, x0, x1) = et.split_sides(lab, points, flip=bool(flip))
    return {"frame": int(frame), "host": int(host), "child": int(number) if number and int(number) > 0 else None,
            "keep": rd.row_runs(part == 1, y0, x0), "cut": rd.row_runs(part == 2, y0, x0)}


def cells_in_box(root, stem, frame: int, y0: int, y1: int, x0: int, x1: int) -> list[dict]:
    """Every number with pixels inside the box in this frame, with one of its pixels there (for box select)."""
    lab = current_frame_fn(_rec(root, stem))(int(frame))
    crop = lab[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    out = []
    for n in np.unique(crop):
        if n:
            ys, xs = np.nonzero(crop == n)
            i = len(ys) // 2
            out.append({"n": int(n), "y": int(ys[i]) + max(0, y0), "x": int(xs[i]) + max(0, x0)})
    return out


def fence_edit(rec, n, points, frame, span, *, key=None, by="jamie", inside_only=False, **extra) -> dict:
    """Fence (Jamie 2026-11-18: "set a boundary the cell never goes into ... not having to split every frame"):
    cell n's pixels inside the drawn area are removed in every frame from the frames 'from' box (default this
    frame) to the 'to' box (default the end of the recording). One paint edit (to 0, only n), so edits made
    after it can still put n there. inside_only (Keep in, Jamie 2026-11-18: "the cell has to be within a certain
    area"): the opposite, n's pixels outside the area are removed; only the pixels n has there are listed."""
    from ..tracking import correction_grow as cg
    if not n:
        return {"ok": False, "message": "Click the cell (or follow it) first, then draw round the area it must not enter"}
    points = [(int(y), int(x)) for y, x in (points or [])]
    if len(points) < 3:
        return {"ok": False, "message": "Draw round the area: hold the button and go all the way round"}
    inside = cg.polygon_mask(points, rec.labels.shape[1:])
    T = rec.data['T']
    first = int(frame) if span[0] is None else span[0]
    last = T - 1 if span[1] is None else span[1]
    first, last = min(first, last), max(first, last)
    current = current_frame_fn(rec)
    if inside_only:
        runs = {}
        for f in range(first, last + 1):
            out = (current(f) == n) & ~inside
            if out.any():
                runs[str(f)] = rd.row_runs(out)
        if not runs:
            return {"ok": False, "message": f"{n} never leaves that area in frames {first}-{last}: nothing to remove"}
        hit = sorted(map(int, runs))
        op = dict(op='paint', to=0, only=int(n), runs=runs, first=hit[0], last=hit[-1], keep_in=True)
        what = f"Kept {n} inside the drawn area: removed outside it in {len(hit)} frames"
    else:
        area = rd.row_runs(inside)
        hit = [f for f in range(first, last + 1) if (inside & (current(f) == n)).any()]
        if not hit:
            return {"ok": False, "message": f"{n} never enters that area in frames {first}-{last}: nothing to fence"}
        op = dict(op='paint', to=0, only=int(n), runs={str(f): area for f in hit}, first=hit[0], last=hit[-1], fence=True)
        what = f"Fenced {n} out of the drawn area: removed from it in {len(hit)} frames"
    return add_edit(rec, op, f"{what} (of {first}-{last})", frame=frame, to_frame=None, key=key, by=by, **extra)


def fill_gap_masks(rec, n, f0, until=None, level=None, taken=None):
    """Fill gaps (Jamie 2026-11-18): wherever cell n is missing after frame f0 and comes back later, seed a cell at
    its centroid in the last frame it was seen, at the grow level (correction_grow.signal), and follow that signal
    frame to frame through the gap with n's number, each frame seeding again at the previous frame's centroid.
    A frame whose centroid is not on free signal ends that gap's fill. The first frame of each gap is a
    plain seed (any size, as Seed); later frames stop where correction_grow.keeps_going does (sizes judged
    against that first seed). After n's last frame it continues to `until` (default the end, while the signal stays the same cell).
    Returns ({frame: mask}, [reasons a gap was not filled])."""
    from ..tracking import correction_grow as cg
    current, T = current_frame_fn(rec), rec.data['T']
    level = float(level) if level not in (None, "") else cg.DEFAULT_LEVEL
    taken = taken or {}

    def labels(f):   # this frame with the fills already chosen for other cells
        lab = current(f)
        extra = [m[f] for m in taken.values() if f in m]
        if extra:
            lab = lab.copy()
            for m in extra:
                lab[m] = -1 if lab.dtype.kind == 'i' else lab.max() + 1
        return lab

    present = [bool((current(f) == n).any()) for f in range(T)]
    seen = [g for g in range(f0 + 1) if present[g]]
    if not seen:
        return {}, [f"{n} is not in frame {f0} or before it"]
    out, why, f = {}, [], seen[-1] + 1   # from the cell's last frame at or before this one (Q pressed mid-gap)
    while f < T:
        if present[f] or not (present[f - 1] or (f - 1) in out):
            f += 1
            continue
        back = next((g for g in range(f + 1, T) if present[g]), None)
        end = back - 1 if back is not None else (min(T - 1, until) if until is not None else T - 1)
        if end < f:
            break
        prev = out[f - 1] if (f - 1) in out else current(f - 1) == n
        filled, stop = 0, ""
        for g in range(f, end + 1):
            ys, xs = np.nonzero(prev)   # Jamie 2026-11-18: every frame seeds at the previous frame's centroid
            cy, cx = int(round(ys.mean())), int(round(xs.mean()))
            lab, fg = labels(g), _signal_at(rec, g, level)
            if not fg[cy, cx]:
                stop = f"no signal at level {level:g} at its centre ({cy}, {cx})"
                break
            if lab[cy, cx]:
                stop = f"its centre ({cy}, {cx}) is inside cell {int(lab[cy, cx])} (merged into it? split that instead)"
                break
            region = cg.seed_region(lab, fg, cy, cx)
            if g == f:
                ref = int(region.sum())   # the first frame is a plain Seed (B) at the centroid, any size
            elif not cg.keeps_going(ref, int(prev.sum()), int(region.sum())):
                stop = f"the signal at its centre is {int(region.sum())} px, not the same cell (was {int(prev.sum())} px)"
                break
            out[g], prev, filled = region, region, filled + 1
        if filled < end - f + 1:
            why.append(f"{n} frames {f + filled}-{end}: {stop}")
        f = end + 1
    return out, why


def track_on_op(rec, cells, f0, last):
    """Run the tracking on from frame f0 for the given cells (Jamie 2026-11-18: "after changes i can run the
    tracking from that instance"): frame by frame each cell gives its number to the number its pixels overlap
    most in the next frame (swapping with any other cell carrying it there), the tracker's own link by overlap.
    Stops for a cell when nothing overlaps it or its area changes by the tracker's sudden-size ratio
    (correction_rules.AREA_RATIO). Returns (one 'keep' edit with the per-frame moves, or None; {cell: last frame})."""
    current = current_frame_fn(rec)
    moves, ends, done = {}, {}, []
    pending = lambda f: rd.apply_ops(current(f), f, done)
    for n in cells:
        lab = pending(f0)
        prev = lab == n
        if not prev.any():
            raise ValueError(f"{n} is not in frame {f0}")
        mine, end = {}, f0
        for f in range(f0 + 1, int(last) + 1):
            lab = pending(f)
            hit = lab[prev & (lab > 0)]
            if not hit.size:
                break
            m = int(np.bincount(hit).argmax())
            groups, _ = _cells_of(lab, m)
            seg = np.isin(groups, np.unique(groups[prev & (groups > 0)])) & (groups > 0)
            a, b = int(prev.sum()), int(seg.sum())
            if max(a, b) / max(1, min(a, b)) >= rd.AREA_RATIO:
                break
            if m != n:
                mv = [[m, n, rd.row_runs(seg)]]
                other = (lab == n) & ~seg
                if other.any():
                    mv.append([n, m, rd.row_runs(other)])
                mine[str(f)] = mv
            prev, end = seg, f
        ends[n] = end
        if mine:
            op = dict(op='keep', moves=mine, first=min(map(int, mine)), last=max(map(int, mine)), cell=n)
            done.append(op)
            for f, mv in mine.items():
                moves.setdefault(f, []).extend(mv)
    if not moves:
        return None, ends
    return dict(op='keep', moves=moves, first=min(map(int, moves)), last=max(map(int, moves)),
                cell=cells[0], track_on=cells), ends


def cell_now(root, stem, frame: int, y: int, x: int) -> dict:
    """The number a full-frame pixel carries now (decisions and edits applied), and where its run ends."""
    rec = _rec(root, stem)
    current = current_frame_fn(rec)
    now = int(current(frame)[y, x])
    return {"frame": frame, "y": y, "x": x, "before": int(rec.labels[frame][y, x]), "now": now,
            "to": et.run_end(current, now, frame, rec.data["T"] - 1) if now else frame}


def piece_op(rec, picked, to, last=None):
    """Edit moving just the clicked connected part of the clicked number, from the frame it was clicked to frame
    last, following that part frame to frame through the parts of the number that overlap it (stops where none
    does). Jamie 2026-11-17: Piece only changed one frame, so a seeded cell needed a give per frame."""
    f0, a = picked['frame'], picked['now']
    current, eight = current_frame_fn(rec), np.ones((3, 3), bool)
    from ..tracking.correction_note_actions import _label_cell   # same pieces, worked out round the cell only
    parts, _ = _label_cell(current(f0), a)
    mask = parts == parts[picked['y'], picked['x']]
    runs, end = {str(f0): rd.row_runs(mask)}, f0
    for f in range(f0 + 1, max(f0, int(f0 if last is None else last)) + 1):
        parts, _ = _label_cell(current(f), a)
        hit = np.unique(parts[mask & (parts > 0)])
        if not len(hit):
            break
        mask = np.isin(parts, hit)
        runs[str(f)], end = rd.row_runs(mask), f
    return dict(op='piece', host=a, child=to, first=f0, last=end, runs=runs)


SEPARATE_PX = 25   # pieces of one number further apart than this are different cells


def _dilate(mask, iterations):
    """binary_dilation(mask, iterations) computed only round the mask's box (same result, far less work)."""
    out = np.zeros(mask.shape, bool)
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if not len(rows):
        return out
    H, W = mask.shape
    y0, y1 = max(0, rows[0] - iterations - 1), min(H, rows[-1] + 1 + iterations + 1)
    x0, x1 = max(0, cols[0] - iterations - 1), min(W, cols[-1] + 1 + iterations + 1)
    out[y0:y1, x0:x1] = ndi.binary_dilation(mask[y0:y1, x0:x1], iterations=iterations)
    return out


def _cells_of(lab, n):
    """Pieces of number n grouped into cells: pieces within SEPARATE_PX of each other are one cell. Worked out
    on n's box widened by the grouping reach (same groups and numbering as on the whole frame; speed pass
    2026-11-20)."""
    mask = lab == n
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    out = np.zeros(lab.shape, np.int32)
    if not len(rows):
        return out, 0
    r = SEPARATE_PX // 2 + 2
    H, W = mask.shape
    box = (slice(max(0, rows[0] - r), min(H, rows[-1] + 1 + r)), slice(max(0, cols[0] - r), min(W, cols[-1] + 1 + r)))
    m = mask[box]
    near, count = ndi.label(ndi.binary_dilation(m, iterations=SEPARATE_PX // 2) if SEPARATE_PX // 2 else m,
                            structure=np.ones((3, 3), bool))
    out[box] = np.where(m, near, 0)
    return out, count


def follow_cell_op(rec, picked, child, last):
    """Give or delete (child 0) only the clicked cell when its number also sits on another cell (not in Motion;
    added 2026-11-11: Jamie's renumbers caught a second cell sharing the old number). The clicked cell is followed
    frame to frame by overlap; returns None when the number is on one cell throughout, so a plain edit is used."""
    a, f0 = picked['now'], picked['frame']
    current = current_frame_fn(rec)
    cells, count = _cells_of(current(f0), a)
    mine = cells == cells[picked['y'], picked['x']]
    shared, moves, end = count > 1, {}, f0
    for f in range(f0, max(f0, int(last)) + 1):
        if f > f0:
            cells, count = _cells_of(current(f), a)
            hit = np.unique(cells[_dilate(mine, 3) & (cells > 0)])
            if not len(hit):
                break
            shared |= count > 1
            mine = np.isin(cells, hit)
        moves[str(f)] = [[a, int(child), rd.row_runs(mine)]]
        end = f
    if not shared:
        return None
    return dict(op='keep', moves=moves, first=f0, last=end, cell=a, to=int(child))


def _refresh(rec) -> None:
    """With the Workbench running (PREWARM), rebuild the timeline, tower and traces in the background."""
    if PREWARM:
        from .correction_cells import refresh_after_edit
        refresh_after_edit(rec)


def add_edit(rec, op, msg, *, frame: int, to_frame: int | None, key: str | None, by: str = "jamie", **extra) -> dict:
    """Record one edit (frames: this one to the 'to' frame unless the edit carries its own)."""
    if 'first' not in op:
        op['first'] = frame
        op['last'] = max(frame, int(to_frame or frame))
    current = current_frame_fn(rec)
    # An edit must change something; otherwise the numbers were not where it covers.
    if not any(not np.array_equal(rd.apply_ops(current(f), f, [op]), current(f)) for f in range(op['first'], op['last']+1)):
        return {"ok": False, "message": f'Nothing to change in frames {op["first"]}–{op["last"]}'}
    op['time'] = _dt.datetime.now().isoformat(timespec='seconds')
    if key:
        op['item'] = base_key(key)
    op['by'] = by
    op.update({k: v for k, v in extra.items() if v not in (None, "")})
    with _LOCK:
        ops = _ops(rec.folder, copy=False) + [_check_edit(op, rec.data["T"])]
        _write(rec.folder / "edits.json", ops)
    _refresh(rec)
    return {"ok": True, "message": f"{msg} (frames {op['first']}–{op['last']}; Ctrl+Z undoes)", "count": len(ops)}


def edit(root, stem, *, action: str, picked: dict | None, frame: int, to_frame: int | None = None,
         key: str | None = None, other: int | None = None, number: int | None = None, piece_only: bool = False,
         points=None, whole: bool = False, by: str = "jamie", **extra) -> dict:
    """The reviewer's edit buttons: swap, give, delete, split, keep (as Reviewer.start_swap ... keep)."""
    rec = _rec(root, stem)
    a = (picked or {}).get("now")
    T = rec.data['T']
    span = [None if extra.get(k) in (None, "") else min(T - 1, max(0, int(extra[k]))) for k in ("range_from", "range_to")]
    extra.pop("range_from", None); extra.pop("range_to", None)
    flip, cells = bool(extra.pop("flip", False)), extra.pop("cells", None)
    if action in ("fence", "keep_in"):
        return fence_edit(rec, int(cells[0]) if cells else a, points, frame, span, key=key, by=by,
                          inside_only=action == "keep_in", **extra)
    if action == "one_island":
        if not a or (picked or {}).get("y") is None:
            return {"ok": False, "message": "Click the piece of the cell to keep first, then ! (One island)"}
        f0 = int(picked.get("frame", frame))
        f1 = max(f0, int(to_frame if to_frame is not None else f0)) if span[1] is None else max(f0, span[1])
        from ..tracking import correction_grow as cg
        free = cg.one_island_track(current_frame_fn(rec), a, f0, f1, int(picked["y"]), int(picked["x"]),
                                   first=span[0] if span[0] is not None and span[0] < f0 else None)
        if not free:
            return {"ok": False, "message": f"{a} is already one piece in frames {f0}\u2013{f1}"}
        op = dict(op='paint', to=0, only=a, runs={str(f): rd.row_runs(m) for f, m in free.items()},
                  first=min(free), last=max(free), island=True)
        px = sum(int(m.sum()) for m in free.values())
        return add_edit(rec, op, f"Kept one piece of {a}: freed {px} px of other pieces in {len(free)} frames",
                        frame=frame, to_frame=None, key=key, by=by, **extra)
    if action == "model_grow":
        from ..tracking.correction_model import model_grow_track
        if not a:
            return {"ok": False, "message": "Click a cell first, then * (Model grow)"}
        try:
            model = model_stack(rec)
        except Exception as err:
            return {"ok": False, "message": f"The outline model could not run here: {err}"}
        g0 = int((picked or {}).get("frame", frame)) if span[0] is None else span[0]
        g1 = max(int(frame), int(to_frame if to_frame is not None else frame)) if span[1] is None else span[1]
        masks = model_grow_track(current_frame_fn(rec), lambda f: model[f], a, min(g0, g1), max(g0, g1))
        if not masks:
            return {"ok": False, "message": f"Nothing to add: the outline model has no free pixels for {a} there"}
        op = dict(op='paint', to=a, only=0, runs={str(f): rd.row_runs(m) for f, m in masks.items()},
                  first=min(masks), last=max(masks), model="lv200")
        return add_edit(rec, op, f"Grew {a} into the outline model's piece of it ({len(masks)} frames)",
                        frame=frame, to_frame=None, key=key, by=by, **extra)
    if action in ("draw", "erase", "seed", "grow"):
        return _shape_edit(rec, action, picked or {}, frame=frame, to_frame=to_frame, key=key, number=number,
                           points=points, level=extra.pop("level", None), by=by, span=span, **extra)
    level = extra.pop("level", None)
    if action == "fill_gaps":
        cells = [int(c) for c in (cells or ([a] if a else []))]
        if not cells:
            return {"ok": False, "message": "Click a cell (or follow cells) first, then Q"}
        masks, notes = {}, []
        for n in cells:
            got, why = fill_gap_masks(rec, n, int(frame), span[1], level, taken=masks)
            for f, m in got.items():
                masks.setdefault(n, {})[f] = m
            notes += why
        if not masks:
            return {"ok": False, "message": "Nothing filled: " + ("; ".join(notes) or "no gaps after this frame")}
        msgs = []
        for n, got in masks.items():
            op = dict(op='paint', to=n, only=0, runs={str(f): rd.row_runs(m) for f, m in got.items()},
                      first=min(got), last=max(got), level=float(level or 0.4))
            r = add_edit(rec, op, f"Filled {n}'s gaps in {len(got)} frames", frame=frame, to_frame=None, key=key,
                         by=by, **extra)
            if not r["ok"]:
                return r
            msgs.append(r["message"])
        return {**r, "message": "; ".join(msgs) + (f" (not filled: {'; '.join(notes)})" if notes else "")}
    if action == "track_on":
        cells = [int(c) for c in (cells or ([a] if a else []))]
        if not cells:
            return {"ok": False, "message": "Click a cell (or follow cells) first, then L"}
        last = span[1] if span[1] is not None else T - 1
        op, ends = track_on_op(rec, cells, int(frame), last)
        if op is None:
            return {"ok": False, "message": f"Nothing to change: {', '.join(map(str, cells))} already continue "
                                            f"as far as {'their signal is followed' if ends else 'frame ' + str(last)}"}
        return add_edit(rec, op, f"Tracked {', '.join(map(str, cells))} on from frame {frame} "
                        f"({'; '.join(f'{c} to frame {e}' for c, e in ends.items())})",
                        frame=frame, to_frame=None, key=key, by=by, **extra)
    if action not in ("split", "trim") and not a:
        return {"ok": False, "message": "Click a cell first"}
    if action == "delete" and whole:   # Motion's "remove complete selected tracks"
        return add_edit(rec, dict(op='delete', cell=a, first=0, last=rec.data['T']-1),
                        f'Deleted {a} from the whole recording', frame=frame, to_frame=to_frame, key=key, by=by, **extra)
    if action == "swap":
        if not other:
            return {"ok": False, "message": "Click on a cell"}
        return add_edit(rec, dict(op='swap', a=a, b=int(other)), f'Swapped {a} and {other}',
                        frame=frame, to_frame=to_frame, key=key, by=by, **extra)
    if action in ("give", "delete"):
        b = int(number or 0) if action == "give" else 0
        if action == "give" and b <= 0:
            return {"ok": False, "message": "Type a number above 0"}
        if a == b:
            return {"ok": False, "message": ""}
        if piece_only:
            msg = f'Piece of {a} is now {b}' if b else f'Deleted a piece of {a}'
            return add_edit(rec, piece_op(rec, picked, b, to_frame if to_frame is not None else frame), msg, frame=frame, to_frame=to_frame, key=key, by=by, **extra)
        only = follow_cell_op(rec, picked, b, to_frame if to_frame is not None else frame)
        if only:
            msg = (f'This {a} is now {b}' if b else f'Deleted this {a}') + ' (another cell also carries ' + str(a) + ')'
            return add_edit(rec, only, msg, frame=frame, to_frame=to_frame, key=key, by=by, **extra)
        if b:
            return add_edit(rec, dict(op='renumber', **{'from': a}, to=b), f'{a} is now {b}', frame=frame,
                            to_frame=to_frame, key=key, by=by, **extra)
        return add_edit(rec, dict(op='delete', cell=a), f'Deleted {a}', frame=frame, to_frame=to_frame,
                        key=key, by=by, **extra)
    if action == "keep":
        lo, hi = (0, rec.data['T']-1) if whole else item_window(rec, _item(rec, key))
        to = int(number) if number and int(number) > 0 else None
        op = na.keep_cell(current_frame_fn(rec), rec.raw_frame, picked['frame'], picked['y'], picked['x'], lo, hi, to=to)
        if not op:
            return {"ok": False, "message": "Nothing to change: the cell already keeps its number"}
        what = f"{a} kept as {to or a}" + (' over the whole recording' if whole else ' split over this snippet')
        return add_edit(rec, op, what, frame=frame, to_frame=to_frame, key=key, by=by, **extra)
    if action in ("split", "trim"):   # trim (Jamie 2026-11-18): a split whose cut-off side is erased, not renumbered
        points = [(int(y), int(x)) for y, x in (points or [])]
        if len(points) < 2:
            return {"ok": False, "message": "Draw a line across the cell to cut it"}
        f0, current = frame, current_frame_fn(rec)
        host = et.host_under(current(f0), points)
        if a == host and int(to_frame or 0) >= f0:
            last = int(to_frame)   # this cell was clicked first: its 'to' frame applies
        else:
            last = et.run_end(current, host, f0, rec.data['T']-1) if host else f0
        known = {n for n, _, _ in rec.data['tracks']} | {o.get('to') or o.get('child') or 0 for o in _ops(rec.folder)}
        if span[1] is not None and span[1] >= f0:
            last = span[1]
        try:
            op, msg = et.split(current, rec.raw_frame, f0, last, points, known, flip=flip,
                               give=int(number) if number and int(number) > 0 else None)
        except ValueError as err:
            return {"ok": False, "message": str(err).replace("cut it", "trim it") if action == "trim" else str(err)}
        if action == "trim":
            stopped = msg.split("; stopped because ", 1)
            op = dict(op='paint', to=0, only=op['host'], runs=op['runs'], first=op['first'], last=op['last'])
            msg = (f"Trimmed {op['only']}: erased the cut-off side in frames {op['first']}-{op['last']}"
                   + (f"; stopped because {stopped[1]}" if len(stopped) > 1 else ""))
        return add_edit(rec, op, msg, frame=frame, to_frame=to_frame, key=key, by=by, **extra)
    raise ValueError(f"Unknown edit {action}")


def marks(root, stem) -> dict:
    """Jamie's own marks for a recording (marks.json, kept apart from the notes the agent reads): cells ticked
    as good {number: {time, frame}} and flags [{id, cell, frame, note, time}] to revisit (Jamie 2026-11-17)."""
    return _marks_of(_rec(root, stem).folder)


def _marks_of(folder) -> dict:
    out = _read(folder / "marks.json", {})
    return {"ticks": out.get("ticks", {}), "flags": out.get("flags", []), "resolved": out.get("resolved", [])}


def mark(root, stem, *, op: str, cell=None, frame=None, id=None, note=None) -> dict:
    """tick / untick a cell, flag a cell at a frame (optional note), change a flag's note, unflag."""
    rec = _rec(root, stem)
    now = _dt.datetime.now().isoformat(timespec='seconds')
    with _LOCK:   # (opening a recording takes _LOCK too, so read the file directly here)
        m = _marks_of(rec.folder)
        m.setdefault("resolved", [])
        if op in ("resolve", "unresolve"):
            ev = [int(cell), int(frame), str(note or "")]
            m["resolved"] = [r for r in m["resolved"] if r[:3] != ev]
            if op == "resolve":
                m["resolved"].append(ev + [now])
            _write(rec.folder / "marks.json", m)
            return m
        if op in ("tick", "untick", "flag") and not cell:
            raise ValueError("Click a cell first")
        if op == "tick":
            m["ticks"][str(int(cell))] = {"time": now, "frame": None if frame is None else int(frame)}
        elif op == "untick":
            m["ticks"].pop(str(int(cell)), None)
        elif op == "flag":
            m["flags"].append({"id": max([f["id"] for f in m["flags"]], default=0) + 1, "cell": int(cell),
                               "frame": int(frame or 0), "note": (note or "").strip(), "time": now})
        elif op in ("note", "unflag"):
            hit = [f for f in m["flags"] if f["id"] == int(id or 0)]
            if not hit:
                raise ValueError("That flag is gone")
            if op == "note":
                hit[0]["note"] = (note or "").strip()
            else:
                m["flags"].remove(hit[0])
        else:
            raise ValueError(f"Unknown mark: {op}")
        _write(rec.folder / "marks.json", m)
    return m


QUEUE_ACTIONS = {"delete", "give", "grow", "seed", "keep", "draw", "erase", "split", "trim", "model_grow", "one_island"}


def edit_batch(root, stem, items, key=None, by="jamie") -> dict:
    """Queued edits run in the order they were queued (Jamie 2026-11-17: "select multiple points to do and then
    have them all run at once"). Each queued spot is looked up again just before its edit, so it acts on the
    cell there after the edits before it; 'to' frames are that cell's run end then, as a click sets them.
    Every edit is saved as usual (one Ctrl+Z each). A failed item is reported and the rest still run."""
    results, items = [], list(items or ())
    seeds = {}   # seeds queued in the same frame and level grow together (Jamie 2026-11-18)
    for i, it in enumerate(items):
        if str(it.get("action")) in ("seed", "grow"):
            seeds.setdefault((int(it.get("frame", 0)), it.get("level")), []).append(i)
    together = {g[0]: g for g in seeds.values() if len(g) > 1}
    skip = {i for g in together.values() for i in g[1:]}
    for i, it in enumerate(items):
        action, frame = str(it.get("action")), int(it.get("frame", 0))
        if i in skip:
            continue
        if i in together:
            try:
                got = seed_group(_rec(root, stem), [items[j] for j in together[i]], key=key, by=by)
            except (ValueError, KeyError, TypeError) as err:
                got = [{"ok": False, "message": str(err) or "Could not seed"}] * len(together[i])
            for j, r in zip(together[i], got):
                results.append({"i": j, "ok": bool(r.get("ok")), "message": r.get("message", ""), "count": r.get("count")})
            continue
        try:
            if action not in QUEUE_ACTIONS:
                raise ValueError(f"{action} cannot be queued")
            spot = it.get("pick") if action in ("draw", "erase", "split", "trim") else (it.get("y"), it.get("x"))
            picked = None
            if spot and spot[0] is not None:
                picked = cell_now(root, stem, frame, int(spot[0]), int(spot[1]))
                picked["now"] = picked["now"] or None
            if action in ("draw", "erase", "split", "trim"):
                to_frame = picked["to"] if action in ("split", "trim") and picked and picked["now"] else frame
            else:
                to_frame = frame if action == "seed" or not picked["now"] else picked["to"]
            r = edit(root, stem, action=action, picked=picked, frame=frame, to_frame=to_frame, key=key,
                     number=it.get("number"), piece_only=bool(it.get("piece_only")), points=it.get("points"),
                     whole=bool(it.get("whole")), level=it.get("level"), by=by, range_from=it.get("range_from"),
                     range_to=it.get("range_to"), flip=it.get("flip"), one_piece=bool(it.get("one_piece")))
        except (ValueError, KeyError, TypeError) as err:
            r = {"ok": False, "message": str(err) or "Could not apply"}
        results.append({"i": i, "ok": bool(r.get("ok")), "message": r.get("message", ""), "count": r.get("count")})
    done = sum(r["ok"] for r in results)
    return {"results": results, "done": done,
            "message": f"Ran {done} of {len(results)} queued edits" +
                       ("" if done == len(results) else
                        ": " + "; ".join(f"#{r['i'] + 1} {r['message']}" for r in results if not r["ok"]))}


def seed_group(rec, items, *, key=None, by="jamie", root=None, stem=None) -> list:
    """Several queued seeds in one frame, grown together so each gets its own territory (correction_grow.
    seed_many). items: [{frame, y, x, level, range_from, range_to}] (same frame). Each seed is saved as its own
    edit (one Ctrl+Z each), numbered like a single seed (a track that vanished there, else a new number).
    Queued Grow clicks in the same frame grow at the same time and share the signal with the seeds and each
    other (each to the end of its run, or the frames 'to' box), saved as a grow edit each. Results in item order."""
    from ..tracking import correction_grow as cg
    T, f0 = rec.data['T'], int(items[0]["frame"])
    level = float(items[0].get("level") or cg.DEFAULT_LEVEL)
    span = [None if items[0].get(k) in (None, "") else min(T - 1, max(0, int(items[0][k])))
            for k in ("range_from", "range_to")]
    lo, hi = item_window(rec, _item(rec, key)) if key else (0, T - 1)
    first = lo if span[0] is None else min(span[0], f0)
    until = hi if span[1] is None else max(span[1], f0)
    current = current_frame_fn(rec)
    lab = current(f0)
    seeds = [i for i in items if i.get("action", "seed") == "seed"]
    grows, grow_items, results_at = [], [], {}
    for i in items:
        if i.get("action") == "grow":
            n = int(lab[int(i["y"]), int(i["x"])])
            if not n:
                results_at[id(i)] = {"ok": False, "message": f"Grow: no cell at ({i['y']}, {i['x']}) in frame {f0}"}
                continue
            g1 = span[1] if span[1] is not None else et.run_end(current, n, f0, T - 1)
            grows.append((n, max(f0, g1), (int(i["y"]), int(i["x"]))) if i.get("one_piece") else (n, max(f0, g1)))
            grow_items.append(i)
    masks, grown = cg.seed_many(current, rec.raw_frame, f0, until, [(int(i["y"]), int(i["x"])) for i in seeds],
                                level, first=first, sig=lambda f: _signal_at(rec, f, level), grows=grows)
    for it, (n, *_), got in zip(grow_items, grows, grown):
        if not got:
            results_at[id(it)] = {"ok": False, "message": f"Nothing to add: no free signal touches {n} at grow level {level:g}"}
            continue
        op = dict(op='paint', to=n, only=0, runs={str(f): rd.row_runs(m) for f, m in got.items()},
                  first=min(got), last=max(got), level=level)
        results_at[id(it)] = add_edit(rec, op, f"Grew {n} together with {len(items) - 1} other edit{'s' * (len(items) > 2)}",
                                      frame=f0, to_frame=None, key=key, by=by)
    used, fresh = set(), new_number(rec)
    for it, got in zip(seeds, masks):
        if not got:
            results_at[id(it)] = {"ok": False, "message": f"No free signal at ({it['y']}, {it['x']}) at grow level {level:g}"}
            continue
        cont = cg.continued_number(current, got, T)
        if cont in used:
            cont = None
        n = cont or fresh
        if not cont:
            fresh += 1
        used.add(n)
        op = dict(op='paint', to=n, only=0, runs={str(f): rd.row_runs(m) for f, m in got.items()},
                  first=min(got), last=max(got), level=level)
        msg = f"Seeded {n} together with {len(items) - 1} other edit{'s' * (len(items) > 2)}" + (
            " (continues the track that vanished here)" if cont else " (new number)")
        results_at[id(it)] = add_edit(rec, op, msg, frame=f0, to_frame=None, key=key, by=by)
    return [results_at[id(i)] for i in items]


def new_number(rec) -> int:
    """A number no track, edit or cell has used."""
    known = {n for n, _, _ in rec.data['tracks']} | {rec.size - 1}
    known |= {int(o.get(k) or 0) for o in _ops(rec.folder) for k in ('to', 'child', 'b', 'a', 'cell', 'host')}
    return max(known) + 1


def _shape_edit(rec, action, picked, *, frame, to_frame, key, number, points, level, by, span=(None, None),
                **extra) -> dict:
    """Draw an outline, erase inside one, seed a new cell or grow a cell (correction_grow; paint edits).

    draw: pixels inside the drawn outline that carry no number go to the clicked cell, else the typed number,
    else the number of a track that vanished where it is (correction_grow.continued_number), else a new number
    (this frame). erase: the clicked cell's pixels inside it are removed (every cell's when none is clicked).
    seed: the free signal at the click becomes a cell, followed both ways through the view's frames (forward
    only to the 'to' frame when that is later than this one) until no free signal overlaps; numbered as draw.
    grow: the clicked cell takes the free signal touching it, this frame to the 'to' frame (Jamie 2026-11-17:
    seeding frame by frame gave every frame a new number)."""
    from ..tracking import correction_grow as cg
    a = picked.get("now")
    current = current_frame_fn(rec)
    one_piece = bool(extra.pop("one_piece", False))
    level = float(level) if level not in (None, "") else cg.DEFAULT_LEVEL
    last = max(int(frame), int(to_frame if to_frame is not None else frame))
    give = int(number) if number and int(number) > 0 else None
    if action in ("draw", "erase"):
        points = [(int(y), int(x)) for y, x in (points or [])]
        if len(points) < 3:
            return {"ok": False, "message": "Draw round the area: hold the button and go all the way round"}
        inside = cg.polygon_mask(points, rec.labels.shape[1:])
        if action == "erase":
            op = dict(op='paint', to=0, only=a or None, runs={str(frame): rd.row_runs(inside)}, first=frame, last=frame)
            msg = f'Erased the drawn part of {a}' if a else 'Erased every cell inside the drawn outline'
        else:
            free = inside & (current(frame) == 0)
            cont = None if a or give or not free.any() else cg.continued_number(current, {frame: free}, rec.data['T'])
            n = a or give or cont or new_number(rec)
            op = dict(op='paint', to=n, only=0, runs={str(frame): rd.row_runs(inside)}, first=frame, last=frame)
            msg = f'Drew an outline for {n}' + ('' if a or give else ' (continues the track that vanished here)'
                                                 if cont else ' (new number)')
        return add_edit(rec, op, msg, frame=frame, to_frame=None, key=key, by=by, **extra)
    if picked.get("y") is None:
        return {"ok": False, "message": "Click a spot first"}
    try:
        if action == "seed":
            if a:
                return {"ok": False, "message": f"That is cell {a}: click empty signal to seed, or use Grow on {a}"}
            lo, hi = item_window(rec, _item(rec, key)) if key else (0, rec.data['T'] - 1)
            f0 = int(picked["frame"])
            first = lo if span[0] is None else min(span[0], f0)            # Jamie 2026-11-18: set the frames
            until = (last if last > f0 else hi) if span[1] is None else max(span[1], f0)
            masks = cg.seed_track(current, rec.raw_frame, f0, until, int(picked["y"]),
                                  int(picked["x"]), level, first=first,
                                  sig=lambda f: _signal_at(rec, f, level))
            cont = None if give else cg.continued_number(current, masks, rec.data['T'])
            n = give or cont or new_number(rec)
            msg = f'Seeded {n}' + ('' if give else ' (continues the track that vanished here)' if cont else ' (new number)')
        else:
            if not a:
                return {"ok": False, "message": "Click a cell first, then Grow"}
            g0 = int(picked["frame"]) if span[0] is None else span[0]
            g1 = last if span[1] is None else span[1]
            piece = (int(picked["frame"]), int(picked["y"]), int(picked["x"])) if one_piece else None
            masks = cg.grow_track(current, rec.raw_frame, a, min(g0, g1), max(g0, g1), level,
                                  sig=lambda f: _signal_at(rec, f, level), piece=piece)
            if not masks:
                return {"ok": False, "message": f"Nothing to add: no free signal touches {a} at grow level {level:g}"}
            n = a
            msg = f'Grew {a} into the signal touching it'
    except ValueError as err:
        return {"ok": False, "message": str(err)}
    op = dict(op='paint', to=n, only=0, runs={str(f): rd.row_runs(m) for f, m in masks.items()},
              first=min(masks), last=max(masks), level=level)
    return add_edit(rec, op, msg, frame=frame, to_frame=None, key=key, by=by, **extra)


def cells_view(root, stem) -> dict:
    """The Gantt chart's data: every cell's frames and flagged events (correction_cells.timeline)."""
    from .correction_cells import timeline
    return timeline(_rec(root, stem))


def tower_view(root, stem, cells, lo=None, hi=None, style="line", preset="cd68") -> dict:
    from .correction_cells import tower
    return tower(_rec(root, stem), sorted({int(c) for c in cells}), lo, hi, style=style, preset=preset)


def undo_edit(root, stem, count: int) -> dict:
    """Undo the last edit if the list still has the length it had after that edit (as Reviewer.undo_last)."""
    rec = _rec(root, stem)
    with _LOCK:
        ops = _ops(rec.folder)
        if len(ops) != count:
            return {"ok": False, "message": "The edits changed since; nothing undone"}
        op = ops.pop()
        _write(rec.folder / "edits.json", ops)
    _refresh(rec)
    return {"ok": True, "message": f"Undid {op['op']} (frames {op['first']}–{op['last']})"}


def add_note(root, stem, *, key: str | None, frame: int, text: str = "", tag: str | None = None,
             picked: dict | None = None, by: str = "jamie", **extra) -> dict:
    """Save one note as the reviewer does: '<tag>: <typed>' attached to the clicked cell if any."""
    typed = (text or "").strip()
    body = f'{tag}: {typed}' if tag and typed else (tag or typed)
    if not body:
        return {"ok": False, "message": ""}
    rec = _rec(root, stem)
    pick_ = picked or {}
    entry = dict(item=base_key(key), frame=pick_.get('frame', frame), cell=pick_.get('cell'), panel=pick_.get('panel'),
                 y=pick_.get('y'), x=pick_.get('x'), text=body, time=_dt.datetime.now().isoformat(timespec='seconds'),
                 tag=tag, by=by, **{k: v for k, v in extra.items() if v not in (None, "")})
    with _LOCK:
        notes = _read(rec.folder / "notes.json", []) + [entry]
        _write(rec.folder / "notes.json", notes)
    on = f"cell {entry['cell']}, " if entry['cell'] else ''
    return {"ok": True, "message": f"Note saved: {body} ({on}frame {entry['frame']}; {len(notes)} for {stem})"}


# ----------------------------------------------------------------------------- export
def composed(root, stem):
    """(tracker labels, current labels, data). The current labels come from the reviewer's whole-recording stack,
    which equals rd.compose of the decisions and edits (speed pass 2026-11-18: Export composed it again, ~2 s)."""
    from .correction_cells import current_stack
    rec = _rec(root, stem)
    return rec.labels, current_stack(rec)["stack"].copy(), rec.data


def export(root, stem, motion_url: str | None, *, user: str = "jamie") -> dict:
    """Write the decided labels as one new Motion editor session (via the Motion service) and record it."""
    if not motion_url:
        raise ValueError("Motion service is not connected; start the workbench with --motion-url")
    rec = _rec(root, stem)
    folder = rec.folder
    if rd.conflicts(rec.data, _decisions(folder)):
        raise ValueError("Some decisions put one number on two cells in a frame; resolve them first")
    before, after, _ = composed(root, stem)
    if np.array_equal(before, after):
        raise ValueError("Nothing to export: the decided labels equal the source")
    exports = _read(folder / "exports.json", [])
    out_dir = Path(root) / f"{stem}_reviewed{len(exports) + 1:02d}"
    decisions = _decisions(folder)
    review = _read(folder / "review.json", {})
    reason = (f"Correction review {stem}: Jamie's join rule (applied joins kept unless denied, unsure joins only if "
              f"approved) with {len(decisions)} decisions and {len(_ops(folder))} edits; review set {folder}")
    body = json.dumps({"action": "apply_identity_patch", "params": {
        "source": review["source"]["folder"], "patch": rd.patch_from(before, after), "reason": reason,
        "output_dir": str(out_dir), "user": user}}).encode()
    request = UrlRequest(motion_url.rstrip("/") + "/api/action", data=body, method="POST",
                         headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=600) as response:
            json.loads(response.read())
    except HTTPError as error:   # Motion answers 422/403 with {"detail": why}
        detail = json.loads(error.read() or b"{}").get("detail", str(error))
        raise ValueError(f"Motion refused the export: {detail}") from None
    record = {"session": str(out_dir), "at_utc": datetime.now(timezone.utc).isoformat(), "user": user,
              "decisions": len(decisions), "edits": len(_ops(folder)),
              "changed_pixels": int((before != after).sum()),
              "labels_sha256": hashlib.sha256(np.ascontiguousarray(after).tobytes()).hexdigest()}
    _write(folder / "exports.json", exports + [record])
    return record
