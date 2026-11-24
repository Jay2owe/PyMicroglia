"""Render, Movie and the trace grid of the correction reviewer (Jamie 2026-11-18: "get the render and the movie
functions working ... a button to see the trace grids as well").

Render: an MP4 of the whole recording with the current corrected outlines (decisions and every edit applied) on
the raw red frames, drawn by video.full_recording.overlay (display only: native pixels, nothing re-measured).
It runs in the background and is written to <review folder>/renders/<stem>_corrected_<time>.mp4 with a JSON
receipt (edit count, frames). Movie opens the newest one.

Trace grid: per cell and frame, the total and mean red signal inside the current outline (raw red pixel values,
as the movie shows them), with gaps where the cell is absent; drawn in the page like the Workbench's All cell
traces grid. Live from the current edits, so it is a review aid, not the measurement export.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

_RENDERS: dict[str, dict] = {}
FPS = 8


def _renders_dir(rec) -> Path:
    return rec.folder / "renders"


def render_movie(root, stem) -> dict:
    """Start rendering the corrected movie in the background; returns at once."""
    from .correction_review import _rec, _ops
    rec = _rec(root, stem)
    key = str(rec.folder)
    job = _RENDERS.get(key)
    if job and job["state"] == "running":
        return {"ok": True, "message": f"Already rendering: frame {job['done']} of {job['total']}", **job}
    T = rec.data["T"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _renders_dir(rec) / f"{rec.stem}_corrected_{stamp}.mp4"
    job = _RENDERS[key] = {"state": "running", "done": 0, "total": T, "path": str(out), "error": "",
                           "edits": len(_ops(rec.folder))}
    threading.Thread(target=_render, args=(rec, out, job), daemon=True, name="correction-render").start()
    return {"ok": True, "message": f"Rendering {rec.stem}: {T} frames with {job['edits']} edits applied", **job}


def _render(rec, out: Path, job: dict) -> None:
    import imageio_ffmpeg
    from ..video.full_recording import overlay
    from .correction_cells import current_stack
    try:
        stack = current_stack(rec)["stack"]
        T = len(stack)
        sample = [rec.raw_frame(f) for f in np.linspace(0, T - 1, min(T, 12)).astype(int)]
        high = float(np.percentile(np.concatenate([np.nan_to_num(s).ravel() for s in sample]), 99.8))
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = out.with_name(out.stem + ".partial.mp4")
        first = overlay(rec.raw_frame(0), stack[0], high, "")
        h, w = first.shape[0] + first.shape[0] % 2, first.shape[1] + first.shape[1] % 2
        writer = imageio_ffmpeg.write_frames(str(partial), (w, h), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                             macro_block_size=1, quality=8)
        writer.send(None)
        for f in range(T):
            title = f"{rec.stem} | corrected | frame {f} | {rec.hours.get(f, '?')} h | {job['edits']} edits"
            img = overlay(rec.raw_frame(f), np.asarray(stack[f]), high, title)
            img = np.pad(img, ((0, h - img.shape[0]), (0, w - img.shape[1]), (0, 0)))
            writer.send(np.ascontiguousarray(img))
            job["done"] = f + 1
        writer.close()
        decoded, _ = imageio_ffmpeg.count_frames_and_secs(str(partial))
        if decoded != T:
            raise ValueError(f"the movie has {decoded} frames, the recording {T}")
        partial.replace(out)
        out.with_suffix(".json").write_text(json.dumps({
            "stem": rec.stem, "frames": T, "fps": FPS, "edits_applied": job["edits"], "display_high": high,
            "made": datetime.now().isoformat(timespec="seconds"),
            "what": "raw red frames with the corrected outlines (decisions + edits); display only"}, indent=1),
            encoding="utf-8")
        job["state"] = "done"
    except Exception as err:   # reported to the page; the partial file is left for inspection
        job["state"], job["error"] = "failed", str(err)


def render_status(root, stem) -> dict:
    from .correction_review import _rec
    rec = _rec(root, stem)
    job = _RENDERS.get(str(rec.folder))
    latest = latest_movie(rec)
    return {"job": job, "latest": latest.name if latest else None}


def latest_movie(rec):
    folder = _renders_dir(rec)
    found = sorted(folder.glob(f"{rec.stem}_corrected_*.mp4")) if folder.is_dir() else []
    found = [p for p in found if not p.name.endswith(".partial.mp4")]
    return found[-1] if found else None


def movie_file(root, stem) -> Path:
    from .correction_review import _rec
    path = latest_movie(_rec(root, stem))
    if path is None:
        raise ValueError("No corrected movie yet: press Render first")
    return path


def traces(root, stem) -> dict:
    """{hours, cells: [{n, total, mean}]} for every cell in the current outlines; None where it is absent."""
    from .correction_review import _rec
    from .correction_cells import traces_for
    return traces_for(_rec(root, stem), _traces)


def _traces(rec) -> dict:
    from .correction_cells import current_stack
    stack = current_stack(rec)["stack"]
    T, size = len(stack), int(max(int(s.max()) for s in stack)) + 1
    total = np.zeros((T, size)); area = np.zeros((T, size))
    for f in range(T):
        lab = np.asarray(stack[f]).ravel()
        raw = np.nan_to_num(rec.raw_frame(f)).ravel()
        total[f] = np.bincount(lab, raw, size)
        area[f] = np.bincount(lab, minlength=size)
    out = []
    for n in np.flatnonzero(area[:, 1:].sum(0)) + 1:
        here = area[:, n] > 0
        out.append({"n": int(n),
                    "total": [round(float(v), 1) if h else None for v, h in zip(total[:, n], here)],
                    "mean": [round(float(t / a), 3) if h else None for t, a, h in zip(total[:, n], area[:, n], here)]})
    hours = [float(rec.hours[f]) if rec.hours.get(f, "?") != "?" else float(f) for f in range(T)]
    return {"hours": hours, "unit": "h" if rec.interval_min else "frame", "cells": out}
