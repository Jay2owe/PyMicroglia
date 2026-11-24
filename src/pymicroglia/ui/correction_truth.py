"""Tracking ground truth from Jamie's corrections, and the timeline's suspicious events for the agent.

Jamie 2026-11-17: "we can use my edits compared to the raw video as training data to improve the base tracking
... these edits and especially the older ones that i have since analysed should be saved to a special folder".
A snapshot is a new dated folder (never overwritten) holding, per recording: verbatim copies of the review state
(decisions, edits, notes, Jamie's marks, exports), the tracker's labels it started from, the corrected labels now,
and tables saying how far each cell and each edit can be trusted. The raw movie is referenced by path and SHA256.

It is identity truth on the k=4 cell-body masks (plus seeded, drawn and grown cells), not full-cell outlines,
and only where Jamie looked: untouched cells are unverified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from . import correction_review as cr

STATE_FILES = ("review.json", "decisions.json", "edits.json", "notes.json", "marks.json", "exports.json")
CELL_KEYS = ("a", "b", "from", "to", "cell", "host", "child", "only")
# Ordered most to least likely to be a tracking mistake worth a look (a number ending or starting mid-recording is
# usually a missed join; a jump with no overlap is usually a swap).
SUSPECT_ORDER = ("end", "start", "jump", "pieces", "size", "gap")


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cells(op) -> list[int]:
    return sorted({int(op[k]) for k in CELL_KEYS if op.get(k)})


def snapshot(root, stem, out, *, held_out_note: str = "") -> dict:
    """Write one recording's ground truth into the new folder out (never overwritten). Read-only on the review
    set."""
    import tifffile
    from . import correction_cells as cc
    rec = cr._rec(root, stem)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name in STATE_FILES:
        if (rec.folder / name).is_file():
            shutil.copy2(rec.folder / name, out / name)
    src = cr._read(rec.folder / "review.json", {}).get("source", {})
    base, after, data = cr.composed(root, stem)
    tifffile.imwrite(out / "tracker_labels.tif", base, compression="zlib")
    tifffile.imwrite(out / "truth_labels.tif", after, compression="zlib")

    ops = cr._ops(rec.folder)
    exports = cr._read(rec.folder / "exports.json", [])
    analysed = int(exports[-1]["edits"]) if exports else 0       # edits inside the last export (used downstream)
    marks = cr._marks_of(rec.folder)
    notes = cr._read(rec.folder / "notes.json", [])
    with open(out / "edits_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "op", "by", "first_frame", "last_frame", "cells", "time", "in_analysed_export"])
        for i, op in enumerate(ops):
            w.writerow([i, op.get("op"), op.get("by", "jamie"), op.get("first"), op.get("last"),
                        " ".join(map(str, _cells(op))), op.get("time", ""), i < analysed])

    tl = cc.timeline(rec)
    jamie_cells = {n for op in ops if op.get("by", "jamie") == "jamie" for n in _cells(op)}
    flagged = {int(f["cell"]) for f in marks["flags"]}
    asked = {int(q["cell"]) for q in tl["questions"] if q.get("cell")}
    rows = []
    for c in tl["cells"]:
        n = c["n"]
        status = ("ticked_good" if str(n) in marks["ticks"] else "flagged" if n in flagged else
                  "open_question" if n in asked else "edited" if n in jamie_cells else "untouched")
        rows.append([n, c["runs"][0][0], c["runs"][-1][1], sum(b - a + 1 for a, b in c["runs"]), len(c["runs"]),
                     status, " ".join(sorted({e[1] for e in c["events"]}))])
    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["number", "first_frame", "last_frame", "frames", "runs", "status", "timeline_flags"])
        w.writerows(rows)
    changed = [int((base[f] != after[f]).sum()) for f in range(len(base))]
    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "pixels_changed_from_tracker", "jamie_edits_covering"])
        for f, k in enumerate(changed):
            w.writerow([f, k, sum(op.get("first", 0) <= f <= op.get("last", -1) for op in ops
                                  if op.get("by", "jamie") == "jamie")])

    counts = {}
    for _, _, _, _, _, s, _ in rows:
        counts[s] = counts.get(s, 0) + 1
    manifest = {
        "recording": stem, "made": datetime.now().isoformat(timespec="seconds"),
        "review_set": str(rec.folder),
        "raw": {"path": src.get("raw"), "sha256": _sha(src["raw"]) if src.get("raw") else None,
                "frames": src.get("raw_frames")},
        "tracker_labels_source": {"path": src.get("labels"), "sha256": src.get("labels_sha256")},
        "truth_labels_sha256": hashlib.sha256(np.ascontiguousarray(after).tobytes()).hexdigest(),
        "shape": list(after.shape),
        "edits": {"total": len(ops), "by_jamie": sum(op.get("by", "jamie") == "jamie" for op in ops),
                  "by_agent": sum(op.get("by") == "agent" for op in ops), "in_analysed_export": analysed,
                  "after_analysed_export": len(ops) - analysed,
                  "last_edit": ops[-1].get("time") if ops else None},
        "analysed_export": exports[-1] if exports else None,
        "decisions": len(cr._decisions(rec.folder)), "notes": len(notes),
        "cells_by_status": counts, "pixels_changed_from_tracker": int(sum(changed)),
        "what_it_is": "Identity corrections on k=4 cell-body masks (plus seeded/drawn/grown cells); not full-cell "
                      "outlines. Trust: ticked_good > edited > flagged/open_question; untouched cells are unverified.",
        "held_out": held_out_note,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


SETTINGS = "truth.json"


def truth_settings(root) -> dict:
    """<root>/agent_review/truth.json: {"folder": where snapshots go, "held_out": {stem: note}}."""
    return cr._read(Path(root) / cr.FOLDER / SETTINGS, {})


def save_truth(root, stem, motion_url=None, *, user="jamie") -> dict:
    """The reviewer's Export button (Jamie 2026-11-17: "export and send to manual truth folder ... make it
    automated"): write the new Motion session when anything changed since the last export, then save a dated
    ground-truth snapshot of this recording to <folder>/<stem>/<YYYYMMDD_HHMMSS>/ and point latest.json at it.
    A failed Motion export is reported but the snapshot is still saved."""
    cfg = truth_settings(root)
    if not cfg.get("folder"):
        raise ValueError(f'No ground-truth folder set: add {{"folder": ...}} to {Path(root) / cr.FOLDER / SETTINGS}')
    folder = Path(cfg["folder"])
    rec = cr._rec(root, stem)
    exports = cr._read(rec.folder / "exports.json", [])
    export, export_note = None, ""
    changed = (not exports or exports[-1].get("edits") != len(cr._ops(rec.folder))
               or exports[-1].get("decisions") != len(cr._decisions(rec.folder)))
    if not motion_url:
        export_note = "Motion export skipped: Motion is not connected"
    elif not changed:
        export_note = "no change since the last Motion export"
    else:
        try:
            export = cr.export(root, stem, motion_url, user=user)
        except ValueError as err:
            export_note = f"Motion export skipped: {err}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    m = snapshot(root, stem, folder / stem / stamp, held_out_note=cfg.get("held_out", {}).get(stem, ""))
    latest_path = folder / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.is_file() else {}
    last_export = (cr._read(rec.folder / "exports.json", []) or [{}])[-1]
    latest[stem] = {"snapshot": f"{stem}/{stamp}", "made": m["made"], "edits": m["edits"]["total"],
                    "cells_by_status": m["cells_by_status"], "motion_session": last_export.get("session")}
    tmp = latest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(latest, indent=1), encoding="utf-8")
    tmp.replace(latest_path)
    msg = (f"Saved truth for {stem} ({m['edits']['total']} edits) to {folder.name}/{stem}/{stamp}; "
           + (f"exported {Path(export['session']).name}" if export else export_note))
    return {"ok": True, "message": msg, "snapshot": str(folder / stem / stamp), "export": export,
            "export_note": export_note}


def suspects(root, stem, *, kind: str = "", limit: int = 40, include_ticked: bool = False,
             include_asked: bool = False) -> dict:
    """The timeline's flagged events (what Jamie scans the Gantt chart for), most suspicious first: numbers
    ending or starting mid-recording, jumps with no overlap, a number on separate pieces, sudden size changes,
    gaps. Cells Jamie ticked as good and cells with an open question are left out unless asked for. Each event
    says whether a Jamie edit already covers that cell there."""
    from . import correction_cells as cc
    rec = cr._rec(root, stem)
    tl = cc.timeline(rec)
    marks = cr._marks_of(rec.folder)
    ops = cr._ops(rec.folder)
    asked = {(int(q["cell"]), int(q["frame"])) for q in tl["questions"] if q.get("cell")}
    out = []
    for c in tl["cells"]:
        n = c["n"]
        ticked = str(n) in marks["ticks"]
        if ticked and not include_ticked:
            continue
        for f, k, text in c["events"]:
            if kind and k != kind:
                continue
            near_q = any(cell == n and abs(fr - f) <= 2 for cell, fr in asked)
            if near_q and not include_asked:
                continue
            edited = any(n in _cells(op) and op.get("first", 0) - 2 <= f <= op.get("last", -1) + 2
                         and op.get("by", "jamie") == "jamie" for op in ops)
            out.append({"cell": n, "frame": int(f), "kind": k, "text": text, "ticked": ticked,
                        "jamie_edited_here": edited, "question_here": near_q})
    out.sort(key=lambda e: (e["jamie_edited_here"], SUSPECT_ORDER.index(e["kind"]) if e["kind"] in SUSPECT_ORDER
                            else 99, e["frame"], e["cell"]))
    counts = {}
    for e in out:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    return {"frames": tl["T"], "total": len(out), "counts": counts, "events": out[:max(1, int(limit))],
            "jamie_flags": marks["flags"], "ticked": sorted(int(n) for n in marks["ticks"]),
            "open_questions": tl["questions"]}
