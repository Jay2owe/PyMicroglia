"""One canonical place to review cell traces, rhythm fits and include/exclude decisions.

A review is a folder (``review.json``) naming the recordings it covers. Each
rhythm *variant* is a set of saved analyses sharing settings: canonical results
from :func:`pymicroglia.ui.exploration_rhythms.run_rhythm_analysis`, or older
figure-export folders converted once by :func:`import_legacy`. Nothing here
fits a rhythm.

Human decisions (include, exclude, star, note) live in ``decisions.json`` with
an append-only ``decisions_log.jsonl``. Agents may only add *suggestions*; they
never write a human mark. :func:`handoff` freezes the current decisions into
``handoff.json`` and ``included_cells.csv`` for downstream analysis, bound to
the hash of every recording's measurement table so later re-tracking shows up
as stale rather than being silently reused.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from base64 import b64encode
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np

from .exploration_data import manifest, number, recording, table_path

SCHEMA = "pymicroglia.trace-review.v1"
STATUSES = ("unreviewed", "include", "exclude")
MAX_RESULT_BYTES = 16 * 1024**2
MAX_TABLE_BYTES = 192 * 1024**2
MAX_RECORDINGS = 256
DESCRIPTOR_KEYS = ("fit_method", "significance_method", "fft_component_test", "detrend",
                   "trace_normalization", "trace_view", "period_min_hours", "period_max_hours",
                   "multiple_testing", "rhythmic_alpha", "window_start_hours", "window_end_hours",
                   "trace_filter", "metrics")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path, budget=MAX_RESULT_BYTES):
    path = Path(path)
    if path.stat().st_size > budget:
        raise ValueError(f"{path.name} exceeds the review metadata budget.")
    return json.loads(path.read_text(encoding="utf-8"))


def cell_key(run, stem, identity):
    return f"{recording_key(run, stem)}#{int(identity)}"


def recording_key(run, stem):
    # Short and stable: the same recording always has the same key in one review.
    return hashlib.sha256(f"{Path(run).resolve()}|{stem}".encode()).hexdigest()[:8] + ":" + stem


def default_home(run):
    root, _ = manifest(run)
    return root / "AI_Exports" / "trace_review"


# --------------------------------------------------------------------- review

def create_review(home, recordings, *, title=None):
    """Create, or extend, a review covering ``[{"run": ..., "stem": ...}]``."""
    home = Path(home).expanduser().resolve()
    if not isinstance(recordings, list) or not recordings or len(recordings) > MAX_RECORDINGS:
        raise ValueError(f"Choose between one and {MAX_RECORDINGS} recordings.")
    existing = open_review(home) if (home / "review.json").is_file() else None
    rows = list(existing["recordings"]) if existing else []
    seen = {(r["run"], r["stem"]) for r in rows}
    for source in recordings:
        if not isinstance(source, dict) or not isinstance(source.get("run"), str) or not isinstance(source.get("stem"), str):
            raise ValueError("Each recording needs its saved run and recording name.")
        root, _, movie = recording(source["run"], source["stem"])
        key = (str(root), movie["stem"])
        if key not in seen:
            seen.add(key)
            rows.append({"run": key[0], "stem": key[1], "added_at": _now()})
    review = {"schema": SCHEMA, "title": title or (existing or {}).get("title") or home.parent.name,
              "created_at": (existing or {}).get("created_at") or _now(), "recordings": rows}
    _atomic_json(home / "review.json", review)
    return review


def open_review(home):
    path = Path(home).expanduser().resolve() / "review.json"
    if not path.is_file():
        raise ValueError("This folder has no trace review; create one first.")
    review = _read_json(path)
    if review.get("schema") != SCHEMA or not isinstance(review.get("recordings"), list):
        raise ValueError("review.json is not a PyMicroglia trace review.")
    return review


def review_for_run(run, *, title=None):
    """Every recording declared by one run, in one review beside it."""
    root, data = manifest(run)
    return default_home(root), create_review(default_home(root), [{"run": str(root), "stem": m["stem"]} for m in data["movies"]], title=title)


def cell_frame_path(root, movie):
    found = table_path(root, movie, "cell_frame")
    if found is not None:
        return found
    # Archived runs keep tables beside each recording.
    for candidate in (root / movie["stem"] / "tables" / "cell_frame.csv", root / movie["stem"] / "cell_frame.csv"):
        if candidate.is_file():
            return candidate
    return None


_hash_memo: dict = {}


def _table_hash(path):
    if path is None:
        return None
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _hash_memo:
        _hash_memo[key] = _sha(path)
    return _hash_memo[key]


def _recording_binding(run, stem):
    root, _, movie = recording(run, stem)
    path = cell_frame_path(root, movie)
    return {"cell_frame": str(path) if path else None, "sha256": _table_hash(path)}


# ------------------------------------------------------------------- variants

def _descriptor(options, settings, label):
    options, settings = options or {}, settings or {}
    out = {}
    for key in DESCRIPTOR_KEYS:
        value = options.get(key, settings.get(key))
        if key == "trace_normalization" and value is None:
            value = settings.get("trace_normalization")
        out[key] = value
    out["label"] = label or None
    return out


def describe_variant(descriptor):
    """A short human description of how a variant was produced."""
    d = descriptor
    parts = [d.get("label")] if d.get("label") else []
    names = {"fft_nlls": "FFT-NLLS", "lomb": "Lomb-Scargle"}
    method = names.get(d.get("fit_method"), d.get("fit_method") or "?")
    test = "component test" if d.get("fft_component_test") else names.get(d.get("significance_method"), d.get("significance_method") or "")
    parts.append(method if test in ("", method) else f"{method} + {test}")
    if d.get("detrend"):
        parts.append(str(d["detrend"]).replace("_", " "))
    if d.get("period_min_hours") is not None and d.get("period_max_hours") is not None:
        parts.append(f"{_short(d['period_min_hours'])}-{_short(d['period_max_hours'])} h")
    start, end = d.get("window_start_hours"), d.get("window_end_hours")
    if start is not None or end is not None:
        parts.append(f"window {_short(start) if start is not None else 'start'}-{_short(end) if end is not None else 'end'} h")
    return " · ".join(parts)


def _short(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _variant_key(descriptor):
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _canonical_results(run, stem):
    from .exploration_rhythms import _folder
    root = Path(run)
    folder = _folder(root, stem)
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob("*/result.json")):
        try:
            result = _read_json(path)
        except (OSError, ValueError):
            continue
        if result.get("stem") != stem or not (path.parent / "statistics.csv").is_file():
            continue
        options = result.get("resolved_options") or {}
        found.append({"path": str(path), "analysis_id": result.get("analysis_id"), "created_at": result.get("created_at"),
                      "source_sha256": result.get("source_sha256"), "metrics": result.get("metrics") or options.get("metrics") or [],
                      "descriptor": _descriptor(options, result.get("settings"), options.get("label")),
                      "settings": result.get("settings") or {}, "origin": "canonical",
                      "reproduce": str(path.with_name("reproduce.py"))})
    return found


def _imported_results(home, run, stem):
    folder = Path(home) / "imported"
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob("*/result.json")):
        try:
            result = _read_json(path)
        except (OSError, ValueError):
            continue
        if result.get("run") != str(run) or result.get("stem") != stem:
            continue
        found.append({"path": str(path), "analysis_id": result["analysis_id"], "created_at": result.get("created_at"),
                      "source_sha256": result.get("source_sha256"), "metrics": result.get("metrics") or [],
                      "descriptor": _descriptor({}, result.get("settings"), result.get("label")),
                      "settings": result.get("settings") or {}, "origin": "imported",
                      "imported_from": result.get("imported_from"), "reproduce": result.get("reproduce")})
    return found


def list_variants(home):
    """Group every saved analysis of the review's recordings by its settings."""
    home = Path(home).expanduser().resolve()
    review = open_review(home)
    variants = {}
    for source in review["recordings"]:
        binding = _recording_binding(source["run"], source["stem"])
        rkey = recording_key(source["run"], source["stem"])
        for found in _canonical_results(source["run"], source["stem"]) + _imported_results(home, source["run"], source["stem"]):
            key = _variant_key(found["descriptor"])
            variant = variants.setdefault(key, {"key": key, "descriptor": found["descriptor"],
                                                "description": describe_variant(found["descriptor"]),
                                                "metrics": found["metrics"], "recordings": {}, "created_at": found["created_at"]})
            previous = variant["recordings"].get(rkey)
            # Identical settings re-run later replace the earlier analysis.
            if previous is None or str(found["created_at"] or "") > str(previous["created_at"] or ""):
                variant["recordings"][rkey] = {**{k: found.get(k) for k in ("path", "analysis_id", "created_at", "origin", "imported_from", "reproduce", "settings")},
                                               "fresh": bool(binding["sha256"]) and found["source_sha256"] == binding["sha256"]}
            variant["created_at"] = max(str(variant["created_at"] or ""), str(found["created_at"] or ""))
    rows = sorted(variants.values(), key=lambda v: v["created_at"], reverse=True)
    for index, row in enumerate(rows):
        row["short"] = chr(ord("A") + index) if index < 26 else f"V{index + 1}"
        row["coverage"] = len(row["recordings"])
        row["stale"] = sum(1 for r in row["recordings"].values() if not r["fresh"])
    return {"review": review, "variants": rows,
            "recordings": [{"key": recording_key(r["run"], r["stem"]), "run": r["run"], "stem": r["stem"]} for r in review["recordings"]]}


# ----------------------------------------------------------------------- grid

def _rows(path):
    path = Path(path)
    if path.stat().st_size > MAX_TABLE_BYTES:
        raise ValueError(f"{path.name} exceeds the review table budget.")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def _round(value, digits=5):
    if value is None:
        return None
    return float(f"{value:.{digits}g}")


def _spike_score(values):
    """Largest single-step jump relative to the typical step: high means a sudden spike."""
    finite = [v for v in values if v is not None]
    if len(finite) < 4:
        return None
    steps = np.abs(np.diff(np.asarray(finite, float)))
    typical = float(np.median(steps))
    return _round(float(steps.max()) / typical, 3) if typical > 0 else None


def window_of(descriptor):
    return (descriptor.get("window_start_hours"), descriptor.get("window_end_hours"))


def load_grid(home, variant_keys, metrics=None):
    """Statistics, displayed traces and fits for every cell, one *series* per variant and metric.

    Series share one time axis, so variants analysed over different recording
    windows are refused together; view them one at a time instead.
    """
    listing = list_variants(home)
    chosen = [v for key in variant_keys for v in listing["variants"] if v["key"] == key]
    if not chosen or len(chosen) != len(set(variant_keys)):
        raise ValueError("Choose saved variants listed by this review.")
    if len({window_of(v["descriptor"]) for v in chosen}) > 1:
        raise ValueError("Variants with different time windows cannot be overlaid; view them one at a time.")
    if isinstance(metrics, str):
        metrics = [m for m in metrics.split(",") if m]
    recordings = {r["key"]: r for r in listing["recordings"]}
    cells = {}
    series = []
    for variant in chosen:
        wanted = [m for m in (metrics or []) if m in variant["metrics"]] or variant["metrics"][:1] or [None]
        for metric in wanted:
            series.append({"id": f"{variant['key']}:{metric or ''}", "variant": variant["key"], "metric": metric})
    for item in series:
        variant = next(v for v in chosen if v["key"] == item["variant"])
        sid, wanted = item["id"], item["metric"]
        for rkey, source in variant["recordings"].items():
            folder = Path(source["path"]).parent
            info = recordings.get(rkey)
            if info is None:
                continue
            for row in _rows(folder / "statistics.csv"):
                if wanted and row.get("metric") not in (None, "", wanted):
                    continue
                identity = int(float(row["identity"]))
                cell = cells.setdefault(f"{rkey}#{identity}", {"key": f"{rkey}#{identity}", "recording": rkey, "stem": info["stem"],
                                                               "run": info["run"], "identity": identity, "stats": {}, "traces": {}})
                cell["stats"][sid] = {
                    "status": row.get("rhythm_status") or "unknown",
                    "period": _round(number(row.get("period_hours"))), "p": _round(number(row.get("p_value")), 4),
                    "q": _round(number(row.get("q_value")), 4), "phase": _round(number(row.get("phase_hours"))),
                    "amplitude": _round(number(row.get("amplitude"))), "cycles": _round(number(row.get("cycles_observed")), 3),
                    "observations": int(number(row.get("observations")) or 0),
                    "supported": str(row.get("supported_period")).lower() in {"true", "1"},
                    "reason": row.get("reason") or ""}
            traces = folder / "traces.csv"
            if not traces.is_file():
                continue
            for row in _rows(traces):
                if wanted and row.get("metric") not in (None, "", wanted):
                    continue
                cell = cells.get(f"{rkey}#{int(float(row['identity']))}")
                if cell is None:
                    continue
                trace = cell["traces"].setdefault(sid, {"frame": [], "hours": [], "value": [], "fit": [], "raw": []})
                trace["frame"].append(int(float(row["frame_index"])))
                trace["hours"].append(_round(number(row.get("hours")), 6))
                trace["value"].append(_round(number(row.get("value"))))
                trace["fit"].append(_round(number(row.get("descriptive_fit"))))
                trace["raw"].append(_round(number(row.get("raw_value"))))
    for cell in cells.values():
        quality = {}
        for key, trace in cell["traces"].items():
            if not any(v is not None for v in trace["fit"]):
                trace["fit"] = None
            raw = [v for v in trace.pop("raw") if v is not None]
            quality[key] = {"spike": _spike_score(trace["value"]), "signal": _round(float(np.mean(raw))) if raw else None,
                            "points": sum(v is not None for v in trace["value"])}
        cell["quality"] = quality
    ordered = sorted(cells.values(), key=lambda c: (list(recordings).index(c["recording"]) if c["recording"] in recordings else 0, c["identity"]))
    return {"variants": chosen, "series": series, "cells": ordered,
            "recordings": listing["recordings"], "decisions": load_decisions(home)}


# ------------------------------------------------------------------ decisions

def _empty_decisions():
    return {"schema": SCHEMA + ".decisions", "revision": 0, "updated_at": None, "preferred_variant": None,
            "cells": {}, "suggestions": {}}


def load_decisions(home):
    path = Path(home).expanduser().resolve() / "decisions.json"
    return _read_json(path) if path.is_file() else _empty_decisions()


def _log(home, entry):
    with (Path(home) / "decisions_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, default=str) + "\n")


def save_decisions(home, changes, *, revision, preferred_variant=..., reviewer="Jamie"):
    """Apply human marks. ``revision`` must match the saved one, so edits are never lost."""
    home = Path(home).expanduser().resolve()
    open_review(home)
    current = load_decisions(home)
    if revision != current["revision"]:
        raise ValueError("Decisions changed elsewhere; reload before saving.")
    if not isinstance(changes, list) or len(changes) > 100_000:
        raise ValueError("Changes must be a list of cell decisions.")
    stamp = _now()
    for change in changes:
        key = change.get("cell") if isinstance(change, dict) else None
        if not isinstance(key, str) or "#" not in key:
            raise ValueError("Each change names a cell key.")
        entry = dict(current["cells"].get(key) or {"status": "unreviewed", "star": False, "note": ""})
        if "status" in change:
            if change["status"] not in STATUSES:
                raise ValueError(f"Status must be one of {', '.join(STATUSES)}.")
            entry["status"] = change["status"]
        if "star" in change:
            entry["star"] = bool(change["star"])
        if "note" in change:
            entry["note"] = str(change["note"])[:2000]
        entry["updated_at"], entry["reviewer"] = stamp, reviewer
        if entry["status"] == "unreviewed" and not entry["star"] and not entry["note"]:
            current["cells"].pop(key, None)
        else:
            current["cells"][key] = entry
    if preferred_variant is not ...:
        current["preferred_variant"] = preferred_variant
    current["revision"] += 1
    current["updated_at"] = stamp
    _atomic_json(home / "decisions.json", current)
    _log(home, {"at": stamp, "revision": current["revision"], "reviewer": reviewer, "changes": changes,
                **({"preferred_variant": preferred_variant} if preferred_variant is not ... else {})})
    return current


def suggest(home, suggestions, *, agent):
    """Record an agent's include/exclude suggestions. They are shown, never applied, until Jamie accepts them."""
    home = Path(home).expanduser().resolve()
    open_review(home)
    if not agent or not isinstance(suggestions, list):
        raise ValueError("Name the suggesting agent and give a list of suggestions.")
    current = load_decisions(home)
    stamp = _now()
    for row in suggestions:
        if not isinstance(row, dict) or row.get("status") not in ("include", "exclude") or "#" not in str(row.get("cell")):
            raise ValueError("Each suggestion needs a cell key and include or exclude.")
        current["suggestions"][row["cell"]] = {"status": row["status"], "reason": str(row.get("reason") or "")[:500],
                                               "agent": str(agent), "at": stamp}
    current["revision"] += 1
    current["updated_at"] = stamp
    _atomic_json(home / "decisions.json", current)
    _log(home, {"at": stamp, "revision": current["revision"], "agent": agent, "suggestions": suggestions})
    return current


# -------------------------------------------------------------------- handoff

def handoff(home, *, note="", variant=None, metric=None):
    """Freeze the decisions for downstream analysis. Unreviewed cells stay listed as unreviewed."""
    home = Path(home).expanduser().resolve()
    listing = list_variants(home)
    decisions = load_decisions(home)
    preferred = variant or decisions.get("preferred_variant")
    chosen = next((v for v in listing["variants"] if v["key"] == preferred), None)
    if preferred and chosen is None:
        raise ValueError("The preferred variant is no longer available in this review.")
    keys = set(decisions["cells"])
    stats = {}
    if chosen:
        if metric is not None and metric not in chosen["metrics"]:
            raise ValueError("The chosen variant did not analyse that metric.")
        grid = load_grid(home, [chosen["key"]], [metric] if metric else None)
        sid = grid["series"][0]["id"]
        metric = grid["series"][0]["metric"]
        keys |= {c["key"] for c in grid["cells"]}
        stats = {c["key"]: c["stats"].get(sid, {}) for c in grid["cells"]}
    recordings = {r["key"]: r for r in listing["recordings"]}
    rows = []
    order = list(recordings)
    for key in sorted(keys, key=lambda k: (order.index(k.rsplit("#", 1)[0]) if k.rsplit("#", 1)[0] in order else len(order), int(k.rsplit("#", 1)[1]))):
        rkey, identity = key.rsplit("#", 1)
        info = recordings.get(rkey, {})
        mark = decisions["cells"].get(key, {})
        stat = stats.get(key, {})
        rows.append({"cell": key, "run": info.get("run"), "stem": info.get("stem"), "identity": int(identity),
                     "status": mark.get("status", "unreviewed"), "representative": bool(mark.get("star")),
                     "note": mark.get("note", ""), "rhythm_status": stat.get("status"),
                     "period_hours": stat.get("period"), "p_value": stat.get("p"), "q_value": stat.get("q"),
                     "phase_hours": stat.get("phase")})
    counts = {status: sum(r["status"] == status for r in rows) for status in STATUSES}
    counts["representative"] = sum(r["representative"] for r in rows)
    stamp = _now()
    record = {"schema": SCHEMA + ".handoff", "created_at": stamp, "note": str(note)[:4000],
              "decisions_revision": decisions["revision"], "counts": counts,
              "variant": None if chosen is None else {"key": chosen["key"], "description": chosen["description"], "metric": metric,
                                                       "descriptor": chosen["descriptor"],
                                                       "analyses": {k: {"analysis_id": v["analysis_id"], "path": v["path"], "origin": v["origin"], "fresh": v["fresh"]}
                                                                    for k, v in chosen["recordings"].items()}},
              "bindings": {r["key"]: _recording_binding(r["run"], r["stem"]) for r in listing["recordings"]},
              "cells": rows,
              "meaning": "status is Jamie's decision: include = use downstream, exclude = drop, unreviewed = not yet judged. "
                         "representative marks example cells for figures. Agent suggestions are not decisions."}
    _atomic_json(home / "handoff.json", record)
    columns = ["cell", "run", "stem", "identity", "status", "representative", "note", "rhythm_status",
               "period_hours", "p_value", "q_value", "phase_hours"]
    temporary = home / f".included_cells.{uuid4().hex[:8]}.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, home / "included_cells.csv")
    _log(home, {"at": stamp, "handoff": counts, "revision": decisions["revision"], "variant": preferred})
    return record


def read_handoff(home):
    """Downstream entry point: the frozen handoff, with ``stale`` recordings whose measurements changed since."""
    path = Path(home).expanduser().resolve() / "handoff.json"
    if not path.is_file():
        raise ValueError("No handoff has been saved for this review yet.")
    record = _read_json(path, MAX_TABLE_BYTES)
    stale = []
    for rkey, binding in record.get("bindings", {}).items():
        current_path = binding.get("cell_frame")
        current = _table_hash(Path(current_path)) if current_path and Path(current_path).is_file() else None
        if current != binding.get("sha256"):
            stale.append(rkey)
    return {**record, "stale_recordings": stale}


def approved_cells(home, *, representative_only=False):
    """``[(run, stem, identity), ...]`` Jamie included; raises if any recording changed after the handoff."""
    record = read_handoff(home)
    if record["stale_recordings"]:
        raise ValueError("Measurements changed after the handoff for: " + ", ".join(record["stale_recordings"]))
    return [(r["run"], r["stem"], r["identity"]) for r in record["cells"]
            if r["status"] == "include" and (r["representative"] or not representative_only)]


# --------------------------------------------------------------- legacy import

def import_legacy(home, folder, *, label=None):
    """Convert an older trace-grid figure folder into a review variant, keeping its provenance."""
    home, folder = Path(home).expanduser().resolve(), Path(folder).expanduser().resolve()
    statistics, points = folder / "statistics.csv", folder / "figure_data.csv"
    if not statistics.is_file() or not points.is_file():
        raise ValueError("A legacy trace folder needs statistics.csv and figure_data.csv.")
    sources = {}
    if (folder / "sources.csv").is_file():
        for row in _rows(folder / "sources.csv"):
            sources[row.get("file_name")] = row
    manifest_row = sources.get("manifest.json")
    cell_row = sources.get("cell_frame.csv")
    if not manifest_row or not cell_row:
        raise ValueError("sources.csv must record the original manifest.json and cell_frame.csv.")
    run = Path(manifest_row["original_path"]).parent
    _, data = manifest(run)
    parts = Path(cell_row["original_path"]).parts
    stem = next((m["stem"] for m in data["movies"] if m["stem"] in parts), None)
    if stem is None:
        raise ValueError("The original cell_frame.csv does not belong to a recording in its manifest.")
    settings = {}
    if (folder / "der_settings.csv").is_file():
        row = next(_rows(folder / "der_settings.csv"), {})
        try:
            settings = json.loads(row.get("settings_json") or "{}")
        except json.JSONDecodeError:
            settings = {}
    settings.pop("identities", None)
    fingerprint = hashlib.sha256((_sha(statistics) + _sha(points)).encode()).hexdigest()[:32]
    target = home / "imported" / fingerprint
    create_review(home, [{"run": str(run), "stem": stem}])
    if not (target / "result.json").is_file():
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(statistics, target / "statistics.csv")
        shutil.copyfile(points, target / "traces.csv")
        metrics = sorted({r["metric"] for r in _rows(statistics) if r.get("metric")})
        _atomic_json(target / "result.json", {
            "schema": SCHEMA + ".import", "analysis_id": fingerprint, "run": str(run), "stem": stem,
            "label": label or folder.name, "metrics": metrics, "settings": settings,
            "created_at": datetime.fromtimestamp(statistics.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "imported_at": _now(), "imported_from": str(folder),
            "source_sha256": cell_row.get("sha256"),
            "reproduce": str(folder / "plot.py") if (folder / "plot.py").is_file() else None,
            "file_sha256": {"statistics.csv": _sha(statistics), "figure_data.csv": _sha(points)},
            "meaning": "Converted once from a legacy figure export; statistics are the export's own, not refitted."})
    return {"analysis_id": fingerprint, "run": str(run), "stem": stem, "path": str(target / "result.json")}


# --------------------------------------------------------------------- images

def cell_crop(run, stem, identity, frame, *, size=72):
    """Fixed-size raw crop following one cell, with its saved mask outline drawn in."""
    from PIL import Image
    from scipy.ndimage import binary_erosion
    from .exploration_images import _contrast, declared_source, plane, signature
    root, _, movie = recording(run, stem)
    labels, raw = declared_source(root, movie, "labels"), declared_source(root, movie, "raw")
    if labels is None or raw is None or not labels.is_file() or not raw.is_file():
        return {"available": False, "message": "No local raw or mask stack is declared for this recording."}
    size = int(min(max(size, 16), 512))
    current = plane(labels, frame)
    mask = current == int(identity)
    present = bool(mask.any())
    if present:
        yy, xx = np.nonzero(mask)
        cy, cx = float(yy.mean()), float(xx.mean())
    else:
        cy = cx = None
        for step in range(1, 25):  # Follow the nearest frame where the cell was segmented.
            for other in (frame - step, frame + step):
                try:
                    found = np.nonzero(plane(labels, other) == int(identity))
                except ValueError:
                    continue
                if found[0].size:
                    cy, cx = float(found[0].mean()), float(found[1].mean())
                    break
            if cy is not None:
                break
        if cy is None:
            return {"available": False, "message": "This cell is not segmented near this frame."}
    height, width = current.shape
    half = size // 2
    y0, x0 = int(round(cy)) - half, int(round(cx)) - half
    offset = int(movie.get("spec", {}).get("source_frame_offset", 0))
    pixels = plane(raw, frame + offset)
    lo, hi = movie.get("display", {}).get("contrast") or _contrast(*signature(raw))
    tile = np.zeros((size, size), np.float64)
    sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(height, y0 + size), min(width, x0 + size)
    tile[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = pixels[sy0:sy1, sx0:sx1]
    grey = np.clip((tile - lo) * 255 / max(hi - lo, 1e-9), 0, 255).astype(np.uint8)
    rgb = np.stack([grey] * 3, -1)
    if present:
        local = np.zeros((size, size), bool)
        local[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = mask[sy0:sy1, sx0:sx1]
        edge = local & ~binary_erosion(local)
        rgb[edge] = (64, 224, 208)
    out = BytesIO()
    Image.fromarray(rgb).save(out, format="PNG")
    return {"available": True, "present": present, "frame_index": int(frame), "size": size,
            "image": "data:image/png;base64," + b64encode(out.getvalue()).decode("ascii")}


# -------------------------------------------------------------- agent actions
# Registered in ``data/actions.json``. ``source`` is a run folder for creation
# and the review folder for everything else.

METHOD_VERSION = "2026-10-27-trace-review-v1"


def _review_home(source):
    path = Path(source).expanduser().resolve()
    return path if (path / "review.json").is_file() else default_home(path)


def trace_review_create(source, review_dir=None, stems=None, review_title=None):
    """Create or extend the review for a run's recordings; returns its folder."""
    root, data = manifest(source)
    chosen = stems or [m["stem"] for m in data["movies"]]
    home = Path(review_dir).expanduser().resolve() if review_dir else default_home(root)
    review = create_review(home, [{"run": str(root), "stem": s} for s in chosen], title=review_title)
    return {"home": str(home), "recordings": review["recordings"], "url_path": "/review?home=" + str(home)}


def trace_review_variant(source, rhythm_options, fit_workers=None):
    """Run one rhythm variant over every recording in the review, reusing identical completed analyses."""
    from .exploration_rhythms import run_rhythm_analysis
    home = _review_home(source)
    review = open_review(home)
    if not isinstance(rhythm_options, dict):
        raise ValueError("options are rhythm-analysis settings, e.g. {'label': '24-96 h', 'window_start_hours': 24}.")
    done = []
    for item in review["recordings"]:
        result = run_rhythm_analysis(item["run"], item["stem"], rhythm_options, workers=fit_workers, reuse=True)
        done.append({"stem": item["stem"], "analysis_id": result["analysis_id"], "reused": bool(result.get("reused"))})
    return {"home": str(home), "analyses": done}


def trace_review_import(source, legacy_folder, variant_label=None):
    """Convert an older trace-grid figure folder into a variant of this review."""
    return import_legacy(_review_home(source), legacy_folder, label=variant_label)


def trace_review_suggest(source, suggestions, agent):
    """Record include/exclude suggestions for Jamie to accept or ignore; never a decision."""
    current = suggest(_review_home(source), suggestions, agent=agent)
    return {"revision": current["revision"], "suggestions": len(current["suggestions"])}


def trace_review_handoff(source):
    """Jamie's frozen decisions for downstream analysis, with any recordings changed since."""
    return read_handoff(_review_home(source))


# ------------------------------------------------------------------------ CLI

def main(argv=None):
    """``python -m pymicroglia.ui.trace_review <command>`` for agents."""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m pymicroglia.ui.trace_review", description=main.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("create", help="Create a review for every recording in a run, or for chosen ones")
    p.add_argument("run")
    p.add_argument("--home")
    p.add_argument("--stem", action="append")
    p.add_argument("--title")
    p = sub.add_parser("variants", help="List rhythm variants available to a review")
    p.add_argument("home")
    p = sub.add_parser("import", help="Convert a legacy trace-grid figure folder into a variant")
    p.add_argument("home")
    p.add_argument("folder")
    p.add_argument("--label")
    p = sub.add_parser("suggest", help="Record agent suggestions from a CSV with cell,status,reason columns")
    p.add_argument("home")
    p.add_argument("csv")
    p.add_argument("--agent", required=True)
    p = sub.add_parser("handoff", help="Read the frozen handoff for downstream analysis")
    p.add_argument("home")
    p = sub.add_parser("open", help="Start the reviewer and open it in the browser (review folder or run folder)")
    p.add_argument("home")
    p.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "open":
        import socket
        import webbrowser
        from urllib.parse import quote
        import uvicorn
        from .server import create_app
        home = Path(args.home).expanduser().resolve()
        if not (home / "review.json").is_file():
            home, _ = review_for_run(home)
        port = args.port
        if not port:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
        url = f"http://127.0.0.1:{port}/review?home={quote(str(home))}"
        print(f"Trace review: {url}", flush=True)

        def open_when_ready():
            # Startup imports take a while; a browser opened earlier shows an error page.
            import time
            from urllib.request import urlopen
            for _ in range(240):
                try:
                    with urlopen(f"http://127.0.0.1:{port}/api/config", timeout=1):
                        break
                except OSError:
                    time.sleep(.5)
            webbrowser.open(url)

        import threading
        threading.Thread(target=open_when_ready, daemon=True).start()
        uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")
        return
    if args.command == "create":
        if args.stem:
            home = Path(args.home) if args.home else default_home(args.run)
            out = create_review(home, [{"run": args.run, "stem": s} for s in args.stem], title=args.title)
        else:
            home, out = review_for_run(args.run, title=args.title)
            if args.home:
                out = create_review(args.home, out["recordings"], title=args.title)
                home = Path(args.home)
        out = {"home": str(Path(home).resolve()), **out}
    elif args.command == "variants":
        out = list_variants(args.home)
        for variant in out["variants"]:
            variant.pop("descriptor", None)
            variant["recordings"] = {k: {"fresh": v["fresh"], "analysis_id": v["analysis_id"]} for k, v in variant["recordings"].items()}
    elif args.command == "import":
        out = import_legacy(args.home, args.folder, label=args.label)
    elif args.command == "suggest":
        out = suggest(args.home, list(_rows(args.csv)), agent=args.agent)
        out = {"revision": out["revision"], "suggestions": len(out["suggestions"])}
    else:
        out = read_handoff(args.home)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
