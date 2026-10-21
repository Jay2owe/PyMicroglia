"""Saved tracker explanations, aligned by the native history module."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .exploration_data import _observations, read_rows, recording, table_path
from .exploration_images import declared_source, signature

TABLES = ("history_gap_frames", "history_lifespans", "history_residency", "history_gap_evidence")
DECISIONS = {"gap_evidence", "termination_evidence", "termination_audit", "residency_registry"}
MAX_METADATA_BYTES = 16 * 1024**2


def _folder(root, stem):
    return root / "AI_Exports/exploration_history" / hashlib.sha256(stem.encode()).hexdigest()[:16]


@lru_cache(maxsize=64)
def _content_hash(path, size, modified):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _fingerprint(path):
    return _content_hash(*signature(Path(path)))


def _alignment(movie):
    return hashlib.sha256(json.dumps(movie, sort_keys=True).encode()).hexdigest()


def _derived(declaration, root, movie, observations, frames):
    """Let the native module interpret source-frame indices and history flags."""
    from pymicroglia.measure.modules.history import derive
    from pymicroglia.tracking.contract import DecisionTables

    if isinstance(declaration, (str, Path)):
        where = Path(declaration)
        tables = DecisionTables.motion(where if where.is_absolute() else root / where)
    else:
        tables = DecisionTables.from_mapping(declaration, root)
    tables = DecisionTables(tables.root, tuple(row for row in tables.tables if row[0] in DECISIONS))
    sources = tables.present()
    # Inspect size before asking the native module to read external decisions.
    if sum(path.stat().st_size for path in sources.values()) > MAX_METADATA_BYTES:
        raise ValueError("Tracker explanations exceed the 16 MiB interactive metadata budget.")
    clock = pd.DataFrame(frames)
    clock["source_imagej_frame"] = clock["source_frame_index"] + 1
    named = pd.DataFrame([(identity, frame) for identity, rows in observations.items() for frame in rows],
                         columns=["identity", "frame_index"])
    context = SimpleNamespace(decisions=tables, frame_table=lambda: clock, n_frames=len(frames),
                              module_params=lambda name: {})
    result = derive(named, context)
    return {name: json.loads(result[name].to_json(orient="records")) for name in TABLES if name in result}, sources


def verify_tracking_alignment(run, stem, tracking_labels, *, expected_tracking_sha256):
    """Verify frozen original masks and exact per-frame measured identity membership."""
    from .exploration_context import _labels

    root, _, movie = recording(run, stem)
    path, observations, frames, _ = _observations(root, movie)
    base = Path(tracking_labels).resolve()
    if _fingerprint(base) != expected_tracking_sha256:
        raise ValueError("The supplied tracking labels do not match the frozen tracking checksum.")
    before = _labels(*signature(base))
    labels_path = declared_source(root, movie, "labels")
    after = _labels(*signature(labels_path))
    if before.shape != after.shape or len(after) != len(frames):
        raise ValueError("Tracking and refined masks do not have the same frame alignment.")
    for index, (old, new) in enumerate(zip(before, after)):
        old_ids = set(map(int, np.unique(old))) - {0}
        new_ids = set(map(int, np.unique(new))) - {0}
        measured = {identity for identity, rows in observations.items() if index in rows}
        if old_ids != new_ids or new_ids != measured:
            raise ValueError(f"Tracking, refined masks and measured identities differ at frame {index}.")
    return root, movie, path, observations, frames, labels_path, base


def attach_tracking_history(run, stem, tracking_labels, decisions, *, expected_tracking_sha256):
    """Freeze explanations from a verified tracking run without changing its data.

    The explicit checksum and exact per-frame identity membership must match.
    Refining cell shapes may change pixels, but this attachment cannot relabel,
    add or remove an observed identity. Only small display metadata is written.
    """
    from .exploration_rhythms import _atomic_json

    root, movie, path, observations, frames, labels_path, base = verify_tracking_alignment(
        run, stem, tracking_labels, expected_tracking_sha256=expected_tracking_sha256)
    tables, sources = _derived(decisions, root, movie, observations, frames)
    result = {"schema": "pymicroglia.exploration-history.v1", "stem": stem,
              "source_sha256": _fingerprint(path), "movie_sha256": _alignment(movie),
              "labels_sha256": _fingerprint(labels_path), "tracking_labels": str(base),
              "tracking_sha256": expected_tracking_sha256,
              "source_frame_offset": movie.get("spec", {}).get("source_frame_offset", 0),
              "sources": {name: {"path": str(p.resolve()), "sha256": _fingerprint(p)} for name, p in sources.items()},
              "tables": tables, "producer": "pymicroglia.measure.modules.history.derive"}
    folder = _folder(root, stem)
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_json(folder / "history.json", result)
    return {"path": str(folder / "history.json"), "tables": {k: len(v) for k, v in tables.items()}}


def load_history(root, movie, observations, frames):
    """Prefer native saved history; allow a declared or checksum-pinned history."""
    tables, sources = {}, {}
    for name in TABLES:
        path = table_path(root, movie, name)
        if path:
            tables[name], sources[name] = read_rows(path), str(path)
    if "history_gap_frames" in tables:
        return tables, sources, ""
    declaration = movie.get("spec", {}).get("history")
    if declaration:
        derived, original = _derived(declaration, root, movie, observations, frames)
        derived.update(tables)
        sources.update({name: str(path) for name, path in original.items()})
        return derived, sources, ""
    attached = _folder(root, movie["stem"]) / "history.json"
    if not attached.is_file():
        return tables, sources, "Tracker gap explanations were not saved with this recording."
    try:
        if attached.stat().st_size > MAX_METADATA_BYTES:
            raise ValueError("The attached tracker history exceeds the metadata budget.")
        record = json.loads(attached.read_text(encoding="utf-8"))
        source = table_path(root, movie, "cell_frame")
        labels = declared_source(root, movie, "labels")
        if (record.get("schema") != "pymicroglia.exploration-history.v1"
                or record.get("stem") != movie["stem"]
                or record.get("movie_sha256") != _alignment(movie)
                or record.get("source_sha256") != _fingerprint(source)
                or record.get("labels_sha256") != _fingerprint(labels)):
            raise ValueError("Attached tracker explanations do not match these saved masks and observations.")
        joined = record["tables"]
        joined.update(tables)
        sources.update({name: entry["path"] for name, entry in record.get("sources", {}).items()})
        return joined, sources, ""
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return tables, sources, str(exc)


def _true(value):
    return value is True or str(value).strip().lower() in {"true", "1", "1.0"}


def timeline_explanations(root, movie, observations, frames):
    """Retain every explanation for actual missing frames, including conflicts."""
    from pymicroglia.figure_tables.presence import mechanism_label, mechanism_palette
    from pymicroglia.visualisation.panels import colour
    from .exploration_data import number

    tables, sources, message = load_history(root, movie, observations, frames)
    index, hosts = {}, {}
    for row in tables.get("history_gap_frames", []):
        identity, frame = number(row.get("identity")), number(row.get("frame_index"))
        if identity is None or frame is None or identity != int(identity) or frame != int(frame):
            continue
        identity, frame = int(identity), int(frame)
        if identity not in observations or not 0 <= frame < len(frames) or frame in observations[identity]:
            continue  # repaired gaps and retired names cannot explain current observations
        if "still_missing_in_accepted_labels" in row and not _true(row["still_missing_in_accepted_labels"]):
            continue
        mechanism = row.get("mechanism")
        if mechanism not in (None, "", "nan"):
            index.setdefault((identity, frame), []).append(row)
    for row in tables.get("history_residency", []):
        identity, frame, host = (number(row.get(key)) for key in ("identity", "frame_index", "host_identity"))
        if identity is not None and frame is not None and host is not None and host > 0:
            hosts.setdefault((int(identity), int(frame)), set()).add(int(host))
    lifespans = {int(row["identity"]): row for row in tables.get("history_lifespans", [])}
    mechanisms = {str(row["mechanism"]) for rows in index.values() for row in rows}
    palette = mechanism_palette(mechanisms)
    legend = {key: {"label": mechanism_label(key), "colour": colour(role)} for key, role in palette.items()}
    legend["no_recorded_cause"] = {"label": "Gap with no recorded explanation", "colour": colour("orange")}
    if any(len({r["mechanism"] for r in rows}) > 1 for rows in index.values()):
        legend["multiple_recorded_causes"] = {"label": "Multiple recorded reasons", "colour": colour("muted")}
    by_cell = {}
    for identity, observed in observations.items():
        gaps = []
        for frame in range(min(observed), max(observed) + 1):
            if frame in observed:
                continue
            evidence = index.get((identity, frame), [])
            reasons = sorted({str(row["mechanism"]) for row in evidence})
            key = reasons[0] if len(reasons) == 1 else "multiple_recorded_causes" if reasons else "no_recorded_cause"
            details = {"reason": key, "mechanisms": reasons, "labels": [mechanism_label(r) for r in reasons],
                       "hosts": sorted(hosts.get((identity, frame), set())),
                       "prior_owners": sorted({int(number(row["prior_owner"])) for row in evidence if number(row.get("prior_owner")) and number(row["prior_owner"]) > 0}),
                       "next_owners": sorted({int(number(row["next_owner"])) for row in evidence if number(row.get("next_owner")) and number(row["next_owner"]) > 0})}
            if gaps and gaps[-1]["end_frame"] + 1 == frame and all(gaps[-1][k] == v for k, v in details.items()):
                gaps[-1]["end_frame"] = frame
            else:
                gaps.append({**details, "start_frame": frame, "end_frame": frame})
        termination = lifespans.get(identity, {})
        current_ending = number(termination.get("last_source_frame")) == frames[max(observed)]["source_frame_index"] + 1
        by_cell[identity] = {"gaps": gaps, "termination": termination, "termination_current": current_ending}
    used = {gap["reason"] for cell in by_cell.values() for gap in cell["gaps"]}
    return by_cell, {key: value for key, value in legend.items() if key in used}, sources, message
