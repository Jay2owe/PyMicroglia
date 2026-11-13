"""From dropped files to the full automated chain, with a person's labels.

A dropped folder that carries a microscope manifest runs as it is. Bare TIFF
stacks get Auto-Organotypic's own local-input manifest, written into
``<folder>/pymicroglia_analysis`` and never beside a recording. Either way the
run is :func:`pymicroglia.pipelines.auto_microglia.run`, unchanged.

Labels (name, condition, treatment, sample, notes) are kept in the folder's
``recording_labels.json`` through Auto-Organotypic, so its workflow, this run
and later grouped comparisons all read one answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from auto_organotypic import recording_labels

#: Where bare stacks get their input manifest and therefore their results.
HOME = "pymicroglia_analysis"
TIFFS = {".tif", ".tiff"}
MAX_RECORDINGS = 500
_SIGNAL_WORDS = ("biolum", "luc", "lum")


def _key(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _manifest_recordings(root: Path) -> list | None:
    from auto_organotypic import sources
    try:
        sources.find_manifest(root)
    except sources.ManifestNotFound:
        return None
    return sources.read_manifest(root)


def _guess_roles(channels: list[dict[str, Any]]) -> dict[str, int]:
    """The signal only: a named luciferase channel, else 1.

    Never the outline channel. This page does not outline, and outlining
    elsewhere needs its channel named by a person.
    """
    signal = next((row["index"] for row in channels
                   if any(word in str(row.get("name") or "").lower() for word in _SIGNAL_WORDS)),
                  1)
    return {"signal": signal}


def _traceable(frames: int, interval_s: float) -> bool:
    """Whether the cells stage's default window and baselines fit, by its own rules."""
    from auto_organotypic import baselines, metadata, region_trace
    hours = [n * interval_s / 3600.0 for n in range(frames)]
    try:
        window = metadata.usable_window(hours, t0=region_trace.DEFAULT_T0)
        baselines.validate_baseline_windows(baselines.DEFAULT_BASELINES,
                                            hours[window.start:window.end])
    except ValueError:  # EmptyWindow is one
        return False
    return True


def _too_short(row: dict[str, Any]) -> dict[str, Any]:
    """Say before a run, not after registration, that cell traces cannot be made.

    The chain still runs such a recording up to the cells stage; the note does
    not block it. Assumes evenly spaced frames; a gap makes the need larger.
    """
    from auto_organotypic import region_trace
    frames, interval = row.get("frames"), row.get("interval_s")
    if not row.get("valid") or not frames or not interval or _traceable(int(frames), float(interval)):
        return row
    low, high = int(frames), int(frames)
    while not _traceable(high, float(interval)):
        low, high = high, high * 2
        if high > 100_000:
            return row
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if _traceable(middle, float(interval)) else (middle, high)
    have, need = (int(frames) - 1) * interval / 3600.0, (high - 1) * interval / 3600.0
    row["issues"] = [*row.get("issues", []),
                     f"{have:.0f} h long: cell traces skip the first {region_trace.DEFAULT_T0:g} h "
                     f"while the slice settles and need about {need:.0f} h in all, so this run "
                     "will stop at the cells stage"]
    return row


def _local_row(root: Path, path: Path, labels: Mapping[str, str],
               roles: Mapping[str, Any] | None = None,
               interval_s: float | None = None) -> dict[str, Any]:
    from auto_organotypic import local_inputs
    try:
        found = local_inputs.inspect_stack(path, roles=roles, interval_s=interval_s)
        if roles is None and not found["roles"].get("signal"):
            found = local_inputs.inspect_stack(path, roles=_guess_roles(found["channels"]),
                                               interval_s=interval_s)
    except (OSError, ValueError) as exc:
        return {"key": _key(root, path), "path": str(path), "stem": path.stem,
                "name": labels.get("name", ""), "labels": dict(labels), "valid": False,
                "issues": [str(exc)], "channels": [], "roles": {}, "frames": None,
                "interval_s": None, "selected": False}
    return {"key": _key(root, path), "path": str(path), "stem": path.stem,
            "name": labels.get("name", ""), "labels": dict(labels),
            "frames": found["frames"], "channels": found["channels"],
            "roles": found["roles"], "interval_s": found["interval_s"]["value"],
            "interval_source": found["interval_s"]["source"],
            "valid": found["valid"], "issues": found["issues"], "selected": True}


def _manifest_row(root: Path, one: Any, labels: Mapping[str, str]) -> dict[str, Any]:
    path = Path(one.path)
    return {"key": _key(root, path), "path": str(path), "stem": path.stem,
            "name": labels.get("name", ""), "labels": dict(labels), "frames": one.frames,
            "channels": [{"index": c.index, "name": c.name} for c in one.channels],
            "roles": {}, "interval_s": one.interval_s.value,
            "interval_source": one.interval_s.source, "valid": True, "issues": [],
            "selected": True, "well": one.field.name}


def _labels(root: Path) -> dict[str, dict[str, str]]:
    try:
        return recording_labels.read(root)
    except (OSError, ValueError):
        return {}


def inspect(paths: Iterable[str]) -> dict[str, Any]:
    """Describe what was dropped: one folder, or TIFF stacks from one folder."""
    given = [Path(str(path).strip().strip('"')).expanduser() for path in paths or ()
             if str(path).strip()]
    if not given:
        raise ValueError("Drop or choose a TIFF stack or a folder of them")
    resolved = []
    for path in given:
        if not path.exists():
            raise ValueError(f"{path} does not exist on this computer")
        resolved.append(path.resolve())
    folders = [path for path in resolved if path.is_dir()]
    files = [path for path in resolved if not path.is_dir()]
    if folders and (len(folders) > 1 or files):
        raise ValueError("Drop one folder, or TIFF stacks from one folder, at a time")
    if folders:
        root, chosen = folders[0], None
    else:
        parents = {path.parent for path in files}
        if len(parents) > 1:
            raise ValueError("Drop TIFF stacks from one folder at a time")
        wrong = [path.name for path in files if path.suffix.lower() not in TIFFS]
        if wrong:
            raise ValueError(f"Only TIFF stacks can be analysed: {', '.join(wrong)}")
        root, chosen = parents.pop(), set(files)
    labels = _labels(root)

    found = _manifest_recordings(root)
    planes = sum(one.is_per_image for one in found or ())
    if planes:
        # The chain refuses these too (sources.refuse_per_image); saying so
        # here costs one manifest read instead of a row per image.
        raise ValueError(
            f"This folder holds {planes:,} single images, one per well, channel "
            f"and time point, not time stacks. Export it again as a "
            f"time-and-channel stack (PyIncucyte layout 'time_channel_stack') "
            f"and drop that folder.")
    if found is not None and (chosen is None or chosen <= {Path(one.path).resolve() for one in found}):
        rows = [_manifest_row(root, one, labels.get(_key(root, Path(one.path)), {}))
                for one in found]
        mode = "manifest"
    else:
        stacks = sorted(chosen) if chosen is not None else sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in TIFFS and not path.name.startswith((".", "~")))
        if not stacks:
            raise ValueError("No TIFF stacks sit directly in this folder. Drop the folder that holds them, or the stacks themselves.")
        if len(stacks) > MAX_RECORDINGS:
            raise ValueError(f"Choose at most {MAX_RECORDINGS} stacks at a time")
        rows = [_local_row(root, path, labels.get(_key(root, path), {})) for path in stacks]
        mode = "local"
    rows = [_too_short(row) for row in rows]
    if chosen is not None:
        for row in rows:
            row["selected"] = row["selected"] and Path(row["path"]) in chosen
    return {"folder": str(root), "mode": mode, "recordings": rows,
            "labels_file": str(root / recording_labels.FILE_NAME),
            "results": results(root)}


def save_labels(folder: str, labels: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Save labels keyed by relative path; empty text clears a field."""
    root = Path(folder).resolve()
    if not isinstance(labels, Mapping) or not labels:
        raise ValueError("Give at least one recording's labels")
    saved = recording_labels.save(root, {root / key: values for key, values in labels.items()})
    return {"folder": str(root), "labels": saved}


def run_params(folder: str, mode: str, recordings: list[Mapping[str, Any]]) -> dict[str, Any]:
    """The auto_microglia keywords for the chosen recordings."""
    root = Path(folder).resolve()
    chosen = [row for row in recordings if row.get("selected")]
    if not chosen:
        raise ValueError("Choose at least one recording to analyse")
    paths = [(root / str(row["key"])).resolve() for row in chosen]
    if any(root not in path.parents for path in paths):
        raise ValueError("A chosen recording is outside the dropped folder")
    if mode == "manifest":
        everything = {Path(one.path).resolve() for one in _manifest_recordings(root) or ()}
        params: dict[str, Any] = {"folder": str(root)}
        if set(paths) != everything:
            params["only"] = [path.name for path in paths]
        return params
    if mode != "local":
        raise ValueError("Unknown recording source")
    from auto_organotypic import local_inputs
    inputs = []
    for row, path in zip(chosen, paths):
        item: dict[str, Any] = {"path": str(path), "roles": row.get("roles") or None}
        if row.get("name"):
            item["name"] = row["name"]
        if row.get("interval_s") not in (None, "") and row.get("interval_source") in (None, "confirmed by user", "not stated"):
            item["interval_s"] = {"value": float(row["interval_s"]), "source": "confirmed by user"}
        inputs.append(item)
    home = root / HOME
    local_inputs.write_manifest(home, inputs)
    return {"folder": str(home)}


def results(folder) -> list[dict[str, Any]]:
    """Finished automated runs under a recordings folder, newest first."""
    root = Path(folder).resolve()
    found = []
    for base in (root, root / HOME):
        runs = base / "auto_microglia"
        if not runs.is_dir():
            continue
        for manifest in runs.glob("*/manifest.json"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            measured = (data.get("outputs") or {}).get("tracked_measurement") or {}
            found.append({
                "run": str(manifest.parent), "label": manifest.parent.name,
                "modified": manifest.stat().st_mtime,
                "recordings": [{"stem": stem, "measurement": row.get("folder")}
                               for stem, row in sorted(measured.items())
                               if isinstance(row, Mapping) and row.get("folder")]})
    return sorted(found, key=lambda row: row["modified"], reverse=True)[:20]
