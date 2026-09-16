"""What the Motion project reads, written where it can read it.

Not a pipeline and not a stage of one: a handoff. The Motion project tracks
microglial identity through movement, merges, splits and temporary invisibility,
and it is a separate piece of work that is not finished. This module is the
seam, written now so that porting Motion is a matter of resolving one dotted
name rather than of agreeing an interface afterwards.

Motion pins every input it reads by a relative path and a SHA-256 and checks
both before it analyses anything, which is a good rule and the reason this
writes a mapping rather than a folder somebody has to describe by hand. Its
vocabulary -- ``dataset``, ``stems``, ``input_space``, ``pinned_files`` -- is
Motion's own ``config.json``, copied deliberately so that what is written here
drops into that file instead of needing translation on the way in.

Two things are deliberately **not** written.

The registered stacks are pinned where they already are rather than copied.
Copying ten gigabytes per well to give it a different name is not a handoff.

The motion-evidence stacks Motion also pins -- ``lag_float``,
``neutral_tracks``, ``trail_labels``, ``trail_ages``, ``motion_composite`` --
are its own first stage's output, computed from the registered stack named here.
Writing empty entries for them would make this file look complete when the work
it describes has not been done.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

__all__ = ["FOLDER", "INPUTS_NAME", "TARGET", "status", "write"]

#: Under the run folder, so a handoff belongs to the run that produced it.
FOLDER = "motion"
INPUTS_NAME = "motion_inputs.json"

#: The dotted name that will one day resolve. Until it does, :func:`status`
#: reports it pending the way Auto-Organotypic reports a missing instrument
#: client -- a line at the top of a run rather than a failure in the middle.
TARGET = "motion.pipeline:run"


def status() -> tuple[str, str]:
    """``("ready", "")`` once the Motion project is importable, else why not."""
    import importlib.util

    module = TARGET.partition(":")[0]
    try:
        if importlib.util.find_spec(module) is None:
            return "pending", f"{module} is not installed."
    except (ImportError, ValueError):
        return "pending", f"{module} is not installed."
    return "ready", ""


def write(recordings, masks: Mapping[str, Any], folder, notes=None, *,
          dataset: str = "", hashes: bool = True) -> dict[str, Any]:
    """Pin the registered stacks and their masks, and say what is still missing.

    ``hashes=False`` writes ``null`` where each SHA-256 goes. Motion will not
    accept that, and the option exists because a full hash is a full read of
    every stack: worth knowing you are choosing to defer it rather than
    discovering the cost inside a run.
    """
    target = Path(folder) / FOLDER
    target.mkdir(parents=True, exist_ok=True)

    pinned: dict[str, Any] = {}
    interval_min = 0.0
    for row in recordings:
        path = Path(str(row["path"]))
        entry = {"registered_raw": _pin(path, hashes)}
        mask = masks.get(str(path)) if masks else None
        if mask is not None:
            entry["cell_mask"] = _pin(Path(mask["outputs"]["mask_cells"]),
                                      hashes)
            still = mask.get("still") or {}
            if still.get("mask_cells_still"):
                entry["cell_mask_still"] = _pin(
                    Path(still["mask_cells_still"]), hashes)
        pinned[path.stem] = entry
        seconds = row.get("interval_s") or {}
        if isinstance(seconds, Mapping) and seconds.get("value"):
            interval_min = float(seconds["value"]) / 60.0

    payload = {
        "dataset": dataset,
        "stems": sorted(pinned),
        "frame_interval_min": interval_min or None,
        "input_space": "registered",
        "registered_input_dir": (str(Path(recordings[0]["path"]).parent)
                                 if recordings else ""),
        "pinned_files": pinned,
        "still_missing": ["lag_float", "neutral_tracks", "trail_labels",
                          "trail_ages", "motion_composite"],
        "note": ("Written by PyMicroglia's auto_microglia pipeline. The stacks "
                 "under still_missing are Motion's own first stage, computed "
                 "from the registered stack pinned here."),
    }
    written = target / INPUTS_NAME
    written.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    state, reason = status()
    if notes is not None and state == "pending":
        notes.note("motion", chosen="handed off, not run", confidence="medium",
                   changes_result=False,
                   why=f"{reason} The registered stacks and their masks are "
                       f"written and pinned; nothing has linked a region in "
                       f"one frame to a region in the next, so no cell here "
                       f"has an identity that outlives a frame.",
                   remedy="Install the Motion project and re-run: the stage "
                          "resolves and this file is what it reads.",
                   evidence=[str(written)])
    return {"status": state, "reason": reason, "stems": len(pinned),
            "outputs": {"motion_inputs": str(written)}}


def _pin(path: Path, hashes: bool) -> dict[str, Any]:
    """One entry of Motion's ``pinned_files``: where it is and what it is."""
    return {"path": str(path), "name": path.name,
            "sha256": _sha256(path) if hashes else None}


def _sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()
