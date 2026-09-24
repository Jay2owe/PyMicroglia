"""Prepare Motion's six pinned inputs, then run its frozen tracking rules.

The original registered photons are saved separately for measurement; only
the scaled, mask-zeroed copy enters Motion. A manifest pins both kinds of
input, and the tracker returns file paths, never arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .. import tracking as _tracking
from ..tracking import prepare as _prepare
from ..tracking import TRACKER_TARGET as TARGET

__all__ = ["FOLDER", "INPUTS_NAME", "TARGET", "status", "write", "track"]

#: Under the run folder, so a handoff belongs to the run that produced it.
FOLDER = "motion"
INPUTS_NAME = "motion_inputs.json"

# ``TARGET`` is :data:`pymicroglia.tracking.TRACKER_TARGET`, re-exported: the
# dotted name of the packaged tracker lives on the seam, and this module only
# writes what that seam's tracker reads.


def status() -> tuple[str, str]:
    """``("ready", "")`` once the tracker resolves, else why not.

    The seam's own answer, so a run record and ``describe track`` say the same
    thing about the same missing module.
    """
    return _tracking.status()


def write(recordings, masks: Mapping[str, Any], folder, notes=None, *,
          dataset: str = "", hashes: bool = True) -> dict[str, Any]:
    """Build and pin Motion inputs for every per-frame mask.

    ``hashes=False`` writes ``null`` where each SHA-256 goes. Motion will not
    accept that, and the option exists because a full hash is a full read of
    every stack: worth knowing you are choosing to defer it rather than
    discovering the cost inside a run.
    """
    target = Path(folder) / FOLDER
    target.mkdir(parents=True, exist_ok=True)

    pinned: dict[str, Any] = {}
    prepared: dict[str, Any] = {}
    interval_min = 0.0
    for row in recordings:
        path = Path(str(row["path"]))
        entry = {"source_registered": _pin(path, hashes)}
        mask = masks.get(str(path)) if masks else None
        if mask is not None:
            entry["cell_mask"] = _pin(Path(mask["outputs"]["mask_cells"]),
                                      hashes)
            still = mask.get("still") or {}
            if still.get("mask_cells_still"):
                entry["cell_mask_still"] = _pin(
                    Path(still["mask_cells_still"]), hashes)
        seconds = row.get("interval_s") or {}
        this_interval = (float(seconds["value"]) / 60.0
                         if isinstance(seconds, Mapping) and seconds.get("value")
                         else 0.0)
        if this_interval:
            interval_min = this_interval
        if mask is not None:
            if this_interval <= 0:
                raise ValueError(f"{path.name}: Motion needs a frame interval")
            from .auto_microglia import _Registered

            made = _prepare.build(
                path.stem, _Registered(path, row).dluc,
                mask["outputs"]["mask_cells"], target / path.stem,
                frame_interval_min=this_interval,
                dataset=dataset or path.stem)
            prepared[path.stem] = made
            prepared[path.stem]["um_per_px"] = _Registered(path, row).um_per_px
            entry["measurement_raw"] = _pin(Path(made["measurement_raw"]), hashes)
            for role, pin in made["pins"].items():
                relative = (pin.get("relative_to_registered_input_dir")
                            or pin.get("relative_to_motion_input_dir"))
                entry[role] = _pin(target / path.stem / relative, hashes)
        else:
            entry["registered_raw"] = _pin(path, hashes)
        pinned[path.stem] = entry

    payload = {
        "dataset": dataset,
        "stems": sorted(pinned),
        "frame_interval_min": interval_min or None,
        "input_space": "registered",
        "registered_input_dir": (str(Path(recordings[0]["path"]).parent)
                                 if recordings else ""),
        "pinned_files": pinned,
        "prepared": prepared,
        "still_missing": ([] if len(prepared) == len(pinned) else
                          ["lag_float", "neutral_tracks", "trail_labels",
                           "trail_ages", "motion_composite"]),
        # What the tracker is expected to write back, per stem, in the words
        # of ``pymicroglia.tracking.contract``: the five files and the
        # decision-table root, relative to the tracker's run folder except for
        # ``raw``, which is the pinned registered stack above. Written so the
        # tracker and the measure step agree by file rather than by convention.
        "expects": {stem: _tracking.expected_files(
                        stem, entry["registered_raw"]["path"])
                    for stem, entry in pinned.items()},
        "note": ("U-Net masks and registered photons prepared by PyMicroglia. "
                 "Motion's frozen accepted stages run on the six pinned stacks; "
                 "the source photons remain separate for measurement."),
    }
    written = target / INPUTS_NAME
    written.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    state, reason = status()
    if notes is not None and prepared:
        notes.note("motion", chosen="automatic frozen Motion tracker",
                   confidence="medium", changes_result=True,
                   why="The current native-frame Motion identities are provisional; "
                       "a completed run still needs identity review before its "
                       "measurements are treated as accepted.",
                   remedy="Review the full-field Motion TIFF and identity tables.",
                   evidence=[str(written)])
    return {"status": state, "reason": reason, "stems": len(pinned),
            "outputs": {"motion_inputs": str(written)}}


def track(handoff: Mapping[str, Any], folder, entry: dict[str, Any]) -> dict[str, Any]:
    """Run the packaged tracker once per stem and return its file records."""
    from ..run import ActionPending

    outputs = dict(handoff["outputs"])
    entry["target"] = TARGET
    manifest = json.loads(Path(outputs["motion_inputs"]).read_text(encoding="utf-8"))
    if manifest["still_missing"]:
        raise ValueError("motion=True needs a per-frame U-Net mask for every "
                         "recording; pass cell_masks=True")
    results = {}
    try:
        for stem in manifest["stems"]:
            result = _tracking.run(outputs["motion_inputs"], folder,
                                   tracker_options={"stem": stem})
            results[stem] = result.as_dict()
    except ActionPending as exc:
        entry["status"] = "pending"
        entry["reason"] = str(exc)
        return outputs
    entry["status"] = "ok"
    entry.pop("reason", None)
    outputs["tracking"] = results
    return outputs


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
