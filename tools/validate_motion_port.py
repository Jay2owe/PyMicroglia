"""One real frozen recording through the packaged Motion engine.

Uses stored prior outputs only as the comparator; never imports old code.
Writes its run under PyMicroglia/tmp, not into the Motion project.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import tifffile

from pymicroglia.tracking import run
from pymicroglia.tracking.prepare import build

ROOT = Path(__file__).resolve().parents[1]
STEM = "20260721_1417"
HANDOFF = ROOT / "development/single_frame_mask_unet/mask_to_motion_handoff"
INPUT = ROOT / "development/single_frame_mask_unet"
OUTPUT = ROOT / "tmp/motion_port_validation_20260922"
OLD = (ROOT.parent / "Motion/m22_accepted_history"
       / f"mcgmask_a000_{STEM}/out" / f"{STEM}.tif")
OLD_RUN = OLD.parents[1] / "run.json"
OUTPUT_PATHS = {
    "labels": f"out/{STEM}.tif",
    "unclaimed": f"out/{STEM}_unclaimed_original_ids.tif",
    "inferred": f"mid/{STEM}_inferred_accepted_history.tif",
    "frame_identities": "out/frame_identities.csv",
    "tracks": "out/tracks.csv",
    "history_summary": "out/accepted_history_summary.json",
    "provenance": f"out/{STEM}_provenance.tif",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare(accepted: Path, source_frame_offset: int) -> dict:
    previous = json.loads(OLD_RUN.read_text(encoding="utf-8"))["outputs"]
    exact = {name: _sha256(accepted / relative) == previous[name]["sha256"]
             for name, relative in OUTPUT_PATHS.items()}
    labels = tifffile.imread(accepted / OUTPUT_PATHS["labels"])
    old_labels = tifffile.imread(OLD)
    return {
        "accepted_run": str(accepted),
        "old_run": str(OLD_RUN.parent),
        "shape_equal": labels.shape == old_labels.shape,
        "exact_outputs": exact,
        "all_seven_outputs_exact": all(exact.values()),
        "new_identities": int(labels.max()),
        "old_identities": int(old_labels.max()),
        "source_frame_offset": source_frame_offset,
    }


def main() -> None:
    if OUTPUT.exists():
        candidates = [
            OUTPUT / "motion_inprocess/engine/m22_accepted_history"
            / f"pymicroglia_{STEM}",
            OUTPUT / "motion/engine/m22_accepted_history" / f"pymicroglia_{STEM}",
        ]
        accepted = next((path for path in candidates if path.is_dir()), None)
        if accepted is None:
            raise FileExistsError(f"incomplete validation run already exists: {OUTPUT}")
        report = _compare(accepted, source_frame_offset=2)
    else:
        pictures = tifffile.imread(
            INPUT / "MCG_mask_annotation/sharp" / STEM / "sharp_w7.tif")
        cells = (INPUT / "MCG_mask_model/runs/stage3_disguise/predictions"
                 / f"{STEM}_sharp_w7_cells.tif")
        input_dir = OUTPUT / "inputs" / STEM
        made = build(STEM, pictures, cells, input_dir,
                     frame_interval_min=166, dataset="Motion port validation")
        handoff = OUTPUT / "motion_inputs.json"
        handoff.write_text(json.dumps({"stems": [STEM],
                                       "prepared": {STEM: made}}), encoding="utf-8")
        result = run(handoff, OUTPUT / "motion", tracker_options={"stem": STEM})
        accepted = Path(result.labels).parents[1]
        report = _compare(accepted, result.source_frame_offset)
    report_path = OUTPUT / "validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_seven_outputs_exact"]:
        raise SystemExit("packaged Motion outputs differ from the accepted run")


if __name__ == "__main__":
    main()
