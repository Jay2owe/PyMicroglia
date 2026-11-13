"""Apply the experiment's validated structural-channel shifts to red signal.

Source frames 1 and 2 are excluded before the new registrations. Established
integer-shift registered movies are reused byte-for-byte for the other wells.
The saved C1 shift table was independently validated on all 101 source frames.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile

import prepare_full_vid95 as base


PROTOCOL = (base.EXPERIMENT / "AI_Exports/registered_unmixed_2026-07-23")
SHIFT_TABLE = PROTOCOL / "registration_shifts.csv"
VALIDATION = PROTOCOL / "registration_validation_all_frames.csv"
NEW = {"95_B6", "95_C1", "95_C2"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def prepare(stem: str, output: Path, registration) -> dict:
    if stem not in NEW:
        return base.prepare(stem, output, registration)
    source = base.EXTRACTED / f"{stem}.tif"
    movie = tifffile.imread(source)
    if movie.ndim != 3 or len(movie) != 101:
        raise ValueError(f"{source}: expected 101 TYX frames")
    original_name = f"VID{stem}_green-red_timestack.tif"
    matching = [row for row in rows(SHIFT_TABLE) if row["file"] == original_name]
    if [int(row["frame"]) for row in matching] != list(range(1, 102)):
        raise ValueError(f"{stem}: structural registration shifts are incomplete")
    checked = [row for row in rows(VALIDATION) if row["file"] == original_name]
    if len(checked) != 1 or checked[0]["pass_all_frames_below_1px"].lower() != "true":
        raise ValueError(f"{stem}: C1 registration did not pass the all-frame validation")
    maximum_residual = float(checked[0]["residual_max_px"])
    if maximum_residual >= 1.0:
        raise ValueError(f"{stem}: C1 residual is {maximum_residual:.3f} px")
    shifts = np.asarray([[float(row["shift_y_px"]), float(row["shift_x_px"])]
                         for row in matching[2:]])
    height, width = movie.shape[1:]
    crop = registration.choose_crop(
        np.zeros((height // 4, width // 4), np.float32), shifts,
        height=height, width=width, downsample=4, margin_px=128,
        content_crop=False)
    x0, y0, x1, y1 = crop
    result = np.stack([
        np.clip(registration.translate(frame, *shift)[y0:y1, x0:x1],
                0, np.iinfo(movie.dtype).max).astype(movie.dtype)
        for frame, shift in zip(movie[2:], shifts)
    ])
    target = output / f"{stem}.tif"
    if target.exists():
        if not np.array_equal(tifffile.imread(target), result):
            raise ValueError(f"{target}: existing registered pixels differ")
    else:
        base.write_stack(target, result)
    return {"stem": stem, "source": str(source), "source_sha256": base.sha256(source),
            "output": str(target), "output_sha256": base.sha256(target),
            "source_frames_excluded_one_based": [1, 2],
            "retained_source_frames_one_based": [3, 101],
            "output_frames": 99, "height": result.shape[1], "width": result.shape[2],
            "registration_method": "saved validated C1 phase-correlation shifts",
            "registration_shift_table_sha256": base.sha256(SHIFT_TABLE),
            "registration_validation_sha256": base.sha256(VALIDATION),
            "registration_residual_max_px": maximum_residual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=base.PROJECT / "registered inputs/full_vid95_c1_trimmed_20260921")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registration = base.registration_module()
    records = []
    for stem in base.STEMS:
        record = prepare(stem, output, registration)
        records.append(record)
        print(f"{stem}: 99 registered frames", flush=True)
    manifest = {"extraction_macro": str(base.EXTRACTION_MACRO),
                "extraction_macro_sha256": base.sha256(base.EXTRACTION_MACRO),
                "reference_channel": 1, "signal_channel": 2,
                "rolling_radius": 10, "saturation": 0.35,
                "normalization_changes_intensity": True,
                "shift_table": str(SHIFT_TABLE), "validation": str(VALIDATION),
                "records": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in records for key in row}))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
