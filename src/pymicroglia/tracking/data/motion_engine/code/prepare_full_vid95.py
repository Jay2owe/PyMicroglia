"""Prepare 99-frame VID95 red-signal stacks for a new Motion batch.

The Fiji extraction macro has already produced the red-minus-reference stacks.
Reuse established 101-frame registrations where present; register remaining
stacks with the validated translation-only phase-correlation protocol.
Source frames 1 and 2 are excluded before any new registration or tracking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import tifffile


PROJECT = Path(__file__).resolve().parents[1]
MICROGLIA = PROJECT.parent
EXPERIMENT = MICROGLIA / "Experiments/Tmem119-CreERT2/Tests/CD68-DIO-mCherry with HCQ"
EXTRACTED = EXPERIMENT / "signal_extracted"
ESTABLISHED = MICROGLIA / "Longitudinal Cell Tracker/registered_inputs/r02_grossfix/out"
REGISTRATION_SCRIPT = MICROGLIA / "Protocols/Analysis/microglia_phase_correlation_registration.py"
EXTRACTION_MACRO = MICROGLIA / "Protocols/Analysis/microglia_signal_extraction_analysis.ijm"
STEMS = [f"95_{row}{column}" for row, columns in (("A", range(1, 7)),
                                                ("B", range(1, 7)),
                                                ("C", range(1, 3))) for column in columns]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registration_module():
    spec = importlib.util.spec_from_file_location("vid95_phase_registration", REGISTRATION_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_stack(path: Path, movie: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".partial.tif")
    tifffile.imwrite(temporary, movie, imagej=True,
                     metadata={"axes": "TYX", "finterval": 1800.0, "tunit": "sec"})
    temporary.replace(path)


def prepare(stem: str, output: Path, registration) -> dict:
    source = ESTABLISHED / f"{stem}.tif"
    if source.is_file():
        movie = tifffile.imread(source)
        if movie.ndim != 3 or movie.shape[0] != 101:
            raise ValueError(f"{source}: expected 101 TYX frames")
        result = movie[2:]
        method = "established integer-shift registration r02_grossfix"
        maximum_residual = None
    else:
        source = EXTRACTED / f"{stem}.tif"
        movie = tifffile.imread(source)
        if movie.ndim != 3 or movie.shape[0] != 101:
            raise ValueError(f"{source}: expected 101 TYX frames")
        trimmed = movie[2:, None]
        reference, shifts = registration.estimate_registration(
            trimmed, channel_index=0, downsample=4)
        crop = registration.choose_crop(
            reference, shifts, height=trimmed.shape[2], width=trimmed.shape[3],
            downsample=4, margin_px=128, content_crop=True)
        residuals = registration.validate_registration(
            trimmed, reference, shifts, channel_index=0, downsample=4)
        maximum_residual = float(np.hypot(residuals[:, 0], residuals[:, 1]).max())
        if maximum_residual > 1.0:
            raise ValueError(f"{stem}: registration residual {maximum_residual:.3f} px > 1 px")
        x0, y0, x1, y1 = crop
        result = np.stack([
            np.clip(registration.translate(frame, *shift[:2])[y0:y1, x0:x1],
                    0, np.iinfo(movie.dtype).max).astype(movie.dtype)
            for frame, shift in zip(movie[2:], shifts)
        ])
        method = "translation-only phase correlation on extracted red signal"
    if result.shape[0] != 99:
        raise AssertionError(f"{stem}: expected 99 retained frames")
    target = output / f"{stem}.tif"
    if target.exists():
        if not np.array_equal(tifffile.imread(target), result):
            raise ValueError(f"{target}: existing output differs from this preparation")
    else:
        write_stack(target, result)
    return {"stem": stem, "source": str(source), "source_sha256": sha256(source),
            "output": str(target), "output_sha256": sha256(target),
            "source_frames_excluded_one_based": [1, 2],
            "retained_source_frames_one_based": [3, 101],
            "output_frames": 99, "height": result.shape[1], "width": result.shape[2],
            "registration_method": method, "registration_residual_max_px": maximum_residual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "registered inputs/full_vid95_trimmed_20260921")
    parser.add_argument("--stems", nargs="+", default=STEMS, choices=STEMS)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registration = registration_module()
    records = []
    for stem in args.stems:
        record = prepare(stem, output, registration)
        records.append(record)
        print(f"{stem}: {record['output_frames']} registered frames", flush=True)
    manifest = {"extraction_macro": str(EXTRACTION_MACRO),
                "extraction_macro_sha256": sha256(EXTRACTION_MACRO),
                "reference_channel": 1, "signal_channel": 2,
                "rolling_radius": 10, "saturation": 0.35,
                "normalization_changes_intensity": True,
                "records": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
