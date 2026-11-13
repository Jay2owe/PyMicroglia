"""Validate Fiji's 99-frame red outputs and copy them into Motion inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
EXPORT = (ROOT.parent / "Experiments/Tmem119-CreERT2/Tests/CD68-DIO-mCherry with HCQ"
          / "AI_Exports/vid95_red_extracted_after_frame2_20260921")
DEST = ROOT / "registered inputs/full_vid95_fresh_20260921"
MACRO = ROOT.parent / "Protocols/Analysis/microglia_signal_extraction_analysis.ijm"
STEMS = [f"95_{row}{col}" for row, cols in
         (("A", range(1, 7)), ("B", range(1, 7)), ("C", range(1, 3)))
         for col in cols]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for stem in STEMS:
        source = EXPORT / f"{stem}.tif"
        target = DEST / source.name
        if not source.is_file():
            raise FileNotFoundError(source)
        with tifffile.TiffFile(source) as tif:
            shape = tif.series[0].shape
            dtype = str(tif.series[0].dtype)
        if len(shape) != 3 or shape[0] != 99 or dtype != "uint16":
            raise ValueError(f"{stem}: expected 99 TYX uint16, got {shape} {dtype}")
        if target.exists():
            if sha256(source) != sha256(target):
                raise ValueError(f"{stem}: existing Motion input differs from Fiji export")
        else:
            shutil.copy2(source, target)
        source_hash = sha256(source)
        if sha256(target) != source_hash:
            raise ValueError(f"{stem}: copy hash mismatch")
        frame_means = [float(np.mean(frame)) for frame in
                       (tifffile.imread(target, key=0),
                        tifffile.imread(target, key=1),
                        tifffile.imread(target, key=98))]
        records.append({"stem": stem, "shape": shape, "dtype": dtype,
                        "sha256": source_hash, "first_second_last_mean": frame_means,
                        "export": str(source), "motion_input": str(target)})
        print(f"{stem} {shape} {source_hash[:12]}", flush=True)
    manifest = {"source_frame_exclusion": [1, 2], "first_source_frame": 3,
                "source_registration": "validated C1 structural-channel registration from AI_Exports/registered_unmixed_2026-07-23",
                "macro": str(MACRO), "macro_sha256": sha256(MACRO),
                "reference_channel": 1, "signal_channel": 2,
                "rolling_radius": 10, "saturation": 0.35,
                "movies": records}
    path = DEST / "manifest.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
