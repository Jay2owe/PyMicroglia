"""Discard the first two frames before Fiji extracts the red channel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile

from prepare_full_vid95 import EXPERIMENT, STEMS


SOURCE = (EXPERIMENT / "AI_Exports/registered_unmixed_2026-07-23"
          / "raw_registered_two_channel_stacks_30min")
OUTPUT = EXPERIMENT / "AI_Exports/vid95_registered_trimmed_before_extraction_20260921"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for stem in STEMS:
        source = SOURCE / f"VID{stem}_green-red_timestack_registered_cropped_raw_both_channels.tif"
        target = OUTPUT / f"{stem}_registered_trimmed_channels.tif"
        with tifffile.TiffFile(source) as tif:
            if tif.series[0].shape[0] != 101 or tif.series[0].shape[1] != 2:
                raise ValueError(f"{stem}: expected 101-frame, two-channel source")
            movie = tif.series[0].asarray()
        trimmed = movie[2:]
        if target.exists():
            if not np.array_equal(tifffile.imread(target), trimmed):
                raise ValueError(f"{target}: existing output differs")
        else:
            temporary = target.with_name(target.stem + ".partial.tif")
            tifffile.imwrite(temporary, trimmed, imagej=True, compression="zlib",
                             metadata={"axes": "TCYX", "finterval": 1800.0,
                                       "tunit": "sec", "unit": "pixel"})
            temporary.replace(target)
        records.append({"stem": stem, "source": str(source), "source_sha256": sha256(source),
                        "output": str(target), "output_sha256": sha256(target),
                        "excluded_source_frames_one_based": [1, 2],
                        "retained_source_frames_one_based": [3, 101],
                        "frames": 99, "channels": 2,
                        "height": trimmed.shape[2], "width": trimmed.shape[3]})
        print(f"{stem}: 99 frames, 2 channels", flush=True)
    (OUTPUT / "manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
