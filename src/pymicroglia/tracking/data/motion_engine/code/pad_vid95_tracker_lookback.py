"""Give the established tracker two clean lookback frames before 99 retained frames.

The Fiji signal movie contains source frames 3-101 only. The accepted Motion
path always drops its first two input frames, so duplicate frame 3 twice as
non-output context. Neither duplicate is analysed or delivered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "registered inputs/full_vid95_fresh_20260921"
TARGET = ROOT / "registered inputs/full_vid95_clean_lookback_20260922"
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
    TARGET.mkdir(exist_ok=False)
    rows = []
    for stem in STEMS:
        source = SOURCE / f"{stem}.tif"
        signal = tifffile.imread(source)
        if signal.ndim != 3 or len(signal) != 99:
            raise ValueError(f"{stem}: expected 99 extracted frames, got {signal.shape}")
        padded = np.concatenate((signal[:1], signal[:1], signal), axis=0)
        target = TARGET / source.name
        tifffile.imwrite(target, padded, imagej=True,
                         metadata={"axes": "TYX", "finterval": 1800})
        with tifffile.TiffFile(target) as tif:
            if tif.series[0].shape != (101, *signal.shape[1:]):
                raise AssertionError(f"{stem}: tracker lookback stack shape wrong")
        rows.append({"stem": stem, "signal_frames": 99,
                     "tracker_input_frames": 101,
                     "lookback_frames": 2,
                     "source_first_output_frame_one_based": 3,
                     "source_last_output_frame_one_based": 101,
                     "signal": str(source), "signal_sha256": sha256(source),
                     "tracker_input": str(target),
                     "tracker_input_sha256": sha256(target)})
        print(stem, flush=True)
    (TARGET / "manifest.json").write_text(json.dumps({
        "purpose": "two duplicate, clean, non-output lookback frames for the established Motion source-frame offset",
        "invariant": "tracker_input[2:] is pixel-identical to the 99-frame Fiji-extracted signal",
        "movies": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
