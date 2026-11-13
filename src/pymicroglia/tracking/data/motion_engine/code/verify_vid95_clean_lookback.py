"""Check the new clean-lookback boundary without changing accepted Motion runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SIGNAL = ROOT / "registered inputs/full_vid95_fresh_20260921"
LOOKBACK = ROOT / "registered inputs/full_vid95_clean_lookback_20260922"
STEMS = [f"95_{row}{col}" for row, cols in
         (("A", range(1, 7)), ("B", range(1, 7)), ("C", range(1, 3)))
         for col in cols]


def main() -> None:
    for stem in STEMS:
        clean = tifffile.imread(SIGNAL / f"{stem}.tif")
        padded = tifffile.imread(LOOKBACK / f"{stem}.tif")
        if len(clean) != 99 or len(padded) != 101 or clean.shape[1:] != padded.shape[1:]:
            raise ValueError(f"{stem}: clean and lookback stack lengths or fields differ")
        if not (np.array_equal(padded[0], clean[0]) and
                np.array_equal(padded[1], clean[0]) and
                np.array_equal(padded[2:], clean)):
            raise ValueError(f"{stem}: Motion output frame mapping is not pixel-identical")
        evidence = (ROOT / "registered inputs/motion evidence" /
                    f"full_vid95_clean_lookback_20260922_{stem}" / "out" /
                    f"{stem}_lag01_float.tif")
        with tifffile.TiffFile(evidence) as tif:
            if tif.series[0].shape != (100, *clean.shape[1:]):
                raise ValueError(f"{stem}: motion evidence does not cover 100 transitions")
        print(f"{stem}: 99 real frames retained after two clean lookback frames", flush=True)
    prior = ROOT / "reviews/vid95_full_fresh_20260921_outlines/accepted_baseline_hash_check.json"
    status = json.loads(prior.read_text(encoding="utf-8"))
    if not status:
        raise ValueError("protected accepted-output hash check is empty")
    print("previous accepted-output hash check retained", flush=True)


if __name__ == "__main__":
    main()
