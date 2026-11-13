"""Pin the 14 trimmed registered movies and their Motion cell labels for grids."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTERED = ROOT / "registered inputs/full_vid95_fresh_20260921"
OUTPUT = ROOT / "registered inputs/full_vid95_fresh_20260921_analysis.json"
NEW = {"95_A6", "95_B6", "95_C1", "95_C2"}
STEMS = [f"95_{row}{column}" for row, columns in (("A", range(1, 7)),
                                                ("B", range(1, 7)),
                                                ("C", range(1, 3))) for column in columns]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    movies = []
    for stem in STEMS:
        run = f"full_vid95_clean_lookback_20260922_{stem}"
        labels = ROOT / "m22_accepted_history" / run / "out" / f"{stem}.tif"
        raw = REGISTERED / f"{stem}.tif"
        if not labels.is_file():
            raise FileNotFoundError(labels)
        if not raw.is_file():
            raise FileNotFoundError(raw)
        movies.append({"stem": stem, "subject": stem,
                       "notes": "Source frames 1-2 excluded before Fiji extraction; first analysed frame is source frame 3. Signal intensities were normalized by the Fiji macro and are processed units.",
                       "source_frame_offset": 0,
                       "labels": str(labels), "raw": str(raw),
                       "sha256": {"labels": sha256(labels), "raw": sha256(raw)}})
    config = {"dataset": "VID95 CD68-DIO-mCherry +/- HCQ; 99 retained frames per well",
              "frame_interval_min": 30, "microns_per_pixel": None,
              "output_root": str(ROOT / "analysis outputs"),
              "verify_hashes": True, "enabled_modules": ["intensity"],
              "movies": movies, "modules": {},
              "plots": [{"figure": "all-cell-trace-grid",
                         "for_each": {"stem": STEMS},
                         "as": "all-cell-trace-grid/{stem}"}]}
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
