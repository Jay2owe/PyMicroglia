"""Pin first-two-frames-excluded VID95 movies and motion evidence for Motion."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
MOTION_ROOT = ROOT / "registered inputs"
NEW = {"95_A6", "95_B6", "95_C1", "95_C2"}
STEMS = [f"95_{row}{column}" for row, columns in (("A", range(1, 7)),
                                                ("B", range(1, 7)),
                                                ("C", range(1, 3))) for column in columns]
MOTION = {"lag_float": "out/{stem}_lag01_float.tif",
          "neutral_tracks": "out/{stem}_neutral_tracks.tif",
          "trail_labels": "out/{stem}_trail_labels.tif",
          "trail_ages": "out/{stem}_trail_ages.tif",
          "motion_composite": "qc/{stem}_motion_evidence.tif"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("clean_lookback", "fresh", "c1", "red"), default="fresh")
    args = parser.parse_args()
    folder = ("full_vid95_clean_lookback_20260922" if args.variant == "clean_lookback" else
              "full_vid95_fresh_20260921" if args.variant == "fresh" else
              "full_vid95_c1_trimmed_20260921" if args.variant == "c1"
              else "full_vid95_trimmed_20260921")
    registered = ROOT / "registered inputs" / folder
    output = ROOT / "registered inputs" / f"{folder}_config.json"
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config.update({"dataset": "VID95 full source; source frames 1-2 excluded before extraction; two duplicate clean lookback frames for Motion" if args.variant == "clean_lookback" else "VID95 full source; source frames 1-2 excluded",
                   "stems": STEMS, "input_space": "registered",
                   "source_frame_offset": 2 if args.variant == "clean_lookback" else 0,
                   "registered_input_dir": str(registered),
                   "motion_input_dir": str(MOTION_ROOT)})
    pins = {}
    for stem in STEMS:
        raw = registered / f"{stem}.tif"
        if args.variant not in ("fresh", "clean_lookback") and stem not in NEW:
            previous = ROOT / "registered inputs" / f"{stem}.tif"
            if not np.array_equal(tifffile.imread(raw), tifffile.imread(previous)):
                raise ValueError(f"{stem}: prepared movie differs from established 99-frame input")
        with tifffile.TiffFile(raw) as tif:
            shape = tif.series[0].shape
        expected_frames = 101 if args.variant == "clean_lookback" else 99
        if len(shape) != 3 or shape[0] != expected_frames:
            raise ValueError(f"{stem}: expected {expected_frames} TYX frames, found {shape}")
        rows = {"registered_raw": {
            "relative_to_registered_input_dir": raw.name, "sha256": sha256(raw)}}
        run = (f"full_vid95_clean_lookback_20260922_{stem}" if args.variant == "clean_lookback" else
               ("full_vid95_fresh_20260921" if stem == "95_A1" else
                "full_vid95_fresh_b_20260921" if stem.startswith("95_B") else
                f"full_vid95_fresh_{stem[3].lower()}_retry_20260921")
               if args.variant == "fresh" else
               "full_vid95_trimmed_20260921" if stem == "95_A6" else
               folder if stem in NEW else "r02")
        for kind, template in MOTION.items():
            relative = f"motion evidence/{run}/{template.format(stem=stem)}"
            path = MOTION_ROOT / relative
            if not path.exists():
                raise FileNotFoundError(path)
            with tifffile.TiffFile(path) as tif:
                evidence_shape = tif.series[0].shape
            if evidence_shape[0] != expected_frames - 1 or evidence_shape[-2:] != shape[-2:]:
                raise ValueError(f"{path}: evidence shape {evidence_shape} disagrees with {shape}")
            rows[kind] = {"relative_to_motion_input_dir": relative,
                          "sha256": sha256(path)}
        pins[stem] = rows
        print(stem, flush=True)
    config["pinned_files"] = pins
    # Fresh inputs have already lost the noisy source frames. Only clean-lookback
    # inputs still carry a removable prefix; every evidence adapter shares it.
    accepted = config.get("accepted_postprocessing", {})
    consensus = accepted.get("field_wide_owner_consensus", {})
    if "motion_frame_offset" in consensus:
        consensus["motion_frame_offset"] = config["source_frame_offset"]
    invasion = accepted.get("field_wide_bracketed_established_owner_invasion", {}).get("parameters", {})
    if "raw_frame_offset" in invasion:
        invasion["raw_frame_offset"] = config["source_frame_offset"]
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
