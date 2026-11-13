from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


MOTION_FILES = {
    "lag_float": ("out/{stem}_lag01_float.tif", "relative_to_motion_input_dir"),
    "neutral_tracks": ("out/{stem}_neutral_tracks.tif", "relative_to_motion_input_dir"),
    "trail_labels": ("out/{stem}_trail_labels.tif", "relative_to_motion_input_dir"),
    "trail_ages": ("out/{stem}_trail_ages.tif", "relative_to_motion_input_dir"),
    "motion_composite": ("qc/{stem}_motion_evidence.tif", "relative_to_motion_input_dir"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--full-registered", type=Path, required=True)
    parser.add_argument("--public-registered", type=Path, required=True)
    parser.add_argument("--motion-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument("--source-frame-offset", type=int,
                        help="Leading input frames excluded; otherwise inherit the base config (default zero)")
    args = parser.parse_args()

    config = json.loads(args.base_config.resolve().read_text(encoding="utf-8"))
    offset = config.get("source_frame_offset", 0) if args.source_frame_offset is None else args.source_frame_offset
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("source_frame_offset must be a non-negative integer")
    full_registered = args.full_registered.resolve()
    public_registered = args.public_registered.resolve()
    motion_cache = args.motion_cache.resolve()
    config.update({
        "dataset": config.get("dataset", "Motion recording batch"),
        "source_frame_offset": offset,
        "stems": args.stems,
        "input_space": "registered",
        "registered_input_dir": str(full_registered),
        "motion_input_dir": str(motion_cache),
    })
    accepted = config.get("accepted_postprocessing", {})
    consensus = accepted.get("field_wide_owner_consensus", {})
    if "motion_frame_offset" in consensus:
        consensus["motion_frame_offset"] = offset
    invasion = accepted.get("field_wide_bracketed_established_owner_invasion", {}).get("parameters", {})
    if "raw_frame_offset" in invasion:
        invasion["raw_frame_offset"] = offset
    pins = {}
    for stem in args.stems:
        registered = full_registered / f"{stem}.tif"
        public = public_registered / f"{stem}.tif"
        full_array = tifffile.imread(registered)
        public_array = tifffile.imread(public)
        if len(full_array) <= offset or not np.array_equal(public_array, full_array[offset:]):
            raise AssertionError(
                f"{stem}: public registered input does not match source_frame_offset={offset}")
        rows = {
            "registered_raw": {
                "relative_to_registered_input_dir": f"{stem}.tif",
                "sha256": sha256(registered),
            }
        }
        for name, (template, key) in MOTION_FILES.items():
            relative = template.format(stem=stem)
            path = motion_cache / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            rows[name] = {key: relative, "sha256": sha256(path)}
        pins[stem] = rows
    config["pinned_files"] = pins
    args.output.resolve().write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
