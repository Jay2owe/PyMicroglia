"""Build the established motion-evidence inputs for newly registered VID95 wells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tifffile


PROJECT = Path(__file__).resolve().parents[1]
LONGITUDINAL_CODE = PROJECT.parent / "Longitudinal Cell Tracker/code"
sys.path.insert(0, str(LONGITUDINAL_CODE))

import prepare_motion_inputs as motion  # noqa: E402
from common import Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered", type=Path,
                        default=PROJECT / "registered inputs/full_vid95_trimmed_20260921")
    parser.add_argument("--run", default="full_vid95_trimmed_20260921")
    parser.add_argument("--stems", nargs="+", default=["95_A6", "95_B6", "95_C1", "95_C2"])
    args = parser.parse_args()
    registered = args.registered.resolve()
    print(f"MOTION_INPUT {registered}", flush=True)
    previous = json.loads((PROJECT / "registered inputs/motion evidence/r02/run.json").read_text())
    print("MOTION_PARAMS_LOADED", flush=True)
    cfg = Config({"input_dir": str(registered), "input_space": "registered",
                  "frame_interval_min": 30, "pixel_size_um": None,
                  "stacks": args.stems})
    motion.ROOT = PROJECT / "registered inputs"
    motion.STAGE = "motion evidence"
    # These inputs have already been registered and cropped to their common
    # valid field. The established lag formula is exact on every retained pixel.
    motion.logratio.log_ratio = lambda stem, config, lag=1: motion.lag_ratio(
        tifffile.imread(config.input_dir / f"{stem}.tif"))
    print(f"MOTION_PREPARE {args.run} {args.stems}", flush=True)
    motion.prepare(args.run, cfg, previous["params"], args.stems)


if __name__ == "__main__":
    main()
