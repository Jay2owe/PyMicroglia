"""Freeze the accepted Motion engine without changing its numerical code.

Build-time only: the resulting engine folder is packaged with PyMicroglia. Runtime
never needs the sibling checkouts. Run from the PyMicroglia repository root.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pymicroglia.tracking import shipped_engines

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
MOTION = PROJECT / "Motion"
SOURCE = MOTION / "code"
OUTPUT = ROOT / "src/pymicroglia/tracking/data/motion_engine"


def build() -> Path:
    files = [MOTION / "config.json"]
    files += sorted(SOURCE.glob("*.py"))
    files += sorted(SOURCE.glob("*.json"))
    files += sorted((SOURCE / "accepted_history_ops").rglob("*.py"))
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("Motion engine source or configuration is missing")
    entries = {}
    for path in files:
        relative = path.relative_to(MOTION).as_posix()
        entries[relative] = path.read_bytes()
    # Preparation replaces these fields for every recording. Keep the accepted
    # tracking settings, but do not ship the lab recording used as the template.
    config = json.loads(entries["config.json"])
    config.update(dataset="Tracked recording", stems=[],
                  registered_input_dir=".", motion_input_dir=".",
                  pinned_files={}, source_frame_offset=0)
    # Acquisition timing is supplied by preparation, never copied from VID95.
    config.pop("frame_interval_min", None)
    entries["config.json"] = (json.dumps(config, indent=2) + "\n").encode()
    provenance = {
        "source": "Motion accepted run_base engine snapshot",
        "accepted_engine": "2026-10-12-native-speed59-recording-audit192",
        "scope": "Native run_base with accepted execution contexts; completed-feedback study replay is not the native engine",
        "files_sha256": {name: hashlib.sha256(data).hexdigest()
                         for name, data in entries.items()},
    }
    entries["source_hashes.json"] = (json.dumps(provenance, indent=2) + "\n").encode()
    return shipped_engines.write(OUTPUT, entries)


if __name__ == "__main__":
    print(build())
