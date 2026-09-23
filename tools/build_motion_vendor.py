"""Freeze the accepted Motion engine without changing its numerical code.

Build-time only: the resulting archive is packaged with PyMicroglia. Runtime
never needs the sibling checkouts. Run from the PyMicroglia repository root.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
MOTION = PROJECT / "Motion"
SOURCE = MOTION / "code"
OUTPUT = ROOT / "src/pymicroglia/tracking/data/motion_engine.zip"


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
                  pinned_files={})
    entries["config.json"] = (json.dumps(config, indent=2) + "\n").encode()
    provenance = {
        "source": "Motion accepted run_base engine snapshot",
        "files_sha256": {name: hashlib.sha256(data).hexdigest()
                         for name, data in entries.items()},
    }
    entries["source_hashes.json"] = (json.dumps(provenance, indent=2) + "\n").encode()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    return OUTPUT


if __name__ == "__main__":
    print(build())
