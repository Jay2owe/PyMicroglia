from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path) -> Path:
    for path in [start.resolve(), *start.resolve().parents]:
        if path.is_dir() and (path / "config.json").exists() and (
                path / "code/pipeline.py").exists():
            return path
    raise FileNotFoundError("Motion project root not found")


