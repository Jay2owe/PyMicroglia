"""Explicit, shallow access to saved run evidence for the local workbench."""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from threading import RLock
from typing import Any

from pymicroglia._optional import kit

VIEWABLE = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif", ".mp4", ".webm", ".tif", ".tiff"}
DOWNLOADABLE = VIEWABLE | {".pdf", ".csv", ".tsv", ".json", ".txt", ".py"}


def recent_runs(limit: int = 30) -> list[dict]:
    audit = kit()
    if audit is None:
        return []
    return audit.audit.runs(limit=min(max(limit, 1), 200), project="pymicroglia")


def saved_record(run_id: str) -> dict[str, Any] | None:
    audit = kit()
    if audit is None:
        return None
    found = audit.audit.result(run_id)
    return found if found.get("ok") else None


class ArtifactIndex:
    """Tokenise only files in a chosen results folder or recorded outputs."""

    def __init__(self):
        self.files: dict[str, Path] = {}
        self.lock = RLock()

    def select_folder(self, folder: str | Path) -> dict:
        root = Path(folder).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Choose a saved results folder")
        rows = []
        # One level only: listing a Dropbox tree must not hydrate its TIFFs.
        for entry in sorted(root.iterdir()):
            if entry.is_file():
                candidates = (entry,)
            elif entry.is_dir() and not entry.name.startswith("."):
                candidates = tuple(entry.iterdir())
            else:
                candidates = ()
            for file in candidates:
                if file.is_file() and file.suffix.lower() in DOWNLOADABLE:
                    rows.append(self._add(file, root))
                if len(rows) >= 1200:
                    break
            if len(rows) >= 1200:
                break
        return {"folder": str(root), "artifacts": rows, "truncated": len(rows) >= 1200}

    def select_record(self, record: dict) -> list[dict]:
        """Expose files named in a completed run record, with no directory walk."""
        rows = []
        for item in record.get("outputs") or []:
            path = item.get("path") if isinstance(item, dict) else item
            if not path:
                continue
            file = Path(str(path)).expanduser()
            if file.is_file() and file.suffix.lower() in DOWNLOADABLE:
                rows.append(self._add(file.resolve(), file.parent.resolve()))
        return rows

    def select_files(self, files, root) -> list[dict]:
        """Expose explicitly named deliverables, never walk a directory tree."""
        root = Path(root).resolve()
        return [self._add(Path(file), root) for file in files
                if Path(file).is_file() and Path(file).suffix.lower() in DOWNLOADABLE]

    def _add(self, file: Path, root: Path) -> dict:
        file = file.resolve()
        if root != file.parent and root not in file.parents:
            raise ValueError("Output leaves selected results folder")
        token = hashlib.sha256(str(file).encode("utf-8")).hexdigest()[:24]
        with self.lock:
            self.files[token] = file
        return {"token": token, "name": file.name, "path": str(file),
                "kind": file.suffix.lower().lstrip("."),
                "preview": file.suffix.lower() in VIEWABLE}

    def get(self, token: str) -> tuple[Path, str]:
        with self.lock:
            file = self.files.get(token)
        if file is None or not file.is_file():
            raise KeyError(token)
        return file, mimetypes.guess_type(file.name)[0] or "application/octet-stream"
