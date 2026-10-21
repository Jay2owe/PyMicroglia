"""Export linked interactive views of existing cell masks and measurements."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

from ..figure_tables.cell_review_data import METHOD_VERSION, measurement_rows, prepare

ASSETS = Path(__file__).with_name("review_assets")
SCRIPTS = ("review.js", "changes.js", "sculpture.js", "cards.js", "barcode.js")


def code_fingerprint() -> str:
    """Fingerprint the actual preparation and browser code shipped together."""
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "figure_tables/cell_review_data.py",
             root / "figure_tables/cell_review_geometry.py", root / "measure/mask_geometry.py",
             root / "measure/signal.py", root / "measure/cell_signal.py", root / "measure/mask_io.py"]
    paths.extend(sorted(ASSETS.glob("*")))
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_text(path: Path, content: str):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def cell_review(raw, labels, *, output, title=None, source_frame_offset=0,
                frame_interval_h=.5, display_lut="red", display_range=None) -> dict:
    """Write a portable offline browser review; leave inputs and identities intact.

    Output is a new .html file. Its sibling CSV and JSON receipt record original
    pixels, saved masks, native times and producer code. No denoising, missing-
    mask interpolation or ID assignment occurs. Labels[0] belongs to the raw
    frame at source_frame_offset.
    """
    target = Path(output).resolve()
    if target.suffix.lower() != ".html":
        raise ValueError("output must be an .html file")
    table = target.with_name(target.stem + "_measurements.csv")
    receipt_path = target.with_name(target.stem + "_review.json")
    if any(path.exists() for path in (target, table, receipt_path)):
        raise FileExistsError("review outputs already exist; choose a new output version")
    target.parent.mkdir(parents=True, exist_ok=True)
    estimated = max(64 * 1024 ** 2, 12 * (Path(raw).stat().st_size + Path(labels).stat().st_size))
    if shutil.disk_usage(target.parent).free < estimated:
        raise OSError(f"insufficient free space for review export; need {estimated} bytes")
    payload = prepare(raw, labels, title=title, source_frame_offset=source_frame_offset,
                      frame_interval_h=frame_interval_h, display_lut=display_lut,
                      display_range=display_range)
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    scripts = "\n".join((ASSETS / name).read_text(encoding="utf-8")
                        for name in SCRIPTS if (ASSETS / name).is_file())
    page = (ASSETS / "review.html").read_text(encoding="utf-8")
    replacements = {"CSS": (ASSETS / "review.css").read_text(encoding="utf-8"),
                    "DATA": encoded, "JS": scripts + "\nwindow.CellReview.start();"}
    page = re.sub(r"__CELL_REVIEW_(CSS|DATA|JS)__", lambda match: replacements[match[1]], page)
    rows = list(measurement_rows(payload))
    temporary_table = table.with_name(table.name + ".tmp")
    with temporary_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("cell_id", "frame", "time_h", "mask_present",
                                                  "area_px", "integrated_density", "integrated_density_mean3"))
        writer.writeheader()
        writer.writerows(rows)
    temporary_table.replace(table)
    _atomic_text(target, page)
    receipt = {key: payload[key] for key in ("schema_version", "method_version", "title", "frame_count",
               "label_frame_count", "source_frame_offset", "frame_interval_h", "source_shape", "ids",
               "display", "sources", "source_sha256")}
    receipt.update({"status": "completed", "display_only": True, "code_sha256": code_fingerprint(),
                    "html": str(target), "table": str(table), "html_bytes": target.stat().st_size,
                    "html_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(), "table_rows": len(rows),
                    "frame_indexing": "CSV is zero-based; review frame labels are one-based",
                    "measurements": "sum of original raw pixels and area inside saved labels; mean3 requires consecutive observations"})
    _atomic_text(receipt_path, json.dumps(receipt, indent=2) + "\n")
    return {"html": str(target), "table": str(table), "receipt": str(receipt_path),
            "frames": payload["frame_count"], "cells": len(payload["ids"]),
            "display_only": True, "method_version": METHOD_VERSION}
