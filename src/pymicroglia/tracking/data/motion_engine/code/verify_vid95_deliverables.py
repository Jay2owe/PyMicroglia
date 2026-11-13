"""Check that every fresh VID95 run has complete-field, 99-frame review TIFFs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import tifffile


ROOT = Path(__file__).resolve().parents[1]
STEMS = [f"95_{row}{col}" for row, cols in
         (("A", range(1, 7)), ("B", range(1, 7)), ("C", range(1, 3)))
         for col in cols]


def shape(path: Path) -> tuple[int, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with tifffile.TiffFile(path) as tif:
        return tuple(int(x) for x in tif.series[0].shape)


def main() -> None:
    rows = []
    for stem in STEMS:
        run = f"full_vid95_clean_lookback_20260922_{stem}"
        source = ROOT / "registered inputs/full_vid95_fresh_20260921" / f"{stem}.tif"
        labels = ROOT / "m22_accepted_history" / run / "out" / f"{stem}.tif"
        outlines = ROOT / "m6_review" / run / "out" / f"{stem}_full_field_outlines.tif"
        review = ROOT / "m6_review" / run / "out" / f"{stem}_full_field_review.tif"
        source_shape, label_shape = shape(source), shape(labels)
        outline_shape, review_shape = shape(outlines), shape(review)
        if source_shape != label_shape or len(source_shape) != 3 or source_shape[0] != 99:
            raise ValueError(f"{stem}: source {source_shape} and labels {label_shape} disagree")
        review_height = source_shape[1] + 24
        if (outline_shape[:3] != (99, review_height, source_shape[2])
                or outline_shape[-1] != 3):
            raise ValueError(f"{stem}: outlines {outline_shape} disagree with source {source_shape}")
        if (review_shape[0] != 99 or review_shape[1] != review_height
                or review_shape[2] != source_shape[2] * 3 or review_shape[-1] != 3):
            raise ValueError(f"{stem}: review {review_shape} has incomplete movie/field")
        manifest = json.loads((ROOT / "m6_review" / run / "run.json").read_text())
        if manifest.get("status") != "done" or manifest.get("summary", {}).get("review_frames") != 99:
            raise ValueError(f"{stem}: review stage not complete")
        rows.append({"stem": stem, "frames": 99, "height": source_shape[1],
                     "width": source_shape[2], "labels": str(labels),
                     "outlines": str(outlines), "review": str(review),
                     "review_stage_status": manifest["status"]})
        print(f"{stem}: 99 frames, {source_shape[2]} x {source_shape[1]} pixels", flush=True)
    path = ROOT / "reviews/vid95_full_fresh_20260921_outlines/review_tiff_validation.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
