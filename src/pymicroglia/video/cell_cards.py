"""Encode independent cell-card movies from existing identities and original pixels."""
from pathlib import Path
import hashlib
import json
import shutil

METHOD_VERSION = "2026-09-29-cell-card-video-shared-lut-v2"


def cell_card_video(run, *, stem=None, identities=(), fps=4., frame_width_px=1280,
                    output_dir=None, output_name=None, dry_run=False,
                    show_mean=False, reporter="gray"):
    """Write an MP4, exact signal table and native-frame receipt without an HTML page."""
    import numpy as np
    from ..figure_tables.saved import Tables
    from ..figure_tables.cell_inputs import load
    from ..figure_tables.cell_cards import analyse
    from ..visualisation.cell_cards import render_frames
    from ..visualisation.panels.cell_cards import reporter_lut, cards
    from auto_organotypic.render import luts
    from auto_organotypic.video.encode import write_video
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if isinstance(frame_width_px, bool) or int(frame_width_px) != frame_width_px or frame_width_px < 640:
        raise ValueError("frame_width_px must be an integer of at least 640")
    if reporter not in {"gray", "cd68", "bioluminescence"}:
        raise ValueError("reporter must be gray, cd68 or bioluminescence")
    source = Tables(run, stem)
    inputs = load(source)
    chosen = tuple(map(int, identities)) or tuple(int(value) for value in np.unique(inputs["labels"]) if value)[:3]
    if len(chosen) > 6:
        raise ValueError("choose at most six cells per readable card movie")
    analysis = analyse(inputs["raw"], inputs["labels"], identities=chosen,
        source_frame_offset=inputs["offset"], frame_interval_h=inputs["cadence"])
    folder = Path(output_dir) if output_dir else source.run / "videos" / "cell_cards"
    name = output_name or f"{source.stem}_cell_cards"
    if Path(name).name != name:
        raise ValueError("output_name must be a single filename stem")
    video = folder / f"{name}.mp4"
    table = folder / f"{name}.csv"
    receipt = folder / f"{name}_video.json"
    report = {"output": str(video), "table": str(table), "receipt": str(receipt),
        "identities": list(chosen), "frames": len(inputs["raw"]), "fps": float(fps),
        "source_frame_offset": inputs["offset"], "frame_interval_h": inputs["cadence"],
        "display_only": True, "display_filter": None, "resampling": "nearest",
        "motion_trails": False, "method_version": METHOD_VERSION,
        "show_mean": bool(show_mean), "reporter": reporter, "lut": reporter_lut(reporter)}
    if dry_run:
        return {**report, "dry_run": True}
    if any(path.exists() for path in (video, table, receipt)):
        raise FileExistsError("cell-card outputs already exist; choose a new output version")
    existing = folder.resolve()
    while not existing.exists():
        existing = existing.parent
    estimated_bytes = int(frame_width_px * round(frame_width_px * .75) * 3 * len(inputs["raw"]))
    if shutil.disk_usage(existing).free < estimated_bytes + 256 * 1024 ** 2:
        raise OSError("Insufficient free space for the cell-card movie and encoding buffer")
    folder.mkdir(parents=True, exist_ok=True)
    analysis["table"].to_csv(table, index=False)
    write_video(str(video), render_frames(analysis, frame_width_px=frame_width_px,
        show_mean=show_mean, reporter=reporter), fps=float(fps))
    def digest(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    report["sources"] = {"raw": str(inputs["raw_path"]), "labels": str(inputs["label_path"])}
    report["source_sha256"] = {role: digest(path) for role, path in report["sources"].items()}
    report["output_sha256"] = digest(video)
    report["table_sha256"] = digest(table)
    import inspect
    from ..measure.cell_signal import analyse as signal
    from ..measure.mask_geometry import observation
    report["producer_sha256"] = {
        Path(inspect.getfile(function)).name: digest(inspect.getfile(function))
        for function in (cell_card_video, analyse, signal, observation, render_frames, cards, luts.paint)
    }
    report["display_range"] = analysis["display_range"]
    receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
