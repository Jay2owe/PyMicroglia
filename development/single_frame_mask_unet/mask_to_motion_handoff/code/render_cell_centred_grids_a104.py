"""Render the accepted native cell grids with full-frame A104 display pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from auto_organotypic import series
from pymicroglia.visualisation.cell_image_grid import cell_image_grid
from pymicroglia.visualisation.cell_video_grid import cell_video_grid


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n",
                    encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--interval-h", type=float, required=True)
    parser.add_argument("--mask-style", required=True)
    parser.add_argument("--mask-opacity", type=float, required=True)
    parser.add_argument("--outline-colour", required=True)
    parser.add_argument("--outline-width-px", type=int, required=True)
    parser.add_argument("--counts-gain", type=float, required=True)
    parser.add_argument("--counts-offset", type=float, required=True)
    parser.add_argument("--video-crf", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    params = {"interval_h": args.interval_h,
              "mask_style": args.mask_style,
              "mask_opacity": args.mask_opacity,
              "outline_colour": args.outline_colour,
              "outline_width_px": args.outline_width_px,
              "counts_gain": args.counts_gain,
              "counts_offset": args.counts_offset,
              "video_crf": args.video_crf}
    display_filter = {
        "method": "a104", "counts_gain": args.counts_gain,
        "counts_offset": args.counts_offset,
        "cache_dir": str(args.outdir / "a104_source")}
    common = dict(
        output_dir=args.outdir, display_filter=display_filter,
        frame_interval_h=args.interval_h, channels=1,
        mask_style=args.mask_style, mask_opacity=args.mask_opacity,
        outline_colour=args.outline_colour,
        outline_width_px=args.outline_width_px)
    image = cell_image_grid(args.raw, args.labels,
                            output_name="image.png", **common)
    write_json(args.outdir / "image_report.json", image)
    video = cell_video_grid(
        args.raw, args.labels, output_name="video.mp4",
        profile="fast", crf=args.video_crf,
        encoder_options=["-level", "6.2"], **common)
    write_json(args.outdir / "video_report.json", video)
    with series.open_series(args.raw) as source:
        source_frames = source.shape[0]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "json",
         str(args.outdir / "video.mp4")],
        check=True, capture_output=True, text=True)
    encoded_frames = int(json.loads(probe.stdout)["streams"][0]["nb_frames"])
    if encoded_frames != source_frames or video["frames"] != source_frames:
        raise ValueError(f"{args.stem}: video frames {encoded_frames} "
                         f"do not match photon frames {source_frames}")
    write_json(args.outdir / "run.json", {
        "status": "completed", "stem": args.stem,
        "params": params, "source": str(args.raw),
        "labels": str(args.labels),
        "source_frames": source_frames,
        "video_frames": encoded_frames,
        "cell_identities": video["cell_grid"]["cell_identities"],
        "outputs": ["image.png", "video.mp4", "image_report.json",
                    "video_report.json"]})
    print(f"{args.stem}: {source_frames} frames, "
          f"{len(video['cell_grid']['cell_identities'])} cells")


if __name__ == "__main__":
    main()
