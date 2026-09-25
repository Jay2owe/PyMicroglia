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
    parser.add_argument("--um-per-px", type=float, required=True)
    parser.add_argument("--no-scale-bar", action="store_true")
    parser.add_argument("--mask-style", required=True)
    parser.add_argument("--mask-opacity", type=float, required=True)
    parser.add_argument("--outline-colour", required=True)
    parser.add_argument("--outline-width-px", type=int, required=True)
    parser.add_argument("--image-centre-method", default="mask")
    parser.add_argument("--image-crop-basis", default="largest_cell")
    parser.add_argument("--exclude-unavailable-cycles", action="store_true")
    parser.add_argument("--no-image-frame-crop", dest="image_frame_crop",
                        action="store_false", default=True)
    parser.add_argument("--no-image-fill-tile", dest="image_fill_tile",
                        action="store_false", default=True)
    parser.add_argument("--video-centre-method", default="mask")
    parser.add_argument("--video-centre-smoothing-frames", type=int, default=0)
    parser.add_argument("--video-centre-deadband-fraction", type=float, default=0)
    parser.add_argument("--video-crop-basis", default="largest_cell")
    parser.add_argument("--well-label-position", default="top-left")
    parser.add_argument("--crop", default="tight")
    parser.add_argument("--counts-gain", type=float, required=True)
    parser.add_argument("--counts-offset", type=float, required=True)
    parser.add_argument("--a104-cache-dir", type=Path)
    media = parser.add_mutually_exclusive_group()
    media.add_argument("--image-only", action="store_true")
    media.add_argument("--video-only", action="store_true")
    parser.add_argument("--video-crf", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    params = {"interval_h": args.interval_h,
              "um_per_px": args.um_per_px,
              "scale_bar": not args.no_scale_bar,
              "mask_style": args.mask_style,
              "mask_opacity": args.mask_opacity,
              "outline_colour": args.outline_colour,
              "outline_width_px": args.outline_width_px,
              "image_centre_method": args.image_centre_method,
              "image_crop_basis": args.image_crop_basis,
              "exclude_unavailable_cycles": args.exclude_unavailable_cycles,
              "image_frame_crop": args.image_frame_crop,
              "image_fill_tile": args.image_fill_tile,
              "video_centre_method": args.video_centre_method,
              "video_centre_smoothing_frames": args.video_centre_smoothing_frames,
              "video_centre_deadband_fraction": args.video_centre_deadband_fraction,
              "video_crop_basis": args.video_crop_basis,
              "well_label_position": args.well_label_position,
              "crop": args.crop,
              "counts_gain": args.counts_gain,
              "counts_offset": args.counts_offset,
              "a104_cache_dir": (str(args.a104_cache_dir)
                                 if args.a104_cache_dir is not None else None),
              "video_crf": args.video_crf}
    params["media"] = ("image" if args.image_only else
                       "video" if args.video_only else "both")
    display_filter = {
        "method": "a104", "counts_gain": args.counts_gain,
        "counts_offset": args.counts_offset,
        "cache_dir": str(args.a104_cache_dir or args.outdir / "a104_source")}
    common = dict(
        output_dir=args.outdir, display_filter=display_filter,
        frame_interval_h=args.interval_h, channels=1,
        um_per_px=args.um_per_px, scale_bar=not args.no_scale_bar,
        well_label_position=args.well_label_position,
        mask_style=args.mask_style, mask_opacity=args.mask_opacity,
        outline_colour=args.outline_colour,
        outline_width_px=args.outline_width_px)
    image = video = None
    outputs = []
    if not args.video_only:
        image = cell_image_grid(args.raw, args.labels,
                                output_name="image.png",
                                crop_basis=args.image_crop_basis, crop=args.crop,
                                fill_tile=args.image_fill_tile,
                                frame_crop=args.image_frame_crop,
                                exclude_unavailable_cycles=args.exclude_unavailable_cycles,
                                centre_method=args.image_centre_method, **common)
        write_json(args.outdir / "image_report.json", image)
        outputs.extend(["image.png", "image_report.json"])
    if not args.image_only:
        video = cell_video_grid(
            args.raw, args.labels, output_name="video.mp4",
            crop_basis=args.video_crop_basis, crop=args.crop,
            centre_method=args.video_centre_method,
            centre_smoothing_frames=args.video_centre_smoothing_frames,
            centre_deadband_fraction=args.video_centre_deadband_fraction,
            profile="fast", crf=args.video_crf,
            encoder_options=["-level", "6.2"], **common)
        write_json(args.outdir / "video_report.json", video)
        outputs.extend(["video.mp4", "video_report.json"])
    with series.open_series(args.raw) as source:
        source_frames = source.shape[0]
    encoded_frames = None
    if video is not None:
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
        "cell_identities": (video or image)["cell_grid"]["cell_identities"],
        "outputs": outputs})
    print(f"{args.stem}: {source_frames} frames, "
          f"{len((video or image)['cell_grid']['cell_identities'])} cells")


if __name__ == "__main__":
    main()
