"""Show accepted cells without a qualifying cycle at six observed source frames."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from auto_organotypic import grid, series
from auto_organotypic.render.tile_source import TileSource
from pymicroglia.visualisation.cell_display import prepare_cell_display
from pymicroglia.visualisation.cell_tiles import cell_tiles


class SelectedFrames:
    def __init__(self, original, frames):
        self.original = original
        self.frames = frames

    @property
    def shape(self):
        return (len(self.frames), *self.original.shape[1:])

    @property
    def meta(self):
        return self.original.meta

    @property
    def dtype(self):
        return self.original.dtype

    @property
    def display_only(self):
        return self.original.display_only

    def frame(self, time, channel):
        return self.original.frame(int(self.frames[int(time)]), int(channel))


def selected_source(tile: TileSource, interval_h: float, moments: int) -> TileSource:
    _times, values = tile.trace
    observed = np.flatnonzero(np.isfinite(values))
    if not len(observed):
        raise ValueError(f"{tile.name} has no observed mask frame")
    positions = np.rint(np.linspace(0, len(observed) - 1, moments)).astype(int)
    frames = observed[positions].astype(int)
    hours = np.round(frames.astype(float) * float(interval_h), 6)

    @contextmanager
    def opened():
        with tile.open_series() as original:
            yield SelectedFrames(original, frames)

    def overlay(rgb, frame):
        actual = int(frames[int(frame)])
        if tile.frame_overlay is not None:
            rgb = tile.frame_overlay(rgb, actual)
        return rgb

    calibration = (None if tile.frame_um_per_px is None else
                   lambda frame: tile.frame_um_per_px(int(frames[int(frame)])))
    return TileSource(
        key=tile.key, name=tile.name, source_path=tile.source_path,
        open_series=opened,
        provenance={**tile.provenance,
                    "display_selection": "six evenly spaced observed mask frames",
                    "observed_source_frames": frames.tolist(),
                    "observed_source_hours": hours.tolist()},
        frame_overlay=overlay, frame_um_per_px=calibration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cycle-report", type=Path, required=True)
    parser.add_argument("--interval-h", type=float, required=True)
    parser.add_argument("--um-per-px", type=float, required=True)
    parser.add_argument("--counts-gain", type=float, required=True)
    parser.add_argument("--counts-offset", type=float, required=True)
    parser.add_argument("--a104-cache-dir", type=Path, required=True)
    parser.add_argument("--moments", type=int, default=6)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.moments < 1:
        parser.error("--moments must be positive")
    cycle_report = json.loads(args.cycle_report.read_text(encoding="utf-8"))
    ids = [int(one) for one in
           cycle_report["cell_grid"]["excluded_cycle_identities"]]
    if not ids:
        raise ValueError(f"{args.stem}: no excluded cycle identities")
    display_raw, display_range, display_report = prepare_cell_display(
        args.raw, {"method": "a104", "counts_gain": args.counts_gain,
                   "counts_offset": args.counts_offset,
                   "cache_dir": str(args.a104_cache_dir)})
    tiles = cell_tiles(
        args.raw, args.labels, frame_interval_h=args.interval_h,
        display_raw=display_raw, include_identities=ids,
        crop_basis="own_cell", crop="tight", fill_tile=True,
        frame_crop=True, centre_method="intensity_weighted",
        um_per_px=args.um_per_px, outline=True,
        mask_style="outline", outline_colour="yellow", outline_width_px=1)
    selected = [selected_source(tile, args.interval_h, args.moments)
                for tile in tiles]
    args.outdir.mkdir(parents=True, exist_ok=True)
    report = grid.stack_to_grid(
        selected, output_dir=args.outdir, output_name="image.png",
        when=list(range(1, args.moments + 1)), columns=args.moments,
        frame_interval_h=1.0, channels=1, lut="dluc_purple",
        display_range=display_range, tile_label="time",
        timestamp_position="header",
        timestamp_format="Obs {total_hours:.0f}", label_size=8,
        well_label_position="top-left", scale_bar=True)
    report["cell_grid_supplement"] = {
        "source": str(args.raw), "labels": str(args.labels),
        "cycle_report": str(args.cycle_report),
        "cell_identities": [tile.key for tile in selected],
        "selection": "six evenly spaced observed mask frames per cell",
        "times": "actual source recording hours in tile-source provenance",
        "display_filter": display_report}
    (args.outdir / "image_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    with series.open_series(args.raw) as source:
        source_frames = source.shape[0]
    params = {"interval_h": args.interval_h, "um_per_px": args.um_per_px,
              "counts_gain": args.counts_gain, "counts_offset": args.counts_offset,
              "a104_cache_dir": str(args.a104_cache_dir),
              "moments": args.moments}
    (args.outdir / "run.json").write_text(json.dumps({
        "status": "completed", "stem": args.stem, "params": params,
        "source": str(args.raw), "labels": str(args.labels),
        "source_frames": source_frames,
        "cell_identities": [tile.key for tile in selected],
        "outputs": ["image.png", "image_report.json"]}, indent=2) + "\n",
        encoding="utf-8")
    print(f"{args.stem}: {len(selected)} cells, {args.moments} observed frames each")


if __name__ == "__main__":
    main()
