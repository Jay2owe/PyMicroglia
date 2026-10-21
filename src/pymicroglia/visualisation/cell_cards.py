"""Stream native-frame cell cards using prepared pixels and fixed trace artists."""
import numpy as np
from matplotlib.colors import to_rgba
from . import panels
from .panels.cell_cards import cards, reporter_lut, outline_overlay
from ..figure_tables.cell_cards import frame_plan
from auto_organotypic.render.luts import paint


def render_frames(analysis, *, frame_width_px=1280, show_mean=False, reporter="gray", comparison=False):
    """Render cropped source pixels above synchronized measured cell traces."""
    from matplotlib import pyplot as plt
    panels._applied("pyflash")
    if int(frame_width_px) != frame_width_px or frame_width_px < 640:
        raise ValueError("frame_width_px must be an integer of at least 640")
    width = int(frame_width_px)
    height = int(round(width * .75))
    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100, layout="constrained")
    first = frame_plan(analysis, 0)
    drawn = cards(figure, {"cards": first, "table": analysis["table"]},
                  show_mean=show_mean, reporter=reporter, comparison=comparison)
    heading = figure.suptitle("Native source frame 1")
    try:
        for frame in range(len(analysis["raw"])):
            prepared = first if frame == 0 else frame_plan(analysis, frame)
            rows_per_card = 4 if show_mean else 3
            for index, card in enumerate(prepared):
                axes = drawn.axes[index * rows_per_card:index * rows_per_card + rows_per_card]
                image, *traces = axes
                image.images[0].set_data(paint(card["image"], reporter_lut(reporter)))
                image.images[1].set_data(outline_overlay(card))
                image.set_title(f"Cell {card['identity']}" + (" · mask absent" if not card["mask_present"] else ""))
                for axis in traces:
                    axis.lines[-1].set_xdata([card["hours"], card["hours"]])
            hours=prepared[0]['hours']
            title=f"Source frame {frame + 1} · {hours:g} h"
            heading.set_text(f"{analysis['title_prefix']} · {title}" if analysis.get('title_prefix') else title)
            if show_mean:
                heading.set_text(f"Frame {frame + 1} | {hours:g} h | lighter = raw; darker = 3-frame mean")
            figure.canvas.draw()
            yield np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    finally:
        plt.close(figure)


def render_still(analysis, frames, *, reporter="gray", tile_size_px=240):
    """Pack native cell crops with the shared LUT and zero-gap grid defaults.

    Each cell keeps its own fixed square crop through time. Display enlargement
    is uniform and uses nearest source pixels; no spatial filter is applied.
    Cell and time labels occupy strips outside the image tiles.
    """
    from PIL import Image
    from auto_organotypic.render.tiles import lay_out, DEFAULT_GAP_PX
    from auto_organotypic.render.annotate import row_labels, column_labels
    if isinstance(tile_size_px, bool) or int(tile_size_px) != tile_size_px or tile_size_px < 32:
        raise ValueError("tile_size_px must be an integer of at least 32")
    frames = tuple(frames)
    if not frames:
        raise ValueError("choose at least one native frame")
    size = int(tile_size_px)
    lut = reporter_lut(reporter)
    prepared = {frame: frame_plan(analysis, frame) for frame in frames}
    identities = list(analysis["cells"])
    pictures, records = [], []
    ink = np.asarray(to_rgba(panels.colour("teal"))[:3]) * 255
    for row, identity in enumerate(identities):
        for column, frame in enumerate(frames):
            card = prepared[frame][row]
            rgb = paint(card["image"], lut)
            rgb[card["boundary"]] = ink.astype(np.uint8)
            height, width = rgb.shape[:2]
            if height != width:
                raise ValueError("cell-card crops must be square; never stretch biological pixels")
            pictures.append(np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)))
            records.append({"identity": identity, "source_frame": int(frame), "hours": card["hours"],
                "mask_present": card["mask_present"], "crop_width_px": width, "crop_height_px": height,
                "tile_width_px": size, "tile_height_px": size, "row": row, "column": column,
                "gap_px": DEFAULT_GAP_PX, "lut": lut, "display_filter": "none", "resampling": "nearest"})
    canvas = lay_out(pictures, columns=len(frames), gap_px=DEFAULT_GAP_PX, background=(255, 255, 255))
    canvas = row_labels(canvas, [f"Cell {identity}" for identity in identities], row_height=size,
                        row_gap=DEFAULT_GAP_PX, font_size=max(14, round(size * .08)))
    left = canvas.shape[1] - len(frames) * size - (len(frames) - 1) * DEFAULT_GAP_PX
    canvas = column_labels(canvas, [f"{prepared[frame][0]['hours']:g} h" for frame in frames],
        column_width=size, column_gap=DEFAULT_GAP_PX, left_px=left, font_size=max(14, round(size * .08)))
    return canvas, records
