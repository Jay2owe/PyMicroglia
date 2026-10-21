"""Draw sharp prepared cell crops with raw signal, mean signal, area and a cursor."""
import numpy as np
from . import colour
from ._contract import Drawn
from auto_organotypic.render.luts import paint, channel_colour


def reporter_lut(reporter):
    """Use the same image lookup table as the shared still/movie renderer."""
    maps = {"cd68": "red", "green": "green", "bioluminescence": "dluc_purple", "gray": "grays"}
    if reporter not in maps:
        raise ValueError("reporter must be cd68, green, bioluminescence or gray")
    return maps[reporter]


def outline_overlay(card):
    """Match original and expanded outline colours to their trace legends."""
    from matplotlib.colors import to_rgba
    outline = np.zeros((*card["image"].shape, 4))
    if "background_boundary" in card:
        outline[card["background_boundary"]] = to_rgba(colour("blue"), .35)
    if "expanded_boundary" in card:
        outline[card["expanded_boundary"]] = to_rgba(colour("orange"))
    outline[card["boundary"]] = to_rgba(colour("teal"))
    return outline


def cards(figure, data, *, style=None, show_mean=False, reporter="gray", comparison=False, **options):
    from matplotlib.colors import to_rgba
    axes = []
    layout = figure.add_gridspec(4 if show_mean else 3, len(data["cards"]),
                                height_ratios=[2.7, 1.1, 1.1, 1.1] if show_mean else [3, 1.3, 1.3])
    lut = reporter_lut(reporter)
    channel = {"cd68": 1, "green": 3, "bioluminescence": 4, "gray": 2}[reporter]
    strong, light = channel_colour(channel), channel_colour(channel, tint=.55)
    for column, card in enumerate(data["cards"]):
        image = figure.add_subplot(layout[0, column])
        image.imshow(paint(card["image"], lut), interpolation="nearest")
        outline = outline_overlay(card)
        image.imshow(outline, interpolation="nearest")
        image.set_axis_off()
        image.set_title(f"Cell {card['identity']}" + (" · mask absent" if not card["mask_present"] else ""))
        signal = figure.add_subplot(layout[1, column])
        table = card["table"]
        if comparison:
            for field, label, ink in (("integrated_density", "Current outline", "teal"),
                                     ("expanded_integrated_density", "Expanded outline", "orange"),
                                     ("corrected_expanded_integrated_density", "Expanded minus background", "blue")):
                signal.plot(table.hours, table[field], color=colour(ink), label=label,
                            linestyle="--" if ink == "blue" else "-")
        else:
            signal.plot(table.hours, table.integrated_density, color=light, marker=".", label="Total light")
            signal.plot(table.hours, table.integrated_density_mean3, color=strong, marker=".", label="3-frame mean")
        if column == 0:
            signal.set_ylabel("Total light")
        bottom = figure.add_subplot(layout[2, column], sharex=signal)
        if show_mean:
            bottom.plot(table.hours, table.spatial_mean, color=light, marker=".", label="Mean per pixel")
            bottom.plot(table.hours, table.spatial_mean_mean3, color=strong, marker=".", label="3-frame mean")
            if column == 0:
                bottom.set_ylabel("Mean / pixel")
            area = figure.add_subplot(layout[3, column], sharex=signal)
            area.plot(table.hours, table.area_px, color=colour("orange"), marker=".")
            area.set_xlabel("Time (h)")
            if column == 0:
                area.set_ylabel("Area (pixels)")
        else:
            bottom.plot(table.hours, table.area_px, color=colour("teal") if comparison else colour("orange"), marker=".")
            if comparison:
                bottom.plot(table.hours, table.expanded_area_px, color=colour("orange"), label="Expanded")
            bottom.set_xlabel("Recording time (hours)")
            if column == 0:
                bottom.set_ylabel("Mask area (pixels)")
        for ax in (signal, bottom, area) if show_mean else (signal, bottom):
            ax.axvline(card["hours"], color=colour("blue"), linestyle="--")
        axes.extend([image, signal, bottom, area] if show_mean else [image, signal, bottom])
    if not show_mean:
        for index in range(len(data["cards"])):
            axes[index * 3 + 1].legend()
    return Drawn(data["table"], axes)
