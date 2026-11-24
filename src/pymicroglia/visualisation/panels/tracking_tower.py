"""Draw a prepared tracking tower: projected snapshot planes, outlines and identity links.

Everything arrives computed from ``figure_tables.tracking_tower``; each track
interval sits behind its earlier plane and above its later one.
"""
from ._contract import Drawn
from . import colour


def tower(figure, data, *, style=None, **options):
    from matplotlib.collections import LineCollection
    from matplotlib.transforms import Affine2D
    layout = data["layout"]
    # Side-by-side views arrive as subfigures; size the whole page for every tower in it.
    root = figure if hasattr(figure, "set_size_inches") else figure.figure
    width, height = layout["figsize"]
    root.set_size_inches(width * max(1, len(getattr(root, "subfigs", []))), height)
    ax = figure.add_axes([.015, .015, .97, .955])
    ax.set_facecolor(colour("white"))
    for plane in data["planes"]:
        ax.imshow(plane["rgba"], origin="upper", extent=plane["extent"], interpolation="nearest",
                  transform=Affine2D.from_values(*plane["transform"]) + ax.transData, zorder=plane["zorder"])
        ax.text(*plane["label_xy"], plane["label"], ha="left", va="center",
                fontsize=layout["time_label_size"], fontweight="bold")
    for line in data["outlines"]:
        ax.plot(*line["points"].T, color=line["colour"], lw=layout["outline_width_pt"], zorder=line["zorder"])
    for group in data["tracks"]:
        ax.add_collection(LineCollection(group["segments"], colors=group["colours"], linewidths=group["widths"],
                                         zorder=group["zorder"], capstyle="round"))
    for surface in data["surfaces"]:
        ax.imshow(surface["rgba"], extent=surface["extent"], origin="lower", interpolation="bilinear",
                  zorder=surface["zorder"])
    ink = colour("black")
    for start, end in layout["arrows"]:
        ax.annotate("", xy=tuple(end), xytext=tuple(start), zorder=100,
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=ink, mutation_scale=13))
    for text in layout["texts"]:
        ax.text(*text["xy"], text["text"], rotation=text["rotation"], ha="center", va="center",
                fontsize=layout["axis_label_size"], fontweight="bold")
    ax.set_xlim(*layout["xlim"])
    ax.set_ylim(*layout["ylim"])
    ax.set_aspect("equal")
    ax.axis("off")
    if layout["title"]:
        ax.set_title(layout["title"], loc="left", fontweight="bold", pad=8)
    return Drawn(data["table"], ax, {"checks": data["checks"]})
