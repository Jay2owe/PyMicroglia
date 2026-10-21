"""Draw already measured gained/lost pixels over an original source image."""
from . import colour
from ._contract import Drawn


def change_map(ax, data, *, style=None, **options):
    from matplotlib.patches import Rectangle, Patch
    low, high = data["range"]
    ax.imshow(data["image"], cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    for kind, name in (("gained", "teal"), ("lost", "orange")):
        for row in data["pixels"].loc[data["pixels"].kind.eq(kind)].itertuples():
            ax.add_patch(Rectangle((row.x_px - .5, row.y_px - .5), 1, 1,
                                   facecolor=colour(name), edgecolor="none"))
    for path in data["contours"]:
        ax.plot([p[0] - .5 for p in path], [p[1] - .5 for p in path], color="white")
    x0, y0, x1, y1 = data["bounds"]
    ax.set(xlim=(x0 - .5, x1 - .5), ylim=(y1 - .5, y0 - .5), aspect="equal",
           xlabel="Source x (pixels)", ylabel="Source y (pixels)")
    ax.legend(handles=[Patch(color=colour("teal"), label="Gained mask pixels"),
                       Patch(color=colour("orange"), label="Lost mask pixels")],
              loc="upper left", bbox_to_anchor=(1.02, 1))
    if not data["summary"].comparison_available.iloc[0]:
        ax.text(.5, -.15, "No consecutive mask comparison", ha="center", transform=ax.transAxes)
    return Drawn(data["table"], ax)
