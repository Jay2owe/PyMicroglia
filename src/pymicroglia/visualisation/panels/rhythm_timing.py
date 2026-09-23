"""Draw saved cycle positions without implying a shared clock between cells."""
import numpy as np
from ._contract import Drawn
from ..roles import role


def phase_dial(ax, data, *, style=None, **options):
    selected = data.loc[data.included_in_plot]
    for row in selected.itertuples():
        angle = float(row.phase_fraction) * 2 * np.pi
        ax.annotate("", xy=(angle, row.amplitude), xytext=(angle, 0),
                    arrowprops={"arrowstyle": "->", "color": role("reporter"), "alpha": .6})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.arange(4) * np.pi / 2, ["0", "1/4", "1/2", "3/4"])
    ax.set_title("Peak within each cell's own period")
    return Drawn(data, ax)


def phase_histogram(ax, data, *, style=None, **options):
    ax.bar(data.bin_left, data["count"], width=data.bin_right-data.bin_left,
           align="edge", color=role("reporter"))
    ax.set_xlabel("Fitted oscillation amplitude")
    ax.set_ylabel("Rhythmic cells")
    return Drawn(data, ax)
