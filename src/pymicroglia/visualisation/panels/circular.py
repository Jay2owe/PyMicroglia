"""Draw pre-binned circular counts."""
import numpy as np
from ._contract import Drawn
from ..roles import role


def rose(ax, data, *, style=None, **options):
    ax.bar(data.bin_left * 2 * np.pi, data["count"],
           width=(data.bin_right-data.bin_left) * 2 * np.pi,
           align="edge", color=role("reporter"), alpha=.7)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.arange(4)*np.pi/2, ["0", "1/4", "1/2", "3/4"])
    ax.set_title("Peak positions within individual cycles")
    return Drawn(data, ax)
