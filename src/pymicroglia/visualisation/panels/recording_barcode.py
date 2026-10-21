"""Draw a prepared identity-by-native-frame matrix with explicit colour scaling."""
import numpy as np
from ._contract import Drawn


def barcode(ax, data, *, style=None, **options):
    from matplotlib import colormaps
    cmap = colormaps["viridis"].copy()
    cmap.set_bad("white")
    image = ax.imshow(np.ma.masked_invalid(data["display"]), aspect="auto",
                      interpolation="nearest", vmin=0, vmax=1, cmap=cmap,
                      extent=(data["frames"][0] - .5, data["frames"][-1] + .5, len(data["ids"]) - .5, -.5))
    stride = max(1, len(data["ids"]) // 20)
    ticks = list(range(0, len(data["ids"]), stride))
    ax.set_yticks(ticks, [str(data["ids"][index]) for index in ticks])
    ax.set(xlabel="Native source frame (zero based)", ylabel="Saved cell identity")
    ax.figure.colorbar(image, ax=ax, label=f"{data['scale']} scale (0–1)")
    return Drawn(data["table"], ax)
