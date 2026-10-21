"""Draw observed XY outlines with time as the third axis; join no missing slices."""
from ._contract import Drawn


def sculpture(figure, data, *, style=None, **options):
    from matplotlib import colormaps
    ax = figure.add_subplot(111, projection="3d")
    table = data["table"]
    low, high = table.hours.min(), table.hours.max()
    cmap = colormaps["viridis"]
    for (frame, component), group in table.groupby(["source_frame_index", "component"], sort=False):
        group = group.sort_values("vertex")
        hours = group.hours.iloc[0]
        ax.plot(group.x_px, group.y_px, group.hours,
                color=cmap((hours - low) / (high - low) if high > low else .5))
    ax.set(xlabel="X (source pixels)", ylabel="Y (source pixels)", zlabel="Recording time (hours)")
    ax.set_box_aspect(None, zoom=.8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.labelpad = 12
    first, last = data["presence"].hours.min(), data["presence"].hours.max()
    if last > first:
        ax.set_zlim(first, last)
    ax.view_init(elev=float(data["elevation"]), azim=float(data["azimuth"]))
    return Drawn(table, ax)
