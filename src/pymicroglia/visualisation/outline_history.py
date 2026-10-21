"""Native Matplotlib outline viewer with draggable 3D axes, independent of HTML."""
from ..figure_tables.saved import Tables
from ..figure_tables.outline_history import prepare
from .panels.outline_history import sculpture
from . import panels


def view(run, *, stem=None, identity=1, coordinates="cell", elevation=25., azimuth=-55.):
    """Return a rotatable Matplotlib figure; call show() in an interactive backend."""
    from matplotlib import pyplot as plt
    options = dict(identity=identity, coordinates=coordinates, elevation=elevation, azimuth=azimuth)
    data, _ = prepare(Tables(run, stem), options)
    panels._applied("pyflash")
    figure = plt.figure(figsize=(12, 9))
    sculpture(figure, data["outlines"])
    figure.suptitle(f"Cell {identity} · observed outlines through time")
    return figure
