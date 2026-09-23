"""Place saved-display panels on the caller's canvas."""


def figure(canvas, *, figsize=None, **kwargs):
    if canvas is None:
        raise ValueError("A saved-display panel needs its caller's figure")
    if figsize is not None:
        canvas.set_size_inches(*figsize)
    return canvas


def subplots(canvas, *args, figsize=None, **kwargs):
    figure(canvas, figsize=figsize)
    return canvas, canvas.subplots(*args, **kwargs)


def finish(axis):
    axis.spines[["top", "right"]].set_visible(False)
