"""A mask for one exposure, from a network rather than from a time axis.

``pymicroglia.segmentation`` finds cells the accepted way: a band-pass, a noise
estimate, a threshold in sigma. It is the better route whenever it can be used,
because every number in it is a measurement you can point at.

It cannot be used on a still. The accepted seedless detector needs a 14 hour
window to separate a cell from static, and a great deal of microglial imaging --
a single frame, a short clip, anything where a cell moves within the window --
has no such window to give it. This package is the answer to that one question:
what is a cell in *this* picture, with nothing before or after it.

The route is four steps, and only the first is learned:

    normalise   put any picture on one scale: take the glow away per pixel,
                work in square-rooted photons, divide by the picture's own noise
    network     a small U-net returns a probability per pixel
    refine      the cut is a seed, not the answer -- grow each seed out into its
                processes and split what is too big to be one cell
    warn        say so when the recording is not the pixel size the weights were
                trained on, because that failure is silent otherwise

What it is for, and what it is not for. The mask is a mask: which pixels are
cell. Pair it with the raw signal and it is what the Motion project tracks
identities through. It is not a measurement of brightness, and nothing here
should be read as one.

**Known limit, measured rather than assumed.** The network is bound to the pixel
size it was trained at. A recording with pixels twice as coarse loses between a
third and a half of its cells, and the mask still looks reasonable while it
happens. Sizes in micrometres do not fix that, and neither did training across a
range of pixel sizes -- both were tried, and both are written up in the
development record this package was ported from. So :func:`apply.scale_warning`
exists, it fires on every mismatch, and the warning travels in the run's own
record rather than only to a terminal.

    from pymicroglia.learned_mask import apply as masking

    model = masking.load_model(masking.weights())
    cells = masking.mask_pictures(model, pictures, um_per_px=2.0)

Torch, SciPy and scikit-image are all imported inside the functions that need
them, so importing this package costs nothing and a machine without them still
imports ``pymicroglia``. Install what this needs with
``pip install "PyMicroglia[mask]"``.
"""

from __future__ import annotations

from . import filters, refine, scale

__all__ = ["filters", "refine", "scale", "apply", "network"]


def __getattr__(name: str):
    """Reach ``apply`` and ``network`` lazily, so torch is never imported early.

    Both modules are usable as ``learned_mask.apply`` and
    ``learned_mask.network`` without either being imported when this package is.
    """
    if name in ("apply", "network"):
        import importlib  # noqa: PLC0415 - deliberately local

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
