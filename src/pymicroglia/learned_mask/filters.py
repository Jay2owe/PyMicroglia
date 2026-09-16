"""The two filters this route spends most of its time in, done on the vector unit.

Distinct from :mod:`pymicroglia.filtering`, which is about *display* -- what a
person should look at. These two are inside the measurement, and they are here
for one reason: of the ~390 ms a 498 x 498 picture cost, 160 was SciPy, and 60 of
that was these two operations.

Nothing here is a new idea. :func:`blur` is the same Gaussian SciPy computes and
:func:`reach` the same Euclidean distance, to the same kernel width and the same
edge rule; OpenCV simply runs them vectorised. Measured against SciPy on a
498 x 498 float32 field: the blur agrees to 1.2e-07 on a spread of 1.44, which is
every significant figure a float32 has, and the distance transform is
bit-identical.

Where that lands in the mask, measured over five recordings and 66 million
pixels: **one pixel moved**, and no cell was gained, lost, split or merged.

Two things worth knowing before touching this file.

**OpenCV's distance transform defaults to an approximation.** ``DIST_L2`` with a
5x5 mask is a chamfer estimate, not the Euclidean distance, and it differs from
SciPy on about 1500 pixels of a 498 x 498 field. ``DIST_MASK_PRECISE`` is the
real thing and is still five times faster, so the approximation is never worth
taking. It has already been mistaken for the fast option once.

**OpenCV is optional.** Without it, both functions fall back to SciPy and give
the older answer; the analysis works either way, one is quicker.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TRUNCATE = 4.0      # SciPy's default kernel width, and what every sigma assumes

_CV2: Any = None
_TRIED = False


def _opencv() -> Any:
    """The ``cv2`` module, or ``None``. Cached, so a missing one costs one try."""
    global _CV2, _TRIED
    if not _TRIED:
        _TRIED = True
        try:
            import cv2  # noqa: PLC0415 - deliberately local
        except Exception:
            _CV2 = None
        else:
            # One picture at a time; the caller owns the thread budget.
            cv2.setNumThreads(0)
            _CV2 = cv2
    return _CV2


def radius(sigma: float) -> int:
    """How far SciPy's Gaussian reaches, so OpenCV is given the same window."""
    return int(TRUNCATE * float(sigma) + 0.5)


def blur(picture: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur: same window, same reflected edge, same answer."""
    sigma = float(sigma)
    cv2 = _opencv()
    if cv2 is None or sigma <= 0 or picture.ndim != 2:
        from scipy import ndimage  # noqa: PLC0415 - deliberately local

        return ndimage.gaussian_filter(picture, sigma)
    width = 2 * radius(sigma) + 1
    return cv2.GaussianBlur(np.ascontiguousarray(picture, np.float32),
                            (width, width), sigma,
                            borderType=cv2.BORDER_REFLECT)


def reach(labels: np.ndarray, distance: float) -> np.ndarray:
    """Which unlabelled pixels lie within ``distance`` of something labelled.

    SciPy measures out from every empty pixel; OpenCV measures out from every
    non-zero pixel of what it is handed, so it is handed the empty ones. The
    same measurement, read from the other end.
    """
    empty = labels == 0
    cv2 = _opencv()
    if cv2 is None:
        from scipy import ndimage  # noqa: PLC0415 - deliberately local

        return ndimage.distance_transform_edt(empty) <= distance
    return cv2.distanceTransform(empty.astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE) <= distance


def agrees(picture: np.ndarray | None = None) -> dict:
    """What the fast path differs from SciPy by, on a picture or on noise.

    For a test or a new machine, never for the analysis itself. A difference
    here that is not at the size of float32 rounding means OpenCV was built
    differently on this machine and should not be trusted on it.
    """
    from scipy import ndimage  # noqa: PLC0415 - deliberately local

    cv2 = _opencv()
    if picture is None:
        picture = np.random.default_rng(0).normal(
            0, 1, (498, 498)).astype(np.float32)
    picture = np.ascontiguousarray(picture, np.float32)
    marks = np.zeros(picture.shape, np.int32)
    marks[::40, ::40] = 1
    out: dict[str, Any] = {"opencv": None if cv2 is None else cv2.__version__}
    for sigma in (2.0, 15.0):
        theirs = ndimage.gaussian_filter(picture, sigma)
        out[f"blur_{sigma:g}_max_difference"] = float(
            np.abs(blur(picture, sigma) - theirs).max())
        out[f"blur_{sigma:g}_spread"] = float(theirs.max() - theirs.min())
    out["reach_identical"] = bool(np.array_equal(
        reach(marks, 8), ndimage.distance_transform_edt(marks == 0) <= 8))
    return out
