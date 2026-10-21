"""Lossless pixels, centres and contours of existing cell masks."""
from __future__ import annotations

import numpy as np

METHOD_VERSION = "2026-09-28-saved-mask-geometry-v1"


def mask_rows(mask: np.ndarray, *, left: int = 0, top: int = 0) -> list[int]:
    """Flatten occupied runs as y, inclusive x start, exclusive x end."""
    rows: list[int] = []
    for y, line in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.pad(line.astype(np.int8), (1, 1))))
        for x0, x1 in edges.reshape(-1, 2):
            rows.extend((int(y + top), int(x0 + left), int(x1 + left)))
    return rows


def _corners(points: np.ndarray) -> np.ndarray:
    """Remove only exactly collinear vertices, preserving the outline."""
    if len(points) < 4:
        return points
    closed = np.array_equal(points[0], points[-1])
    body = points[:-1] if closed else points
    before = body - np.roll(body, 1, axis=0)
    after = np.roll(body, -1, axis=0) - body
    keep = (before[:, 0] * after[:, 1] != before[:, 1] * after[:, 0])
    keep |= np.sum(before * after, axis=1) < 0
    if not closed:
        keep[0] = keep[-1] = True
    result = body[keep]
    return np.vstack((result, result[:1])) if closed else result


def observation(yy: np.ndarray, xx: np.ndarray) -> dict:
    """Encode all pixels of one saved identity in one observed frame."""
    from skimage.measure import find_contours

    left, top = int(xx.min()), int(yy.min())
    right, bottom = int(xx.max()) + 1, int(yy.max()) + 1
    crop = np.zeros((bottom - top, right - left), bool)
    crop[yy - top, xx - left] = True
    centre = [float(xx.mean()) + .5, float(yy.mean()) + .5]
    # Record integer rounding once; Python and JavaScript have different ties.
    anchor = [int(np.floor(xx.mean() + .5)), int(np.floor(yy.mean() + .5))]
    contours = []
    for path in find_contours(np.pad(crop, 1), .5, fully_connected="high"):
        xy = np.column_stack((path[:, 1] + left - .5, path[:, 0] + top - .5))
        contours.append(_corners(xy).tolist())
    return {"centre": centre, "anchor": anchor, "bounds": [left, top, right, bottom],
            "runs": mask_rows(crop, left=left, top=top), "contours": contours}


def crop_size(observations: list[dict | None], *, margin: int = 10) -> int:
    """One native-pixel crop size which contains every observed mask."""
    reach = 10
    for item in observations:
        if item is None:
            continue
        x, y = item["anchor"]
        x0, y0, x1, y1 = item["bounds"]
        reach = max(reach, x - x0, x1 - x - 1, y - y0, y1 - y - 1)
    return int(2 * (reach + margin) + 1)
