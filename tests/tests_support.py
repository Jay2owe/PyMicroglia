"""Small synthetic stacks the segmentation tests share.

Kept out of ``fixtures.py`` because that file builds OME-TIFFs for the reader
tests, where the metadata is the point; here the *pixels* are the point and the
metadata only has to be enough to open.
"""

from __future__ import annotations

from pathlib import Path


def two_channel_stack(tmp_path, *, frames: int = 10, height: int = 120,
                      width: int = 120, cells=((40, 40, 20.0), (80, 75, 12.0)),
                      seed: int = 3) -> Path:
    """A dLuc channel with a few cells on it, and a structural channel.

    Channel 1 is the bioluminescence, channel 2 the structure. The structural
    channel carries a broad blob so ``off_tissue`` has a tissue to find and the
    background can be read beside it rather than from the corners.
    """
    import numpy as np
    import tifffile

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]

    tissue = 900.0 * np.exp(
        -((yy - height / 2) ** 2 + (xx - width / 2) ** 2) / (2 * 28.0 ** 2))
    dluc = np.zeros((height, width), np.float32)
    for y, x, height_sigma in cells:
        dluc += (height_sigma * 40.0
                 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 3.0 ** 2)))

    data = np.zeros((frames, 2, height, width), np.uint16)
    for frame in range(frames):
        data[frame, 0] = np.clip(
            2000 + dluc + rng.normal(0, 40, (height, width)), 0, 65535)
        data[frame, 1] = np.clip(
            2000 + tissue + rng.normal(0, 20, (height, width)), 0, 65535)

    target = Path(tmp_path) / "synthetic_cells.ome.tif"
    delta_t = [float(f * 1800 + c) for f in range(frames) for c in (0, 1)]
    tifffile.imwrite(
        target, data, ome=True,
        metadata={"axes": "TCYX",
                  "Channel": {"Name": ["dluc", "structural"]},
                  "Plane": {"DeltaT": delta_t,
                            "DeltaTUnit": ["s"] * len(delta_t)}},
    )
    return target
