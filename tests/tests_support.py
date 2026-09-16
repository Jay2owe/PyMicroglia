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


def oscillating_stack(folder, *, frames: int = 72, height: int = 130,
                      width: int = 130, seed: int = 7) -> Path:
    """Two cells that actually vary in time, on a structural blob.

    The temporal variation is the point. Every admissibility test in this
    package reads *amplitude*, not brightness, so a static synthetic cell --
    however bright -- is correctly rejected by the decoy test and would make
    this fixture prove nothing.

    Here rather than in one test file because two of them now need a stack a
    whole pipeline run will actually get through: the run tests and the opt-in
    mask tests. A second copy of it would drift from the first.
    """
    import numpy as np
    import tifffile

    rng = np.random.default_rng(seed)
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    hours = np.arange(frames) * 0.5

    tissue = 900.0 * np.exp(
        -((grid_y - height / 2) ** 2 + (grid_x - width / 2) ** 2)
        / (2 * 30.0 ** 2))

    def spot(y, x, sigma_px):
        return np.exp(-((grid_y - y) ** 2 + (grid_x - x) ** 2)
                      / (2 * sigma_px ** 2))

    cells = [(58, 58, 4.0, 900.0, 24.0), (92, 84, 3.4, 520.0, 24.0)]
    data = np.zeros((frames, 2, height, width), np.uint16)
    for frame in range(frames):
        plane = np.zeros((height, width), np.float64)
        for y, x, sigma_px, amplitude, period in cells:
            phase = 1.0 + 0.6 * np.sin(2 * np.pi * hours[frame] / period)
            plane += amplitude * phase * spot(y, x, sigma_px)
        data[frame, 0] = np.clip(
            2000 + plane + rng.normal(0, 25, (height, width)), 0, 65535)
        data[frame, 1] = np.clip(
            2000 + tissue + rng.normal(0, 15, (height, width)), 0, 65535)

    target = Path(folder) / "oscillating.ome.tif"
    delta_t = [float(f * 1800 + c) for f in range(frames) for c in (0, 1)]
    tifffile.imwrite(
        target, data, ome=True,
        metadata={"axes": "TCYX",
                  "Channel": {"Name": ["dluc", "structural"]},
                  "Plane": {"DeltaT": delta_t,
                            "DeltaTUnit": ["s"] * len(delta_t)}})
    return target
