"""Putting a picture on one scale, and the small U-net that reads it.

The scaling matters more than the network does. Two wells of the same sample
glow differently, and a mask trained on one arrives at the other several times
too bright or too dim unless something puts them both on one footing. Three
things do that here, and each fixed a specific failure:

**Take the glow away per pixel, not per frame.** Subtracting one median left a
pedestal under the cells in a bright corner that nothing in training had shown
the network. So the background is a low percentile of each tile, stretched back
smoothly -- a surface, not a number.

**Work in square-rooted photons.** Photon counting is Poisson, so a place twice
as bright carries root-two more noise, and a single noise figure for a frame is
wrong in both directions at once. The spread of ``2*sqrt(x + 3/8)`` does not
depend on brightness, which is what lets one threshold mean the same thing in a
dim well and a glowing one.

**Divide by the picture's own pixel noise.** Measured on the band-passed
picture, so a glow across the well cannot set the scale. The naive version --
the spread of the whole frame -- read 4.2x the pixel noise on one recording and
8.4x on another purely because those wells glow differently, and sent the same
cell to the network at four times the value in one as in the other.

The network itself is deliberately small. A microglial cell is about 3.5 px
across at its core and under 40 px including processes, so the widest view the
bottom of the U needs is a few tens of pixels, not hundreds: three shrink steps,
0.48 M parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import filters, scale as scaling
from .scale import NATIVE

BASE_CHANNELS = 16
CROP_PX = NATIVE.px_int(scaling.CROP_UM)                       # 128 px
DOG_INNER_PX = NATIVE.px(scaling.DOG_INNER_UM)                 # 2.0 px
DOG_OUTER_PX = NATIVE.px(scaling.DOG_OUTER_UM)                 # 15.0 px
BACKGROUND_TILE = NATIVE.px_int(scaling.BACKGROUND_TILE_UM)    # 32 px
BACKGROUND_PCT = 25.0    # cells sit above this, so they cannot lift their own floor

RULES = ("frame", "noise", "flat", "flat_anscombe")
ACCEPTED_RULE = "flat_anscombe"


def _mad(values: np.ndarray) -> float:
    """A robust spread: the median deviation, scaled to match a Gaussian's."""
    return float(np.median(np.abs(values - np.median(values))) * 1.4826)


def anscombe(photons: np.ndarray) -> np.ndarray:
    """Square-root the photons so the noise is the same size everywhere."""
    return (2.0 * np.sqrt(np.clip(photons, 0.0, None) + 0.375)).astype(np.float32)


def background(image: np.ndarray, tile: int = BACKGROUND_TILE,
               percentile: float = BACKGROUND_PCT) -> np.ndarray:
    """The slow glow under a picture, measured where the cells are not.

    A low percentile of each tile: cells are bright and cover a few per cent of
    the field, so they sit well above the quarter-way mark of their own tile and
    cannot raise it. The coarse grid is stretched back from tile centres, so what
    comes out is a surface rather than a staircase.
    """
    from scipy import ndimage  # noqa: PLC0415 - deliberately local

    height, width = image.shape
    pad_y, pad_x = (-height) % tile, (-width) % tile
    padded = np.pad(image, ((0, pad_y), (0, pad_x)), mode="reflect")
    tiles = padded.reshape(padded.shape[0] // tile, tile,
                           padded.shape[1] // tile, tile)
    coarse = np.percentile(tiles, percentile, axis=(1, 3))
    rows = (np.arange(height) + 0.5) / tile - 0.5
    columns = (np.arange(width) + 0.5) / tile - 0.5
    grid = np.meshgrid(rows, columns, indexing="ij")
    return ndimage.map_coordinates(coarse, grid, order=1,
                                   mode="nearest").astype(np.float32)


def normalise(image: np.ndarray, rule: str = ACCEPTED_RULE) -> np.ndarray:
    """Put any picture on the same footing: middle at zero, pixel noise at one.

    ``flat_anscombe`` is the accepted rule and does all three things described in
    this module's docstring. ``flat`` skips the square root, ``noise`` skips the
    per-pixel glow removal, and ``frame`` is the original rule kept only so that
    weights trained under it are still predicted under it. A model carries the
    rule it was trained with, so no caller has to know which is which.
    """
    image = image.astype(np.float32)
    if rule in ("flat", "flat_anscombe"):
        picture = anscombe(image) if rule == "flat_anscombe" else image
        flat = picture - background(picture)
        band = filters.blur(flat, DOG_INNER_PX) - filters.blur(flat, DOG_OUTER_PX)
        return (flat / max(_mad(band), 1e-6)).astype(np.float32)
    middle = float(np.median(image))
    if rule == "noise":
        band = filters.blur(image, DOG_INNER_PX) - filters.blur(image, DOG_OUTER_PX)
        spread = _mad(band)
    elif rule == "frame":
        spread = _mad(image)
    else:
        raise ValueError(f"unknown normalise rule {rule!r}; one of {RULES}")
    return ((image - middle) / max(spread, 1e-6)).astype(np.float32)


def build(base: int = BASE_CHANNELS) -> Any:
    """The U-net: look, shrink, look wider, then rebuild.

    Built by a function rather than declared at module scope so that importing
    this module does not import torch.
    """
    import torch  # noqa: PLC0415 - deliberately local
    from torch import nn  # noqa: PLC0415
    from torch.nn import functional as F  # noqa: PLC0415

    class Block(nn.Sequential):
        def __init__(self, inputs: int, outputs: int):
            super().__init__(
                nn.Conv2d(inputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
                nn.Conv2d(outputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self, base: int = BASE_CHANNELS):
            super().__init__()
            widths = [base, base * 2, base * 4, base * 8]
            self.down = nn.ModuleList()
            channels = 1
            for width in widths[:-1]:
                self.down.append(Block(channels, width))
                channels = width
            self.middle = Block(channels, widths[-1])
            self.up = nn.ModuleList()
            self.join = nn.ModuleList()
            channels = widths[-1]
            for width in reversed(widths[:-1]):
                self.up.append(nn.ConvTranspose2d(channels, width, 2, stride=2))
                self.join.append(Block(width * 2, width))
                channels = width
            self.head = nn.Conv2d(channels, 1, 1)

        def forward(self, x):
            skips = []
            for block in self.down:
                x = block(x)
                skips.append(x)
                x = F.max_pool2d(x, 2)
            x = self.middle(x)
            for up, join, skip in zip(self.up, self.join, reversed(skips)):
                x = up(x)
                x = join(torch.cat([x, skip], dim=1))
            return self.head(x)

    return UNet(base)
