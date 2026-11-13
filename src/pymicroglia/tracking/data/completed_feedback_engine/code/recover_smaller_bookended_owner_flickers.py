"""Extend approved two-sided soma continuity to smaller one-frame outlines.

The established rule checks both neighboring names, continuity of every
resident strong soma, raw peak retention, independent raw soma valleys,
accepted-pixel protection, foreground and original identity preservation.
The new boundary changes only component area and overlap thresholds. It has
no recording, identity, track, frame, region or feedback selector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import recover_large_bookended_owner_flickers as prior


PARAMS = prior.Params(minimum_component_area=50,
                      minimum_two_sided_overlap=.5)


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    return prior.correct(labels, raw, points, protected, PARAMS)
