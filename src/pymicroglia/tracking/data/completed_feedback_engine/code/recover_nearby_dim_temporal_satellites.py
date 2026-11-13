"""Extend the accepted dim, soma-free branch reunion to nearby fragments.

The accepted R130 evidence and all protected edits remain intact. Only the
newly eligible geometric and temporal boundary is widened; no feedback,
recording, identity, frame, track, or spatial target enters this producer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import recover_soma_free_temporal_satellites as prior


PARAMS = prior.Params(
    maximum_fragment_area=150,
    minimum_host_area=250,
    maximum_gap_pixels=3,
    maximum_raw_q95_ratio=.7,
    minimum_direct_temporal_overlap=.35,
    minimum_expanded_temporal_overlap=.8,
)


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    return prior.correct(labels, raw, points, protected, PARAMS)
