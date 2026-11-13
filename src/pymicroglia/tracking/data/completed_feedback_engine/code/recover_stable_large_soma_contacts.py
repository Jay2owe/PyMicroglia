"""Restrict the target-free dual-soma rule to relatively large stable cells.

Size is ranked against independently observed somata in the same recording,
so this criterion scales with an unseen image set's pixel size and cell sizes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import recover_stable_dual_soma_contacts as dual


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray | None = None
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    return dual.correct(
        labels, raw, points, protected,
        params=dual.Params(minimum_reference_area_percentile=0.6))
