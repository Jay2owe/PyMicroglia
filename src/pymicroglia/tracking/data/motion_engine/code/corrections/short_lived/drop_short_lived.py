"""Convert short-lived numerical identities into explicitly unclaimed foreground."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
import sys
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402


def choose_dropped(profiles: pd.DataFrame, maximum_frames: int,
                   preserve_edge_censored: bool) -> np.ndarray:
    selected = profiles.frames_present.astype(int) <= int(maximum_frames)
    if preserve_edge_censored:
        selected &= ~profiles.edge_censored.astype(bool)
    return profiles.loc[selected, "identity"].astype(int).to_numpy()


def drop(labels: np.ndarray, identities: np.ndarray
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isin(labels, identities)
    ownership = labels.copy()
    ownership[mask] = 0
    original_ids = np.where(mask, labels, 0).astype(labels.dtype, copy=False)
    foreground = (labels > 0).astype(np.uint8)
    return ownership, original_ids, foreground

