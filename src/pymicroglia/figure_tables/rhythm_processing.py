from __future__ import annotations
from typing import Sequence,Any
import numpy as np
import pandas as pd
from .. import workbench

def detrended_z(hours: Sequence[float], values: Sequence[float], *, method: str='linear', window_hours: float=24.0, detrend_options: dict | None=None) -> np.ndarray:
    """A workbench-detrended trace, in units of its own spread.

    Two things at once, and both are what the rhythm test does before it looks
    for a cycle. Subtracting the line stops a cell that simply gets brighter all
    recording from reading as the first half of a wave; dividing by the spread
    puts a dim cell and a bright one on one colour scale, so a raster row is
    comparable to the row above it.
    """
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    residual = np.asarray(workbench.detrend_trace(hours, values, {**dict(detrend_options or {}), 'detrend': method, 'detrend_window_hours': window_hours})['values'], dtype=float)
    return workbench.scale_detrended(residual, values)

def detrended_traces(cell_frame: pd.DataFrame, identities: Sequence[Any], column: str, *, method: str='linear', window_hours: float=24.0, detrend_options: dict | None=None) -> pd.DataFrame:
    """Long table of every named cell's detrended trace: identity, hours, value."""
    pieces = []
    wanted = cell_frame[cell_frame['identity'].isin(list(identities))]
    for identity, group in wanted.groupby('identity'):
        usable = group[['hours', column]].dropna().sort_values('hours')
        if len(usable) < 2:
            continue
        pieces.append(pd.DataFrame({'identity': identity, 'hours': usable['hours'].to_numpy(float), 'detrended_z': detrended_z(usable['hours'], usable[column], method=method, window_hours=window_hours, detrend_options=detrend_options)}))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=['identity', 'hours', 'detrended_z'])
