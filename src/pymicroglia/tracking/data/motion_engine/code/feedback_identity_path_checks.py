"""Evaluation only: explicit feedback identities along declared physical paths.

The producer must never import this module or read these declarations. A
declared expected identity comes from user feedback or a recorded human
adjudication, never from the first label assigned by the tracker. Existing
positional cue scores are kept separate and are not rewritten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from resident_takeover import _disk_owner


def evaluate_declared_paths(labels: np.ndarray, points: pd.DataFrame,
                            declarations: list[dict]) -> pd.DataFrame:
    rows = []
    for declaration in declarations:
        if not declaration.get('feedback_source'):
            raise ValueError('Every expected name requires a feedback source')
        expected = int(declaration['expected_identity'])
        if expected <= 0:
            raise ValueError('Expected biological identity must be positive')
        selected = points[points.track_id.isin(declaration['diagnostic_raw_tracks'])
                          & points.frame.between(declaration['first_review_frame']-1,
                                                 declaration['last_review_frame']-1)
                          & points.state.isin(['observed', 'latent_visible'])]
        if selected.empty:
            raise ValueError('Declared diagnostic path has no visible observations')
        for point in selected.sort_values('frame').itertuples(index=False):
            name, coverage, purity = _disk_owner(
                labels[int(point.frame)], float(point.x), float(point.y),
                float(point.radius_px))
            rows.append(dict(case_id=declaration['case_id'],
                             feedback_source=declaration['feedback_source'],
                             frame=int(point.frame)+1, raw_track=int(point.track_id),
                             expected_identity=expected, actual_identity=name,
                             status=('matches' if name == expected else
                                     'missing_outline' if name == 0 else 'wrong_identity'),
                             coverage=coverage, purity=purity))
    return pd.DataFrame(rows)
