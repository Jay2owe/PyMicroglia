"""Regression checks for the protected component boundary of round three."""
from __future__ import annotations

import numpy as np
import pandas as pd

import local_fragment_reunion as reunion


def _example():
    labels = np.zeros((3, 30, 30), dtype=np.uint16)
    labels[:, 5:15, 5:15] = 10
    labels[:, 10:12, 15:17] = 20
    labels[:, 25:27, 25:27] = 20  # Same identity, distinct untouched body.
    points = pd.DataFrame([
        {"frame": frame, "state": "observed", "strong": True,
         "x": 9, "y": 9, "radius_px": 5}
        for frame in range(3)
    ])
    lineage = pd.DataFrame([{
        "canonical_identity": 10, "alias_identity": 20,
        "track_handoffs": 4,
        "minimum_shared_track_pair_purity": 1.0,
        "union_coverage": 1.0,
        "cooccurrence_fraction": 0.2,
    }])
    return labels, points, lineage


def test_only_local_fragment_pixels_change():
    labels, points, lineage = _example()
    candidate, audit, applications = reunion.correct(labels, points, lineage)
    expected = labels.copy()
    expected[:, 10:12, 15:17] = 10
    np.testing.assert_array_equal(candidate, expected)
    np.testing.assert_array_equal(candidate > 0, labels > 0)
    assert len(applications) == 3
    assert int(audit.applied.sum()) == 3


def test_unreliable_or_already_used_lineage_keeps_every_pixel():
    labels, points, lineage = _example()
    lineage.loc[0, "minimum_shared_track_pair_purity"] = 0.49
    candidate, _, applications = reunion.correct(labels, points, lineage)
    np.testing.assert_array_equal(candidate, labels)
    assert applications.empty
    lineage.loc[0, "minimum_shared_track_pair_purity"] = 1.0
    prior_union = pd.DataFrame([{
        "canonical_identity": 10, "alias_identity": 91,
    }])
    candidate, _, applications = reunion.correct(
        labels, points, lineage, applied_unions=prior_union)
    np.testing.assert_array_equal(candidate, labels)
    assert applications.empty
