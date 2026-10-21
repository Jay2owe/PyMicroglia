"""The default reporter trace is unsubtracted total light averaged in time."""

import numpy as np

from pymicroglia.measure.context import MeasurementContext, Scale
from pymicroglia.measure.defaults import MICROGLIA_METRICS
from pymicroglia.measure.modules.intensity import measure
from pymicroglia.measure.signal import DEFAULT_REPORTER_METRIC


def test_default_total_preserves_expanding_cell_and_does_not_bridge_missing_mask():
    labels = np.zeros((7, 12, 12), dtype=np.uint16)
    raw = np.full((7, 12, 12), 5.0)
    for frame, area in ((0, 1), (1, 2), (2, 3), (4, 4), (5, 5), (6, 6)):
        labels[frame, 3, 3:3 + area] = 1
        raw[frame, 3, 3:3 + area] = 10.0
    context = MeasurementContext(
        stem="test", labels=labels, raw=raw,
        scale=Scale(30.0), identities=[1],
    )
    cells = measure(context)["intensity"]
    assert DEFAULT_REPORTER_METRIC in MICROGLIA_METRICS
    np.testing.assert_allclose(cells.signal_mean, [10.] * 6)
    np.testing.assert_allclose(cells.integrated_density, [10., 20., 30., 40., 50., 60.])
    np.testing.assert_allclose(
        cells.integrated_density_mean3,
        [np.nan, 20., np.nan, np.nan, 50., np.nan], equal_nan=True,
    )
