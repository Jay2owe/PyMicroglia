import numpy as np
import pandas as pd
import pytest
from pymicroglia.figure_tables.cell_signal_ratio import recording_mean_ratio


def table(values, identity=1):
    return pd.DataFrame(dict(identity=identity, frame_index=np.arange(len(values)),
                             corrected_integrated_density=values))


def test_ratio_of_recording_means_includes_early_and_late_frames():
    result = recording_mean_ratio(table([100.,1.,1.,1.,100.]), table([1.]*5), [1]).iloc[0]
    assert result.mean_cd68 == pytest.approx(34.)
    assert result.mean_cry1 == 1.
    assert result.cd68_cry1_ratio == pytest.approx(34.)
    assert result.recording_start_hours == 0 and result.recording_end_hours == 2
    assert result.mean_window_frames == 3


def test_ratio_uses_matched_observations_and_is_not_mean_of_ratios():
    n, d = table([2.,6.,10.]), table([1.,2.,4.])
    result = recording_mean_ratio(n,d,[1]).iloc[0]
    expected_n = np.mean([4.,6.,8.])
    expected_d = np.mean([1.5,7/3,3.])
    assert result.cd68_cry1_ratio == pytest.approx(expected_n / expected_d)
    missing = recording_mean_ratio(table([2.,100.,10.]), table([1.,np.nan,4.]), [1], mean_window_frames=1).iloc[0]
    assert missing.paired_frames == 2
    assert missing.mean_cd68 == 6. and missing.mean_cry1 == 2.5
    assert missing.cd68_cry1_ratio == 2.4


def test_nonpositive_or_missing_denominator_is_explicitly_undefined():
    zero = recording_mean_ratio(table([2.,3.]),table([-1.,-1.]),[1]).iloc[0]
    assert zero.ratio_status == 'nonpositive denominator mean'
    assert np.isnan(zero.cd68_cry1_ratio)
    absent = recording_mean_ratio(table([2.,3.]),table([1.,1.],identity=2),[1]).iloc[0]
    assert absent.ratio_status == 'no paired observations'
    assert absent.paired_frames == 0 and np.isnan(absent.cd68_cry1_ratio)
