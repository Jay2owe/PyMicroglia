"""A correct marked frame cannot conceal a later wrong biological name."""
import numpy as np
import pandas as pd
import pytest
from feedback_identity_path_checks import evaluate_declared_paths


def fixture():
    labels = np.zeros((6, 20, 20), np.uint16)
    labels[:, 5:12, 5:12] = 3
    points = pd.DataFrame([dict(frame=t, track_id=37, x=8., y=8.,
                                radius_px=3., state='observed') for t in range(6)])
    declarations = [dict(case_id='explicit_identity', feedback_source='saved_mark.jsonl',
                         expected_identity=3, diagnostic_raw_tracks=[37],
                         first_review_frame=2, last_review_frame=6)]
    return labels, points, declarations


def test_marked_frame_pass_does_not_hide_wrong_later_name():
    labels, points, declarations = fixture()
    labels[2:, 5:12, 5:12] = 96
    result = evaluate_declared_paths(labels, points, declarations)
    assert result.status.tolist() == ['matches'] + ['wrong_identity']*4


def test_missing_outline_is_reported_separately_from_wrong_name():
    labels, points, declarations = fixture()
    labels[-1] = 0
    result = evaluate_declared_paths(labels, points, declarations)
    assert result.status.tolist() == ['matches']*4 + ['missing_outline']


def test_tracker_first_name_cannot_replace_explicit_expectation():
    labels, points, declarations = fixture()
    labels[0, 5:12, 5:12] = 96
    result = evaluate_declared_paths(labels, points, declarations)
    assert result.actual_identity.eq(3).all()
    assert result.expected_identity.eq(3).all()


def test_inferred_identity_without_feedback_source_is_refused():
    labels, points, declarations = fixture()
    declarations[0].pop('feedback_source')
    with pytest.raises(ValueError, match='feedback source'):
        evaluate_declared_paths(labels, points, declarations)
