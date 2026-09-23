"""Automatic counts retain cell boundaries and can exceed three states."""
from dataclasses import replace

import numpy as np
import pytest

from pymicroglia.states.features import CELL, frame_features
from pymicroglia.states.mixture import StateOptions, fit_states, group_split
from pymicroglia.states.selection import candidate_counts, search_report
from test_states import frames


@pytest.fixture(autouse=True)
def bounded_math_threads():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        yield


def options(**kwargs):
    return StateOptions(initializations=1, stability_repeats=1, training_samples=3000,
                        outlier_fraction=0, persistence_surrogates=0, rhythm_enabled=False,
                        auto_initial_states=2, auto_max_states=8, **kwargs)


def test_adaptive_search_expands_and_reports_a_competitive_limit():
    settings = options()
    rows = []
    for count in candidate_counts(settings, rows, "score", True, 100):
        rows.append({"states": count, "status": "ok", "score": count, "selection_standard_error": 0})
    assert [r['states'] for r in rows] == list(range(1, 9))
    report = search_report(settings, rows, "score", 8)
    assert report['selected_at_search_boundary'] and report['search_boundary_competitive']
    assert report['computational_state_ceiling'] == 8
    assert list(candidate_counts(replace(settings, candidate_states=(3,)), [], "score", True, 100)) == [3]


def test_cell_holdouts_are_complete_and_do_not_claim_independent_animals():
    table = frames(animals=1, cells=20, count=36)
    meta, snapshot, _, families, _ = frame_features({'cell_frame': table})
    settings = options(candidate_states=(2,))
    first = fit_states(meta, snapshot, families, settings)
    assignments = first['assignments']
    assert assignments.groupby(CELL).split.nunique().eq(1).all()
    assert set(assignments.split) == {'training', 'selection', 'test'}
    assert first['report']['selection_unit'] == 'cell'
    assert first['report']['independent_test_groups'] == 0
    # Final test measurements must affect neither preprocessing nor state selection.
    changed = snapshot.copy()
    changed.loc[assignments.split.eq('test')] += 1e6
    second = fit_states(meta, changed, families, settings)
    assert first['preprocessing'] == second['preprocessing']
    np.testing.assert_array_equal(first['model']['means'], second['model']['means'])
    assert len(np.unique(group_split(meta, replace(settings, selection_scope='independent_groups')))) == 1


@pytest.mark.parametrize('populations', [1, 4])
def test_one_or_four_states_can_be_selected_from_one_recording(populations):
    table = frames(animals=1, cells=30, count=72)
    rng = np.random.default_rng(52)
    state = (table.frame_index // 6) % populations
    table['area_px'] = 100 + 100 * (state % 2) + rng.normal(0, 2, len(table))
    table['corrected_mean'] = 200 + 200 * (state // 2) + rng.normal(0, 2, len(table))
    meta, snapshot, _, families, _ = frame_features({'cell_frame': table})
    selected = tuple(c for c in snapshot if '|area_px|' in c or '|corrected_mean|' in c)
    fitted = fit_states(meta, snapshot, families, options(snapshot_features=selected))
    assert fitted['report']['states'] == populations
    assert fitted['report']['count_selection_mode'] == 'adaptive'
    assert fitted['report']['independent_test_groups'] == 0
    if populations == 4:
        assert max(fitted['report']['counts_tested']) > 3


def test_automatic_history_groups_are_not_fixed_to_three():
    from pymicroglia.states.trajectory import group_trajectories
    table = frames(animals=1, cells=14, count=16)
    rng = np.random.default_rng(99)
    table['area_px'] = np.where(table.identity < 6, 20, np.where(table.identity < 12, 100, 500)) + rng.normal(0, .3, len(table))
    table['corrected_mean'] = np.where((table.identity >= 6) & (table.identity < 12), 500, 100) + rng.normal(0, .3, len(table))
    meta, snapshot, change, _, _ = frame_features({'cell_frame': table})
    selected = [c for c in snapshot if '|area_px|' in c or '|corrected_mean|' in c]
    fitted = group_trajectories(meta, snapshot[selected], change.iloc[:, :0],
        options(trajectory_groups=None, dynamic_min_cluster_size=3, dynamic_min_samples=2))
    assert fitted['report']['status'] == 'fitted'
    assert fitted['report']['groups'] == 2
    assert fitted['report']['requested_groups'] is None
    assert fitted['report']['count_selection_mode'] == 'density_hierarchy'
    assert fitted['assignments'].groupby('dynamic_group').size().loc[lambda s: s.index >= 0].min() >= 3


@pytest.mark.parametrize('kwargs', [{'auto_max_states': 0}, {'auto_initial_states': 25},
    {'selection_scope': 'frames'}, {'trajectory_groups': 0}, {'candidate_states': []}])
def test_invalid_search_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        StateOptions(**kwargs)
