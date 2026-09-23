from tests.panel_helpers import reopen_page
"""Population pages never replace all eligible cells with selected reports."""
from pymicroglia._results import read_document
import json
import numpy as np
import pandas as pd
import pytest
import pymicroglia.pipelines.relationships.population_figures as figures
from pymicroglia.pipelines._contracts import Settings
from pymicroglia.pipelines.relationships.inputs import PAIR_KEYS
from pymicroglia.pipelines.relationships.options import run_request
from pymicroglia.pipelines._screening import read_table
from tests.test_relationship_between import inputs, declaration, question
from tests.test_relationship_consistency import context, run, settings
from tests.test_rhythm_relationship_figures import capture_figures

@pytest.fixture(autouse=True)
def restore_style():
    import matplotlib as mpl
    with mpl.rc_context():
        yield

def saved_data(tmp_path, changes=None, unconfirmed=False):
    ctx = context(tmp_path, changes, unconfirmed)
    saved, tables = run(ctx)
    return {'population_members': tables['members'], 'population_summaries': tables['summaries'], 'population_units': tables['units'], 'preparation': {'resolved_request': ctx.request.as_dict()}, 'summary_provenance': read_document(saved.artifact('provenance'))}

def test_opposite_nonsignificant_effects_and_true_sample_counts_survive(tmp_path):
    data = saved_data(tmp_path)
    opts = {'population_groups_per_page': 4, 'population_group_order': None}
    pages = [p for p in figures.pages(data, opts) if p['view'] == 'within_cell']
    frames = [figures.values(data, p)[0] for p in pages]
    cells = pd.concat(frames).loc[lambda f: f.kind.eq('cell_effect')]
    samples = pd.concat(frames).loc[lambda f: f.kind.eq('sample_effect')]
    assert len(cells) == 36 and set(cells.effect) == {0.8, -0.4, 0.2}
    assert cells.original_status.eq('no-detected-association').sum() == 12
    assert len(samples) == 6 and samples.requested_recordings.eq(2).all() and samples.requested_cells.eq(6).all()
    assert set(samples.reference) == set(cells.reference) == {'corrected_mean'}
    assert cells[PAIR_KEYS].drop_duplicates().shape[0] == 36

def test_unresolved_delays_stay_missing_and_unconfirmed_movies_are_distinct(tmp_path):

    def changed(within, delayed, profiles):
        for row in delayed:
            if row['identity'] == 2:
                row.update(significant=True, status='negative-association', resolution_status='broad', delay_hours=None)
    data = saved_data(tmp_path, changed, unconfirmed=True)
    page = next((p for p in figures.pages(data, {'population_groups_per_page': 16, 'population_group_order': None}) if p['view'] == 'delay'))
    values, statistics = figures.values(data, page)
    assert 'recording:m0_0' in page['groups'] and 'sample:s0' in page['groups']
    cells = values.loc[values.kind.eq('cell_effect')]
    assert cells.loc[cells.identity.eq(2), 'delay_hours'].isna().all()
    summary = statistics.loc[statistics.level.eq('all_cells')].iloc[0]
    assert summary.delay_unresolved_cells == 12 and summary.delay_supported_cells == 12

def test_full_registered_population_pages_reopen_without_scientific_calls(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    import pymicroglia.measure.relationship_statistics as relationship_statistics
    import pymicroglia.measure.relationship_lag_statistics as relationship_lag_statistics
    import pymicroglia.measure.relationship_population_statistics as relationship_population_statistics
    import pymicroglia.measure.relationship_consistency_statistics as relationship_consistency_statistics
    frame = pd.DataFrame([{'stem': f'm{s}_{movie}', 'identity': cell, 'frame_index': i, 'hours': 50.0 + i, 'corrected_mean': float(s) + i + cell * 0.1, 'area_px': s * 2 + {1: 1, 2: -1, 3: 1}[cell] * i + cell * 0.1} for s in range(6) for movie in range(2) for cell in (1, 2, 3) for i in range(6)])
    req = declaration(within_cell={'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}}, between_cells=question(evidence={'method': 'none'}), sample_summary=settings(), biological_samples={f'm{s}_{m}': f's{s}' for s in range(6) for m in range(2)})
    resolved, paths, _ = inputs(tmp_path, req, frame)
    science = run_request(resolved, paths, tmp_path / 'pipeline', only=('sample-consistency',))
    assert science.successful, {k: v.outcome.reason for k, v in science.results.items()}
    captured = capture_figures(monkeypatch)

    def forbidden(*a, **k):
        raise AssertionError('Population display repeated scientific analysis')
    for obj, name in ((circadian, 'estimate_one'), (circadian, 'detrend_trace'), (circadian, 'adjust_pvalues'), (relationship_statistics, 'same_time_evidence'), (relationship_lag_statistics, 'evaluate'), (relationship_population_statistics, 'sample_association'), (relationship_consistency_statistics, 'sign_evidence')):
        monkeypatch.setattr(obj, name, forbidden)
    result = run_request(resolved, paths, tmp_path / 'pipeline', only=('relationship-populations',))
    assert result.successful, {k: v.outcome.reason for k, v in result.results.items()}
    assert len(captured) == 4
    saved = result.results['relationship-populations']
    entries = read_table(saved.artifact('entries.json'))
    assert len(entries) == 24 and entries.entry_id.is_unique
    within = captured[0][1].figure_data
    cells = within.loc[within.kind.eq('cell_effect')]
    assert len(cells) == 36 and cells.effect.min() < 0 and (cells.effect.max() > 0)
    scalar = captured[2][1].figure_data
    assert scalar.kind.eq('cell_scalar').sum() == 36 and scalar.kind.eq('sample_scalar').sum() == 6
    for ctx, drawing in captured:
        ctx.cached.clear()
        cold = reopen_page(ctx)
        pd.testing.assert_frame_equal(cold.figure_data, drawing.figure_data)
        import matplotlib.pyplot as plt
    changed = run_request(resolved, paths, tmp_path / 'pipeline', only=('relationship-populations',), presentation={'relationship_populations': {'population_groups_per_page': 3, 'population_group_order': [f'sample:s{s}' for s in reversed(range(6))]}})
    assert changed.successful, {k: v.outcome.reason for k, v in changed.results.items()}
    updated = read_table(changed.results['relationship-populations'].artifact('entries.json'))
    assert set(updated.entry_id) == set(entries.entry_id)
    assert all((saved.outcome.status == 'reused' for name, saved in changed.results.items() if name != 'relationship-populations'))

@pytest.mark.parametrize('value', [{'population_groups_per_page': 0}, {'population_group_order': ['a', 'a']}, {'significant_only': True}])
def test_presentation_cannot_hide_the_eligible_population(value):
    with pytest.raises(ValueError):
        figures.options(Settings({'relationship_populations': value}))
