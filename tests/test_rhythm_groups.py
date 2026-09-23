"""Group-specific selection, complete exclusion accounting and biological pairs."""
import numpy as np
import pandas as pd
import pytest
from pymicroglia.pipelines import parse
from pymicroglia.pipelines._contracts import Settings, StepSpec, content_id
from pymicroglia.pipelines.rhythm.discovery import resolve_request, run_request
from pymicroglia.pipelines._runner import ExecutionContext
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_rhythm_reports import make_screen

def fixture(tmp_path, monkeypatch, *, mapping=True, tests=True, measurements=None, aggregate='median'):

    def transform(frame):
        base = frame[frame.stem.eq('a')].copy()
        rows = []
        for movie in ('a', 'b', 'c', 'd', 'e'):
            part = base.copy()
            part['stem'] = movie
            part['signal'] = np.where(part.identity.eq(1), 10.0, 30.0) + part.hours / 100
            part.loc[part.hours.isin([24, 25]) & part.identity.eq(1), 'signal'] = np.nan
            part['other'] = np.where(part.identity.eq(1), 40.0, 10.0) + part.hours / 100
            if movie == 'e':
                part = part[part.identity.eq(1)]
            rows.append(part)
        invalid = base[base.identity.eq(1)].copy()
        invalid['identity'], invalid['signal'], invalid['other'] = (3, 55.0, 55.0)
        rows.append(invalid)
        return pd.concat(rows, ignore_index=True)
    original, paths, _ = make_screen(tmp_path / 'run', monkeypatch, transform=transform)
    tables = {k: pd.read_csv(p) for k, p in paths.items()}
    values = tables['cell_summary'].copy()
    values['size'] = values.stem.map({'a': 10, 'b': 20, 'c': 15, 'd': 30, 'e': 100}) + values.identity * values.stem.map({'a': 2, 'b': 4, 'c': 5, 'd': 6, 'e': 7})
    values['absent'] = np.nan
    tables['comparison'] = values
    paths['comparison'] = tmp_path / 'run/comparison.csv'
    values.to_csv(paths['comparison'], index=False)
    declaration = {'pipeline': 'rhythm-discovery', 'test_measurements': ['signal', 'other'], 'comparison_measurements': measurements or [{'column': 'size', 'table': 'comparison'}], 'biological_samples': {'a': 's1', 'b': 's1', 'c': 's2', 'd': 's3', 'e': 's4'} if mapping else {}, 'analysis_options': original.request.analysis_options.as_dict(), 'group_comparisons': {'aggregate': aggregate, 'resamples': 100, 'tests': [{'name': 'size_by_signal', 'grouping_measurement': 'signal', 'metrics': ['size'], 'unit': 'subject', 'aggregate': 'median', 'test': 'paired_t', 'family': 'followups', 'correction': 'bonferroni'}] if tests else []}}
    resolved = resolve_request(parse([declaration])[0], source_run=original.inputs.source_run, tables=tables, input_hashes={k: file_hash(p) for k, p in paths.items()}, table_grains={'comparison': ('identity',)})
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=('group-comparisons',))
    assert execution.successful, [(k, v.outcome.reason) for k, v in execution.results.items()]
    return (resolved, paths, execution)

def test_metric_groups_pair_samples_and_preserve_exclusions(tmp_path, monkeypatch):
    _, _, execution = fixture(tmp_path, monkeypatch)
    result = execution.results['group-comparisons']
    cells = read_table(result.artifact('cells'))
    size = cells[cells.comparison.eq('size')]
    assert set(size[size.grouping_measurement.eq('signal') & size.group.eq('significant')].identity) == {1}
    assert set(size[size.grouping_measurement.eq('other') & size.group.eq('significant')].identity) == {2}
    assert size[size.identity.eq(3)].group.eq('untestable').all()
    units = read_table(result.artifact('units'))
    formal = units[units.contrast.eq('size_by_signal')]
    assert len(formal) == 4 and formal.paired.sum() == 3
    sample1 = formal[formal.unit_key.str.contains('s1')].iloc[0]
    assert sample1.cells_a == sample1.cells_b == 2
    assert sample1.value_a == 18 and sample1.value_b == 21
    sample4 = formal[formal.unit_key.str.contains('s4')].iloc[0]
    assert not sample4.paired and sample4.cells_b == 0 and sample4.reason
    stats = read_table(result.artifact('statistics')).iloc[0]
    assert stats.n_a == stats.n_b == 3 and np.isfinite(stats.p_value)
    assert stats.cells_a == stats.cells_b == 4
    context = cells[cells.comparison.eq('screen_invalid_fraction') & cells.grouping_measurement.eq('signal')]
    assert np.allclose(context[context.identity.eq(1)].value, 2 / 72)
    assert len(context) == 10
    summary = read_table(result.artifact('summary'))
    selected = summary[summary.comparison.eq('size') & summary.grouping_measurement.eq('signal')]
    assert selected.group_cells.sum() == 10 and selected.population_cells.eq(10).all()
    from pymicroglia.pipelines.rhythm.groups import unit_ledger, descriptive_summary
    from pymicroglia.pipelines.rhythm.group_figures import pages
    cell_units = unit_ledger(cells, name='declared-cells', unit='cell', aggregate=None)
    assert cell_units and all(('"identity"' in row['unit_key'] for row in cell_units))
    collisions = cells.copy()
    collisions.loc[collisions.grouping_measurement.eq('signal'), 'grouping_measurement'] = 'any-significant'
    collisions.loc[collisions.comparison.eq('size'), 'comparison'] = 'screen_duration_hours'
    rebuilt = descriptive_summary(collisions)
    assert set(rebuilt.comparison_kind) == {'measurement', 'context'}
    inventory = pages(collisions, {'measurements': [{'column': 'screen_duration_hours'}]})
    assert len(inventory) == 3 and len(set(inventory)) == 3

def test_changed_comparison_table_reuses_screen_and_preserves_missing_values(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    from pymicroglia.pipelines._screening import screen_identity
    resolved, paths, execution = fixture(tmp_path, monkeypatch, tests=False)
    original = execution.results['rhythm-screen']
    frame = pd.read_csv(paths['comparison'])
    frame['size'] += 100
    frame.to_csv(paths['comparison'], index=False)
    declaration = resolved.request.declaration.as_dict()
    declaration['comparison_measurements'] = [{'column': 'absent', 'table': 'comparison'}]
    newer = resolve_request(parse([declaration])[0], source_run=resolved.inputs.source_run, tables={k: pd.read_csv(p) for k, p in paths.items()}, input_hashes={k: file_hash(p) for k, p in paths.items()}, table_grains={'comparison': ('identity',)})
    assert screen_identity(newer) == original.outcome.scientific_id

    def forbidden(*a, **k):
        raise AssertionError('Saved screen must not be refitted')
    monkeypatch.setattr(circadian, 'estimate_one', forbidden)
    repeated = run_request(newer, paths, tmp_path / 'pipeline', only=('group-comparisons',))
    assert repeated.successful
    assert repeated.results['rhythm-screen'].outcome.status == 'reused'
    assert repeated.results['rhythm-screen'].outcome.selections == original.outcome.selections
    summary = read_table(repeated.results['group-comparisons'].artifact('summary'))
    absent = summary[summary.comparison.eq('absent')]
    assert absent.finite_cells.eq(0).all() and absent.missing_values.sum() == 30

def test_unknown_mapping_and_unpaired_declarations_are_refused(tmp_path, monkeypatch):
    from pymicroglia.pipelines.rhythm.groups import design_reason, _contrast
    resolved, _, execution = fixture(tmp_path, monkeypatch, mapping=False)
    stats = read_table(execution.results['group-comparisons'].artifact('statistics')).iloc[0]
    assert 'unconfirmed' in stats.note and pd.isna(stats.p_corrected)
    cells = read_table(execution.results['group-comparisons'].artifact('cells'))
    cells = cells[cells.group.isin(('significant', 'not-significant')) & cells.comparison.eq('size') & cells.grouping_measurement.eq('signal')].copy()
    cells['sample_confirmed'] = True
    cells['sample'] = cells.movie.map({'a': 's1', 'b': 's1', 'c': 's2', 'd': 's3', 'e': 's4'})
    item = resolved.request.group_comparisons.as_dict()['tests'][0]
    assert 'paired' in design_reason(cells, _contrast({**item, 'test': 'welch_t'}))
    assert 'Multiple movies' in design_reason(cells, _contrast({**item, 'unit': 'movie'}))

def test_declared_family_counts_unavailable_questions():
    import pymicroglia.measure.contrast_statistics as contrasts
    from pymicroglia.pipelines.rhythm.groups import _contrast
    config = _contrast({'name': 'example', 'metrics': ['size'], 'unit': 'subject', 'aggregate': 'mean', 'test': 'paired_t', 'correction': 'bonferroni'})
    rows = [contrasts._blank_row(config, 'size', ''), contrasts._blank_row(config, 'other', 'Unavailable')]
    rows[0].update(p_value=0.03)
    result = contrasts.correct_declared_results(rows)
    assert result.iloc[0].p_corrected == pytest.approx(0.06)
    assert pd.isna(result.iloc[1].p_corrected) and (not result.significant.any())

@pytest.mark.parametrize('with_units', [True, False])
def test_saved_renderer_never_tests_and_reopens_registered_pages(tmp_path, monkeypatch, with_units):
    from pymicroglia import workbench as circadian
    import pymicroglia.measure.contrast_statistics as contrasts
    import pymicroglia.pipelines.rhythm.group_figures as rhythm_group_figures
    resolved, paths, execution = fixture(tmp_path, monkeypatch, tests=with_units, aggregate='median' if with_units else None)
    import matplotlib.pyplot as plt

    def forbidden(*a, **k):
        raise AssertionError('No scientific calculation while drawing')
    monkeypatch.setattr(circadian, 'estimate_one', forbidden)
    monkeypatch.setattr(circadian, 'adjust_pvalues', forbidden)
    monkeypatch.setattr(contrasts, 'evaluate_contrast_rows', forbidden)
    from tests.panel_helpers import capture_pages, reopen_page
    captured = capture_pages(monkeypatch)
    context = ExecutionContext(StepSpec('group-comparison-figures', 'group-comparison-figures', ('group-comparisons',), kind='render'), resolved, Settings(), paths, {'group-comparisons': execution.results['group-comparisons']}, None, tmp_path / 'render/renders/groups/id/inv', 'saved-groups', Settings(), content_id({}))
    result = rhythm_group_figures.produce(context)
    assert result.status == 'completed' and len(captured) == 3
    for ctx, result in captured:
        assert not result.figure_data.empty
        if with_units:
            assert result.auxiliary['units.csv'].unit.eq('subject').all()
        else:
            assert 'units.csv' not in result.auxiliary
    ctx = captured[0][0]
    reopened = reopen_page(ctx)
    assert len(reopened.figure_data) == len(captured[0][1].figure_data)

def test_trace_summary_is_explicit_and_union_cannot_be_a_formal_test(tmp_path, monkeypatch):
    from pymicroglia.pipelines.rhythm.groups import comparison_values
    resolved, paths, _ = fixture(tmp_path, monkeypatch, tests=False, measurements=[{'column': 'signal', 'summary': 'median'}])
    values = comparison_values(resolved, {k: pd.read_csv(p) for k, p in paths.items()})
    assert values.comparison_summary.eq('median').all()
    assert values[values.movie.eq('a') & values.identity.eq(1)].comparison_usable.iloc[0] == 70
    bad = resolved.request.declaration.as_dict()
    bad['group_comparisons'] = {'tests': [{'name': 'bad', 'grouping_measurement': 'any-significant', 'metrics': ['signal'], 'test': 'paired_t', 'unit': 'subject', 'aggregate': 'mean'}]}
    with pytest.raises(ValueError, match='descriptive only'):
        parse([bad])
