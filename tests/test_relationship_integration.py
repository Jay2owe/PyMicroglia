"""Relationship recipe integration: scientific identities, recovery and honest failure."""
from pymicroglia._results import read_document
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pymicroglia.pipelines import parse
import tests.pipelines.relationships.demo as demo
import pymicroglia.pipelines.relationships.options as options
from pymicroglia.pipelines._contracts import PipelineRecipe
from pymicroglia.pipelines._runner import Producer, Unavailable, run_pipeline
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_relationship_inputs import table
from tests.test_relationship_options import declaration, lag

def inputs(tmp_path, **changes):
    frame = table(50 + np.arange(80) * 0.5, np.arange(80), np.arange(80)[::-1])
    path = tmp_path / 'cell_frame.csv'
    frame.to_csv(path, index=False)
    block = declaration(lag=lag(), **changes)
    return (block, frame, {'cell_frame': path})

def resolve(block, frame, paths):
    return options.resolve_request(parse([block])[0], source_run='controlled-source', tables={'cell_frame': frame}, input_hashes={key: file_hash(path) for key, path in paths.items()})

def test_public_example_is_reproducible_and_preserves_original_population(tmp_path):
    first, second = (demo.prepare(tmp_path / 'one'), demo.prepare(tmp_path / 'two'))
    assert read_document(first / 'fixture.json') == read_document(second / 'fixture.json')
    resolved, paths = demo.resolve(first)
    assert len(resolved.inputs.cells) == 7 and len(resolved.request.pairs) == 1
    assert resolved.request.biological_samples['a'] == resolved.request.biological_samples['b']
    execution = options.run_request(resolved, paths, tmp_path / 'prepared', only=('paired-inputs',))
    assert execution.successful
    traces = read_table(execution.results['paired-inputs'].artifact('traces'))
    assert traces.hours.min() == 50 and traces.hours.max() == 649.5
    assert traces.raw_kind.eq('missing').any()
    with pytest.raises(ValueError, match='new empty folder'):
        demo.prepare(first)
    example = json.loads(__import__('importlib.resources',fromlist=['files']).files('pymicroglia').joinpath('data/relationships.example.json').read_text(encoding='utf-8'))
    request = parse([example])[0]
    assert request.within_cell['evidence']['method'] == 'none' and (not request.lag['enabled'])

@pytest.mark.parametrize('constant', [False, True])
def test_real_command_records_no_selected_reports_and_disabled_lag(tmp_path, constant):
    from argparse import Namespace
    from pymicroglia.pipelines.rhythm.discovery import command
    from tests.pipelines.audit.demo import _latest
    source = tmp_path / 'run'
    tables = source / 'pooled/tables'
    tables.mkdir(parents=True)
    x = np.zeros(80) if constant else np.arange(80)
    frame = table(50 + np.arange(80) * 0.5, x, np.arange(80)[::-1])
    frame.to_csv(tables / 'cell_frame.csv', index=False)
    (source / 'manifest.json').write_text(json.dumps({'synthetic': True, 'movies': [{'stem': 'a', 'modules': []}]}))
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(declaration(name='no-selected-example')))
    args = Namespace(run=str(source), request=str(path), config=None, name=None, presentation=None, out=str(tmp_path / 'pipeline'), step=['relationship-reports'])
    assert command(args) == 0
    _, saved = _latest(tmp_path / 'pipeline/no-selected-example')
    assert saved['relationship-reports'].outcome.status == 'skipped-empty'
    assert read_table(saved['lag-association'].artifact('results')).status.eq('disabled').all()
    assert read_table(saved['within-cell-association'].artifact('results')).status.eq('untestable' if constant else 'descriptive').all()
    assert read_table(saved['relationship-report-selection'].artifact('report_members')).empty

@pytest.mark.parametrize('change', ['input', 'window', 'processing', 'direction', 'lag_grid', 'method', 'sample'])
def test_actual_science_cache_changes_with_scientific_meaning(tmp_path, change):
    block, frame, paths = inputs(tmp_path)
    original = resolve(block, frame, paths)
    first = options.run_request(original, paths, tmp_path / 'pipeline', only=('paired-inputs',))
    assert first.successful
    if change == 'input':
        frame.loc[0, 'corrected_mean'] = 123.0
        frame.to_csv(paths['cell_frame'], index=False)
    elif change == 'window':
        block['time_range_hours'] = [51, 85]
    elif change == 'processing':
        block.update(representation='detrended', detrending={'detrend': 'linear'})
    elif change == 'direction':
        block['pairs'] = {'mode': 'explicit', 'pairs': [['area_px', 'corrected_mean']]}
    elif change == 'lag_grid':
        block['lag']['resolution_hours'] = 1.0
    elif change == 'method':
        block['within_cell']['statistic'] = 'spearman'
    elif change == 'sample':
        block['biological_samples'] = {'a': 'confirmed-sample'}
    changed = resolve(block, frame, paths)
    second = options.run_request(changed, paths, tmp_path / 'pipeline', only=('paired-inputs',))
    assert second.successful, {k: v.outcome.reason for k, v in second.results.items()}
    assert original.scientific_id != changed.scientific_id
    assert first.results['paired-inputs'].root != second.results['paired-inputs'].root
    assert second.results['paired-inputs'].outcome.status == 'completed'
    reused = options.run_request(changed, paths, tmp_path / 'pipeline', presentation={'irrelevant_display': True}, only=('paired-inputs',))
    assert reused.successful and reused.results['paired-inputs'].outcome.status == 'reused'

@pytest.mark.parametrize('damage', ['corrupt', 'missing', 'unfinished'])
def test_invalid_completion_is_recomputed_without_overwriting_the_old_result(tmp_path, damage):
    block, frame, paths = inputs(tmp_path)
    resolved = resolve(block, frame, paths)
    first = options.run_request(resolved, paths, tmp_path / 'pipeline', only=('within-cell-association',))
    original = first.results['within-cell-association']
    path = original.artifact('results')
    if damage == 'corrupt':
        path.write_text('damaged controlled artifact')
    elif damage == 'missing':
        path.unlink()
    else:
        from pymicroglia.pipelines import _records
        ledger, key, _ = _records.locate(original.root)
        from pymicroglia._results import read_document, write_document
        records = read_document(ledger)
        del records['completions'][key]
        write_document(ledger, records)
    second = options.run_request(resolved, paths, tmp_path / 'pipeline', only=('within-cell-association',))
    assert second.successful
    repaired = second.results['within-cell-association']
    assert repaired.outcome.status == 'completed' and repaired.root != original.root
    assert read_table(repaired.artifact('results')).status.eq('descriptive').all()
    if damage == 'corrupt':
        assert path.read_text() == 'damaged controlled artifact'
    if damage == 'missing':
        assert not path.exists()

@pytest.mark.parametrize('failure', ['failed', 'unavailable'])
def test_failed_science_leaves_a_readable_index_without_empty_selection_claim(tmp_path, failure):
    block, frame, paths = inputs(tmp_path)
    resolved = resolve(block, frame, paths)
    names = ('relationship-design', 'paired-inputs', 'within-cell-association', 'lag-association', 'relationship-report-selection')
    steps = [step for step in options.RECIPE.steps if step.name in names]
    index = next((step for step in options.RECIPE.steps if step.name == 'linked-results-index'))
    recipe = PipelineRecipe('relationship-failure-verification', 1, (*steps, replace(index, prerequisites=names)))
    registry = options.producers()

    def fail(context):
        if failure == 'unavailable':
            raise Unavailable('Controlled missing scientific capability')
        raise RuntimeError('Controlled scientific operation failure')
    registry['within-cell-association'] = Producer(fail, version='controlled-failure')
    execution = run_pipeline(recipe, registry, request=resolved, scientific_settings={'relationships': resolved.scientific_id}, table_paths=paths, output=tmp_path / 'pipeline')
    assert not execution.successful and execution.results['within-cell-association'].outcome.status == failure
    assert execution.results['lag-association'].outcome.status == 'completed'
    assert execution.results['relationship-report-selection'].outcome.status != 'skipped-empty'
    report = execution.results['linked-results-index']
    assert report.outcome.status == 'completed'
    html = report.artifact('index.html').read_text(encoding='utf-8')
    assert 'Some requested branches are unfinished' in html and 'Controlled' in html
    navigation = read_document(report.artifact('navigation.json'))
    assert len(navigation['cells']) == 1 and (not navigation['pages'])
