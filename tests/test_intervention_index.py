from pymicroglia._results import document, output_files
"""Original population, exact figure/context membership and portable evidence."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import shutil
import pandas as pd
import pytest
import pymicroglia.pipelines.intervention.index as index, pymicroglia.pipelines.intervention.sample_figures as figures
from pymicroglia.pipelines.intervention.options import run_request, RECIPE, producers
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, Settings, PipelineRecipe, content_id, result_to_dict
from pymicroglia.pipelines._runner import SavedResult, run_pipeline
from pymicroglia.pipelines.audit.index import copy_evidence
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, write_table

def export(output, saved):
    output.mkdir()
    inventory = copy_evidence(saved, output)
    directory = output / 'execution-records'
    directory.mkdir()
    for name, value in saved.items():
        path = directory / (name + '.json')
        _write_json(path, result_to_dict(value.outcome))
        inventory[name].update(record=document(path).relative_to(output).as_posix(), record_sha256=file_hash(path))
    return inventory

def fixture(tmp_path):
    from tests.test_intervention_patterns import fixture, resolve
    request, tables, paths, _ = fixture(tmp_path)
    request['coordinated']['pairs'] = {'mode': 'explicit', 'pairs': [['first', 'second'], ['first', 'missing']]}
    resolved = resolve(request, tables, paths)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=['coordinated-responses'])
    assert execution.successful
    return (resolved, paths, dict(execution.results))

def registry(tmp_path, resolved, saved, wrong=False, view='sample_pairs'):
    context = SimpleNamespace(request=resolved, dependencies=saved, saved=saved.__getitem__)
    values, _, pages = figures.pages(figures.collect(context), {'views': ['sample_pairs', 'joint_outcomes'], 'rows_per_page': 3})
    page = next((p for p in pages if p['definition']['target'] == 'second' and p['view'] == view))
    page = {**page, 'master': 'saved-page.txt'}
    rows = _json_value(values.loc[values.entry_id.isin(page['entry_ids'])].to_dict('records'))
    for row in rows:
        row.update(master='saved-page.txt', view=page['view'])
    if wrong:
        next((r for r in rows if r['kind'] == 'sample_pair'))['reference_value'] = 123456.0
    root = tmp_path / 'registry'
    root.mkdir()
    (root / 'saved-page.txt').write_text('Saved semantic registry target')
    _write_json(root / 'intervention_display_manifest.json', {'pages': [page], 'settings': {}})
    _write_json(root / 'intervention_display_provenance.json', {'source_outcomes': {name: result_to_dict(value.outcome) for name, value in saved.items()}})
    write_table(root / 'entries.csv', pd.DataFrame(rows))
    for prefix in ['figure_data_', 'statistics_', 'der_display_']:
        (root / (prefix + 'saved-page.csv')).write_text('status\nsemantic registry fixture\n')
    sid = content_id('sample-figure-registry')
    refs = tuple((ArtifactRef(path.name, path.relative_to(root).as_posix(), file_hash(path), sid) for path in output_files(root)))
    saved['sample-and-pattern-figures'] = SavedResult(root, StepResult('sample-and-pattern-figures', sid, 'completed', 'Saved semantic registry fixture', refs))

def test_original_inventory_portability_and_displayed_versus_context_samples(tmp_path, monkeypatch):
    resolved, _, saved = fixture(tmp_path)
    registry(tmp_path, resolved, saved)

    def forbidden(*a, **k):
        raise AssertionError('Index recomputed analysis')
    import pymicroglia.pipelines.intervention.windows as intervention_windows, pymicroglia.pipelines.intervention.patterns as intervention_patterns, pymicroglia.pipelines.intervention.evidence as intervention_evidence
    monkeypatch.setattr(intervention_windows, 'prepare', forbidden)
    monkeypatch.setattr(intervention_patterns, 'analyse', forbidden)
    monkeypatch.setattr(intervention_evidence, 'analyse', forbidden)
    output = tmp_path / 'report'
    inventory = export(output, saved)
    nav = index.build(output, inventory, resolved.as_dict(), title='Controlled <test>')
    assert len(nav['cells']) == 10 and len(nav['recordings']) == 9 and (len(nav['samples']) == 8)
    assert len(nav['windows']) == 100 and len(nav['comparisons']) == len(nav['effects']) == 50
    assert len(nav['cell_pairs']) == 20 and len(nav['sample_pairs']) == 16 and (len(nav['associations']) == 2)
    assert len({r['id'] for r in nav['cells']}) == 10 and sum((r['record']['identity'] == 7 for r in nav['cells'])) == 9
    page = nav['pages'][0]
    assert sum((key.startswith('samples-') for key in page['displayed_targets'])) == 3
    assert sum((key.startswith('samples-') for key in page['context_targets'])) == 8
    first = next((r for r in nav['samples'] if r['record']['sample'] == 'sample-1'))
    assert len(first['record']['movies']) == 2
    assert nav['original_prepared_settings'] == resolved.as_dict() and (not nav['analysis_recomputed'])
    assert index.validate_links(output)['all_internal_links_resolve'] and '&lt;test&gt;' in (output / 'index.html').read_text(encoding='utf-8')
    moved = tmp_path / 'copied'
    shutil.copytree(output, moved)
    assert index.build(moved, inventory, resolved.as_dict()) == nav and index.validate_links(moved) == index.validate_links(output)
    target = moved / inventory['response-evidence']['artifacts']['effects']['path']
    target.write_text('changed')
    with pytest.raises(ValueError, match='changed index source'):
        index.build(moved, inventory, resolved.as_dict())

def test_changed_figure_value_cannot_keep_plausible_original_sample_id(tmp_path):
    resolved, _, saved = fixture(tmp_path)
    registry(tmp_path, resolved, saved, wrong=True)
    output = tmp_path / 'report'
    inventory = export(output, saved)
    with pytest.raises(ValueError, match='Figure changed original sample_pairs meaning'):
        index.build(output, inventory, resolved.as_dict())

def test_mixed_table_keeps_numeric_local_ids_with_their_original_recordings(tmp_path):
    resolved, _, saved = fixture(tmp_path)
    registry(tmp_path, resolved, saved, view='joint_outcomes')
    output = tmp_path / 'report'
    inventory = export(output, saved)
    nav = index.build(output, inventory, resolved.as_dict())
    page = nav['pages'][0]
    cells = [r['record'] for r in nav['cells'] if r['id'] in page['displayed_targets']]
    assert len(cells) == 3 and len({(r['movie'], r['identity']) for r in cells}) == 3
    assert all((r['identity'] == 7 for r in cells)) and index.validate_links(output)['all_internal_links_resolve']

def test_unavailable_preparation_preserves_declared_cells_and_windows(tmp_path):
    from tests.test_intervention_evidence import fixture
    _, _, _, resolved = fixture(tmp_path)
    saved = {'aligned-windows': SavedResult(tmp_path / 'absent', StepResult('aligned-windows', 'absent', 'unavailable', 'Original preparation capability unavailable'))}
    output = tmp_path / 'report'
    inventory = export(output, saved)
    nav = index.build(output, inventory, resolved.as_dict())
    assert len(nav['cells']) == 4 and len(nav['declared_windows']) == 16 and (not nav['effects'])
    assert all((row['record']['status'] == 'not_prepared' for row in nav['declared_windows']))
    assert index.validate_links(output)['all_internal_links_resolve']

def test_failed_paired_branch_keeps_every_requested_pair(tmp_path):
    resolved, _, saved = fixture(tmp_path)
    saved['coordinated-responses'] = SavedResult(tmp_path / 'absent', StepResult('coordinated-responses', 'failed-pairs', 'failed', 'Controlled original paired-analysis failure'))
    output = tmp_path / 'report'
    inventory = export(output, saved)
    nav = index.build(output, inventory, resolved.as_dict())
    assert len(nav['declared_pairs']) == 20 and len(nav['cell_pairs']) == 0
    assert {r['record']['target'] for r in nav['declared_pairs']} == {'second', 'missing'}
    assert index.validate_links(output)['all_internal_links_resolve']

def test_native_numeric_reference_is_a_fitted_value_not_a_measurement_name(tmp_path):
    from tests.test_intervention_rhythms import fixture
    _, _, paths, resolved = fixture(tmp_path, cases=['period'])
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=['rhythm-changes'])
    assert execution.successful
    output = tmp_path / 'report'
    inventory = export(output, execution.results)
    nav = index.build(output, inventory, resolved.as_dict())
    assert len(nav['rhythm_windows']) == 2 and len(nav['direct_changes']) == 3
    changed = next((row for row in nav['direct_changes'] if row['record']['property'] == 'period'))
    assert changed['record']['change_supported'] and isinstance(changed['record']['reference'], float)
    assert index.validate_links(output)['all_internal_links_resolve']
    html = (output / 'index.html').read_text(encoding='utf-8')
    assert 'Window comparison' in html and 'Direct rhythm change' in html and ('period hours' in html)

def test_failed_science_index_does_not_make_execution_successful(tmp_path, monkeypatch):
    from tests.test_intervention_evidence import fixture
    import pymicroglia.pipelines.intervention.windows as intervention_windows
    _, _, paths, resolved = fixture(tmp_path)

    def failed(*a, **k):
        raise RuntimeError('Controlled scientific producer failure')
    monkeypatch.setattr(intervention_windows, 'prepare', failed)
    scientific = tuple((step for step in RECIPE.steps if step.kind != 'render'))
    report = replace(RECIPE.steps[-1], prerequisites=tuple((step.name for step in scientific)))
    recipe = PipelineRecipe(RECIPE.name, RECIPE.version, (*scientific, report))
    result = run_pipeline(recipe, producers(), request=resolved, scientific_settings={'intervention': resolved.scientific_id}, table_paths=paths, output=tmp_path / 'pipeline')
    assert not result.successful and result.results['aligned-windows'].outcome.status == 'failed'
    saved = result.results['linked-results-index']
    assert saved.outcome.status == 'completed'
    assert 'Controlled scientific producer failure' in saved.artifact('index.html').read_text(encoding='utf-8')
