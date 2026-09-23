"""Portable relationship navigation rejects real-but-wrong semantic targets."""
from pymicroglia._results import read_document
from dataclasses import replace
import copy
import json
import shutil
from types import SimpleNamespace
import pandas as pd
import pytest
import pymicroglia.pipelines.relationships.index as index
from pymicroglia.pipelines._contracts import Settings, StepSpec, content_id
from pymicroglia.pipelines._runner import ExecutionContext, SavedResult
from pymicroglia.pipelines._screening import file_hash, read_table, write_table
from tests.test_rhythm_index import saved as original_saved, check_links

def saved(tmp_path, step, files, **kwargs):
    value = original_saved(tmp_path, step, files, **kwargs)
    sid = content_id(step)
    return SavedResult(value.root, replace(value.outcome, scientific_id=sid, artifacts=tuple((replace(ref, scientific_id=sid) for ref in value.outcome.artifacts))))

def fixture(tmp_path):
    cells = [{'source_run': 'source <literal>', 'movie': movie, 'identity': 7} for movie in ('a <tag>', 'b & space')]
    pair = {'reference': 'signal <x>', 'target': 'area & y'}
    pid = content_id(pair)
    keys = [{**cell, 'pair_id': pid, **pair} for cell in cells]
    request = {'inputs': {'source_run': cells[0]['source_run'], 'cells': cells}, 'request': {'pairs': [pair], 'representation': 'raw'}}
    outcomes = pd.DataFrame([{**key, 'status': 'positive-association', 'reason': 'Saved evidence', 'representation': 'raw'} for key in keys])
    deps = {'paired-inputs': saved(tmp_path, 'paired-inputs', {'provenance': {'resolved_request': request}}), 'within-cell-association': saved(tmp_path, 'within-cell-association', {'results': outcomes}), 'lag-association': saved(tmp_path, 'lag-association', {'results': outcomes}), 'between-cell-association': saved(tmp_path, 'between-cell-association', {'results': pd.DataFrame([{'source_run': cells[0]['source_run'], 'pair_id': pid, **pair}])}), 'sample-consistency': saved(tmp_path, 'sample-consistency', {'summaries': pd.DataFrame([{'source_run': cells[0]['source_run'], 'pair_id': pid, **pair, 'question': 'within_cell', 'level': 'all_cells'}])}), 'relationship-report-selection': saved(tmp_path, 'relationship-report-selection', {'report_members': pd.DataFrame(keys)})}
    overview = []
    overview_files = {}
    overview_pages = []
    for view in ('within_cell', 'between_cells', 'delay'):
        master = view + '.txt'
        overview_files[master] = 'Navigation test stand-in; no chart generated'
        overview_pages.append({'master': master, 'view': view, 'rows': [pair['reference']], 'columns': [pair['target']]})
        overview.append({'source_run': cells[0]['source_run'], 'pair_id': pid, **pair, 'view': view, 'value': 0.4, 'status': 'descriptive', 'members': cells, 'master': master, 'entry_id': content_id(view), 'source_scientific_id': deps['between-cell-association' if view == 'between_cells' else 'sample-consistency'].outcome.scientific_id})
    deps['relationship-overview'] = saved(tmp_path, 'relationship-overview', {**overview_files, 'relationship_overview_manifest.json': {'pages': overview_pages}, 'entries.json': pd.DataFrame(overview)})
    for step, manifest, view, source in [('relationship-reports', 'relationship_reports_manifest.json', 'cells', 'relationship-report-selection'), ('relationship-lag-profiles', 'relationship_lag_profiles_manifest.json', 'lag-profile', 'lag-association')]:
        entries = []
        pages = []
        files = {}
        for i, key in enumerate(keys):
            master = f'cell {i}.txt'
            files[master] = 'Navigation test stand-in; no chart generated'
            page = {'master': master, 'members': [key]}
            if step == 'relationship-reports':
                page['view'] = view
            pages.append(page)
            entries.append({**key, 'view': view, 'master': master, 'entry_id': content_id({'step': step, **key}), 'selection_id' if step == 'relationship-reports' else 'source_scientific_id': deps[source].outcome.scientific_id})
        deps[step] = saved(tmp_path, step, {**files, manifest: {'pages': pages}, 'entries.json': pd.DataFrame(entries)})
    deps['relationship-populations'] = saved(tmp_path, 'relationship-populations', {'population.txt': 'Navigation test stand-in; no chart generated', 'relationship_populations_manifest.json': {'pages': [{'master': 'population.txt', 'view': 'within_cell', 'pair_id': pid, 'groups': ['sample:s0']}]}, 'entries.json': pd.DataFrame([{'source_run': cells[0]['source_run'], 'pair_id': pid, **pair, 'view': 'within_cell', 'group_id': 'sample:s0', 'source_scientific_id': deps['sample-consistency'].outcome.scientific_id, 'entry_id': content_id('population'), 'master': 'population.txt', 'members': keys}])})
    return (deps, request)

def context(tmp_path, deps, request):
    return ExecutionContext(StepSpec('linked-results-index', 'index', tuple(deps), kind='render'), SimpleNamespace(as_dict=lambda: request), Settings(), {}, deps, None, tmp_path / 'report', 'index-science', Settings({'report': {'title': 'Saved <script>alert(1)</script> & evidence'}}), 'presentation')

def test_full_keys_links_moving_escaping_and_immutable_sources(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    import pymicroglia.measure.relationship_statistics as relationship_statistics
    import pymicroglia.measure.relationship_lag_statistics as relationship_lag_statistics

    def forbidden(*a, **k):
        raise AssertionError('Index attempted scientific calculations')
    for obj, name in ((circadian, 'estimate_one'), (circadian, 'detrend_trace'), (circadian, 'adjust_pvalues'), (relationship_statistics, 'same_time_evidence'), (relationship_lag_statistics, 'evaluate')):
        monkeypatch.setattr(obj, name, forbidden)
    deps, request = fixture(tmp_path)
    before = {str(p): file_hash(p) for s in deps.values() for p in s.root.rglob("*") if p.is_file()}
    ctx = context(tmp_path, deps, request)
    outcome = index.produce(ctx)
    assert outcome.status == 'completed'
    assert check_links(ctx.output) > 50
    navigation = read_document(ctx.output / 'navigation.json')
    assert len(navigation['cells']) == 2 and len(navigation['pairs']) == 1 and (len(navigation['pages']) == 8)
    assert len(set((c['id'] for c in navigation['cells']))) == 2
    reports = [e for e in navigation['entries'] if e['step'] == 'relationship-reports']
    assert len(set((e['path'] for e in reports))) == 2 and all((len(e['cells']) == 1 for e in reports))
    assert {str(p): file_hash(p) for s in deps.values() for p in s.root.rglob("*") if p.is_file()} == before
    html = (ctx.output / 'index.html').read_text()
    assert '<script>alert(1)</script>' not in html and '&lt;script&gt;' in html
    moved = tmp_path / 'moved report'
    shutil.copytree(ctx.output, moved)
    assert check_links(moved) == check_links(ctx.output)

@pytest.mark.parametrize('change', ['existing_wrong_page', 'reversed_pair', 'wrong_science', 'wrong_representation'])
def test_rejects_semantically_wrong_existing_destinations(tmp_path, change):
    deps, request = fixture(tmp_path)
    step = 'relationship-reports' if change == 'existing_wrong_page' else 'relationship-overview'
    source = deps[step]
    path = source.artifact('entries.json')
    frame = read_table(path)
    if change == 'existing_wrong_page':
        frame.loc[1, 'master'] = frame.loc[0, 'master']
    if change == 'reversed_pair':
        frame.loc[0, ['reference', 'target']] = frame.loc[0, ['target', 'reference']].tolist()
    if change == 'wrong_science':
        frame.loc[0, 'source_scientific_id'] = 'different-result'
    if change == 'wrong_representation':
        frame['representation'] = 'detrended'
    path = write_table(path, frame)
    deps[step] = SavedResult(source.root, replace(source.outcome, artifacts=tuple((replace(ref, sha256=file_hash(path)) if ref.name == 'entries.json' else ref for ref in source.outcome.artifacts))))
    with pytest.raises(ValueError):
        index.produce(context(tmp_path, deps, request))

@pytest.mark.parametrize('status', ['skipped-empty', 'unavailable', 'failed'])
def test_missing_render_branch_has_no_placeholder_links(tmp_path, status):
    deps, request = fixture(tmp_path)
    source = deps['relationship-reports']
    deps['relationship-reports'] = SavedResult(source.root, replace(source.outcome, status=status, reason='Exact saved branch explanation', artifacts=()))
    ctx = context(tmp_path, deps, request)
    index.produce(ctx)
    assert check_links(ctx.output) > 30
    navigation = read_document(ctx.output / 'navigation.json')
    assert not any((page['step'] == 'relationship-reports' for page in navigation['pages']))
    html = (ctx.output / 'index.html').read_text()
    assert status in html and 'Exact saved branch explanation' in html
    assert ('Some requested branches are unfinished' in html) == (status in {'failed', 'unavailable'})


def test_current_csv_entry_names_preserve_exact_report_membership(tmp_path):
    deps, request = fixture(tmp_path)
    for step, saved_result in list(deps.items()):
        refs = tuple(replace(ref, name="entries.csv") if ref.name == "entries.json" else ref
                     for ref in saved_result.outcome.artifacts)
        deps[step] = SavedResult(saved_result.root, replace(saved_result.outcome, artifacts=refs))
    ctx = context(tmp_path, deps, request)
    index.produce(ctx)
    navigation = read_document(ctx.output / "navigation.json")
    assert len(navigation["entries"]) == 8
    assert all(page["entries"] for page in navigation["pages"])
