"""Portable semantic navigation and failure reporting use only saved evidence."""
from pymicroglia._results import read_document
import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit
import pandas as pd
import pytest
import pymicroglia.pipelines.rhythm.index as rhythm_index
from pymicroglia.pipelines._contracts import ArtifactRef, CellKey, PipelineRecipe, SelectionRecord, Settings, StepResult, StepSpec, content_id
from pymicroglia.pipelines._runner import ExecutionContext, Producer, SavedResult, dependency_order, run_pipeline
from pymicroglia.pipelines._screening import _write_json, file_hash, write_table

class Links(HTMLParser):

    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.links.extend((attrs[k] for k in ('href', 'src') if k in attrs))
        if 'id' in attrs:
            self.ids.append(attrs['id'])

def check_links(output):
    parser = Links()
    parser.feed((output / 'index.html').read_text(encoding='utf-8'))
    assert len(parser.ids) == len(set(parser.ids))
    for link in parser.links:
        parsed = urlsplit(link)
        assert not parsed.scheme and (not parsed.netloc)
        if parsed.path:
            path = (output / unquote(parsed.path)).resolve()
            assert path.is_relative_to(output.resolve()) and path.is_file(), link
        if parsed.fragment:
            assert unquote(parsed.fragment) in parser.ids
    return len(parser.links)

def saved(tmp_path, step, files, *, selections=(), status='completed', reason='Saved controlled records'):
    root = tmp_path / step
    root.mkdir()
    refs = []
    for name, value in files.items():
        path = root / name
        if isinstance(value, pd.DataFrame):
            path = write_table(path, value)
        elif isinstance(value, dict):
            path = _write_json(path, value)
        else:
            path.write_text(value, encoding='utf-8')
        refs.append(ArtifactRef(name, str(path.relative_to(root)), file_hash(path), 'science'))
    return SavedResult(root, StepResult(step, 'science', status, reason, tuple(refs), selections))

def controlled_inventory(tmp_path):
    cells = [{'source_run': 'source & <literal>', 'movie': movie, 'identity': 7} for movie in ('a <tag>', 'b space')]
    pair = {'pair_id': 'pair-evidence', 'reference': 'speed <x>', 'target': 'area & y'}
    request = {'inputs': {'cells': cells}, 'test_measurements': [{'column': pair['reference']}, {'column': pair['target']}], 'request': {'pairs': [{k: pair[k] for k in ('reference', 'target')}]}}
    selection = SelectionRecord('any-significant', 'science', Settings({'rule': 'saved'}), tuple((CellKey(**row) for row in cells)))
    outcomes = pd.DataFrame([{**cell, 'measurement': metric, 'status': 'significant', 'significant': True, 'method': 'saved estimator', 'significance_method': 'saved test', 'family_id': 'original'} for cell in cells for metric in (pair['reference'], pair['target'])])
    deps = {'rhythm-screen': saved(tmp_path, 'rhythm-screen', {'provenance': {'resolved_request': request}, 'rhythm_results': outcomes}, selections=(selection,))}
    deps['selected-cell-evidence'] = saved(tmp_path, 'selected-cell-evidence', {'evidence_manifest.json': {'pages': [{'kind': 'reports', **cell, 'master': f'card {i}.txt'} for i, cell in enumerate(cells)]}, 'card 0.txt': 'navigation test stand-in', 'card 1.txt': 'navigation test stand-in'})
    deps['detection-agreement'] = saved(tmp_path, 'detection-agreement', {'summary': pd.DataFrame([{**pair, 'both': 2}])})
    deps['timing-relationship-figures'] = saved(tmp_path, 'timing-relationship-figures', {'relationship_manifest.json': {'pages': [{'kind': 'pair-traces', **pair, 'cells': cells, 'master': 'pair.txt'}]}, 'pair.txt': 'navigation test stand-in'})
    deps['time-matrices'] = saved(tmp_path, 'time-matrices', {}, status='unavailable', reason='Declared trace representation has no saved values')
    return (deps, cells)

def test_exact_cells_pairs_escaping_moved_folder_and_no_analysis(tmp_path, monkeypatch):
    import pymicroglia.workbench as circadian

    def forbidden(*a, **k):
        raise AssertionError('Index attempted scientific calculations')
    for name in ('estimate_one', 'rhythm_pair_timing', 'rhythm_timing_summary', 'adjust_pvalues'):
        monkeypatch.setattr(circadian, name, forbidden)
    deps, cells = controlled_inventory(tmp_path)
    output = tmp_path / 'report'
    ctx = ExecutionContext(StepSpec('linked-results-index', 'index', tuple(deps), kind='render'), None, Settings(), {}, deps, None, output, 'report-id', Settings({'report': {'title': 'Saved <script>alert(1)</script> & evidence'}}), 'display')
    outcome = rhythm_index.produce(ctx)
    assert outcome.status == 'completed'
    report = SavedResult(output, outcome)
    short = rhythm_index.openable_report(report, directory=tmp_path / 'short')
    assert file_hash(short) == file_hash(report.artifact('index.html'))
    assert check_links(short.parent) > 20
    assert rhythm_index.openable_report(report, directory=tmp_path / 'short') == short
    if short.parent != output:
        short.write_text('User-edited browser copy', encoding='utf-8')
        replacement = rhythm_index.openable_report(report, directory=tmp_path / 'short')
        assert replacement != short and short.read_text() == 'User-edited browser copy'
        assert file_hash(replacement) == file_hash(report.artifact('index.html'))
    navigation = read_document(output / 'navigation.json')
    assert len(navigation['cells']) == 2 and len(navigation['pairs']) == 1
    assert len({cell['id'] for cell in navigation['cells']}) == 2
    by_page = {page['id']: page for page in navigation['pages']}
    for cell in navigation['cells']:
        assert len(cell['report_pages']) == 1
        assert by_page[cell['report_pages'][0]]['semantics']['movie'] == cell['movie']
    pair = navigation['pairs'][0]
    assert pair['reference'] == 'speed <x>' and len(pair['report_pages']) == 1
    assert by_page[pair['report_pages'][0]]['cells'] == [rhythm_index.cell_id(cell) for cell in cells]
    document = (output / 'index.html').read_text(encoding='utf-8')
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in document and '<script>alert(1)</script>' not in document
    assert 'Declared trace representation has no saved values' in document
    assert check_links(output) > 20
    moved = tmp_path / 'copied elsewhere'
    shutil.copytree(output, moved)
    assert check_links(moved) == check_links(output)
    original = deps['selected-cell-evidence'].artifact('card 0.txt')
    assert file_hash(moved / 'evidence/selected-cell-evidence/card 0.txt') == file_hash(original)
    record = read_document(moved / 'execution-records/rhythm-screen.json')
    assert len(record['selections'][0]['members']) == 2
    current = {**navigation['resolved_settings'], 'comparison_measurements': [{'column': 'new independent comparison'}]}
    changed = rhythm_index.build(moved, navigation['inputs'], requested=current)
    assert changed['resolved_settings']['comparison_measurements'][0]['column'] == 'new independent comparison'
    assert 'comparison_measurements' not in changed['screen_resolved_settings']
    (moved / 'evidence/selected-cell-evidence/card 0.txt').write_text('changed')
    with pytest.raises(ValueError, match='missing or changed'):
        rhythm_index.build(moved, navigation['inputs'])

def failed(context):
    raise RuntimeError('Controlled required calculation failure')

def test_index_runs_after_a_failure_without_certifying_the_pipeline(tmp_path):
    from types import SimpleNamespace
    recipe = PipelineRecipe('failure-report', 1, (StepSpec('rhythm-screen', 'failed'), StepSpec('linked-results-index', 'index', ('rhythm-screen',), kind='render')))
    producers = {'failed': Producer(failed), 'index': Producer(rhythm_index.produce, accepts_unavailable_dependencies=True)}
    request = {'inputs': {'cells': [{'source_run': 'failed-source', 'movie': 'a', 'identity': 1}]}, 'test_measurements': [{'column': 'a'}, {'column': 'b'}], 'request': {'pairs': [{'reference': 'a', 'target': 'b'}]}}
    result = run_pipeline(recipe, producers, request=SimpleNamespace(as_dict=lambda: request), scientific_settings={}, output=tmp_path)
    assert not result.successful
    assert result.results['rhythm-screen'].outcome.status == 'failed'
    report = result.results['linked-results-index']
    assert report.outcome.status == 'completed', report.outcome.reason
    assert 'Controlled required calculation failure' in report.artifact('index.html').read_text(encoding='utf-8')
    assert check_links(report.root) > 5
    navigation = read_document(report.artifact('navigation.json'))
    assert len(navigation['cells']) == len(navigation['pairs']) == 1
    invalid = PipelineRecipe('invalid', 1, (StepSpec('science', 'index'),))
    with pytest.raises(ValueError, match='only a render'):
        dependency_order(invalid, producers)
