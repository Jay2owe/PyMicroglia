from pymicroglia._results import document, output_files
"""Portable links and semantic targets over actual saved no-inference results."""
from pathlib import Path
from copy import deepcopy
import shutil
import pandas as pd
import pytest
from tests.test_coordination_options import request
from pymicroglia.pipelines.coordination.options import resolve_request, run_request
from pymicroglia.pipelines.coordination.index import build, validate_links
from pymicroglia.pipelines.coordination.evidence import read_evidence
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id, result_to_dict
from pymicroglia.pipelines._runner import SavedResult
from pymicroglia.pipelines.audit.index import copy_evidence
from pymicroglia.pipelines._screening import _write_json, file_hash, write_table

def fixture(tmp_path):
    frame = pd.DataFrame([{'stem': movie, 'identity': cell, 'frame_index': i, 'hours': 50.0 + i * 0.5, 'custom_signal': float(cell + i), 'custom_shape': float(100 - cell - i), 'centroid_x': cell * 4.0, 'centroid_y': 0.0} for movie in ['one', 'two'] for cell in [7, 8] for i in range(8)])
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    paths = {}
    for name, table in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        table.to_csv(paths[name], index=False)
    resolved = resolve_request(request(), source_run='original-source', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=['coordination-evidence', 'coordination-samples'])
    assert execution.successful
    return (resolved, dict(execution.results))

def export(tmp_path, resolved, saved):
    tmp_path.mkdir()
    inventory = copy_evidence(saved, tmp_path)
    records = tmp_path / 'execution-records'
    records.mkdir()
    for name, result in saved.items():
        path = records / (name + '.json')
        _write_json(path, result_to_dict(result.outcome))
        inventory[name].update(record=document(path).relative_to(tmp_path).as_posix(), record_sha256=file_hash(path))
    return inventory

def add_controlled_registry(tmp_path, saved, wrong=False):
    source = read_evidence(saved['coordination-evidence'])
    entry = source['effects'].iloc[0].to_dict()
    entry['entry_id'] = 'controlled-entry'
    entry['master'] = 'saved-render-target.txt'
    entry['view'] = 'effects'
    if wrong:
        entry['reference_endpoint_id'] = 'different-original-measurement-endpoint'
    folder = tmp_path / 'registry'
    folder.mkdir()
    (folder / 'saved-render-target.txt').write_text('Controlled saved render target')
    manifest = {'pages': [{'master': 'saved-render-target.txt', 'title': 'Controlled saved registry', 'view': 'effects', 'entry_ids': ['controlled-entry']}], 'settings': {}}
    _write_json(folder / 'coordination_display_manifest.json', manifest)
    _write_json(folder / 'coordination_display_provenance.json', {'evidence_id': saved['coordination-evidence'].outcome.scientific_id, 'pair_inputs_id': saved['pair-inputs'].outcome.scientific_id})
    write_table(folder / 'entries.csv', pd.DataFrame([entry]))
    for prefix in ['figure_data_', 'statistics_', 'der_display_']:
        (folder / (prefix + 'saved-render-target.csv')).write_text('status\ncontrolled registry fixture\n')
    sid = content_id('controlled-render-registry')
    refs = tuple((ArtifactRef(path.name, path.relative_to(folder).as_posix(), file_hash(path), sid) for path in output_files(folder)))
    saved['coordination-overview'] = SavedResult(folder, StepResult('coordination-overview', sid, 'completed', 'Controlled saved registry fixture', refs))

def test_no_support_keeps_full_pairs_samples_and_portable_negative_results(tmp_path):
    resolved, saved = fixture(tmp_path)
    output = tmp_path / 'report'
    inventory = export(output, resolved, saved)
    navigation = build(output, inventory, resolved.as_dict(), title='Controlled <test> title')
    assert len(navigation['pairs']) == 6 and len(navigation['cells']) == 4 and (len(navigation['recordings']) == 2)
    assert len(navigation['samples']) == 2 and all((not row['sample_confirmed'] for row in navigation['samples']))
    assert all((not row['decision']['supported_effect_ids'] for row in navigation['pairs'])) and (not navigation['pages'])
    assert len({row['id'] for row in navigation['pairs']}) == 6
    assert validate_links(output)['all_internal_links_resolve']
    moved = tmp_path / 'moved'
    shutil.copytree(output, moved)
    assert validate_links(moved) == validate_links(output)
    assert '&lt;test&gt;' in (output / 'index.html').read_text()

def test_registry_resolves_full_measurement_effect_and_rejects_plausible_wrong_endpoint(tmp_path):
    resolved, saved = fixture(tmp_path)
    add_controlled_registry(tmp_path, saved)
    output = tmp_path / 'report'
    inventory = export(output, resolved, saved)
    navigation = build(output, inventory, resolved.as_dict())
    assert len(navigation['pages']) == 1 and len(navigation['pages'][0]['pairs']) == 1
    page = navigation['pages'][0]
    pair = next((row for row in navigation['pairs'] if row['id'] == page['pairs'][0]))
    assert page['id'] in pair['pages'] and pair['effects'][0] in page['effects']
    assert validate_links(output)['all_internal_links_resolve']
    entries = output / inventory['coordination-overview']['artifacts']['entries.csv']['path']
    from pymicroglia.pipelines._screening import read_table
    frame = read_table(entries)
    frame['reference_endpoint_id'] = 'another-plausible-measurement'
    entries = write_table(entries, frame)
    inventory['coordination-overview']['artifacts']['entries.csv']['sha256'] = file_hash(entries)
    with pytest.raises(ValueError, match='swapped its original pair endpoints'):
        build(output, inventory, resolved.as_dict())

def test_unavailable_preparation_remains_an_honest_requested_inventory(tmp_path):
    resolved, saved = fixture(tmp_path)
    unavailable = SavedResult(tmp_path / 'absent', StepResult('pair-inputs', 'unavailable-inputs', 'unavailable', 'Recorded source capability is unavailable'))
    saved = {'pair-inputs': unavailable}
    output = tmp_path / 'report'
    inventory = export(output, resolved, saved)
    navigation = build(output, inventory, resolved.as_dict())
    assert len(navigation['cells']) == 4 and len(navigation['samples']) == 2 and (navigation['pairs'] == [])
    assert validate_links(output)['all_internal_links_resolve']
    assert 'Recorded source capability is unavailable' in (output / 'index.html').read_text()
