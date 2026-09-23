"""Public generated inputs preserve original units through scientific handoffs."""
from pymicroglia._results import read_document
from copy import deepcopy
from pathlib import Path
import json
import pandas as pd
from pymicroglia.pipelines import parse
import tests.pipelines.intervention.demo as demo
import pymicroglia.pipelines.intervention.windows as intervention_windows, pymicroglia.pipelines.intervention.evidence as intervention_evidence, pymicroglia.pipelines.intervention.controls as intervention_controls, pymicroglia.pipelines.intervention.patterns as intervention_patterns
from pymicroglia.pipelines.intervention.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash

def test_native_public_response_example_complete_families_units_and_reuse(tmp_path, monkeypatch):
    frame, clock, request, sources, truth = demo.response_inputs()
    tables = {'cell_frame': frame, 'frame_summary': clock, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    paths = {name: tmp_path / (name + '.csv') for name in tables}
    for name, table in tables.items():
        table.to_csv(paths[name], index=False)
    resolved = resolve_request(parse([request])[0], source_run='native-generated-software', tables=tables, input_hashes={key: file_hash(path) for key, path in paths.items()})
    assert sources and len(resolved.inputs.cells) == truth['cells'] == 19
    execution = run_request(resolved, paths, tmp_path / 'results', only=['control-comparisons', 'coordinated-responses'])
    assert execution.successful
    evidence = intervention_evidence.read_evidence(execution.results['response-evidence'])
    assert len(evidence['effects']) == 38 and evidence['families'].requested.sum() == 38
    assert evidence['effects'].loc[lambda x: x.movie.eq('target0'), 'p_value'].isna().all()
    assert evidence['effects'].loc[lambda x: x.movie.eq('target7')].set_index('measurement').outcome.to_dict() == {'reporter_custom': 'increase', 'shape_custom': 'decrease'}
    controls = intervention_controls.read_controls(execution.results['control-comparisons'])
    assert controls['comparisons'].complete_matches.eq(7).all()
    patterns = intervention_patterns.read_patterns(execution.results['coordinated-responses'])
    assert len(patterns['cell_pairs']) == 19 and len(patterns['sample_pairs']) == 16
    original = {str(value.artifact(ref.name)): file_hash(value.artifact(ref.name)) for value in execution.results.values() for ref in value.outcome.artifacts}

    def forbidden(*args, **kwargs):
        raise AssertionError('Unchanged generated inputs repeated scientific calculations')
    for module, name in [(intervention_windows, 'prepare'), (intervention_evidence, 'analyse'), (intervention_controls, 'compare_units'), (intervention_patterns, 'analyse')]:
        monkeypatch.setattr(module, name, forbidden)
    reopened = run_request(resolved, paths, tmp_path / 'results', only=['control-comparisons', 'coordinated-responses'])
    assert reopened.successful and all((row.outcome.status == 'reused' for row in reopened.results.values()))
    assert all((file_hash(Path(path)) == sha for path, sha in original.items()))
    changed = deepcopy(request)
    changed['anchors']['target7']['hours'] += 0.5
    changed_request = resolve_request(parse([changed])[0], source_run='native-generated-software', tables=tables, input_hashes={key: file_hash(path) for key, path in paths.items()})
    assert changed_request.scientific_id != resolved.scientific_id

def test_editable_example_resolves_free_columns_and_per_measurement_summaries(tmp_path):
    request = json.loads(__import__('importlib.resources',fromlist=['files']).files('pymicroglia').joinpath('data/intervention-response.example.json').read_text(encoding='utf-8'))
    frame = pd.DataFrame({'stem': ['example-recording'] * 4, 'identity': [7] * 4, 'frame_index': [0, 1, 2, 3], 'hours': [47.0, 47.5, 48.0, 48.5], 'reporter_custom': [1.0, 2.0, 3.0, 4.0], 'shape_custom': [4.0, 3.0, 2.0, 1.0]})
    source = tmp_path / 'cell_frame.csv'
    frame.to_csv(source, index=False)
    resolved = resolve_request(parse([request])[0], source_run='editable-example-software', tables={'cell_frame': frame}, input_hashes={'cell_frame': file_hash(source)})
    assert {m.column: m.summary for m in resolved.measurements} == {'reporter_custom': 'mean', 'shape_custom': 'median'}
    assert resolved.request.evidence['method'] == 'none' and (not resolved.rhythm_analysis)
    assert not resolved.inputs.samples[0].confirmed
