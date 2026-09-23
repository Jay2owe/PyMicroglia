"""Scientific descendants invalidate independently of display and other questions."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pymicroglia.pipelines import parse
import pymicroglia.pipelines.coordination.inputs as coordination_inputs
import pymicroglia.pipelines.coordination.simultaneous as coordination_simultaneous
import pymicroglia.pipelines.coordination.evidence as coordination_evidence
from pymicroglia.pipelines.coordination.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash
from tests.test_coordination_delay import temporal_fixture

def setup(tmp_path, formal=False):
    resolved, tables, _ = temporal_fixture(values=np.random.default_rng(518).normal(size=(2, 180 if formal else 60)), formal=False, uncertainty=False)
    request = resolved.request
    if formal:
        declaration = request.declaration.as_dict()
        declaration['questions']['delay']['evidence'] = {'method': 'truncated_time_shift', 'radius_hours': 12.0, 'stationary_series': 'target', 'stationarity_justification': 'Constructed stationary independent software sequences'}
        declaration['inference'] = {'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'question'}
        request = parse([declaration])[0]
    tables['cell_frame']['alternate_x'] = tables['cell_frame'].centroid_x + 1
    paths = {}
    for name, frame in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        frame.to_csv(paths[name], index=False)

    def resolve(request):
        return resolve_request(request, source_run='original-source', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    return (request, tables, paths, resolve)

def test_lag_search_representation_and_geometry_invalidate_their_own_descendants(tmp_path):
    request, tables, paths, resolve = setup(tmp_path)
    root = tmp_path / 'pipeline'
    first = run_request(resolve(request), paths, root, only=['lagged-coordination'])
    assert first.successful
    original = {str(item.artifact(ref.name)): file_hash(item.artifact(ref.name)) for item in first.results.values() for ref in item.outcome.artifacts}
    original.update({str(path): file_hash(path) for path in paths.values()})
    specification = request.declaration.as_dict()
    changed = deepcopy(specification)
    changed['questions']['delay']['range_hours'] = [-3.0, 3.0]

    def forbidden(*a, **k):
        raise AssertionError('An unrelated scientific dependency was recalculated')
    with patch.object(coordination_inputs, 'prepare', forbidden), patch.object(coordination_simultaneous, 'analyse', forbidden):
        lag = run_request(resolve(parse([changed])[0]), paths, root, only=['lagged-coordination'])
    assert lag.successful and lag.results['pair-inputs'].outcome.status == 'reused' and (lag.results['simultaneous-coordination'].outcome.status == 'reused')
    assert lag.results['lagged-coordination'].outcome.scientific_id != first.results['lagged-coordination'].outcome.scientific_id
    changed['representation'] = 'changes'
    with patch.object(coordination_inputs, 'prepare', forbidden):
        representation = run_request(resolve(parse([changed])[0]), paths, root, only=['lagged-coordination'])
    assert representation.successful and representation.results['pair-inputs'].outcome.status == 'reused'
    assert representation.results['simultaneous-coordination'].outcome.scientific_id != lag.results['simultaneous-coordination'].outcome.scientific_id
    changed['geometry']['x'] = 'alternate_x'
    geometry = run_request(resolve(parse([changed])[0]), paths, root, only=['lagged-coordination'])
    assert geometry.successful
    assert geometry.results['pair-inputs'].outcome.scientific_id != representation.results['pair-inputs'].outcome.scientific_id
    assert all((file_hash(Path(path)) == value for path, value in original.items()))

def test_family_change_reuses_original_inputs_and_temporal_estimates(tmp_path):
    request, tables, paths, resolve = setup(tmp_path, formal=True)
    root = tmp_path / 'pipeline'
    first = run_request(resolve(request), paths, root, only=['coordination-evidence'])
    assert first.successful
    before = coordination_evidence.read_evidence(first.results['coordination-evidence'])
    changed = request.declaration.as_dict()
    changed['inference'] = {'alpha': 0.05, 'multiple_testing': 'bh', 'correction_scope': 'all'}

    def forbidden(*a, **k):
        raise AssertionError('Correction change repeated original preparation or estimates')
    import pymicroglia.pipelines.coordination.delay as coordination_delay
    with patch.object(coordination_inputs, 'prepare', forbidden), patch.object(coordination_simultaneous, 'analyse', forbidden), patch.object(coordination_delay, 'analyse', forbidden):
        updated = run_request(resolve(parse([changed])[0]), paths, root, only=['coordination-evidence'])
    assert updated.successful and updated.results['lagged-coordination'].outcome.status == 'reused'
    assert updated.results['coordination-evidence'].outcome.scientific_id != first.results['coordination-evidence'].outcome.scientific_id
    after = coordination_evidence.read_evidence(updated.results['coordination-evidence'])
    assert before['effects'].effect_id.tolist() == after['effects'].effect_id.tolist()
    assert before['effects'].estimated_delay_hours.tolist() == after['effects'].estimated_delay_hours.tolist()
