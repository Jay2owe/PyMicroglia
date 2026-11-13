"""Repairs recompute stale accepted history instead of refusing, and fit the shipped engine."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pymicroglia.tracking import (checkpoints, lineage_overlaps, owner_consensus_conflicts,
                                  owner_identity_accounting, reciprocal_overlaps, repair_composition,
                                  residency_frames)

ENGINE = Path(__file__).parents[1] / 'src/pymicroglia/tracking/data/motion_engine'


def engine_source(name):
    return (ENGINE / f'code/{name}.py').read_bytes().decode().replace('\r\n', '\n')


@pytest.mark.parametrize('repair, module', [
    (residency_frames, 'accepted_history'), (repair_composition, 'accepted_history'),
    (lineage_overlaps, 'local_lineage_reservations'), (owner_consensus_conflicts, 'owner_consensus'),
    (reciprocal_overlaps, 'reciprocal_lineages'), (owner_identity_accounting, 'accepted_history')])
def test_every_installed_repair_recognises_the_shipped_engine(repair, module):
    # An unrecognised engine would stop every recording, so check them all here.
    original, patched = repair.repaired_function(engine_source(module))
    assert original and patched


def history(root, name='pymicroglia_A1', produced='mid/accepted_history/16_issue032_owner_consensus.tif'):
    run = root / 'm22_accepted_history' / name
    (run / produced).parent.mkdir(parents=True)
    (run / produced).write_bytes(b'old')
    return run


def test_changed_repair_sets_history_aside_and_ignores_earlier_set_aside_copies(tmp_path, capsys):
    run = history(tmp_path)
    kept = history(tmp_path, 'pymicroglia_A1_failed_1792796663771034000')
    record = tmp_path / 'repair.json'
    record.write_text(json.dumps({'version': 'old'}))
    assert checkpoints.reconcile_repair(tmp_path, record, {'version': 'new'}, 'checkpoint.json', 'test repair')
    assert not run.exists() and kept.exists()
    assert len(list((tmp_path / 'm22_accepted_history').glob('pymicroglia_A1_superseded_*'))) == 1
    assert 'test repair changed' in capsys.readouterr().out


def test_history_made_before_the_repair_existed_is_recomputed(tmp_path):
    run = history(tmp_path)
    record = tmp_path / 'repair.json'
    produced = 'mid/accepted_history/16_issue032_owner_consensus.tif'
    assert checkpoints.reconcile_repair(tmp_path, record, {'version': 'new'}, produced, 'test repair')
    assert not run.exists()


def test_matching_or_absent_history_is_left_alone(tmp_path):
    record = tmp_path / 'repair.json'
    assert not checkpoints.reconcile_repair(tmp_path, record, {'version': 'new'}, 'checkpoint.json', 'repair')
    run = history(tmp_path)
    record.write_text(json.dumps({'version': 'new'}))
    assert not checkpoints.reconcile_repair(tmp_path, record, {'version': 'new'}, 'checkpoint.json', 'repair')
    record.write_text('{not json')
    assert checkpoints.reconcile_repair(tmp_path, record, {'version': 'new'}, 'checkpoint.json', 'repair')
    assert not run.exists()


OWNER_STAGE = '''
import numpy as np
def _owner_consensus(labels, candidate):
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    if candidate_identities != baseline_identities:
        raise AssertionError("owner consensus changed the active identity set")
    metrics = {"active_identities": int(len(candidate_identities))}
    return candidate, metrics
'''


@pytest.fixture
def owner_history(tmp_path, monkeypatch):
    path = tmp_path / 'engine' / 'accepted_history.py'
    path.parent.mkdir()
    path.write_text(OWNER_STAGE)
    module = ModuleType('accepted_history')
    module.__file__ = str(path)
    exec(OWNER_STAGE, module.__dict__)
    cache = ModuleType('physical_evidence_cache')
    cache._owner_function_with_evidence = lambda history, entry: history
    monkeypatch.setitem(sys.modules, 'physical_evidence_cache', cache)

    class Stages:
        signature = {'base': 'same'}

        def call(self, method, *args, **kwargs):
            return self.signature.copy()

    return module, cache, SimpleNamespace(Stages=Stages)


def test_owner_accounting_lets_combined_corrections_retire_a_name(owner_history, tmp_path):
    history_module, cache, context = owner_history
    labels, candidate = np.array([[1, 2]]), np.array([[1, 1]])
    with pytest.raises(AssertionError, match='active identity set'):
        history_module._owner_consensus(labels, candidate)
    root = tmp_path / 'run'
    root.mkdir()
    stale = history(root)
    old_builder, old_call = cache._owner_function_with_evidence, context.Stages.call
    with owner_identity_accounting.install(history_module, context, root) as signature:
        result, metrics = history_module._owner_consensus(labels, candidate)
        np.testing.assert_array_equal(result, candidate)
        assert metrics['retired_identities'] == [2] and metrics['introduced_identities'] == []
        assert context.Stages().call('stage_m22_accepted_history')['owner_identity_accounting_repair'] == signature
        assert context.Stages().call('stage_m1') == {'base': 'same'}
        assert cache._owner_function_with_evidence(history_module, None).__file__.endswith('_stage.py')
    assert not stale.exists()
    assert cache._owner_function_with_evidence is old_builder and context.Stages.call is old_call
    with pytest.raises(AssertionError, match='active identity set'):
        history_module._owner_consensus(labels, candidate)
