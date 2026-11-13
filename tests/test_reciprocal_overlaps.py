"""Conflicting pair histories must not abort or choose an iteration-order winner."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
from pymicroglia.tracking import reciprocal_overlaps as repair


@pytest.fixture
def native(tmp_path, monkeypatch):
    engine = Path(__file__).parents[1]/'src/pymicroglia/tracking/data/motion_engine'
    source = (engine/'code/reciprocal_lineages.py').read_bytes().decode().replace('\r\n', '\n')
    path = tmp_path/'reciprocal_lineages.py'; path.write_text(source)
    module = ModuleType('reciprocal_lineages'); module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(source.replace('from local_lineage_reservations import discover_duplicate_handoffs', ''), module.__dict__)
    tail = source[source.index('    candidate = labels.copy()'):]
    pre = '''def baseline(labels, accepted, unclaimed, offset, minimum_area):
    pair_rows = [dict(candidate_id=e['candidate_id'], accepted=True, decision='accepted_recurrent_reciprocal_lineages') for e in accepted]
    duplicate_audit = predecessor_audit = pd.DataFrame()
    all_components, all_assignments, all_mismatches = [], [], []
'''
    exec(pre+tail, module.__dict__)
    return module, source


def event(module, labels, pair, target, number, frames=None):
    comps, assignments = [], []
    for frame in frames or range(1, len(labels)+1):
        for row in module.frame_components(labels[frame-1], pair, frame, 1):
            comps.append(row)
            assignments.append(dict(source_frame=frame, component_id=row['component_id'],
                                    identity=target, observed=True))
    return dict(candidate_id=str(number), pair=pair, assignments=assignments,
                components=comps, mismatches=[])


def run(module, labels, events):
    rows = [dict(candidate_id=e['candidate_id'], accepted=True,
                 decision='accepted_recurrent_reciprocal_lineages') for e in events]
    return repair.compose(labels, events, rows, 0, 1, module.frame_components, module.CONNECTIVITY), rows


def fixture(module):
    labels = np.zeros((4, 12, 16), np.uint16)
    labels[:, 2:5, 2:5] = 2
    labels[:, 8:10, 12:14] = 8
    a = event(module, labels, (1, 2), 1, 'a')
    b = event(module, labels, (2, 3), 3, 'b')
    control = event(module, labels, (8, 9), 9, 'control')
    return labels, a, b, control


def test_original_crash_and_independent_correction_survives(native):
    m, _ = native; labels, a, b, control = fixture(m)
    with pytest.raises(AssertionError, match='repair domains overlap'):
        m.baseline(labels, [a, b, control], np.zeros_like(labels), 0, 1)
    result, rows = run(m, labels, [a, b, control])
    expected = labels.copy(); expected[expected == 8] = 9
    np.testing.assert_array_equal(result[0], expected)
    assert result[3] == {'a', 'b'}
    assert [r['accepted'] for r in rows] == [False, False, True]
    assert len(result[2]) == 4


def test_nonoverlap_labels_domains_and_action_tables_are_exact(native):
    m, _ = native; labels, a, b, control = fixture(m)
    before = m.baseline(labels, [a, control], np.zeros_like(labels), 0, 1)
    result, rows = run(m, labels, [a, control])
    assert result[0].tobytes() == before.labels.tobytes()
    assert result[1].tobytes() == before.repair_domains.tobytes()
    pd.testing.assert_frame_equal(pd.DataFrame(result[2]), before.actions, check_exact=True)
    pd.testing.assert_frame_equal(pd.DataFrame(rows), before.pair_audit, check_exact=True)


def test_agreeing_repeated_proposals_count_pixels_once(native):
    m, _ = native; labels, a, b, control = fixture(m)
    repeated = {**a, 'candidate_id': 'repeat'}
    result, rows = run(m, labels, [a, repeated, control])
    expected = labels.copy(); expected[expected == 2] = 1; expected[expected == 8] = 9
    np.testing.assert_array_equal(result[0], expected)
    assert sum(r['changed_pixels'] for r in result[2]) == np.count_nonzero(expected != labels)
    assert not result[3]


def test_transitive_conflict_rejects_whole_histories(native):
    m, _ = native; labels, a, b, control = fixture(m)
    a = event(m, labels, (1, 2), 1, 'a', [1, 2])
    b = event(m, labels, (1, 2), 1, 'b', [2, 3])
    c = event(m, labels, (2, 3), 3, 'c', [3, 4])
    result, _ = run(m, labels, [a, b, c])
    np.testing.assert_array_equal(result[0], labels)
    assert result[3] == {'a', 'b', 'c'}


def test_order_identity_renaming_translation_and_input_preservation(native):
    m, _ = native; labels, a, b, control = fixture(m); original = labels.copy()
    events = [a, b, control]
    first, _ = run(m, labels, events); reverse, _ = run(m, labels, list(reversed(events)))
    np.testing.assert_array_equal(first[0], reverse[0])
    mapping = np.arange(20, dtype=np.uint16); mapping[1:] = mapping[:0:-1]
    shifted = np.pad(mapping[labels], ((0, 0), (3, 3), (4, 4)))
    moved = [event(m, shifted, tuple(int(mapping[x]) for x in e['pair']),
                   int(mapping[e['assignments'][0]['identity']]), e['candidate_id']) for e in events]
    actual, _ = run(m, shifted, moved)
    np.testing.assert_array_equal(actual[0], np.pad(mapping[first[0]], ((0, 0), (3, 3), (4, 4))))
    np.testing.assert_array_equal(labels, original)


def test_adapter_scope_and_restoration(native, tmp_path):
    m, source = native
    class Stages:
        signature = {'base': 'same'}
        def call(self, method): return self.signature.copy()
    original = m.reconcile_reciprocal_lineages
    old_code = original.__code__
    with repair.install(None, SimpleNamespace(Stages=Stages), tmp_path) as signature:
        assert Stages().call('stage_m1') == {'base': 'same'}
        assert Stages().call('stage_m22_accepted_history')['reciprocal_overlap_repair'] == signature
        assert original.__code__ is not old_code
    assert m.reconcile_reciprocal_lineages is original
    assert original.__code__ is old_code
    assert Path(m.__file__).read_text() == source
