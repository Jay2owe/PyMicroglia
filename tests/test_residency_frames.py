"""Real residency calculations keep movie indices independent of source offsets."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi
from pymicroglia.tracking import residency_frames as repair

ARCHIVE = Path(__file__).parents[1] / 'src/pymicroglia/tracking/data/motion_engine.zip'


def fixture_code():
    with zipfile.ZipFile(ARCHIVE) as z:
        source = z.read('code/accepted_history.py').decode().replace('\r\n', '\n')
        ops = {}
        exec(z.read('code/accepted_history_ops/issue015/m5_blob_residency.py'), ops)
        params = json.loads(z.read('code/accepted_history_parameters.json'))['issue015']['blob_residency']
    module = SimpleNamespace(**ops)
    ns = dict(np=np, pd=pd, ndi=ndi, I015=None,
              _issue_module=lambda *a: module, _params=lambda *a: params)
    tree = ast.parse(source)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in ('blob_residency_precondition', 'select_blob_residency_events'):
            exec(compile(ast.Module(body=[n], type_ignores=[]), '<native>', 'exec'), ns)
    return source, ns


def movie(donor=11, host=27, padding=0):
    a = np.zeros((10, 20, 24), dtype=np.uint16)
    a[:, 4:9, 5:10] = host
    a[:7, 4:9, 10:14] = donor
    a[7:, 4:9, 10:14] = host
    a[:, 15:18, 18:21] = 33  # unrelated correct body
    if padding:
        a = np.pad(a, ((0, 0), (padding, padding), (padding, padding)))
    event = dict(event_id='synthetic', donor_identity=donor, host_identity=host,
                 imagej_frame=8, donor_core_ancestry=1., host_core_ancestry=1.,
                 direct_transfer_fraction=1., consecutive_prior_presence_frames=7,
                 donor_still_present=False)
    return a, pd.DataFrame([event])


def evaluate(offset, *, old=False, donor=11, host=27, padding=0, empty=False, canonical=False):
    source, ns = fixture_code()
    original, patched = repair.repaired_function(source)
    if canonical:
        path = Path(__file__).parents[2] / 'Motion/code/accepted_history.py'
        text = path.read_text()
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == repair.FUNCTION)
        patched = ast.get_source_segment(text, node)
    exec(original if old else patched, ns)
    labels, ledger = movie(donor, host, padding)
    ns['detect_identity_ledger_transfers'] = lambda *a: (ledger.iloc[:0] if empty else ledger, None, None)
    registry = pd.DataFrame(columns=['source_imagej_frame', 'identity', 'host_identity', 'held_px'])
    result = ns[repair.FUNCTION](labels, np.zeros_like(labels), labels.astype(float), registry, offset)
    return labels, result


def test_reproduces_original_failure():
    with pytest.raises(AssertionError, match='before configured activation'):
        evaluate(0, old=True)


@pytest.mark.parametrize('offset', [0, 2, 13])
def test_real_event_survives_all_remaining_frames(offset):
    labels, (out, unclaimed, registry, ledger, manual) = evaluate(offset)
    np.testing.assert_array_equal(out, labels)
    assert not unclaimed.any() and manual.empty
    assert ledger.source_frame.tolist() == [8 + offset, 9 + offset, 10 + offset]
    assert registry.source_imagej_frame.tolist() == ledger.source_frame.tolist()
    assert set(registry.identity) == {11}
    assert ledger.iloc[-1].decision == 'retained_inside_stored_branch_at_movie_end'


def test_identity_rename_and_translation_do_not_change_decisions():
    _, base = evaluate(0)
    labels, moved = evaluate(0, donor=3, host=7, padding=4)
    np.testing.assert_array_equal(moved[0], labels)
    assert moved[3].decision.tolist() == base[3].decision.tolist()
    assert set(moved[2].identity) == {3}


def test_unaffected_movie_is_identical():
    _, old = evaluate(0, old=True, empty=True)
    _, new = evaluate(0, empty=True)
    for a, b in zip(old, new):
        if isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
        else: pd.testing.assert_frame_equal(a, b)


def test_canonical_and_archive_adapter_match():
    _, native = evaluate(2, canonical=True)
    _, adapted = evaluate(2)
    for a, b in zip(native, adapted):
        if isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
        else: pd.testing.assert_frame_equal(a, b)


def test_only_changed_stage_gets_new_signature(tmp_path):
    source, ns = fixture_code()
    original, _ = repair.repaired_function(source)
    exec(original, ns)
    path = tmp_path / 'history.py'; path.write_text(source)
    history = SimpleNamespace(**ns, __file__=str(path))
    original_fn = getattr(history, repair.FUNCTION)
    seen = []
    class Stages:
        signature = {'original': 'unchanged'}
        def call(self, method):
            seen.append(self.signature.copy())
            if method == 'stage_m22_accepted_history': raise RuntimeError('test failure')
    stages = Stages()
    with repair.install(history, SimpleNamespace(Stages=Stages), tmp_path):
        stages.call('stage_m20_accepted')
        with pytest.raises(RuntimeError): stages.call('stage_m22_accepted_history')
        assert stages.signature == {'original': 'unchanged'}
    assert seen[0] == {'original': 'unchanged'}
    assert seen[1]['stage_repair']['version'] == repair.VERSION
    assert getattr(history, repair.FUNCTION) is original_fn
