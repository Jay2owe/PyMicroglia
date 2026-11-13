"""Parallel repairs abstain on conflicting complete transfers, not one pixel."""
import ast
from functools import wraps
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from pymicroglia.tracking import repair_composition as r


def movie():
    base = np.zeros((4, 7, 9), dtype=np.uint16)
    base[:, 1:3, 1:3] = 10
    base[:, 4:6, 5:7] = 40
    base[:, 0, 8] = 90  # unrelated correct control
    ledger = np.zeros_like(base)
    return base, ledger


def original_merge(base, ledger, a, au, b, bu):
    ad, bd, aud, bud = a != base, b != base, au != ledger, bu != ledger
    if np.any(ad & bd & (a != b)): raise AssertionError('field-derived alias and continuity repairs conflict')
    if np.any(aud & bud & (au != bu)): raise AssertionError('field-derived alias and continuity unclaimed ledgers conflict')
    labels, unclaimed = a.copy(), au.copy()
    labels[bd], unclaimed[bud] = b[bd], bu[bud]
    return labels, unclaimed


def test_reproduces_crash_and_rejects_whole_transfer_not_just_overlap():
    base, u = movie()
    a, b = base.copy(), base.copy()
    a[base == 10] = 20
    b[2, 1, 1] = 30
    b[base == 40] = 50
    with pytest.raises(AssertionError, match='repairs conflict'):
        original_merge(base, u, a, u, b, u)
    labels, ledger, audit = r.compose_identity_repairs(base, u, a, u, b, u)
    expected = base.copy(); expected[base == 40] = 50
    np.testing.assert_array_equal(labels, expected)
    np.testing.assert_array_equal(ledger, u)
    assert set(audit.status) == {'applied', 'unresolved_conflicting_proposals_parent_retained'}
    assert (audit[audit.branch == 'alias'].pixels == 16).all()


@pytest.mark.parametrize('agreement', [False, True])
def test_uncontested_results_match_original_bytes(agreement):
    base, u = movie(); a, b = base.copy(), base.copy()
    a[base == 10] = 20
    b[base == (10 if agreement else 40)] = 20 if agreement else 50
    expected = original_merge(base, u, a, u, b, u)
    actual = r.compose_identity_repairs(base, u, a, u, b, u)
    assert all(x.tobytes() == y.tobytes() for x, y in zip(actual, expected))
    assert set(actual[2].status) == {'applied'}


def test_labels_and_unclaimed_are_one_state():
    base, u = movie(); u[base == 10] = 10; base[base == 10] = 0
    a, au, b, bu = base.copy(), u.copy(), base.copy(), u.copy()
    a[u == 10] = 20; au[u == 10] = 0
    bu[u == 10] = 30
    actual = r.compose_identity_repairs(base, u, a, au, b, bu)
    np.testing.assert_array_equal(actual[0], base)
    np.testing.assert_array_equal(actual[1], u)


def test_ledger_only_disagreement_is_recorded():
    base, u = movie(); u[base == 10] = 10; base[base == 10] = 0
    au, bu = u.copy(), u.copy(); au[u == 10] = 20; bu[u == 10] = 30
    with pytest.raises(AssertionError, match='ledgers conflict'):
        original_merge(base, u, base, au, base, bu)
    labels, ledger, audit = r.compose_identity_repairs(base, u, base, au, base, bu)
    np.testing.assert_array_equal(labels, base); np.testing.assert_array_equal(ledger, u)
    assert len(audit) == 2


def test_renaming_translation_branch_order_and_inputs_preserved():
    base, u = movie(); a, b = base.copy(), base.copy()
    a[base == 10] = 20; b[2, 1, 1] = 30; b[base == 40] = 50
    inputs = [base, u, a, u, b, u]
    before = [x.tobytes() for x in inputs]
    result = r.compose_identity_repairs(*inputs)
    swapped = r.compose_identity_repairs(base, u, b, u, a, u)
    for x,y in zip(result, swapped):
        if isinstance(x,np.ndarray): np.testing.assert_array_equal(x,y)
    mapping = np.arange(100,dtype=np.uint16); mapping[1:] = mapping[:0:-1]
    renamed = [np.pad(mapping[x], ((0,0),(3,3),(5,5))) for x in inputs]
    actual = r.compose_identity_repairs(*renamed)
    for x,y in zip(actual[:2], result[:2]):
        np.testing.assert_array_equal(x,np.pad(mapping[y],((0,0),(3,3),(5,5))))
    assert before == [x.tobytes() for x in inputs]


def test_empty_and_invalid_foreground():
    base,u = movie()
    assert r.compose_identity_repairs(base,u,base,u,base,u)[2].empty
    bad=base.copy(); bad[0,0,0]=3
    with pytest.raises(ValueError,match='foreground ownership'):
        r.compose_identity_repairs(base,u,bad,u,base,u)


def test_canonical_and_frozen_patch_match():
    root=Path(__file__).parents[2]
    canonical=(root/'Motion/code/parallel_identity_repairs.py').read_text()
    ns={}; exec(canonical,ns)
    code=ast.get_source_segment(Path(r.__file__).read_text(),next(n for n in ast.parse(Path(r.__file__).read_text()).body
                 if isinstance(n,ast.FunctionDef) and n.name=='compose_identity_repairs'))
    assert code in canonical
    engine=root/'PyMicroglia/src/pymicroglia/tracking/data/motion_engine'
    original,patched=r.repaired_function((engine/'code/accepted_history.py').read_bytes().decode())
    assert 'field-derived alias and continuity repairs conflict' in original
    assert 'field-derived alias and continuity repairs conflict' not in patched
    current=(root/'Motion/code/accepted_history.py').read_text()
    assert r.repaired_function(current)[0]==r.repaired_function(current)[1]


def test_install_patches_captured_wrappers_and_only_stage22_signature(tmp_path):
    source='''def run_accepted_history():
        alias_delta = alias.labels != issue015_labels
        raise AssertionError("old conflict")
        _save(issue017_checkpoint, labels, interval)
        return labels, unclaimed
'''
    path=tmp_path/'history.py'; path.write_text(source)
    base,u=movie(); a,b=base.copy(),base.copy(); a[base==10]=20; b[2,1,1]=30
    history=ModuleType('fixture_history'); history.__file__=str(path)
    history.__dict__.update(alias=SimpleNamespace(labels=a,unclaimed=u),issue015_labels=base,
        issue015_unclaimed=u,continuity_labels=b,continuity_unclaimed=u,output_root=tmp_path,
        issue017_checkpoint=None,interval=1,_save=lambda *a:None)
    exec(source,history.__dict__)
    function=history.run_accepted_history
    @wraps(function)
    def wrapped(): return function()
    history.run_accepted_history=wrapped
    captured=wrapped
    class Stages:
        signature={'base':'same'}
        def call(self,method): return self.signature.copy()
    checkpoints=SimpleNamespace(Stages=Stages)
    before=function.__code__
    with r.install(history,checkpoints,tmp_path) as signature:
        np.testing.assert_array_equal(captured()[0],base)
        assert Stages().call('stage_m1')=={'base':'same'}
        assert Stages().call('stage_m22_accepted_history')['composition_repair']==signature
        assert Stages.signature=={'base':'same'}
    assert function.__code__ is before
    with pytest.raises(AssertionError,match='old conflict'): captured()
    record=tmp_path/'parallel_repair_composition.json'
    value=json.loads(record.read_text()); value['version']='changed'; record.write_text(json.dumps(value))
    # A changed repair recomputes accepted history instead of refusing the run.
    with r.install(history,checkpoints,tmp_path) as again: pass
    assert json.loads(record.read_text())==again
