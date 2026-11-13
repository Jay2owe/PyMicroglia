"""Repeated parent components must not duplicate a fragment observation or vote."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pymicroglia.tracking import fragment_observations as repair


@pytest.fixture
def modules(tmp_path,monkeypatch):
    engine=Path(__file__).parents[1]/'src/pymicroglia/tracking/data/completed_feedback_engine'
    result=[]
    for name in ('reunite_recurrent_weak_branch_paths','extend_recurrent_weak_branch_paths'):
        source=(engine/('code/'+name+'.py')).read_bytes().decode()
        path=tmp_path/(name+'.py');path.write_text(source)
        module=ModuleType(name);module.__file__=str(path)
        monkeypatch.setitem(sys.modules,name,module)
        exec(compile(source,str(path),'exec'),module.__dict__)
        result.append(module)
    return result


def movie(parent=10,fragment=20,padding=0):
    labels=np.zeros((8,32,60),np.uint16)
    labels[:,5:20,5:17]=parent
    labels[:,5:20,25:37]=parent
    labels[:,9:15,18:24]=fragment
    labels[[0,7],9:15,18:24]=parent
    labels[:,26:29,51:54]=99
    raw=np.where(labels==fragment,10,100).astype(float)
    points=pd.DataFrame(columns=['state','strong','frame','x','y'])
    if padding:
        pad=((0,0),(padding,padding),(padding,padding))
        labels=np.pad(labels,pad);raw=np.pad(raw,pad)
    return labels,raw,points,np.zeros_like(labels,bool)


def test_real_scan_reproduces_failure_and_applies_shared_observation_once(modules):
    base,extension=modules;labels,raw,points,protected=movie()
    inventory=base.scan_nearby(labels,raw,points,protected)
    assert inventory.duplicated(['frame','small_component','large_name']).any()
    with pytest.raises(AssertionError,match='stale, protected or overlapping'):
        extension.correct(labels,raw,points,protected,inventory)
    with repair.install(base):
        actual,audit,apps=extension.correct(labels,raw,points,protected,inventory)
    expected=labels.copy();expected[labels==20]=10
    np.testing.assert_array_equal(actual,expected)
    assert len(apps)==6 and apps.changed_pixels.sum()==216
    assert audit.pixels.sum()==216
    np.testing.assert_array_equal(actual>0,labels>0)


def test_unique_evidence_is_exactly_unchanged(modules):
    base,_=modules;labels,raw,points,protected=movie()
    inventory=base.scan_nearby(labels,raw,points,protected).drop_duplicates(['frame','small_component','large_name'])
    original=base.correct(labels,protected,inventory)
    with repair.install(base):actual=base.correct(labels,protected,inventory)
    assert original[0].tobytes()==actual[0].tobytes()
    for a,b in zip(original[1:],actual[1:]):pd.testing.assert_frame_equal(a,b,check_exact=True)


def test_duplicate_contacts_cannot_inflate_path_area_or_brightness_evidence(modules):
    base,_=modules;labels,raw,points,protected=movie()
    inventory=base.scan_nearby(labels,raw,points,protected)
    params=replace(base.Params(),minimum_path_pixels=300)
    assert len(base.select_paths(inventory,params)[1])
    with repair.install(base):
        audit,selected=base.select_paths(inventory,params)
        assert selected.empty and audit.pixels.sum()==216
        # Two views of the same parent disagree on its brightness: require the
        # weak-fragment evidence to hold for both, not their weighted average.
        inventory['q95_ratio']=np.tile([.1,.4],len(inventory)//2)
        audit,selected=base.select_paths(inventory)
        assert selected.empty
        assert (audit.median_raw_q95_ratio==.4).all()


def test_inconsistent_repeats_and_truly_stale_or_protected_inputs_still_rejected(modules):
    base,_=modules;labels,raw,points,protected=movie()
    inventory=base.scan_nearby(labels,raw,points,protected)
    bad=inventory.copy();bad.loc[bad.index[0],'small_area']+=1
    with repair.install(base):
        with pytest.raises(ValueError,match='inconsistent repeated'):
            base.select_paths(bad)
        unique=inventory.drop_duplicates(['frame','small_component','large_name'])
        stale=unique.copy();stale['small_area']+=1
        with pytest.raises(AssertionError,match='stale, protected'):
            base.correct(labels,protected,stale)
        protected[1,9,18]=True
        with pytest.raises(AssertionError,match='stale, protected'):
            base.correct(labels,protected,unique)


def test_row_order_identity_renaming_and_translation(modules):
    base,_=modules
    with repair.install(base):
        labels,raw,points,protected=movie()
        inventory=base.scan_nearby(labels,raw,points,protected)
        expected=base.correct(labels,protected,inventory)[0]
        actual=base.correct(labels,protected,inventory.sample(frac=1,random_state=2))[0]
        np.testing.assert_array_equal(actual,expected)
        moved,raw2,points2,protected2=movie(parent=71,fragment=13,padding=4)
        actual=base.correct(moved,protected2,base.scan_nearby(moved,raw2,points2,protected2))[0]
        changed=expected.copy();changed[changed==10]=71;changed[changed==20]=13
        np.testing.assert_array_equal(actual,np.pad(changed,((0,0),(4,4),(4,4))))


def test_context_source_record_and_canonical_parity(modules,tmp_path):
    base,_=modules
    @contextmanager
    def methods(root,**kwargs):yield SimpleNamespace(repeated=SimpleNamespace(branches=base))
    feedback=SimpleNamespace(methods=methods)
    old=base.select_paths
    with repair.activate(feedback,tmp_path):
        with feedback.methods(tmp_path):assert base.select_paths is not old
    assert feedback.methods is methods and base.select_paths is old
    path=tmp_path/'fragment_observation_repair.json'
    assert json.loads(path.read_text())['version']==repair.VERSION
    canonical=(Path(__file__).parents[2]/'Motion/issues/ISSUE-009-completed-feedback-rule-optimization/analysis/completed_feedback_tuning/code/reunite_recurrent_weak_branch_paths.py').resolve()
    canonical=Path('\\\\?\\'+str(canonical)) if sys.platform=='win32' else canonical
    assert repair.repaired_function(canonical.read_text())[0]==repair.repaired_function(Path(base.__file__).read_text())[1]
    value=json.loads(path.read_text());value['version']='changed';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='changed within recording'):
        with repair.activate(feedback,tmp_path), feedback.methods(tmp_path):pass
