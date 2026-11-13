"""Empty temporal evidence is insufficient evidence, never a missing schema."""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pymicroglia.tracking import core_replacement_evidence as repair


@pytest.fixture
def source():
    root=Path(__file__).parents[1]/'src/pymicroglia/tracking/data/completed_feedback_engine'
    return next(root.rglob('reviewed/R208-A005/core_replacement_ownership.py')).read_bytes().decode()


def nested(source, name, namespace):
    tree=ast.parse(source)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'nested original operation','exec'),namespace)
    return namespace[name]


def holder_fixture(prior):
    points=pd.DataFrame(dict(frame=[0,1,2,3] if prior else [3],node=[1,1,0,2] if prior else [2],
        owner=[9,9,0,7] if prior else [7],track_id=4,x=2.,y=2.,radius_px=2.))
    labels=np.zeros((4,5,5),np.uint16);labels[3,2,2]=7
    nodes=np.zeros_like(labels);nodes[3,2,2]=2
    return dict(np=np,pd=pd,nodes=nodes,strong=points,strong_tracks={4:points},
        by_node={1:points.iloc[:1],2:points.iloc[-1:]},labels=labels,original_labels=labels.copy(),
        pixels=lambda n:np.array([12]),independent_optical_source=lambda p:True)


def test_reproduces_missing_column_and_preserves_empty_evidence(source):
    original,patched=repair.repaired_function(source)
    with pytest.raises(AttributeError,match='owner'):
        nested(original,'recover_displaced_holder',holder_fixture(False))(3,3,7,np.array([12]))
    assert nested(patched,'recover_displaced_holder',holder_fixture(False))(3,3,7,np.array([12])) is None


def test_valid_measured_holder_decision_is_identical(source):
    original,patched=repair.repaired_function(source)
    a=nested(original,'recover_displaced_holder',holder_fixture(True))(3,3,7,np.array([12]))
    b=nested(patched,'recover_displaced_holder',holder_fixture(True))(3,3,7,np.array([12]))
    assert a is not None and b is not None
    assert a[:2]==b[:2] and a[3:]==b[3:]
    np.testing.assert_array_equal(a[2],b[2])


@pytest.mark.parametrize('table,predicate',repair.FILTERS)
@pytest.mark.parametrize('empty',[True,False])
def test_all_filters_preserve_columns_and_nonempty_values(table,predicate,empty):
    data=pd.DataFrame(dict(node=pd.Series([] if empty else [0,1,2],dtype='int64'),
                           owner=pd.Series([] if empty else [4,5,6],dtype='int64')))
    env={table:data,'by_node':{1:[1],2:[1,2]}}
    result=eval(f'{table}.loc[{table}.node.map(lambda n:{predicate}).astype(bool)]',env)
    assert list(result.columns)==list(data.columns)
    if not empty:
        expected=eval(f'{table}[{table}.node.map(lambda n:{predicate})]',env)
        pd.testing.assert_frame_equal(result,expected,check_exact=True)
    else:
        pd.testing.assert_frame_equal(result,data,check_exact=True)


def test_adapter_pins_function_restores_loader_and_restarts_after_changed_repair(source,tmp_path):
    original,patched=repair.repaired_function(source)
    path=tmp_path/'core.py';path.write_text(original)
    namespace={};exec(compile(original,str(path),'exec'),namespace)
    method=namespace[repair.FUNCTION];old_code=method.__code__
    @contextmanager
    def loader(spec):yield method
    sequence=SimpleNamespace(reviewed=SimpleNamespace(frozen_method=loader))
    @contextmanager
    def methods(root,**kw):yield sequence
    feedback=SimpleNamespace(methods=methods)
    with repair.activate(feedback,tmp_path):
        with feedback.methods(tmp_path) as selected:
            with selected.reviewed.frozen_method({}) as loaded:
                assert loaded is method and loaded.__code__ is not old_code
            assert method.__code__ is old_code
    assert feedback.methods is methods and sequence.reviewed.frozen_method is loader
    record=tmp_path/'core_replacement_evidence_repair.json';data=json.loads(record.read_text())
    assert len(data['functions'])==1
    # Unfinished correction work is never reused, so a changed repair starts a fresh record.
    data['signature']['version']='changed';record.write_text(json.dumps(data))
    with repair.activate(feedback,tmp_path):pass
    data=json.loads(record.read_text())
    assert data['signature']['version']==repair.VERSION and data['functions']=={}


def test_canonical_matches_frozen_repair(source):
    p=Path(__file__).parents[2]/'Motion/code/core_replacement_ownership.py'
    if not p.exists():pytest.skip('canonical Motion checkout unavailable')
    before,after=repair.repaired_function(p.read_text())
    assert before==after==repair.repaired_function(source)[1]
