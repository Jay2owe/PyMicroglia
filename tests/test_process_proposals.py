import dataclasses
from contextlib import contextmanager
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymicroglia.tracking import process_proposals as repair


@pytest.fixture
def groups(tmp_path):
    archive = Path(__file__).parents[1] / 'src/pymicroglia/tracking/data/completed_feedback_engine.zip'
    with zipfile.ZipFile(archive) as z:
        source = z.read('code/reunite_large_process_groups.py').decode()
    path = tmp_path / 'groups.py'
    path.write_text(source)
    module = types.ModuleType('process_proposal_fixture')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    yield module
    sys.modules.pop(module.__name__, None)


def fixture(conflict=True, padding=0, names=(11, 27, 43, 59, 71)):
    a = np.zeros((20, 40, 100), np.uint16)
    points, neighbours = [], []
    bodies = [(5,25,names[0]), (29,49,names[1]), (26,28,names[2]),
              (65,85,names[3]), (86,88,names[4])]
    for t in range(len(a)):
        for x1,x2,name in bodies:
            a[t,5:25,x1:x2] = name
            points.append(dict(track_id=name,frame=t,x=x1+padding,y=10+padding))
        for owner,small in [(names[0],names[2]),(names[3],names[4])] + ([(names[1],names[2])] if conflict else []):
            neighbours.append(dict(frame=t+1,large_track=owner,small_track=small,aligned_exact_max=1.))
    a = np.pad(a, ((0,0),(padding,padding),(padding,padding)))
    return a, np.ones(a.shape), pd.DataFrame(points), np.zeros(a.shape,bool), pd.DataFrame(neighbours)


def params(groups):
    return dataclasses.replace(groups.Params(), minimum_group_satellites=1)


def test_conflict_reproduces_old_failure_and_retains_ambiguous_body(groups):
    args = fixture()
    with pytest.raises(AssertionError, match='overlapping process proposals'):
        groups.correct(*args, params(groups))
    with repair.install(groups):
        out,evidence,audit,apps = groups.correct(*args, params(groups))
    assert np.array_equal(out[args[0]==43],args[0][args[0]==43])
    assert np.all(out[args[0]==71] == 59)  # unrelated valid corrections still apply
    assert np.array_equal(out>0,args[0]>0)
    assert int(apps.changed_pixels.sum()) == int(np.count_nonzero(out != args[0]))
    assert int(evidence.proposal_conflict.sum()) == 40
    assert audit.ambiguous_pixels.sum() == 1600  # each competing parent records its claim


def test_no_conflict_is_identical_to_original(groups):
    args = fixture(False)
    expected = groups.correct(*args, params(groups))
    with repair.install(groups):
        actual = groups.correct(*args, params(groups))
    np.testing.assert_array_equal(actual[0],expected[0])
    for a,b in zip(actual[1:],expected[1:]):
        pd.testing.assert_frame_equal(a,b)


def test_order_translation_and_identity_renaming(groups):
    with repair.install(groups):
        baseline = groups.correct(*fixture(),params(groups))[0]
        args=list(fixture())
        args[-1]=args[-1].iloc[::-1]
        np.testing.assert_array_equal(groups.correct(*args,params(groups))[0],baseline)
        translated=groups.correct(*fixture(padding=3),params(groups))[0]
        np.testing.assert_array_equal(translated[:,3:-3,3:-3],baseline)
        renamed=groups.correct(*fixture(names=(71,43,27,11,59)),params(groups))[0]
        mapped=np.zeros_like(baseline)
        for old,new in zip((11,27,43,59,71),(71,43,27,11,59)):
            mapped[baseline==old]=new
        np.testing.assert_array_equal(renamed,mapped)


def test_protected_and_empty_inventory(groups):
    args=list(fixture(False));args[3][:]=True
    with repair.install(groups):
        np.testing.assert_array_equal(groups.correct(*args,params(groups))[0],args[0])
        args[-1]=args[-1].iloc[:0]
        np.testing.assert_array_equal(groups.correct(*args,params(groups))[0],args[0])


def test_restoration_after_exception_and_unknown_source(groups):
    old=groups.correct
    with pytest.raises(RuntimeError):
        with repair.install(groups):
            raise RuntimeError('test')
    assert groups.correct is old
    with pytest.raises(ValueError,match='Unrecognised'):
        repair.repaired_function('def correct():\n    return None\n')


def test_frozen_runner_integration_records_and_restores(groups,tmp_path):
    @contextmanager
    def methods(root, **kwargs):
        yield types.SimpleNamespace(process=types.SimpleNamespace(groups=groups))
    feedback=types.SimpleNamespace(methods=methods)
    old=groups.correct
    with repair.activate(feedback,tmp_path):
        with feedback.methods(tmp_path) as sequence:
            output=sequence.process.groups.correct(*fixture(),params(groups))[0]
            assert np.all(output[:,5:25,26:28]==43)
    assert feedback.methods is methods and groups.correct is old
    record=tmp_path/'process_proposal_repair.json'
    assert json.loads(record.read_text())['version']==repair.VERSION
    record.write_text('{}')
    with pytest.raises(ValueError,match='changed within'):
        with repair.activate(feedback,tmp_path):
            with feedback.methods(tmp_path):
                pass


def test_canonical_motion_rule_matches_archive_adapter(groups):
    path=Path(__file__).parents[2]/'Motion/issues/ISSUE-009-completed-feedback-rule-optimization/analysis/completed_feedback_tuning/code/reunite_large_process_groups.py'
    canonical=path.read_text()
    old,new=repair.repaired_function(canonical)
    assert old==new
    with repair.install(groups):
        expected=groups.correct(*fixture(),params(groups))
    namespace={}
    exec(compile(new,str(path),'exec'),groups.__dict__,namespace)
    actual=namespace['correct'](*fixture(),params(groups))
    np.testing.assert_array_equal(actual[0],expected[0])
    for a,b in zip(actual[1:],expected[1:]):
        pd.testing.assert_frame_equal(a,b)
