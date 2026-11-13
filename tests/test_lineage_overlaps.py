"""Real local-body tracing tolerates repeated events without inventing ownership."""
import ast
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pymicroglia.tracking import lineage_overlaps as repair


@pytest.fixture
def native(tmp_path, monkeypatch):
    engine=Path(__file__).parents[1]/'src/pymicroglia/tracking/data/motion_engine'
    source=(engine/'code/local_lineage_reservations.py').read_bytes().decode().replace('\r\n','\n')
    path=tmp_path/'local_lineage_reservations.py'; path.write_text(source)
    module=ModuleType('local_lineage_reservations'); module.__file__=str(path)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    exec(source.replace('from tracking import motion_support',''),module.__dict__)
    # Isolate trace composition; geometric body selection is the real algorithm.
    module.motion_support=lambda a,b,lag: dict(motion_support=1.,supportive_px=int((a&b).sum()),distance_px=0.)
    return module,source


def fixture():
    labels=np.zeros((7,20,24),np.uint16)
    labels[:,2:6,2:6]=7
    labels[1,2:6,2:6]=9
    labels[4,2:6,2:6]=9
    labels[:,13:17,16:20]=40
    labels[2,13:17,16:20]=41
    seed=np.zeros((20,24),bool);seed[2:6,2:6]=True
    control=np.zeros_like(seed);control[13:17,16:20]=True
    events=[dict(event_number=1,event_id='first',review_frame=2,owner=7,seed=seed),
            dict(event_number=2,event_id='repeat',review_frame=5,owner=7,seed=seed),
            dict(event_number=3,event_id='independent',review_frame=3,owner=40,seed=control)]
    params=dict(maximum_link_fraction_of_diagonal=.2,minimum_trace_motion_support=.5,
        source_frame_offset=0,minimum_component_px=1,minimum_area_ratio=.5,maximum_area_ratio=2.,
        minimum_substantial_core_px=1,minimum_fragment_px=1,minimum_expanded_area_ratio=.5,
        maximum_expanded_area_ratio=2.,minimum_area_continuity_improvement=0.)
    lag=np.zeros((6,20,24),np.float32)
    return labels,lag,events,params


def patch(module,source):
    old,new=repair.repaired_function(source)
    exec(new,module.__dict__)


def run(module, labels, lag, events, params):
    module.discover_duplicate_handoffs=lambda *a: (events,pd.DataFrame(),pd.DataFrame(),np.zeros_like(labels))
    return module.reserve_local_lineages(labels,np.zeros_like(labels),lag,params)


def test_reproduces_original_same_owner_crash_and_restores_both_seeds(native):
    module,source=native;args=fixture()
    with pytest.raises(AssertionError,match='independent local-lineage events overlap'):
        module._trace_events(*args)
    patch(module,source)
    result=run(module,*args)
    expected=args[0].copy(); expected[:,2:6,2:6]=7; expected[:,13:17,16:20]=40
    np.testing.assert_array_equal(result.labels,expected)
    assert not result.unclaimed.any()
    assert 'overlap_same_owner_shared_path' in set(result.boundaries.reason)
    assert set(result.actions.event_id)=={'first','repeat','independent'}


def test_disagreeing_connected_events_retain_whole_parent_and_independent_repairs(native):
    module,source=native;labels,lag,events,params=fixture();patch(module,source)
    events.append(dict(event_number=4,event_id='conflict',review_frame=3,owner=80,seed=events[0]['seed']))
    result=run(module,labels,lag,events,params)
    expected=labels.copy();expected[:,13:17,16:20]=40
    np.testing.assert_array_equal(result.labels,expected)
    assert set(result.actions.event_id)=={'independent'}
    assert set(result.boundaries[result.boundaries.direction=='composition'].reason)=={'overlap_different_owners_parent_retained'}


def test_nonoverlapping_trace_and_tables_are_byte_identical(native):
    module,source=native;labels,lag,events,params=fixture();events=[events[0],events[2]]
    original=module._trace_events(labels,lag,events,params)
    patch(module,source);actual=module._trace_events(labels,lag,events,params)
    assert actual[0].tobytes()==original[0].tobytes()
    pd.testing.assert_frame_equal(actual[1],original[1],check_exact=True)
    pd.testing.assert_frame_equal(actual[2],original[2],check_exact=True)


@pytest.mark.parametrize('conflicting',[False,True])
def test_event_order_identity_renaming_and_translation(native,conflicting):
    module,source=native;labels,lag,events,params=fixture();patch(module,source)
    if conflicting:
        events.append(dict(event_number=4,event_id='conflict',review_frame=3,owner=80,seed=events[0]['seed']))
    before=labels.copy(); result=run(module,labels,lag,events,params)
    permuted=run(module,labels,lag,list(reversed(events)),params)
    np.testing.assert_array_equal(permuted.labels,result.labels)
    mapping=np.arange(100,dtype=np.uint16);mapping[1:]=mapping[:0:-1]
    pad=((0,0),(3,3),(4,4))
    moved=[{**e,'owner':int(mapping[e['owner']]),'seed':np.pad(e['seed'],pad[1:])} for e in events]
    translated=run(module,np.pad(mapping[labels],pad),np.pad(lag,pad),moved,params)
    np.testing.assert_array_equal(translated.labels,np.pad(mapping[result.labels],pad))
    np.testing.assert_array_equal(labels,before)


def test_adapter_signature_restore_and_canonical_parity(native,tmp_path):
    module,source=native
    class Stages:
        signature={'base':'same'}
        def call(self,method):return self.signature.copy()
    context=SimpleNamespace(Stages=Stages)
    old=module._trace_events
    with repair.install(None,context,tmp_path) as signature:
        run(module,*fixture())
        assert (tmp_path/'local_lineage_overlap_audit.csv').is_file()
        assert Stages().call('stage_m1')=={'base':'same'}
        assert Stages().call('stage_m22_accepted_history')['lineage_overlap_repair']==signature
    assert module._trace_events is old
    with pytest.raises(AssertionError,match='events overlap'): old(*fixture())
    assert Path(module.__file__).read_text()==source
    canonical=(Path(__file__).parents[2]/'Motion/code/local_lineage_reservations.py').read_text()
    assert repair.repaired_function(canonical)[0]==repair.repaired_function(source)[1]
    p=tmp_path/'local_lineage_overlap_repair.json';d=json.loads(p.read_text());d['version']='changed';p.write_text(json.dumps(d))
    with repair.install(None,context,tmp_path) as again:pass
    assert json.loads(p.read_text())==again
