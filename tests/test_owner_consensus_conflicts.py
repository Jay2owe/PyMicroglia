"""Simultaneous body renames must carry attached zero-core fragments."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pymicroglia.tracking import owner_consensus_conflicts as repair


@pytest.fixture
def native(tmp_path, monkeypatch):
    engine = Path(__file__).parents[1]/'src/pymicroglia/tracking/data/motion_engine'
    for name in ('detector_spatial_units', 'ownership_spatial_units', 'component_accounting', 'owner_consensus'):
        p = tmp_path/(name+'.py'); p.write_bytes((engine/('code/'+name+'.py')).read_bytes())
        spec = importlib.util.spec_from_file_location(name, p)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module, Path(module.__file__).read_text()


def fixture(moving=True):
    labels = np.zeros((3,32,42), np.uint16)
    labels[:,10:14,10:14] = 1
    labels[1:,24:28,32:36] = 2
    labels[0,9,10:12] = 2
    labels[0,14,12:14] = 2
    if not moving:
        labels[0,10:14,10:14] = 3
        labels[0,20:24,3:7] = 1
    point = dict(track_id=10, frame=0, x=11., y=11., radius_px=2., physically_visible=True, candidate_owner=1 if moving else 3)
    points = [point]
    if not moving:
        points.append(dict(point, track_id=11, x=4., y=21., candidate_owner=1))
    proposals = pd.DataFrame([dict(proposal_id='first', track_id=10 if moving else 11, source_owner=1,
        proposed_owner=2, first_frame=0, last_frame=0, discovery_status='eligible')])
    return labels, pd.DataFrame(points), proposals, dict(absorb_zero_core_blockers=True)


def patch(module, source):
    old,new = repair.repaired_function(source)
    exec(new, module.__dict__)


def test_original_duplicate_and_correct_final_attachment(native):
    module,source = native
    args = fixture()
    old,old_audit = module.apply_dependency_ordered_proposals(*args)
    assert module.count_new_duplicate_components(args[0],old)==1
    patch(module,source)
    new,audit = module.apply_dependency_ordered_proposals(*args)
    expected=args[0].copy(); expected[0,10:14,10:14]=2
    np.testing.assert_array_equal(new,expected)
    assert module.count_new_duplicate_components(args[0],new)==0
    assert audit.changed_pixels.sum()==np.count_nonzero(new!=args[0])==16
    assert (audit[audit.proposal_id.str.startswith('O')].proposed_owner==2).all()
    np.testing.assert_array_equal(new>0,args[0]>0)


@pytest.mark.parametrize('orphans', [True,False])
def test_stationary_recipient_and_no_orphans_byte_parity(native,orphans):
    module,source=native;args=fixture(moving=False)
    if not orphans: args[3]['absorb_zero_core_blockers']=False
    before,audit_before=module.apply_dependency_ordered_proposals(*args)
    patch(module,source)
    after,audit_after=module.apply_dependency_ordered_proposals(*args)
    np.testing.assert_array_equal(before,after)
    pd.testing.assert_frame_equal(audit_before,audit_after,check_exact=True)


def test_rename_translation_and_order_invariance_with_independent_control(native):
    module,source=native;patch(module,source)
    labels,points,proposals,params=fixture()
    labels[:,20:24,20:24]=4;labels[1:,3:7,30:34]=5
    points=pd.concat([points,pd.DataFrame([dict(points.iloc[0],track_id=22,x=21.,y=21.,candidate_owner=4)])],ignore_index=True)
    proposals=pd.concat([proposals,pd.DataFrame([dict(proposals.iloc[0],proposal_id='independent',track_id=22,source_owner=4,proposed_owner=5)])],ignore_index=True)
    expected,_=module.apply_dependency_ordered_proposals(labels,points,proposals,params)
    assert np.all(expected[0,20:24,20:24]==5)
    reverse,_=module.apply_dependency_ordered_proposals(labels,points,proposals.iloc[::-1],params)
    np.testing.assert_array_equal(reverse,expected)
    mapping=np.arange(100,dtype=np.uint16);mapping[[1,2,4,5]]=[70,9,88,3]
    shifted=np.pad(mapping[labels],((0,0),(3,0),(4,0)))
    shifted_points=points.copy();shifted_points.x+=4;shifted_points.y+=3
    shifted_points.candidate_owner=shifted_points.candidate_owner.map(lambda n:int(mapping[n]))
    renamed=proposals.copy()
    for col in ('source_owner','proposed_owner'):renamed[col]=renamed[col].map(lambda n:int(mapping[n]))
    result,_=module.apply_dependency_ordered_proposals(shifted,shifted_points,renamed,params)
    np.testing.assert_array_equal(result,np.pad(mapping[expected],((0,0),(3,0),(4,0))))


def test_rejected_proposal_does_not_absorb_fragments(native):
    module,source=native;patch(module,source)
    labels,points,proposals,params=fixture();params['minimum_component_area_ratio']=100
    candidate,audit=module.apply_dependency_ordered_proposals(labels,points,proposals,params)
    np.testing.assert_array_equal(candidate,labels)
    assert audit.changed_pixels.sum()==0


def test_install_provenance_signature_restoration_and_canonical(native,tmp_path):
    module,source=native
    class Stages:
        signature={'original':True}
        def call(self,method,*a,**k): return self.signature.copy()
    original=getattr(module,repair.FUNCTION).__code__;old_call=Stages.call
    checkpoints=SimpleNamespace(Stages=Stages)
    with repair.install(None,checkpoints,tmp_path) as record:
        assert Stages().call('stage_m22_accepted_history')['owner_consensus_repair']==record
        assert Stages().call('other')=={'original':True}
        assert Stages.signature=={'original':True}
        assert getattr(module,repair.FUNCTION).__code__ is not original
    assert getattr(module,repair.FUNCTION).__code__ is original
    assert Stages.call is old_call
    path=tmp_path/'owner_consensus_attachment_repair.json';data=json.loads(path.read_text());data['version']='changed';path.write_text(json.dumps(data))
    with repair.install(None,checkpoints,tmp_path) as again:pass
    assert json.loads(path.read_text())==again
    canonical=Path(__file__).parents[2]/'Motion/code/owner_consensus.py'
    if canonical.exists():
        old,new=repair.repaired_function(canonical.read_text())
        assert old==new==repair.repaired_function(source)[1]
