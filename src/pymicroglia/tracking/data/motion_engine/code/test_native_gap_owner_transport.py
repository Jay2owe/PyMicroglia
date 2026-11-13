import numpy as np
import pandas as pd

from native_gap_owner_transport import carry_measured_gap_owners


def example():
    parent=np.zeros((5,20,20),np.uint16)
    parent[:,5:8,5:8]=81;parent[:2,5:8,5:8]=42
    parent[:,12:15,12:15]=90
    points=pd.DataFrame([dict(frame=t,x=6.,y=6.,radius_px=1.,state='observed',strong=True) for t in range(5)])
    traces=[]
    for name in (7,9):
        before=np.zeros_like(parent);before[:,5:8,5:8]=name;before[:2,5:8,5:8]=5
        after=before.copy();after[:2,5:8,5:8]=name
        links=pd.DataFrame([
            dict(kind='link',identity=name,**{'pass':'backward'},from_t=4,to_t=3),
            dict(kind='link',identity=name,**{'pass':'backward'},from_t=2,to_t=1,
                 soma_source_frame=3,gap_source_evidence='preceding_soma_sized_body'),
            dict(kind='link',identity=name,**{'pass':'backward'},from_t=1,to_t=0)])
        traces.append((before,after,links))
    return parent,parent.copy(),points,traces


def test_native_names_are_replaced_by_the_accepted_measured_source_owner():
    parent,original,points,traces=example()
    result,audit=carry_measured_gap_owners(parent,original,points,traces,source_frame_offset=0)
    assert np.all(result[:,5:8,5:8]==81)
    assert np.array_equal(result[:,12:15,12:15],parent[:,12:15,12:15])
    assert int((result!=parent).sum())==18
    assert not np.isin(result,[7,9]).any()


def test_prior_revision_protects_the_complete_body():
    parent,original,points,traces=example();original[1,5,5]=77
    result,_=carry_measured_gap_owners(parent,original,points,traces,source_frame_offset=0)
    assert np.array_equal(result[1],parent[1])
    assert np.all(result[0,5:8,5:8]==81)


def test_no_ownership_change_cannot_become_a_feedback_style_edit():
    parent,original,points,traces=example()
    traces=[(after.copy(),after,links) for before,after,links in traces]
    result,_=carry_measured_gap_owners(parent,original,points,traces,source_frame_offset=0)
    assert np.array_equal(result,parent)


def test_existing_independent_holder_prevents_duplicate_identity():
    parent,original,points,traces=example()
    parent[1,12:15,12:15]=81;original=parent.copy()
    for before,after,links in traces:after[1,12:15,12:15]=99
    result,_=carry_measured_gap_owners(parent,original,points,traces,source_frame_offset=0)
    assert np.array_equal(result[1],parent[1])


def test_disagreement_between_tracking_passes_preserves_the_parent():
    parent,original,points,traces=example()
    before,after,links=traces[1];traces[1]=(after.copy(),after,links)
    result,_=carry_measured_gap_owners(parent,original,points,traces,source_frame_offset=0)
    assert np.array_equal(result,parent)
