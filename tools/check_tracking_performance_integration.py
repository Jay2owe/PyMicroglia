"""Exercise composed performance contexts against exact native/correction inputs."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('source','engine','output','scratch'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fixture',type=Path)
    p.add_argument('--native-only',action='store_true')
    a=p.parse_args()
    if not a.native_only and a.fixture is None:p.error('--fixture is required unless --native-only')
    platform.platform=lambda *a,**k:'Windows-tracking-integration'
    sys.path[:0]=[str(a.source.resolve()),str((a.engine/'code').resolve())]
    import numpy as np
    import pandas as pd
    import tifffile
    from pymicroglia.tracking import frozen_io,completed_feedback,identity_performance
    from pymicroglia.tracking.correction_audit import array_sha256
    common=importlib.import_module('common')
    tracking=importlib.import_module('tracking')
    history=importlib.import_module('accepted_history')
    original_features=tracking.link_features
    original_assign=tracking._assign
    original_motion=tracking.motion_support
    territory=importlib.import_module('territory_reconciliation')
    original_territory=territory._observations
    original_loader=history._issue_module
    original_cached=history.cached_dormant_detect
    original_components=history._cropped_frame_components
    confidence=importlib.import_module('lineage_confidence')
    original_confidence=confidence.identity_confidence_table
    reconnect=importlib.import_module('reconnected_companion_merge_partition')
    original_reconnect=reconnect.discover
    merges=importlib.import_module('persistent_merges')
    motion=importlib.import_module('motion_handoffs')
    original_merge_geometry=merges._frame_geometry
    original_merge_split=merges._split_interval
    original_motion_split=motion._split_interval
    original_motion_history=motion.carry_established_identities_along_motion
    original_motion_lobes=motion._motion_lobes
    oscillatory=importlib.import_module('oscillatory_reconciliation')
    original_host=history.optimise_host_merge_graph
    pipeline=importlib.import_module('pipeline')
    original_pipeline_host=pipeline.optimise_host_merge_graph
    original_pipeline_confidence=pipeline.identity_confidence_table
    persistence=importlib.import_module('persistence')
    ledger=importlib.import_module('global_soma_ledger')
    original_green=persistence.green_evidence_for_fixed_objects
    original_physical=ledger.physical_soma_tracking_evidence
    original_conflicts=ledger.substantial_identity_conflicts
    quarantine=history._issue_module(history.I009,'m2_ordered_quarantine')
    original_cleanup=quarantine.clean_duplicates
    early_quarantine=history._issue_module(history.I008,'m7_global_quarantine')
    original_early_detect=early_quarantine.detect_events
    early_duplicates=history._issue_module(history.I008,'m9_duplicate_cleanup')
    original_early_cleanup=early_duplicates.clean_duplicates
    sampler=importlib.import_module('separable_merge_recovery')
    original_sampler=sampler.attach_owners
    resident=importlib.import_module('resident_takeover')
    original_resident=resident.attach_owners
    stationary=importlib.import_module('stationary_reconciliation')
    original_stationary=stationary.discover_stationary_takeovers
    original_history_stationary=history.discover_stationary_takeovers
    hierarchy=history._issue_module(history.I009,'m6_hierarchy_reconcile')
    original_ordered=hierarchy.apply_ordered
    territory_memory=importlib.import_module('territory_memory')
    original_peak=territory_memory._centre_surround_peak
    original_frames=oscillatory._frames
    original_transition=oscillatory._transition_evidence
    excursion=importlib.import_module('bounded_owner_excursion')
    original_excursion=excursion.discover
    original_marker=excursion._marker
    duplicate=importlib.import_module('duplicate_soma_exclusivity')
    flip=importlib.import_module('complete_flip_recovery')
    a.scratch.mkdir(parents=True,exist_ok=True)
    discover=flip.discover
    components=flip.component_accounting.label_value_components
    shape=(5,24,29)
    labels=np.zeros(shape,np.uint16)
    labels[:,2:6,3:7]=1
    labels[2:,2:6,3:7]=2
    visible=pd.DataFrame([dict(frame=t,track_id=1,candidate_owner=1 if t<2 else 2,
        x=4,y=4,strong=True,radius_px=2) for t in range(5)])
    expected=flip._terminal_successor_proposals(labels,visible,{},1.)
    original_apply=flip.apply
    proposals=[dict(eligible=True,source_identity=1,lineage_identity=3,
        proposal_id='fixture',atomic_group_id='fixture',_proof='fixture',
        _component_keys=[flip.ComponentKey(0,1)])]
    expected_application=original_apply(labels,proposals)
    with frozen_io.install(common,a.engine/'code',a.scratch):
        assert history._cropped_frame_components is not original_components
        assert confidence.identity_confidence_table is not original_confidence
        assert reconnect.discover is not original_reconnect
        assert merges._frame_geometry is not original_merge_geometry
        assert merges._split_interval is not original_merge_split
        assert motion._split_interval is merges._split_interval
        assert motion.carry_established_identities_along_motion is not original_motion_history
        assert motion._motion_lobes is not original_motion_lobes
        assert tracking.motion_support is not original_motion
        assert territory._observations is not original_territory
        assert history.optimise_host_merge_graph is not original_host
        assert pipeline.optimise_host_merge_graph is history.optimise_host_merge_graph
        assert pipeline.identity_confidence_table is confidence.identity_confidence_table
        assert persistence.green_evidence_for_fixed_objects is not original_green
        assert ledger.green_evidence_for_fixed_objects is pipeline.green_evidence_for_fixed_objects
        assert ledger.physical_soma_tracking_evidence is not original_physical
        assert ledger.substantial_identity_conflicts is not original_conflicts
        checked_quarantine=history._issue_module(history.I009,'m2_ordered_quarantine')
        assert checked_quarantine.clean_duplicates is not original_cleanup
        assert history._issue_module(history.I008,'m7_global_quarantine').detect_events is not original_early_detect
        assert history._issue_module(history.I008,'m9_duplicate_cleanup').clean_duplicates is not original_early_cleanup
        checked_profile=history._issue_module(history.I010,'m1_profile_persistence')
        checked_reclaimer=history._issue_module(history.I010,'m3_reclaim_gaps')
        assert Path(checked_profile.profile.__code__.co_filename).name=='persistence_profile_reuse.py'
        assert Path(checked_reclaimer.reclaim.__code__.co_filename).name=='persistence_profile_reuse.py'
        checked_tenure=history._issue_module(history.I008,'m16_persistent_quarantine')
        assert Path(checked_tenure.physical_chain_backward.__code__.co_filename).name=='tenure_geometry_reuse.py'
        assert sampler.attach_owners is not original_sampler
        assert resident.attach_owners is not original_resident
        assert stationary.discover_stationary_takeovers is not original_stationary
        assert history.discover_stationary_takeovers is stationary.discover_stationary_takeovers
        assert history._issue_module(history.I009,'m6_hierarchy_reconcile').apply_ordered is not original_ordered
        assert territory_memory._centre_surround_peak is not original_peak
        probe=np.arange(400,dtype=np.float32).reshape(20,20)
        assert territory_memory._centre_surround_peak(probe,9.,9.,2,2,3,5)==original_peak(probe,9.,9.,2,2,3,5)
        assert oscillatory._frames is not original_frames
        assert oscillatory._transition_evidence is not original_transition
        before_index,after_index=original_frames(labels),oscillatory._frames(labels)
        assert list(before_index)==list(after_index)
        for identity in before_index:np.testing.assert_array_equal(before_index[identity],after_index[identity])
        restorer=history._issue_module(history.I012,'m9_reconcile_dormant_releases')
        assert hasattr(restorer.reconcile,'__wrapped__')
        assert excursion.discover is not original_excursion
        assert excursion._marker is not original_marker
        assert tracking.link_features is not original_features
        assert tracking._assign is not original_assign
        detector=history._issue_module(history.I012,'m8_detect_dormant_releases')
        assert getattr(detector.detect,'_compact_dormant_cache',False)
        assert history.cached_dormant_detect(detector) is detector.detect
        detector.detect(labels,np.ones_like(labels,dtype=float),np.zeros_like(labels,dtype=float),
            pd.DataFrame([dict(action='dormancy_started',source_imagej_frame=3,prior_identity=1,host_identity=2)]),0,{})
        ownership=history._issue_module(history.I012,'m1_reconcile_domain_ownership')
        assert ownership.apply_dormancy.__code__.co_filename=='<reused-ownership-observations>'
        assert flip.discover is not discover
        assert flip.component_accounting.label_value_components is not components
        assert flip._terminal_successor_proposals(labels,visible,{},1.)==expected
        assert flip.apply is not original_apply
        application=flip.apply(labels,proposals)
        np.testing.assert_array_equal(application[0],expected_application[0])
        np.testing.assert_array_equal(application[2],expected_application[2])
        assert application[1].to_csv(index=False)==expected_application[1].to_csv(index=False)
    assert flip.discover is discover
    assert history._cropped_frame_components is original_components
    assert confidence.identity_confidence_table is original_confidence
    assert reconnect.discover is original_reconnect
    assert merges._frame_geometry is original_merge_geometry
    assert merges._split_interval is original_merge_split and motion._split_interval is original_motion_split
    assert motion.carry_established_identities_along_motion is original_motion_history
    assert motion._motion_lobes is original_motion_lobes
    assert tracking.link_features is original_features and tracking._assign is original_assign
    assert tracking.motion_support is original_motion and territory._observations is original_territory
    assert history._issue_module is original_loader and history.cached_dormant_detect is original_cached
    assert excursion.discover is original_excursion
    assert excursion._marker is original_marker
    assert flip.apply is original_apply
    assert flip.component_accounting.label_value_components is components
    assert history.optimise_host_merge_graph is original_host and oscillatory._frames is original_frames
    assert oscillatory._transition_evidence is original_transition
    assert pipeline.optimise_host_merge_graph is original_pipeline_host
    assert pipeline.identity_confidence_table is original_pipeline_confidence
    assert persistence.green_evidence_for_fixed_objects is original_green
    assert ledger.physical_soma_tracking_evidence is original_physical
    assert ledger.substantial_identity_conflicts is original_conflicts
    assert quarantine.clean_duplicates is original_cleanup
    assert early_quarantine.detect_events is original_early_detect
    assert early_duplicates.clean_duplicates is original_early_cleanup
    assert sampler.attach_owners is original_sampler
    assert resident.attach_owners is original_resident
    assert stationary.discover_stationary_takeovers is original_stationary
    assert history.discover_stationary_takeovers is original_history_stationary
    assert Path(checked_profile.profile.__code__.co_filename).name=='m1_profile_persistence.py'
    assert Path(checked_reclaimer.reclaim.__code__.co_filename).name=='m3_reclaim_gaps.py'
    assert Path(checked_tenure.physical_chain_backward.__code__.co_filename).name=='m16_persistent_quarantine.py'
    assert hierarchy.apply_ordered is original_ordered
    assert territory_memory._centre_surround_peak is original_peak
    if a.native_only:
        from benchmark_working_storage import load
        storage=load(a.source.resolve()/'array_storage.py')
        raw=np.ones(labels.shape,np.float32)
        expected_pixels=common.outline_overlay(raw,labels,thick=2)
        pool=storage.Pool(a.scratch/'display',minimum_bytes=1,temporary_ram_bytes=1024**2)
        with storage.install(pool):
            with frozen_io.install(common,a.engine/'code',a.scratch,array_pool=pool):
                assert hasattr(common.outline_overlay,'__wrapped__')
                actual_pixels=common.outline_overlay(raw,labels,thick=2)
                np.testing.assert_array_equal(actual_pixels,expected_pixels)
                assert pool._resident_ndim.get()==2
                assert pool.temporary_ram_peak_bytes>0
                del actual_pixels
        report=dict(status='pass',native_entry_patched_and_restored=True,
            host_partition_and_dormant_path_enabled=True,
            display_budget_entry_pixels_and_restoration=True,
            scope='Small native integration; no complete-movie or correction-group claim')
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
        return
    fixture=json.loads(a.fixture.read_text())
    for path,expected_hash in fixture['files'].items():
        with Path(path).open('rb') as handle:
            assert hashlib.file_digest(handle,'sha256').hexdigest()==expected_hash
    parent=tifffile.imread(fixture['parent'])
    original=np.load(fixture['original'],allow_pickle=False)
    points=pd.read_csv(fixture['points'],low_memory=False)
    traces=[(np.load(r['before'],allow_pickle=False),np.load(r['after'],allow_pickle=False),
        pd.read_pickle(r['links'])) for r in fixture['traces']]
    spec=fixture['spec']
    evidence={spec['stage']:dict(original_labels=original,points=points,traces=traces)}
    start=time.perf_counter()
    with completed_feedback.methods(Path(fixture['correction_root'])) as sequence:
        old=sequence.final.frozen_method
        with identity_performance.corrections(sequence.final):
            with sequence.final.frozen_method(spec) as method:
                from pymicroglia.tracking.compact_components import CompactLabeller
                assert isinstance(sys.modules[method.__module__].ndi,CompactLabeller)
            rows=list(sequence.final.apply(parent,np.zeros_like(parent),evidence,[spec],
                source_frame_offset=fixture['source_frame_offset']))
        assert sequence.final.frozen_method is old
    _,result,ledger,tables=rows[0]
    assert array_sha256(result)==fixture['expected_label_sha256']
    assert tables['source_branch_audit'].to_csv(index=False).encode()==Path(fixture['expected_audit']).read_bytes()
    assert not np.any(ledger)
    report=dict(status='pass',native_entry_patched_and_restored=True,
        terminal_proposals_equal=True,compact_gap_entry_enabled=True,
        lazy_application_entry_equal=True,
        complete_gap_labels_and_audit_equal=True,gap_seconds=time.perf_counter()-start,
        sources={str(p.relative_to(a.source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in a.source.rglob('*.py')})
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
