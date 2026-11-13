"""Run the existing pipeline with exact-output physical-stage acceleration.

Scientific settings and the ordinary pipeline arguments are unchanged. The
optional cache contains raw evidence only; label ownership is always recomputed.
This entry point can be disabled without changing analysis configuration.
"""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import hashlib
from pathlib import Path
import runpy
import sys


def resolve_config_time_units(config):
    """Resolve only the six declared timing fields; keep other choices intact."""
    owner=config.values.get('accepted_postprocessing',{}).get('field_wide_owner_consensus',{})
    specification=owner.get('time_units')
    if specification is None:
        return config
    from tracking_time_units import resolve
    translated=resolve(specification,config.values['frame_interval_min'])
    values=deepcopy(config.values)
    target=values['accepted_postprocessing']['field_wide_owner_consensus']['latent_tracking']
    for key in ('maximum_track_gap_frames','maximum_reconnection_gap_frames',
                'base_link_distance_px','additional_link_px_per_gap',
                'reconnection_base_distance_px','reconnection_distance_px_per_gap'):
        target[key]=translated[key]
    return type(config)(values,config.path)


def main(argv=None, *, runner=None):
    """Install accepted execution contexts around the CLI or an embedded runner."""
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--evidence-cache',type=Path,
                        default=Path(__file__).resolve().parents[1]/'.cache/physical_evidence')
    parser.add_argument('--no-evidence-cache',action='store_true')
    parser.add_argument('--no-physical-speedups',action='store_true')
    parser.add_argument('--stage-timings',type=Path,help='Write correction-operation wall times to a new JSONL file')
    options,remaining=parser.parse_known_args(argv)
    target=Path(__file__).with_name('pipeline.py')
    previous_argv=sys.argv
    sys.argv=[str(target),*remaining]
    if '--help' in remaining or '-h' in remaining:
        print('Execution options: --evidence-cache PATH, --no-evidence-cache, --no-physical-speedups, --stage-timings PATH')
        try:runpy.run_path(str(target),run_name='__main__')
        finally:sys.argv=previous_argv
        return
    import common
    import accepted_history as history
    import latent_body_tracks as linker
    import owner_consensus as owner
    from physical_tracking_speed import physical_tracking_speed
    from physical_evidence_cache import persistent_evidence
    modules=('optimized_pipeline.py','physical_tracking_speed.py','physical_evidence_cache.py',
             'tracking_time_units.py','encounter_speed.py','valley_sampling_speed.py','reconnection_speed_spatial.py','motion_sampling_speed.py','relative_detection_visibility.py','relative_latent_visibility.py','relative_valley_ratio.py','fractional_raw_peaks.py','peak_search_speed.py','owner_consensus.py','owner_sampling_speed.py','owner_run_speed.py','parallel_detection_context.py','accepted_history.py','latent_body_tracks.py','raw_physical_hypotheses.py','pipeline.py')
    modules=(*modules, 'detector_spatial_units.py', 'tracking_physical_units.py', 'adaptive_link_search.py', 'detector_core_radius_units.py', 'ownership_spatial_units.py', 'contact_spatial_units.py', 'ownership_time_units.py')
    modules=(*modules,'relay_pair_speed.py','cross_reference_owner_relay.py','partial_body_transfers.py','lineage_confidence.py')
    modules=(*modules,'array_bounds.py','model_review.py','global_soma_ledger.py')
    modules=(*modules,'midpoint_reconciliation.py')
    modules=(*modules,'persistent_merges.py','merge_partition_speed.py')
    modules=(*modules,'dormant_component_speed.py')
    modules=(*modules,'component_feature_speed.py')
    modules=(*modules,'recurrent_two_body_fusion.py')
    modules=(*modules,'component_accounting.py','complete_flip_recovery.py','tracking.py','motion_handoffs.py')
    modules=(*modules,'corrections/ownership_quarantine/global_quarantine.py','corrections/ownership_quarantine/general_merge_bridge.py','corrections/ownership_quarantine/duplicate_cleanup.py','corrections/ownership_quarantine/component_tenure_audit.py','corrections/ownership_quarantine/tenure_quarantine.py')
    modules=(*modules,'inferred_raw_soma_evidence.py','complete_stationary_soma_path.py','corrections/seat_identity/seat_identity.py')
    modules=(*modules,'transiently_misowned_body_recovery.py')
    hashes={name:hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in modules}
    modules=(*modules,'reciprocal_lookup_speed.py','lifetime_endpoint_speed.py','history_stage_timing.py')
    hashes.update({name:hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in modules[-3:]})
    history_speed_statistics={'reciprocal_discovery':[],'lifetime_discovery':[]}
    previous_finish=common.Run.finish
    previous_load=common.Config.load
    previous_load_descriptor=common.Config.__dict__['load']
    resolved_time_configurations=[]
    resolved_spatial_configurations=[]
    resolved_core_radius_configurations=[]
    resolved_ownership_spatial_configurations=[]
    resolved_ownership_time_configurations=[]
    resolved_contact_spatial_configurations=[]
    resolved_tracking_physical_configurations=[]
    def load(cls,path=None):
        original=previous_load(path)
        from tracking_physical_units import resolve_config as resolve_tracking_physical
        physical=resolve_tracking_physical(original)
        if physical is not original:
            owner=physical.values['accepted_postprocessing']['field_wide_owner_consensus']
            resolved_tracking_physical_configurations.append(dict(source_config=str(physical.path),
                pixel_size_um=physical.values['pixel_size_um'],frame_interval_min=physical.values['frame_interval_min'],
                specification=owner['tracking_physical_units'],resolved_latent_tracking=owner['latent_tracking']))
        resolved=resolve_config_time_units(physical)
        if resolved is not physical:
            owner=resolved.values['accepted_postprocessing']['field_wide_owner_consensus']
            resolved_time_configurations.append(dict(source_config=str(resolved.path),
                frame_interval_min=resolved.values['frame_interval_min'],specification=owner['time_units'],
                resolved_latent_tracking=owner['latent_tracking']))
        from detector_spatial_units import resolve_config
        spatial=resolve_config(resolved)
        if spatial is not resolved:
            owner=spatial.values['accepted_postprocessing']['field_wide_owner_consensus']
            resolved_spatial_configurations.append(dict(source_config=str(spatial.path),
                pixel_size_um=spatial.values['pixel_size_um'],
                specification=owner['detector_spatial_units'],
                resolved_raw_detection=owner['raw_detection'],
                scope='Four detector lengths only; no whole-pipeline spatial invariance claim'))
        from detector_core_radius_units import resolve_config as resolve_core_radius
        resolved=resolve_core_radius(spatial)
        if resolved is not spatial:
            owner=resolved.values['accepted_postprocessing']['field_wide_owner_consensus']
            resolved_core_radius_configurations.append(dict(source_config=str(resolved.path),
                pixel_size_um=resolved.values['pixel_size_um'],
                specification=owner['detector_core_radius_units'],
                resolved_raw_detection=owner['raw_detection'],
                scope='Existing core-radius bounds expressed as physical lengths; no new detector operation'))
        from ownership_spatial_units import resolve_config as resolve_ownership
        ownership=resolve_ownership(resolved)
        if ownership is not resolved:
            owner=ownership.values['accepted_postprocessing']['field_wide_owner_consensus']
            resolved_ownership_spatial_configurations.append(dict(source_config=str(ownership.path),
                pixel_size_um=ownership.values['pixel_size_um'],
                specification=owner['ownership_spatial_units'],
                resolved_owner_consensus=owner['owner_consensus'],
                scope='Existing ownership disk bounds and component seed floor expressed as physical lengths'))
        from ownership_time_units import resolve_config as resolve_ownership_time
        timed_ownership=resolve_ownership_time(ownership)
        if timed_ownership is not ownership:
            owner=timed_ownership.values['accepted_postprocessing']['field_wide_owner_consensus']
            resolved_ownership_time_configurations.append(dict(source_config=str(timed_ownership.path),
                frame_interval_min=timed_ownership.values['frame_interval_min'],
                specification=owner['ownership_time_units'],resolved_owner_consensus=owner['owner_consensus'],
                scope='Two existing ownership elapsed-time bounds; supporting observations remain counts'))
        ownership=timed_ownership
        from contact_spatial_units import resolve_config as resolve_contact
        contact=resolve_contact(ownership)
        if contact is not ownership:
            owner=contact.values['accepted_postprocessing']['field_wide_owner_consensus']
            resolved_contact_spatial_configurations.append(dict(source_config=str(contact.path),
                pixel_size_um=contact.values['pixel_size_um'],specification=owner['contact_spatial_units'],
                resolved_latent_tracking=owner['latent_tracking'],
                scope='Existing encounter distance floors only; inherited valley-sampling translation sensitivity remains'))
        return contact
    statistics={};entries=[]
    def finish(run,summary):
        run.manifest['normalisation'] = dict(relative_detection_visibility=True,relative_latent_visibility=True,fractional_raw_peak_exports=True,relative_valley_ratio=False,scope='Detector and gap visibility plus fractional peak exports: nine native stage tables exact; benchmark physical-stage raw-gain checks; no full-pipeline or offset invariance claim')
        used={event.get('source',event.get('directory')) for event in statistics.get('events',[])}
        run.manifest['execution_acceleration']=dict(
            exact_physical_speedups=not options.no_physical_speedups,
            exact_history_speedups=not options.no_physical_speedups,
            history_speed_statistics=history_speed_statistics,
            stage_timing_log=str(options.stage_timings) if options.stage_timings else None,
            evidence_cache_enabled=not options.no_evidence_cache,
            code_sha256=hashes,cache_statistics=dict(statistics),
            resolved_time_configurations=list(resolved_time_configurations),
            resolved_spatial_configurations=list(resolved_spatial_configurations),
            resolved_core_radius_configurations=list(resolved_core_radius_configurations),
            resolved_ownership_spatial_configurations=list(resolved_ownership_spatial_configurations),
            resolved_ownership_time_configurations=list(resolved_ownership_time_configurations),
            resolved_contact_spatial_configurations=list(resolved_contact_spatial_configurations),
            resolved_tracking_physical_configurations=list(resolved_tracking_physical_configurations),
            used_evidence_entries=[entry for entry in entries if entry['directory'] in used])
        return previous_finish(run,summary)
    try:
        with ExitStack() as stack:
            if not options.no_physical_speedups:
                from peak_search_speed import bucketed_peak_search
                stack.enter_context(bucketed_peak_search(history.raw_physical_hypotheses))
            from relative_detection_visibility import relative_detection_visibility
            stack.enter_context(relative_detection_visibility(history.raw_physical_hypotheses,reuse_core_structure=not options.no_physical_speedups))
            from fractional_raw_peaks import fractional_raw_peaks
            stack.enter_context(fractional_raw_peaks(history.raw_physical_hypotheses))
            from relative_latent_visibility import relative_latent_visibility
            stack.enter_context(relative_latent_visibility(linker))
            if not options.no_physical_speedups:
                from owner_sampling_speed import batched_owner_samples
                from owner_run_speed import cached_owner_row_types
                stack.enter_context(batched_owner_samples(owner))
                stack.enter_context(cached_owner_row_types(owner))
                from parallel_detection_context import parallel_detection
                stack.enter_context(parallel_detection(history))
                stack.enter_context(physical_tracking_speed(linker))
                import gap_tolerant_reciprocal_seat_exchange as reciprocal
                from reciprocal_lookup_speed import cached_reciprocal_discovery
                from lifetime_endpoint_speed import cached_lifetime_endpoints
                history_speed_statistics['reciprocal_discovery']=stack.enter_context(cached_reciprocal_discovery(reciprocal))
                history_speed_statistics['lifetime_discovery']=stack.enter_context(cached_lifetime_endpoints(history.isolated_unowned_lifetime))
                from relay_pair_speed import faster_relay_pairs
                stack.enter_context(faster_relay_pairs(history.cross_reference_owner_relay))
            if not options.no_evidence_cache:
                statistics,entries=stack.enter_context(persistent_evidence(history,options.evidence_cache))
            if options.stage_timings is not None:
                from history_stage_timing import time_history_stages
                stack.enter_context(time_history_stages(history,options.stage_timings))
            common.Run.finish=finish
            common.Config.load=classmethod(load)
            if runner is None:
                runpy.run_path(str(target),run_name='__main__')
            else:
                runner()
    finally:
        common.Run.finish=previous_finish
        common.Config.load=previous_load_descriptor
        sys.argv=previous_argv


if __name__=='__main__':
    main()
