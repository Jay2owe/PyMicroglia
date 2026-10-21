"""Compose the verified isolated adapters; never mutate a live installation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--adapters',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--storage-source',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    shutil.copytree(a.base,a.output,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    target=a.output/'pymicroglia/tracking'
    names=('identity_reuse','frame_component_reuse','terminal_feature_reuse','lazy_flip_components','compact_components',
           'dormant_component_reuse','ownership_observation_reuse','tracking_feature_batch','companion_lookup_reuse',
           'dormant_path_reuse','host_partition_reuse','diagnostic_shortlist','hierarchy_metric_reuse',
           'territory_observation_reuse','lazy_motion_counts','confidence_geometry_reuse',
           'hierarchy_component_batch','reconnection_candidate_index','reconnection_area_reuse',
           'merge_geometry_reuse','merge_proposal_storage','motion_history_reuse',
           'merge_foreground_scope','motion_overlap_batch','motion_lobe_batch','tracking_plain_batch',
           'hierarchy_overlap_reuse','fixed_object_overlap','relay_transition_geometry','conflict_core_batch',
           'duplicate_fragment_geometry','hierarchy_candidate_filter','territory_peak_geometry','measured_gap_path_reuse','quarantine_overlap_batch',
           'correction_geometry_storage','correction_owner_sampling',
           'tenure_geometry_reuse','persistence_profile_reuse','stationary_candidate_filter','bounded_marker_geometry',
           'display_block_residency','point_disk_votes','ownerless_coverage_batch','bounded_morphology_loader','convex_hull_image_fast','merge_split_feature_index','reciprocal_component_crop','component_marker_prefilter','census_distance_crop','merge_continuation_frames','two_body_fusion_index','gap_bracket_memo','tracking_matrix_batch','identity_name_batch','territory_overlap_index','midpoint_bookend_reuse','relay_pair_reuse','merge_bridge_objects','observation_boxes','presence_contrast_crop','census_presence_table','unclaimed_body_boxes','ownership_region_rows','sparse_pair_table','classify_disc_boxes','presence_area_table','end_path_counts','seat_overlap_counts','display_lookup','flip_boundary_crop','unclaimed_frame_domains','lineage_frame_components','plain_component_frames','structural_reference_boxes','stationary_pair_median','nested_alias_reuse','post_split_reuse','branched_lineage_reuse','bracketed_handoff_reuse','owner_consensus_presence','terminal_companion_reuse','retroactive_separation_reuse','observation_cache','soma_separation_boxes','presence_mask_boxes','motion_union_bounds','merge_mask_counts','alias_host_boxes','merge_partition_boxes','events_host_boxes','two_core_audit_boxes','presence_settle_boxes','physical_core_boxes','relay_pair_batch','start_duplicate_excess','exclusive_pair_index','censored_area_boxes','speckle_alias_presence','presence_descriptor_medians','raw_support_memo','reference_area_boxes','core_replacement_index','destination_support_boxes','birth_lineage_index','process_extension_boxes','recording_step_memo','gap_component_boxes','bounded_label','origin_footprint_boxes','guard_scan','tracking_pass_record','fragment_pair_scan','link_paint_boxes','bounded_morphology','occupancy_overlap_boxes','spatial_path_views','optical_physical_paths','identity_performance')
    for name in names:shutil.copyfile(a.adapters/(name+'.py'),target/(name+'.py'))
    def replace(name,old,new):
        path=target/name
        source=path.read_text()
        if source.count(old)!=1:raise ValueError('Unexpected source boundary: '+name)
        path.write_text(source.replace(old,new))
    replace('frozen_io.py','    try:yield\n','    try:\n        from .identity_performance import native\n        with native(code_root,scratch):\n            yield\n')
    replace('frozen_io.py','    replacements={old_observation:',
            '    from .display_block_residency import render_with_budget\n    from .display_lookup import lookup_renderer\n    replacements={old_observation:')
    replace('frozen_io.py','old_overlay:overlay_renderer(common,scratch)}',
            'old_overlay:render_with_budget(lookup_renderer(common,scratch),array_pool)}')
    replace('completed_feedback.py','    with methods(root) as sequence:\n',
            '    from .identity_performance import corrections\n    with methods(root) as sequence, corrections(sequence.final):\n')
    replace('completed_evidence.py',
            "                    after, events, stages = collect_tracking_prefix(cfg,physical,reservation,raw,lag,\n                        observations,observation_table,anchor,reserved,points=native_points,\n                        rule_context=memory.measured_soma_gap_memory,prepared_cache=cache)\n",
            "                    from .measured_gap_path_reuse import install as reuse_paths\n"
            "                    # Only the labels and the stage's seconds are kept from this prefix.\n"
            "                    from .sparse_pair_table import gap_events_discarded\n"
            "                    with reuse_paths(memory), gap_events_discarded():\n                        after, events, stages = collect_tracking_prefix(cfg,physical,reservation,raw,lag,\n                            observations,observation_table,anchor,reserved,points=native_points,\n                            rule_context=memory.measured_soma_gap_memory,prepared_cache=cache)\n")
    # The gap reference is Issue 005's first two tracking passes, which the
    # accepted history records; use them only on an exact input digest match.
    replace('completed_evidence.py',
            "    before, _, stages = collect_tracking_prefix(cfg,physical,reservation,raw,lag,\n        observations,observation_table,anchor,reserved,points=native_points,prepared_cache=cache)\n",
            "    from .tracking_pass_record import load as recorded_passes\n"
            "    recorded = recorded_passes(history,ah,cfg,physical,reservation,raw,lag,\n"
            "        observations,observation_table,anchor,reserved)\n"
            "    if recorded is not None:\n"
            "        before, stages = recorded\n"
            "    else:\n"
            "        before, _, stages = collect_tracking_prefix(cfg,physical,reservation,raw,lag,\n"
            "            observations,observation_table,anchor,reserved,points=native_points,prepared_cache=cache)\n")
    # 72 of 114 accepted corrections change no label pixel. An unchanged integer
    # movie has the same bytes, so the same digest, as at the previous record.
    replace('correction_audit.py',
            "        self.inferred = np.zeros(labels.shape, dtype=bool)\n",
            "        self.inferred = np.zeros(labels.shape, dtype=bool)\n"
            "        self.label_sha256 = None\n")
    replace('correction_audit.py',
            "        result = dict(changed_pixels=changed,\n"
            "                      label_array_sha256=array_sha256(labels),\n",
            "        if (changed == 0 and self.label_sha256 is not None and labels.dtype == self.previous.dtype\n"
            "                and labels.dtype.kind in 'biu'):\n"
            "            label_sha256 = self.label_sha256\n"
            "        else:\n"
            "            label_sha256 = array_sha256(labels)\n"
            "        self.label_sha256 = label_sha256\n"
            "        result = dict(changed_pixels=changed,\n"
            "                      label_array_sha256=label_sha256,\n")
    replace('checkpoints.py',"'engine.py', 'engine_bounded.py', 'frozen_io.py',",
            "'engine.py', 'engine_bounded.py', 'frozen_io.py', "+', '.join(repr(n+'.py') for n in names)+',')
    replace('display_scale.py',"    if raw.dtype.kind=='f' and raw.dtype.itemsize in (2,4,8) and raw.nbytes>32*1024**2:\n",
            "    if (raw.dtype.kind=='f' and raw.dtype.itemsize in (2,4,8) and raw.nbytes>32*1024**2\n            and np.lib.NumpyVersion(np.__version__) >= '2.0.0'):\n")
    # The ordinary route reuses the bounded runner but must keep the frozen
    # runner's worker threads and native review. The bounded route keeps the
    # live contract: m22 on the calling thread (NumPy 2.1 gives fresh threads a
    # different scalar-promotion state, which changes float32 evidence tables)
    # and review deferred to PyMicroglia.
    replace('engine_bounded.py','def run(pipeline, common, run_name, stem, config_path,*,array_pool=None,resume=False):',
            'def run(pipeline, common, run_name, stem, config_path,*,array_pool=None,resume=False,native_review=False):')
    replace('engine_bounded.py','            # Keep both branches in worker-thread numerical contexts. Ordinary\n',
            '            if not native_review:\n'
            '                # Match the frozen runner\'s worker-thread numeric context. NumPy\n'
            '                # 2.1 initialises scalar promotion differently in a fresh thread.\n'
            '                # Wait here so the bounded runner still runs one heavy stage at a time.\n'
            '                with ThreadPoolExecutor(max_workers=1) as executor:\n'
            '                    future=executor.submit(call,\'stage_m20_accepted\',cfg,stem,run_name,sources,final.copy())\n'
            '                    future.result()\n'
            '                call(\'stage_m22_accepted_history\',cfg,stem,run_name,sources,final.copy(),observations,table,anchor,history_inputs=history_inputs,m20_result_future=future,upstream_inferred=pre_inferred,unresolved=unresolved)\n'
            '                # New data have not been approved by a human; don\'t write inherited approval text.\n'
            '                (pipeline.ROOT/"m22_accepted_history"/run_name/"review_pending.json").write_text(json.dumps(dict(status="pending",reason="Full-duration new-input tracking requires a source-linked review; full-field RGB preparation deferred to PyMicroglia."),indent=2))\n'
            '                print(f"DONE {stem}: accepted engine stages completed; new-input review pending",flush=True)\n'
            '                return\n'
            '            # Keep both branches in worker-thread numerical contexts. Ordinary\n')
    replace('engine.py','                run_staged(pipeline, common, run_name, stem, config)',
            '                run_staged(pipeline, common, run_name, stem, config, native_review=True)')
    shutil.copyfile(a.adapters/'test_engine_bounded_routes.py',a.output/'tests/test_engine_bounded.py')
    if a.storage_source:
        shutil.copyfile(a.storage_source,a.output/'array_storage.py')
        replace('engine.py',"pool=Pool(Path(scratch)/'working_arrays',minimum_bytes=working_array_min_bytes)",
                "pool=Pool(Path(scratch)/'working_arrays',minimum_bytes=working_array_min_bytes,temporary_ram_bytes=32*1024**2)")
        replace('correction_storage.py',"minimum_bytes=minimum_bytes)",
                "minimum_bytes=minimum_bytes,temporary_ram_bytes=32*1024**2)")
    def hashes(folder):
        return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.zip')}
    report=dict(status='composed_pending_validation',base=str(a.base.resolve()),
        adapters=str(a.adapters.resolve()),output=str(a.output.resolve()),
        storage_source=str(a.storage_source.resolve()) if a.storage_source else None,
        base_sources=hashes(a.base),adapter_sources=hashes(a.adapters),sources=hashes(a.output),
        scope='Isolated composition; no live production source or active checkpoints changed')
    (a.output/'composition.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],files=len(report['sources']),output=str(a.output))))


if __name__=='__main__':main()
