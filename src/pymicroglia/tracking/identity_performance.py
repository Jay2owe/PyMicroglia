"""Scope verified performance adapters to the frozen scientific implementation."""
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import importlib.util
from pathlib import Path
try:
    from . import resolved_paths as _paths
except ImportError:
    import resolved_paths as _paths
import sys


@contextmanager
def native(code_root, scratch):
    from . import identity_reuse, frame_component_reuse, terminal_feature_reuse, lazy_flip_components
    from . import ownership_observation_reuse, tracking_feature_batch
    from . import companion_lookup_reuse
    from . import host_partition_reuse
    from . import lazy_motion_counts
    from . import confidence_geometry_reuse
    from . import hierarchy_component_batch
    from . import motion_history_reuse
    from . import motion_overlap_batch, motion_lobe_batch
    from . import fixed_object_overlap
    from . import relay_transition_geometry
    from . import conflict_core_batch
    from . import tenure_geometry_reuse
    from . import correction_owner_sampling
    from . import bounded_morphology_loader
    from . import convex_hull_image_fast
    from . import identity_name_batch
    from . import relay_pair_reuse
    from . import ownership_region_rows
    from .frozen_io import LabelMask
    with ExitStack() as stack:
        # First, so every module's ndi.label labels a whole-frame mask inside its rectangle.
        from . import bounded_label
        native_labels=stack.enter_context(bounded_label.install())
        stack.callback(lambda: print('native bounded label',native_labels,file=sys.stderr))
        from . import csv_joined_rows
        csv_rows=stack.enter_context(csv_joined_rows.install())
        stack.callback(lambda: print('csv joined rows',csv_rows,file=sys.stderr))
        from . import disk_owner_batch
        disk_owners=stack.enter_context(disk_owner_batch.install(
            importlib.import_module('merge_residency') if (Path(code_root)/'merge_residency.py').is_file() else None,
            importlib.import_module('persistent_body_identity') if (Path(code_root)/'persistent_body_identity.py').is_file() else None))
        stack.callback(lambda: print('disk owner batch',disk_owners,file=sys.stderr))
        if (Path(code_root)/'merge_residency.py').is_file():
            from . import residency_component_boxes
            residency=stack.enter_context(residency_component_boxes.install(importlib.import_module('merge_residency')))
            stack.callback(lambda: print('residency component boxes',residency,file=sys.stderr))
        stack.enter_context(convex_hull_image_fast.install())
        from . import perimeter_shifts
        stack.enter_context(perimeter_shifts.install())
        from . import mask_centre_counts
        stack.enter_context(mask_centre_counts.install())
        from . import row_tuples
        stack.enter_context(row_tuples.install())
        if (Path(code_root)/'partial_body_transfers.py').is_file():
            from . import two_core_audit_boxes
            stack.enter_context(two_core_audit_boxes.install(importlib.import_module('model_review'),
                importlib.import_module('partial_body_transfers'),importlib.import_module('pipeline')))
        if (Path(code_root)/'global_soma_ledger.py').is_file():
            from . import soma_separation_boxes
            stack.enter_context(soma_separation_boxes.install(importlib.import_module('global_soma_ledger')))
        if (Path(code_root)/'separable_merge_recovery.py').is_file():
            stack.enter_context(correction_owner_sampling.install(importlib.import_module('separable_merge_recovery')))
        if (Path(code_root)/'resident_takeover.py').is_file():
            stack.enter_context(correction_owner_sampling.install(importlib.import_module('resident_takeover')))
            from . import resident_run_rows
            stack.enter_context(resident_run_rows.install(importlib.import_module('resident_takeover')))
        if (Path(code_root)/'owner_consensus.py').is_file():
            consensus_stats=stack.enter_context(correction_owner_sampling.install(importlib.import_module('owner_consensus')))
            stack.callback(lambda: print('consensus owner votes',consensus_stats,file=sys.stderr))
        if (Path(code_root)/'physical_lineage_aliases.py').is_file():
            from . import physical_core_boxes
            stack.enter_context(physical_core_boxes.install(importlib.import_module('physical_lineage_aliases')))
            from . import one_core_union_boxes
            stack.enter_context(one_core_union_boxes.install(importlib.import_module('physical_lineage_aliases')))
            stack.enter_context(mask_centre_counts.install_lineage(importlib.import_module('physical_lineage_aliases')))
        if (Path(code_root)/'persistence.py').is_file():
            stack.enter_context(fixed_object_overlap.install(importlib.import_module('persistence'),
                importlib.import_module('global_soma_ledger'),importlib.import_module('pipeline')))
            stack.enter_context(conflict_core_batch.install(importlib.import_module('global_soma_ledger'),
                importlib.import_module('pipeline')))
        if (Path(code_root)/'tracking.py').is_file():
            tracking=importlib.import_module('tracking')
            stack.enter_context(tracking_feature_batch.install(tracking))
            stack.enter_context(lazy_motion_counts.install(tracking,LabelMask))
            from . import link_paint_boxes
            stack.enter_context(link_paint_boxes.install(tracking,LabelMask))
            from . import occupancy_overlap_boxes
            stack.enter_context(occupancy_overlap_boxes.install(tracking))
            from . import motion_union_bounds
            stack.enter_context(motion_union_bounds.install(tracking, code_root))
        if (Path(code_root)/'accepted_history.py').is_file():
            history=importlib.import_module('accepted_history')
            # Correction evidence reuses Issue 005's first two tracking passes.
            from . import tracking_pass_record
            stack.enter_context(tracking_pass_record.install(history,importlib.import_module('pipeline')))
            if (Path(code_root)/'motion_handoffs.py').is_file():
                stack.enter_context(motion_lobe_batch.install(importlib.import_module('motion_handoffs')))
                stack.enter_context(motion_history_reuse.install(
                    importlib.import_module('motion_handoffs'),importlib.import_module('pipeline'),history))
                stack.enter_context(motion_overlap_batch.install(
                    importlib.import_module('motion_handoffs'),importlib.import_module('pipeline'),history))
            stack.enter_context(hierarchy_component_batch.install(history))
            stack.enter_context(identity_name_batch.install(importlib.import_module('global_soma_ledger'),
                importlib.import_module('lineage_confidence'),history,importlib.import_module('pipeline')))
            from . import ledger_reference_boxes
            reference_stats=stack.enter_context(ledger_reference_boxes.install(importlib.import_module('global_soma_ledger')))
            stack.callback(lambda: print('native ledger references',reference_stats,file=sys.stderr))
            if (Path(code_root)/'post_split_backfill.py').is_file():
                from . import post_split_reuse
                stack.enter_context(post_split_reuse.install(importlib.import_module('post_split_backfill')))
            if (Path(code_root)/'terminal_companion_assimilation.py').is_file():
                from . import terminal_companion_reuse
                stack.enter_context(terminal_companion_reuse.install(importlib.import_module('terminal_companion_assimilation')))
            if (Path(code_root)/'owner_consensus.py').is_file():
                from . import owner_consensus_presence
                stack.enter_context(owner_consensus_presence.install(importlib.import_module('owner_consensus')))
            if (Path(code_root)/'bracketed_handoff_recovery.py').is_file():
                from . import bracketed_handoff_reuse
                stack.enter_context(bracketed_handoff_reuse.install(importlib.import_module('bracketed_handoff_recovery')))
            if (Path(code_root)/'lineage_confidence.py').is_file():
                stack.enter_context(confidence_geometry_reuse.install(importlib.import_module('lineage_confidence'),history,importlib.import_module('pipeline')))
            # First, so later loader adapters copy the bounded ndimage into their globals.
            stack.enter_context(bounded_morphology_loader.install_loader(history))
            stack.enter_context(ownership_region_rows.install_loader(history))
            stack.enter_context(ownership_observation_reuse.install_loader(history))
            stack.enter_context(tenure_geometry_reuse.install_loader(history))
            # Last loader: it takes only functions no other loader replaced.
            from . import sparse_nonzero
            native_sparse=stack.enter_context(sparse_nonzero.install_loader(history))
            stack.callback(lambda: print('native sparse nonzero',native_sparse,file=sys.stderr))
            if (Path(code_root)/'recording_start_duplicate_release.py').is_file():
                from . import start_duplicate_excess
                stack.enter_context(start_duplicate_excess.install(importlib.import_module('recording_start_duplicate_release')))
            if (Path(code_root)/'right_censored_seat_partition.py').is_file():
                from . import censored_area_boxes
                stack.enter_context(censored_area_boxes.install(importlib.import_module('right_censored_seat_partition')))
            if (Path(code_root)/'projection_tolerant_seat_preservation.py').is_file():
                from . import seat_projection_boxes
                seat_areas=stack.enter_context(seat_projection_boxes.install(importlib.import_module('projection_tolerant_seat_preservation')))
                stack.callback(lambda: print('seat projection boxes',seat_areas,file=sys.stderr))
            if (Path(code_root)/'oscillatory_reconciliation.py').is_file():
                stack.enter_context(relay_transition_geometry.install(importlib.import_module('oscillatory_reconciliation')))
                stack.enter_context(relay_pair_reuse.install(importlib.import_module('oscillatory_reconciliation'),relay_transition_geometry))
                from . import relay_metric_reuse
                stack.enter_context(relay_metric_reuse.install(importlib.import_module('oscillatory_reconciliation')))
                stack.enter_context(host_partition_reuse.install(
                    importlib.import_module('oscillatory_reconciliation'), history, importlib.import_module('pipeline')))
        if (Path(code_root)/'bounded_owner_excursion.py').is_file():
            stack.enter_context(companion_lookup_reuse.install(importlib.import_module('bounded_owner_excursion')))
        if (Path(code_root)/'duplicate_soma_exclusivity.py').is_file():
            stack.enter_context(identity_reuse.install(importlib.import_module('duplicate_soma_exclusivity')))
        if (Path(code_root)/'complete_flip_recovery.py').is_file():
            module=importlib.import_module('complete_flip_recovery')
            stack.enter_context(frame_component_reuse.install(module,scratch))
            stack.enter_context(terminal_feature_reuse.install(module))
            stack.enter_context(lazy_flip_components.install(module,scratch))
            from . import flip_boundary_crop
            stack.enter_context(flip_boundary_crop.install(module))
            from . import nested_alias_reuse
            stack.enter_context(nested_alias_reuse.install(module))
        if (Path(code_root)/'latent_body_tracks.py').is_file():
            from . import threaded_evidence_images
            threaded=stack.enter_context(threaded_evidence_images.install(
                importlib.import_module('latent_body_tracks'),importlib.import_module('raw_physical_hypotheses')))
            stack.callback(lambda: print('threaded evidence images',threaded,file=sys.stderr))
        if (Path(code_root)/'raw_physical_hypotheses.py').is_file():
            # Entered inside optimized_pipeline, so the production detector is already composed.
            from . import batched_detection_cores
            detection_cores=stack.enter_context(batched_detection_cores.install(
                importlib.import_module('raw_physical_hypotheses')))
            stack.callback(lambda: print('batched detection cores',detection_cores,file=sys.stderr))
            if (Path(code_root)/'parallel_detection_context.py').is_file():
                detection_pool=stack.enter_context(batched_detection_cores.detection_workers(
                    importlib.import_module('parallel_detection_context')))
                stack.callback(lambda: print('detection workers',detection_pool,file=sys.stderr))
        # Track-row memos for two later identity stages, each scoped to one call.
        if (Path(code_root)/'concurrent_duplicate_invasion.py').is_file():
            from . import invasion_donor_rows
            donor_rows=stack.enter_context(invasion_donor_rows.install(
                importlib.import_module('concurrent_duplicate_invasion')))
            stack.callback(lambda: print('invasion donor rows',donor_rows,file=sys.stderr))
        if (Path(code_root)/'bracketed_ownerless_seat_completion.py').is_file():
            from . import bracketed_seat_rows
            seat_rows=stack.enter_context(bracketed_seat_rows.install(
                importlib.import_module('bracketed_ownerless_seat_completion')))
            stack.callback(lambda: print('bracketed seat rows',seat_rows,file=sys.stderr))
        # Last, so the cache wraps the installed observation reader in every alias.
        from . import observation_cache
        stack.enter_context(observation_cache.install(code_root, LabelMask,
            'fea06f4e36c47e9673fe1b55badf543a3a1315eef012c6728a29bff6e0b2a5ad'))
        yield


COMPOSED=('embedded_handoff_composition','optical_source_composition',
          'owner_episode_composition','ownerless_contact_composition')


def preload_composed():
    """Import the corrections these compositions import only when they run.

    Each composition calls importlib.import_module(name).correct stage by stage,
    after the correction adapters have looked for loaded modules, so those
    modules ran unadapted. Importing them first gives the composition the same
    module objects from sys.modules. A module that fails to import here is left
    out of sys.modules by Python and raises again where the composition imports it.
    """
    stats=dict(loaded=0,imported=0,failed=0)
    for composition in COMPOSED:
        try:
            stages=importlib.import_module(composition).STAGES
        except Exception:
            stats['failed']+=1
            continue
        for stage in stages:
            name=stage[1]
            if name in sys.modules:
                stats['loaded']+=1
                continue
            try:
                importlib.import_module(name)
                stats['imported']+=1
            except Exception:
                stats['failed']+=1
    return stats


@contextmanager
def corrections(final):
    from . import compact_components, correction_owner_sampling
    original=final.frozen_method
    raw_support=None
    try:
        from . import raw_support_memo
        raw_support=raw_support_memo.Memo()
    except ImportError:
        pass
    reference_stats=dict(calls=0,labelled=0,reused=0,fallback=0)
    core_stats=dict(adapted=0,indexes=0,lookups=0)
    destination_stats=dict(versions=0,frames=0,boxed=0,fallback=0)
    birth_stats=dict(versions=0,tables=0,fallback=0)
    gap_stats=dict(versions=0,boxed=0,fallback=0)
    origin_stats=dict(versions=0,frames=0,boxed=0,fallback=0)
    measured_core_stats=dict(versions=0,vectorised=0,original=0)
    spatial_stats={}
    from . import spatial_path_views as _spatial_views
    spatial_parts=_spatial_views.PartsMemo()
    optical_stats={}
    gap_owner_stats={}
    gap_trace_stats={}
    core_window_stats={}
    coupled_stats={}
    flicker_stats={}
    bracketed_stats={}
    track_sort_stats={}
    sparse_stats={}
    core_stable_stats={}
    centroid_stats={}
    smoothed_stats={}
    handoff_stats={}
    try:
        from . import reference_area_boxes
    except ImportError:
        reference_area_boxes=None
    shared_samplers={name:importlib.import_module(name) for name in
                     ('separable_merge_recovery','resident_takeover')}
    @contextmanager
    def method(spec):
        with original(spec) as function, ExitStack() as stack:
            from . import module_scan
            with module_scan.held():
                adapted=_install_stage(spec,function,stack,module_scan)
            yield adapted
    def _install_stage(spec,function,stack,module_scan):
        module=sys.modules[function.__module__]
        # This version has been audited for read-only component consumers.
        # Future scientific implementations retain their native storage.
        supported=(spec.get('kind')=='gap' and module_scan.file_sha256(module.__file__)==
                   '81a7217a8ef0b56fc768522f1f797c0fd597e25235138f267526ef50ab212fa5')
        if supported:stack.enter_context(compact_components.install(module))
        from . import gap_component_boxes
        stack.enter_context(gap_component_boxes.install(module,gap_stats))
        from . import gap_trace_walk
        stack.enter_context(gap_trace_walk.install(module,gap_trace_stats))
        from . import gap_owner_source_memo
        stack.enter_context(gap_owner_source_memo.install(module,gap_owner_stats))
        for name,shared in shared_samplers.items():
            sampler=sys.modules.get(name)
            if sampler is not None and sampler is not shared:
                stack.enter_context(correction_owner_sampling.install(sampler))
        if raw_support is not None:stack.enter_context(raw_support_memo.install(raw_support))
        from . import spatial_path_views
        stack.enter_context(spatial_path_views.install(spatial_stats,spatial_parts))
        from . import optical_physical_paths
        stack.enter_context(optical_physical_paths.install(optical_stats))
        from . import core_intersects_window
        stack.enter_context(core_intersects_window.install(core_window_stats))
        from . import coupled_peer_memo
        stack.enter_context(coupled_peer_memo.install(coupled_stats))
        from . import flicker_component_boxes
        stack.enter_context(flicker_component_boxes.install(flicker_stats))
        from . import bracketed_consensus_tracks
        stack.enter_context(bracketed_consensus_tracks.install(bracketed_stats))
        from . import track_sort_skips
        stack.enter_context(track_sort_skips.install(track_sort_stats))
        from . import handoff_component_box
        stack.enter_context(handoff_component_box.install(handoff_stats))
        # Before the rewrites that copy module globals, so they hold the swap.
        from . import measured_cores
        stack.enter_context(measured_cores.install(measured_core_stats))
        from . import destination_support_boxes
        stack.enter_context(destination_support_boxes.install(destination_stats))
        from . import convex_hull_image_fast
        stack.enter_context(convex_hull_image_fast.aliases())
        from . import birth_lineage_index
        stack.enter_context(birth_lineage_index.install(birth_stats))
        if reference_area_boxes is not None:stack.enter_context(reference_area_boxes.install_loaded(reference_stats))
        from . import origin_footprint_boxes
        stack.enter_context(origin_footprint_boxes.install(origin_stats))
        from . import core_stable_memo
        stack.enter_context(core_stable_memo.install(core_stable_stats))
        from . import reciprocal_centroid_sums
        stack.enter_context(reciprocal_centroid_sums.install(centroid_stats))
        from . import smoothed_frame_memo
        stack.enter_context(smoothed_frame_memo.install(smoothed_stats))
        # Last: it takes only functions no other adapter replaced.
        from . import sparse_nonzero
        stack.enter_context(sparse_nonzero.install(sparse_stats))
        # frozen_method fetched the stage function before these installs;
        # a stage whose own function an adapter replaced uses the replacement.
        current=getattr(module,function.__name__,function)
        if getattr(current,'__wrapped__',None) is function:function=current
        from . import core_replacement_index
        return core_replacement_index.adapt(function,core_stats)
    aliases=[];root=_paths.resolve(final.__file__).parent
    with ExitStack() as stack:
        composed_stats=preload_composed()
        print('correction composed preload',composed_stats,file=sys.stderr)
        from . import csv_joined_rows
        correction_csv_rows=stack.enter_context(csv_joined_rows.install())
        stack.callback(lambda: print('correction csv joined rows',correction_csv_rows,file=sys.stderr))
        # Corrections run outside the native context, where row tuples are installed.
        from . import row_tuples
        row_stats=stack.enter_context(row_tuples.install())
        stack.callback(lambda: print('correction row tuples',row_stats,file=sys.stderr))
        from . import process_track_features
        track_feature_stats=stack.enter_context(process_track_features.install(sys.modules.get('process_inventory_tracked')))
        stack.callback(lambda: print('correction process track features',track_feature_stats,file=sys.stderr))
        for sampler in shared_samplers.values():
            stack.enter_context(correction_owner_sampling.install(sampler))
        # Families applied directly by the sequence (fragments, contacts and the
        # other owner episodes) never pass through method(spec); their modules
        # are already loaded here, so these adapters cover every stage.
        if raw_support is not None:stack.enter_context(raw_support_memo.install(raw_support))
        from . import spatial_path_views
        stack.enter_context(spatial_path_views.install(spatial_stats,spatial_parts))
        stack.callback(lambda: print('correction spatial path views',spatial_stats,file=sys.stderr))
        from . import optical_physical_paths
        stack.enter_context(optical_physical_paths.install(optical_stats))
        stack.callback(lambda: print('correction optical physical paths',optical_stats,file=sys.stderr))
        from . import core_intersects_window
        stack.enter_context(core_intersects_window.install(core_window_stats))
        stack.callback(lambda: print('correction core windows',core_window_stats,file=sys.stderr))
        from . import coupled_peer_memo
        stack.enter_context(coupled_peer_memo.install(coupled_stats))
        stack.callback(lambda: print('correction coupled peers',coupled_stats,file=sys.stderr))
        from . import flicker_component_boxes
        stack.enter_context(flicker_component_boxes.install(flicker_stats))
        from . import bracketed_consensus_tracks
        stack.enter_context(bracketed_consensus_tracks.install(bracketed_stats))
        from . import track_sort_skips
        stack.enter_context(track_sort_skips.install(track_sort_stats))
        stack.callback(lambda: print('correction flicker boxes',flicker_stats,file=sys.stderr))
        stack.callback(lambda: print('correction bracketed consensus tracks',bracketed_stats,file=sys.stderr))
        stack.callback(lambda: print('correction track sort skips',track_sort_stats,file=sys.stderr))
        from . import track_row_index
        track_row_stats=stack.enter_context(track_row_index.install())
        stack.callback(lambda: print('correction track row index',track_row_stats,file=sys.stderr))
        from . import handoff_runs
        handoff_run_stats=stack.enter_context(handoff_runs.install())
        stack.callback(lambda: print('handoff owner runs',handoff_run_stats,file=sys.stderr))
        from . import density_overlap_reuse
        density_stats=stack.enter_context(density_overlap_reuse.install())
        stack.callback(lambda: print('correction density overlap',density_stats,file=sys.stderr))
        from . import process_inventory_speed
        inventory_stats=stack.enter_context(process_inventory_speed.install())
        stack.callback(lambda: print('correction process inventories',inventory_stats,file=sys.stderr))
        from . import resident_run_rows
        resident_row_stats=stack.enter_context(resident_run_rows.install(shared_samplers['resident_takeover']))
        stack.callback(lambda: print('correction resident run rows',resident_row_stats,file=sys.stderr))
        from . import handoff_component_box
        stack.enter_context(handoff_component_box.install(handoff_stats))
        stack.callback(lambda: print('correction handoff boxes',handoff_stats,file=sys.stderr))
        from . import movie_label_counts
        count_stats=stack.enter_context(movie_label_counts.install())
        stack.callback(lambda: print('correction movie counts',count_stats,file=sys.stderr))
        if reference_area_boxes is not None:stack.enter_context(reference_area_boxes.install_loaded(reference_stats))
        from . import fragment_pair_scan
        print('correction fragment pair scan',stack.enter_context(fragment_pair_scan.install_loaded()),file=sys.stderr)
        from . import convex_hull_image_fast
        # Corrections run outside the native context, where the hull is installed.
        hull_stats=stack.enter_context(convex_hull_image_fast.install())
        stack.callback(lambda: print('correction fast hull',hull_stats,file=sys.stderr))
        hull_aliases=stack.enter_context(convex_hull_image_fast.aliases())
        print('correction fast hull aliases',hull_aliases,file=sys.stderr)
        from . import perimeter_shifts
        perimeter_stats=stack.enter_context(perimeter_shifts.install())
        stack.callback(lambda: print('correction perimeter shifts',perimeter_stats,file=sys.stderr))
        from . import mask_centre_counts
        centre_stats=stack.enter_context(mask_centre_counts.install())
        stack.callback(lambda: print('correction mask centres',centre_stats,file=sys.stderr))
        from . import process_extension_boxes
        extension_stats=stack.enter_context(process_extension_boxes.install())
        stack.callback(lambda: print('correction process extension',extension_stats,file=sys.stderr))
        from . import predecessor_starts_map
        starts_stats=stack.enter_context(predecessor_starts_map.install())
        stack.callback(lambda: print('correction predecessor starts',starts_stats,file=sys.stderr))
        # The sequence's contact families hold this module directly.
        from . import run_blocks
        run_block_stats=stack.enter_context(run_blocks.install())
        stack.callback(lambda: print('correction run blocks',run_block_stats,file=sys.stderr))
        from . import bounded_label
        label_stats=stack.enter_context(bounded_label.install())
        stack.callback(lambda: print('correction bounded label',label_stats,file=sys.stderr))
        from . import bounded_morphology
        morphology_stats=stack.enter_context(bounded_morphology.install())
        stack.callback(lambda: print('correction bounded morphology',morphology_stats,file=sys.stderr))
        from . import guard_scan
        guard_stats=stack.enter_context(guard_scan.install())
        print('correction stage guards',guard_stats,file=sys.stderr)
        stack.callback(lambda: print('correction stage guards',guard_stats,file=sys.stderr))
        from . import core_stable_memo
        stack.enter_context(core_stable_memo.install(core_stable_stats))
        from . import reciprocal_centroid_sums
        stack.enter_context(reciprocal_centroid_sums.install(centroid_stats))
        from . import smoothed_frame_memo
        stack.enter_context(smoothed_frame_memo.install(smoothed_stats))
        # Last: it takes only functions no other adapter replaced.
        from . import sparse_nonzero
        stack.enter_context(sparse_nonzero.install(sparse_stats))
        stack.callback(lambda: print('correction sparse nonzero',sparse_stats,file=sys.stderr))
        stack.callback(lambda: print('correction stable cores',core_stable_stats,file=sys.stderr))
        stack.callback(lambda: print('correction named centroid sums',centroid_stats,file=sys.stderr))
        stack.callback(lambda: print('correction smoothed frames',smoothed_stats,file=sys.stderr))
        if raw_support is not None:stack.callback(lambda: print('correction raw support',raw_support.stats,file=sys.stderr))
        stack.callback(lambda: print('correction reference areas',reference_stats,'core replacement',core_stats,
                                     'destination support',destination_stats,'birth lineage',birth_stats,'gap components',gap_stats,'gap owner sources',gap_owner_stats,'gap trace walks',gap_trace_stats,'origin footprints',origin_stats,'measured cores',measured_core_stats,file=sys.stderr))
        for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
            path=getattr(module,'__file__',None)
            if path is None:continue
            # A plain module without __getattr__ that does not hold the frozen
            # loader is never re-pointed: skip resolving its (string) path.
            names=getattr(module,'__dict__',None)
            if (type(path) is str and names is not None and '__getattr__' not in names
                    and names.get('frozen_method') is not original):continue
            try:inside=_paths.resolve(path).is_relative_to(root)
            except (ValueError,OSError):inside=False
            if inside and getattr(module,'frozen_method',None) is original:
                aliases.append(module);module.frozen_method=method
        stack.callback(lambda: [setattr(module,'frozen_method',original) for module in aliases])
        yield


@contextmanager
def corrections_activate(feedback):
    """Enter corrections() right after the frozen methods() yields, inside any runtime repair.

    Repairs that wrap methods() replace composition loaders (frozen_method)
    after the sequence is built; corrections() re-points only the loaders that
    are still the final composition's, so entering first keeps every stage's
    adapters, and a repair's loader then wraps the adapted one.
    """
    old_methods=feedback.methods
    @contextmanager
    def methods(root,**kwargs):
        with old_methods(root,**kwargs) as sequence, corrections(sequence.final):
            yield sequence
    feedback.methods=methods
    try:
        yield
    finally:
        feedback.methods=old_methods
