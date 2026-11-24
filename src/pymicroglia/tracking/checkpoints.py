"""Verified reuse of Motion's completed stage files and accepted-history work.

Array checkpoints reference existing TIFFs: no duplicate movie cache. Small
table sidecars preserve exact pandas values and types across interruption.
Resume only one's own run directory; table sidecars are trusted local pickles.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import scipy
import skimage
import tifffile

METHOD_VERSION = '2026-10-14-motion-stage-resume-1'
STAGES = {
    'stage_m0': 'm0_inputs', 'stage_m1': 'm1_motion', 'stage_m2': 'm2_anchor',
    'stage_m3': 'm3_track', 'stage_m4': 'm4_events', 'stage_m5': 'm5_reconcile',
    'stage_m7_accepted': 'm7_persistence', 'stage_m8_accepted': 'm8_merge',
    'stage_m9_accepted': 'm9_alias', 'stage_m12_accepted': 'm12_identity_reservation',
    'stage_m19_accepted': 'm19_layered_lineage', 'stage_m20_accepted': 'm20_oscillatory_lineage',
    'stage_m22_accepted_history': 'm22_accepted_history',
}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def retry(function, *args, seconds=60):
    """Repeat a file move or write while Windows, Dropbox or a virus scanner holds it."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            return function(*args)
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.5)


def write(path, value):
    temporary = path.with_suffix('.tmp')
    retry(temporary.write_text, json.dumps(value, indent=2), 'utf8')
    retry(os.replace, temporary, path)


def set_aside(directory, reason):
    """Rename an unusable stage folder beside itself; its evidence is kept, never deleted."""
    target = directory.with_name(f'{directory.name}_{reason}_{time.time_ns()}')
    retry(directory.rename, target)
    return target


def history_runs(root):
    """Accepted-history run folders still in use; set-aside copies are kept evidence only."""
    folder = Path(root) / 'm22_accepted_history'
    if not folder.is_dir():
        return []
    return [run for run in sorted(folder.iterdir())
            if run.is_dir() and not re.search(r'_[a-z]+_\d{15,}$', run.name)]


def reconcile_repair(root, record, signature, produced, name):
    """Recompute, rather than refuse, accepted history made without this repair.

    History made under a different version of the repair, or before it existed
    (``produced`` names a file inside a run folder written after the repaired
    step), is set aside and recomputed. Returns whether anything was set aside.
    """
    record = Path(record)
    try:
        stale = json.loads(record.read_text()) != signature
    except FileNotFoundError:
        stale = any(any(run.glob(produced)) for run in history_runs(root))
    except (OSError, ValueError):
        stale = True
    if stale:
        for run in history_runs(root):
            kept = set_aside(run, 'superseded')
            print(f'CHECKPOINTS superseded {kept.name}: the {name} changed, so accepted history is recomputed',
                  flush=True)
    return stale


@contextmanager
def exclusive(path, *, seconds=120):
    """Do not let two processes resume or preserve the same tracking stage.

    A brief hold by a sync client or scanner is waited out; a second tracking
    process still holding the run after ``seconds`` is refused.
    """
    with Path(path).open('a+b') as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        deadline = time.monotonic() + seconds
        while True:
            handle.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise FileExistsError('Another process is using this tracking run') from error
                time.sleep(2)
        try:
            yield
        finally:
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class Stages:
    def __init__(self, pipeline, cfg, stem, run_name, loader):
        self.pipeline, self.cfg, self.stem = pipeline, cfg, stem
        self.root, self.run_name, self.loader = Path(pipeline.ROOT), run_name, loader
        self.root.mkdir(parents=True, exist_ok=True)
        self.reused = []
        # Verify real source bytes, rather than trusting paths or saved pins.
        pipeline._verify_pins(cfg, stem)
        code = {str(p.relative_to(self.root)): sha(p)
                for p in sorted((self.root / 'code').rglob('*.py'))}
        package = {p.name: sha(p) for p in (
            Path(__file__), *(Path(__file__).with_name(name) for name in (
                'engine.py', 'engine_bounded.py', 'frozen_io.py', 'identity_reuse.py', 'frame_component_reuse.py', 'terminal_feature_reuse.py', 'lazy_flip_components.py', 'compact_components.py', 'ownership_observation_reuse.py', 'tracking_feature_batch.py', 'companion_lookup_reuse.py', 'host_partition_reuse.py', 'diagnostic_shortlist.py', 'lazy_motion_counts.py', 'confidence_geometry_reuse.py', 'hierarchy_component_batch.py', 'motion_history_reuse.py', 'motion_overlap_batch.py', 'motion_lobe_batch.py', 'tracking_plain_batch.py', 'fixed_object_overlap.py', 'relay_transition_geometry.py', 'conflict_core_batch.py', 'measured_gap_path_reuse.py', 'perimeter_shifts.py', 'mask_centre_counts.py', 'correction_owner_sampling.py', 'tenure_geometry_reuse.py', 'display_block_residency.py', 'point_disk_votes.py', 'bounded_morphology_loader.py', 'convex_hull_image_fast.py', 'tracking_matrix_batch.py', 'identity_name_batch.py', 'ledger_reference_boxes.py', 'relay_pair_reuse.py', 'relay_metric_reuse.py', 'ownership_region_rows.py', 'sparse_pair_table.py', 'display_lookup.py', 'flip_boundary_crop.py', 'nested_alias_reuse.py', 'post_split_reuse.py', 'bracketed_handoff_reuse.py', 'owner_consensus_presence.py', 'terminal_companion_reuse.py', 'observation_cache.py', 'soma_separation_boxes.py', 'motion_union_bounds.py', 'two_core_audit_boxes.py', 'physical_core_boxes.py', 'one_core_union_boxes.py', 'handoff_runs.py', 'module_scan.py', 'resident_run_rows.py', 'gap_trace_walk.py', 'process_track_features.py', 'start_duplicate_excess.py', 'censored_area_boxes.py', 'raw_support_memo.py', 'reference_area_boxes.py', 'core_replacement_index.py', 'destination_support_boxes.py', 'birth_lineage_index.py', 'process_extension_boxes.py', 'predecessor_starts_map.py', 'run_blocks.py', 'point_disks.py', 'measured_cores.py', 'track_row_index.py', 'bracketed_consensus_tracks.py', 'track_sort_skips.py', 'sparse_nonzero.py', 'core_stable_memo.py', 'reciprocal_centroid_sums.py', 'smoothed_frame_memo.py', 'gap_component_boxes.py', 'bounded_label.py', 'origin_footprint_boxes.py', 'guard_scan.py', 'tracking_pass_record.py', 'fragment_pair_scan.py', 'link_paint_boxes.py', 'bounded_morphology.py', 'occupancy_overlap_boxes.py', 'spatial_path_views.py', 'optical_physical_paths.py', 'gap_owner_source_memo.py', 'core_intersects_window.py', 'coupled_peer_memo.py', 'flicker_component_boxes.py', 'handoff_component_box.py', 'movie_label_counts.py', 'row_tuples.py', 'density_overlap_reuse.py', 'process_inventory_speed.py', 'resolved_paths.py', 'narrow_labels.py', 'threaded_evidence_images.py', 'seat_projection_boxes.py', 'disk_owner_batch.py', 'residency_component_boxes.py', 'csv_joined_rows.py', 'batched_detection_cores.py', 'invasion_donor_rows.py', 'bracketed_seat_rows.py', 'identity_performance.py', 'display_scale.py', 'detector_acceleration.py',
                'completed_feedback.py', 'completed_evidence.py', 'execution.py', 'parallel_handoff.py', 'legacy_execution.py')))}
        from . import shipped_engines
        # The key keeps its earlier name so saved signatures stay comparable; it holds the folder's fingerprint.
        package['completed_feedback_engine.zip'] = shipped_engines.fingerprint(
            shipped_engines.folder(shipped_engines.COMPLETED_FEEDBACK))
        import inspect
        from auto_organotypic.array_storage import Pool
        package['array_storage.py'] = sha(inspect.getfile(Pool))
        from auto_organotypic.array_selection import select_into
        package['array_selection.py'] = sha(inspect.getfile(select_into))
        runtime = {}
        for name in ('r12_runtime', 'r13_cache_runtime', 'r13_copy_runtime'):
            source = getattr(sys.modules.get(name), '__file__', None)
            if source:
                runtime[name] = sha(source)
        self.signature = dict(method=METHOD_VERSION, stem=stem, config_sha256=sha(cfg.path),
            code=code, adapter=package, runtime=runtime, software=dict(python=platform.python_version(),
            numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
            skimage=skimage.__version__, tifffile=tifffile.__version__))
        self.state_path = self.root / 'tracking_resume.json'
        # A changed input, setting or program recomputes every stage: no saved
        # stage or partial accepted-history work is reused, and the old state
        # is kept beside the new one.
        self.fresh = False
        if self.state_path.is_file():
            try:
                previous = json.loads(self.state_path.read_text())
            except ValueError:
                previous = {}
            if previous.get('signature') != self.signature:
                self.fresh = True
                retry(self.state_path.rename, self.state_path.with_name(
                    f'tracking_resume_superseded_{time.time_ns()}.json'))
                print('CHECKPOINTS superseded: inputs, settings, software or code changed; '
                      'every stage is recomputed', flush=True)
        self.state = dict(signature=self.signature, run=run_name, reused=[], status='running')
        write(self.state_path, self.state)

    def _file(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()) or any(
                parent.is_symlink() for parent in (self.root / relative, *(self.root / relative).parents)
                if parent.is_relative_to(self.root)):
            raise ValueError('Checkpoint file lies outside the owned engine directory')
        return path

    def _verify(self, files):
        for relative, expected in files.items():
            path = self._file(relative)
            if not path.is_file() or sha(path) != expected:
                raise ValueError('Tracking checkpoint file changed or is incomplete: ' + relative)

    def _load(self, value):
        if isinstance(value, list):
            return tuple(self._load(v) for v in value)
        if not isinstance(value, dict):
            return value
        if 'array' in value:
            array = self.loader(self._file(value['array']))
            if list(array.shape) != value['shape']:
                raise ValueError('Checkpoint array shape changed')
            return array.astype(np.dtype(value['dtype']), copy=False)
        if 'table' in value:
            # Verified hash and owned path are checked before unpickling.
            return pd.read_pickle(self._file(value['table']))
        if 'set' in value:
            return set(value['set'])
        if 'path' in value:
            return Path(value['path'])
        return {k: self._load(v) for k, v in value.items()}

    def _capture(self, directory, manifest, result, history, method):
        files = {}
        for output in manifest['outputs'].values():
            path = Path(output['path']).resolve()
            relative = str(path.relative_to(self.root.resolve()))
            files[relative] = output['sha256']
        tables = 0

        def array(value, role):
            if value is None:
                return None
            path = Path(manifest['outputs'][role]['path']).resolve()
            return dict(array=str(path.relative_to(self.root.resolve())),
                        dtype=str(value.dtype), shape=list(value.shape))

        def small(value):
            nonlocal tables
            if isinstance(value, pd.DataFrame):
                path = directory / f'checkpoint_table_{tables}.pkl'
                tables += 1
                value.to_pickle(path)
                relative = str(path.relative_to(self.root))
                files[relative] = sha(path)
                return dict(table=relative)
            if isinstance(value, Path):
                return dict(path=str(value))
            if isinstance(value, set):
                return dict(set=sorted(value))
            if isinstance(value, dict):
                return {k: small(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return [small(v) for v in value]
            return value

        if method in ('stage_m0', 'stage_m1', 'stage_m5'):
            saved = small(result)
        elif method == 'stage_m2':
            # The bounded caller never consumes gained/lost full movies.
            saved = [array(result[0], 'observations'), small(result[1]), int(result[2]), None, None]
        elif method == 'stage_m3':
            saved = [array(result[0], 'initial_identities'), small(result[1])]
        elif method == 'stage_m4':
            saved = [array(result[0], 'event_identities'), small(result[1]),
                     array(result[2], 'inferred_pixels'), array(result[3], 'unresolved_pixels')]
        elif method == 'stage_m7_accepted':
            saved = [array(result[0], 'labels'), small(result[1])]
        elif method == 'stage_m22_accepted_history':
            saved = [array(result[0], 'labels'), array(result[1], 'unclaimed'),
                     array(result[2], 'inferred'), small(result[3])]
        else:
            saved = [array(result[0], 'labels'), array(result[1], 'inferred'), small(result[2])]
        attributes = {}
        if history is not None:
            # Save only this stage's side effects. Earlier evidence is restored
            # by its own checkpoint before the next stage consumes it.
            if method == 'stage_m12_accepted':
                attributes = dict(reservation_labels=array(history.reservation_labels, 'labels'),
                    pre_reservation_labels=array(history.pre_reservation_labels, 'pre_reservation_labels'),
                    host_audit=small(history.host_audit), motion_reserved=small(history.motion_reserved),
                    motion_reservation_events=small(history.motion_reservation_events))
            elif method == 'stage_m19_accepted':
                attributes = dict(physical_somas=array(history.physical_somas, 'physical_somas'))
        self._verify(files)
        receipt = dict(signature=self.signature, result=saved, history=attributes, files=files)
        write(directory / 'checkpoint.json', receipt)
        return receipt

    def call(self, method, *args, **kwargs):
        directory = self.root / STAGES[method] / self.run_name
        marker = directory / 'checkpoint.json'
        history = kwargs.get('history_inputs')
        if marker.is_file():
            try:
                receipt = json.loads(marker.read_text())
                if receipt['signature'] != self.signature:
                    raise ValueError('Tracking stage checkpoint has different inputs, settings or code')
                self._verify(receipt['files'])
                restored = {name: self._load(value) for name, value in receipt['history'].items()}
                result = self._load(receipt['result'])
            except (ValueError, KeyError, OSError) as error:
                # Recompute the stage rather than stop: its old folder is kept.
                print(f'CHECKPOINT not reusable for {STAGES[method]}: {error}; recomputing', flush=True)
                set_aside(directory, 'superseded')
            else:
                for name, value in restored.items():
                    setattr(history, name, value)
                self.reused.append(STAGES[method])
                self.state.update(reused=self.reused, current_stage=STAGES[method], status='resuming')
                write(self.state_path, self.state)
                print('CHECKPOINT reused', STAGES[method], flush=True)
                return result
        if directory.exists() and self.fresh:
            set_aside(directory, 'superseded')
        if directory.exists():
            try:
                old = json.loads((directory / 'run.json').read_text())
            except (OSError, ValueError):
                old = {}  # A hard stop may interrupt Motion's own metadata write.
            # Metadata-only stages from an earlier adapter need no array/table
            # reconstruction. The same source pins have just been verified.
            if method in ('stage_m0', 'stage_m1') and old.get('status') == 'done':
                sources = self.pipeline._pinned_paths(self.cfg, self.stem)
                source_record = self.root / 'm0_inputs' / self.run_name / 'out' / f'{self.stem}_sources.json'
                try:
                    same = json.loads(source_record.read_text())['config_sha256'] == self.signature['config_sha256']
                except (OSError, ValueError, KeyError):
                    same = False
                if same:
                    result = sources if method == 'stage_m0' else None
                    self._capture(directory, old, result, None, method)
                    return self.call(method, *args, **kwargs)
            archived = set_aside(directory, 'interrupted')
            if method == 'stage_m22_accepted_history' and (archived / 'mid/accepted_history').is_dir():
                kwargs['checkpoint_run_name'] = archived.name
        self.state.update(current_stage=STAGES[method], status='running')
        write(self.state_path, self.state)
        result = getattr(self.pipeline, method)(*args, **kwargs)
        manifest = json.loads((directory / 'run.json').read_text())
        if manifest.get('status') != 'done':
            raise RuntimeError('Motion stage did not finish; no checkpoint published')
        self._capture(directory, manifest, result, history, method)
        return result
