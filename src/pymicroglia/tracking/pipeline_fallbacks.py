"""Keep a recording's tracking going when a whole stage cannot handle it.

``stage_fallbacks`` skips single corrections inside the accepted-history
stage. This module does the same one level up, like a relay runner who
stumbles and hands the baton on unchanged: when an identity stage from event
resolution to accepted history raises on a recording's data, its failed folder
is set aside and its input labels are published as its output, in the stage's
own folder and file names, with nothing inferred. Every skip is printed and
written to ``pipeline_fallback_audit.json`` and to ``carried_forward.json`` in
the stage folder.

Two stages hand extra evidence to accepted history. A skipped identity
reservation hands over "nothing reserved"; a skipped layered lineage has no
physical somas, so accepted history is then skipped as well rather than run
without them. The final correction stage publishes the accepted-history labels
when it cannot run, or when accepted history was skipped, marked ``degraded``
in its ``run.json``. When no stage raises, every output is unchanged.
"""
from contextlib import contextmanager
from functools import wraps
import inspect
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .checkpoints import retry, set_aside
from .stage_fallbacks import _location, data_error

VERSION = '2026-10-28-pipeline-fallbacks'
AUDIT = 'pipeline_fallback_audit.json'
MARKER = 'carried_forward.json'

# Stage function: (stage folder, {output role: file under the stage folder}).
# Roles are the names the checkpoint code reads back after a stage finishes.
STAGES = {
    'stage_m4': ('m4_events', {'event_identities': 'out/{stem}_event_identities.tif',
                               'inferred_pixels': 'mid/{stem}_inferred_pixels.tif',
                               'unresolved_pixels': 'mid/{stem}_unresolved_pixels.tif'}),
    'stage_m5': ('m5_reconcile', {'labels': 'out/{stem}.tif'}),
    'stage_m7_accepted': ('m7_persistence', {'labels': 'out/{stem}.tif'}),
    'stage_m8_accepted': ('m8_merge', {'labels': 'out/{stem}.tif',
                                       'inferred': 'mid/{stem}_inferred_merge_pixels.tif'}),
    'stage_m9_accepted': ('m9_alias', {'labels': 'out/{stem}.tif',
                                       'inferred': 'mid/{stem}_inferred_global_alias_pixels.tif'}),
    'stage_m12_accepted': ('m12_identity_reservation', {
        'labels': 'out/{stem}.tif', 'inferred': 'mid/{stem}_inferred_identity_reservation.tif',
        'pre_reservation_labels': 'mid/{stem}_pre_reservation_long_memory.tif'}),
    'stage_m19_accepted': ('m19_layered_lineage', {'labels': 'out/{stem}.tif',
                                                   'inferred': 'mid/{stem}_inferred_layered_lineage.tif'}),
    'stage_m20_accepted': ('m20_oscillatory_lineage', {
        'labels': 'out/{stem}.tif', 'inferred': 'mid/{stem}_inferred_oscillatory_lineage.tif'}),
    'stage_m22_accepted_history': ('m22_accepted_history', {
        'labels': 'out/{stem}.tif', 'unclaimed': 'out/{stem}_unclaimed_original_ids.tif',
        'inferred': 'mid/{stem}_inferred_accepted_history.tif'}),
    'stage_m6': ('m6_review', {}),
}
FINAL = 'm23_completed_feedback'


class Audit:
    """Thread-safe skip record, appended across the recordings of one engine folder."""

    def __init__(self, root):
        self.path = Path(root) / AUDIT
        self.lock = threading.Lock()
        self.skipped = []

    def record(self, row):
        with self.lock:
            try:
                rows = json.loads(self.path.read_text())
            except (OSError, ValueError):
                rows = []
            rows.append(dict(row, version=VERSION, time=time.strftime('%Y-%m-%dT%H:%M:%S')))
            retry(self.path.write_text, json.dumps(rows, indent=2) + '\n')


def _carry(pipeline, common, method, bound, error):
    """Publish a stage's input labels as its output; return what the stage returns."""
    folder, roles = STAGES[method]
    names = list(bound.signature.parameters)
    cfg, stem, run_name, sources = (bound.arguments[name] for name in names[:4])
    labels = bound.arguments[names[4]]
    history = bound.arguments.get('history_inputs')
    run = common.Run(folder, run_name, dict(stem=stem, carried_forward=True, version=VERSION),
                     upstream=None)
    try:
        interval = cfg.values['frame_interval_min']
        if method == 'stage_m22_accepted_history':
            # Accepted history starts from the oscillatory-lineage labels.
            future = bound.arguments.get('m20_result_future')
            if future is not None:
                labels = future.result()[0]
            offset = cfg.source_frame_offset
            labels = labels[offset:]
        labels = np.asarray(labels)
        nothing = np.zeros(labels.shape, bool)
        written = {}
        for role, relative in roles.items():
            path = run.dir / relative.format(stem=stem)
            value = (labels if role in ('labels', 'event_identities', 'pre_reservation_labels')
                     else np.zeros_like(labels) if role == 'unclaimed' else nothing.astype(np.uint8))
            common.save_stack(path, value, interval)
            written[role] = path
        tracks = frame_table = None
        if method != 'stage_m6':
            raw = common.load_stack(sources['registered_raw'])
            if method == 'stage_m22_accepted_history':
                raw = raw[offset:offset + len(labels)]
            frame_table, tracks = pipeline._identity_tables(labels, raw, nothing)
            for name, table in (('frame_identities', frame_table), ('tracks', tracks)):
                written[name] = run.dir / 'out' / f'{name}.csv'
                table.to_csv(written[name], index=False)
        reason = dict(stage=folder, version=VERSION, error_type=type(error).__name__,
                      error=str(error)[:4000], at=_location(error),
                      meaning='this stage changed nothing; its input labels are its output')
        (run.dir / MARKER).write_text(json.dumps(reason, indent=2) + '\n')
        for role, path in written.items():
            run.record(role, path)
        run.finish(dict(carried_forward=True, error=f'{type(error).__name__}: {error}'[:1000]))
    except BaseException as failure:
        run.fail(failure)
        raise
    if method == 'stage_m12_accepted' and history is not None:
        # "Nothing reserved": the reservation is the unchanged input.
        history.reservation_labels = labels
        history.pre_reservation_labels = labels
        history.host_audit = pd.DataFrame()
        history.motion_reserved = set()
        history.motion_reservation_events = pd.DataFrame()
    if method == 'stage_m19_accepted' and history is not None:
        history.physical_somas = None
    if method == 'stage_m4':
        return labels, pd.DataFrame(), nothing, nothing.copy()
    if method == 'stage_m5':
        return frame_table, tracks
    if method == 'stage_m7_accepted':
        return labels, tracks
    if method == 'stage_m22_accepted_history':
        return labels, np.zeros_like(labels), nothing, tracks
    if method == 'stage_m6':
        return None
    return labels, nothing, tracks


def guarded(method, function, pipeline, common, audit):
    signature = inspect.signature(function)
    folder = STAGES[method][0]

    @wraps(function)
    def stage(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            if not data_error(error, audit.skipped):
                raise
            bound = signature.bind(*args, **kwargs)
            run_name = bound.arguments[list(signature.parameters)[2]]
            failed = Path(pipeline.ROOT) / folder / run_name
            kept = set_aside(failed, 'failed') if failed.exists() else None
            audit.skipped.append(folder)
            audit.record(dict(run=run_name, stage=folder, action='input_carried_forward',
                              error_type=type(error).__name__, error=str(error)[:4000],
                              at=_location(error), failed_folder=str(kept) if kept else None))
            print(f'STAGE CARRIED FORWARD {folder}: {type(error).__name__}: {error}; '
                  'its input labels pass on unchanged', flush=True)
            return _carry(pipeline, common, method, bound, error)
    return stage


@contextmanager
def install(pipeline, common, root):
    """Guard every identity stage of ``pipeline`` for the duration of one run."""
    audit = Audit(root)
    saved = []
    for method in STAGES:
        function = getattr(pipeline, method, None)
        if callable(function):
            saved.append((method, function))
            setattr(pipeline, method, guarded(method, function, pipeline, common, audit))
    try:
        yield audit
    finally:
        for method, function in reversed(saved):
            setattr(pipeline, method, function)


def _labels_path(stage):
    manifest = json.loads((stage / 'run.json').read_text())
    if manifest.get('status') != 'done':
        return None
    outputs = manifest.get('outputs', {})
    role = outputs.get('labels') or outputs.get('event_identities')
    return Path(role['path']) if role else None


def deepest_labels(root, run_name):
    """The last finished stage's labels: ``(stage folder, labels path)``, or ``None``."""
    order = ['m22_accepted_history', 'm20_oscillatory_lineage', 'm19_layered_lineage',
             'm12_identity_reservation', 'm9_alias', 'm8_merge', 'm7_persistence', 'm4_events']
    for folder in order:
        stage = Path(root) / folder / run_name
        try:
            path = _labels_path(stage)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if path is not None and path.is_file():
            return stage, path
    return None


def publish_without_corrections(root, stem, run_name, reason, *, source_frame_offset=0,
                                frame_interval_min=30.0, resume=False):
    """Publish the deepest finished labels as the final result, marked degraded."""
    import tifffile
    root = Path(root)
    found = deepest_labels(root, run_name)
    if found is None:
        raise RuntimeError(f'No tracking stage finished for {stem}; nothing can be published') from None
    stage, labels_path = found
    output = root / FINAL / run_name
    if output.exists():
        set_aside(output, 'failed')
    (output / 'out').mkdir(parents=True)
    labels_out = output / 'out' / f'{stem}.tif'
    if stage.parent.name == 'm22_accepted_history':
        # Accepted history is already trimmed to the analysed frames.
        retry(shutil.copyfile, labels_path, labels_out)
        for name in (f'{stem}_unclaimed_original_ids.tif', f'{stem}_provenance.tif',
                     'frame_identities.csv', 'tracks.csv'):
            if (stage / 'out' / name).is_file():
                retry(shutil.copyfile, stage / 'out' / name, output / 'out' / name)
    else:
        labels = tifffile.imread(labels_path)[source_frame_offset:]
        tifffile.imwrite(labels_out, labels, imagej=True, compression='zlib',
                         metadata={'axes': 'TYX', 'finterval': frame_interval_min * 60.0,
                                   'tunit': 'sec', 'unit': 'pixel'})
    manifest = dict(status='complete', degraded=True, version=VERSION,
                    carried_forward_from=str(stage), source_frame_offset=source_frame_offset,
                    reason=reason,
                    meaning='Corrections after this stage were not applied; review before use.')
    (output / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / MARKER).write_text(json.dumps(manifest, indent=2) + '\n')
    if resume:
        from .checkpoints import sha, write
        files = {str(path.relative_to(output)): sha(path) for path in output.rglob('*') if path.is_file()}
        write(output / 'completion_files.json', files)
    print(f'DEGRADED RESULT {stem}: published {stage.parent.name} labels; {reason}', flush=True)
    return output


def final_corrections(run_current, root, stem, run_name, cfg, *, resume=False):
    """Run the final correction stage, or publish accepted history when it cannot run."""
    root = Path(root)
    options = dict(source_frame_offset=cfg.source_frame_offset,
                   frame_interval_min=cfg.values['frame_interval_min'], resume=resume)
    if (root / 'm22_accepted_history' / run_name / MARKER).is_file():
        return publish_without_corrections(
            root, stem, run_name, 'accepted history was skipped, so its corrections have no input',
            **options)
    try:
        return run_current()
    except Exception as error:
        # A missing file here is the data's problem, not the disk's: accepted
        # history finished, so this stage's own inputs exist.
        if not data_error(error, ['accepted history finished']):
            raise
        Audit(root).record(dict(run=run_name, stage=FINAL, action='accepted_history_published',
                                error_type=type(error).__name__, error=str(error)[:4000],
                                at=_location(error)))
        return publish_without_corrections(
            root, stem, run_name, f'final corrections failed: {type(error).__name__}: {error}'[:2000],
            **options)
