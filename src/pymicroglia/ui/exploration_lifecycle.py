"""Attach native lifecycle review of frozen tracking masks to refined measurements.

Classification runs explicitly through the existing measurement action. Reading
a saved review only verifies provenance and returns its existing tables.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from .exploration_data import number, read_rows, recording, table_path
from .exploration_history import _alignment, _fingerprint, verify_tracking_alignment
from .exploration_images import declared_source
from .exploration_rhythms import _atomic_json

from pymicroglia.measure.lifecycle_records import (
    SCHEMA, TABLES, review_folder as _folder, review_scope, review_table)


def _scope(root, movie):
    return review_scope(root, movie, table_path(root, movie, 'cell_frame'),
                        declared_source(root, movie, 'labels'))


def attached_table(root, movie, name):
    return review_table(root, movie, name, measurement_path=table_path(root, movie, 'cell_frame'),
                        labels_path=declared_source(root, movie, 'labels'))


def review_details(root, movie):
    """Expose the classification basis after the shared table lookup verifies it."""
    index = _folder(root, movie['stem']) / 'review.json'
    if not index.is_file():
        return None
    attached_table(root, movie, 'lifecycle_events')
    record = json.loads(index.read_text(encoding='utf-8'))
    return {key: record[key] for key in ('classification_masks', 'method_version', 'tracking_sha256', 'candidate_status_preserved')}


def attach_lifecycle_review(run, stem, review_run, *, expected_tracking_sha256):
    """Publish two native tables after validating original masks, IDs and clock."""
    root, _, movie = recording(run, stem)
    review, manifest, original = recording(review_run, stem)
    if not review.is_relative_to(root):
        raise ValueError('Save the lifecycle review inside the selected run')
    base = declared_source(review, original, 'labels')
    root, movie, _, observations, frames, _, _ = verify_tracking_alignment(
        root, stem, base, expected_tracking_sha256=expected_tracking_sha256)
    provenance = original['provenance']
    interval = provenance['scale']['minutes_per_frame'] / 60
    offset = original['spec'].get('source_frame_offset', 0)
    if (provenance['field']['frames'] != len(frames)
            or any(f['source_frame_index'] != index+offset
                   or not np.isclose(f['hours'], (index+offset)*interval) for index, f in enumerate(frames))):
        raise ValueError('Lifecycle review time coordinates do not match this recording')
    module = next((m for m in original['modules'] if m['module'] == 'lifecycle' and m['status'] == 'done'), None)
    if not module or manifest['settings']['enabled_modules'] != ['lifecycle']:
        raise ValueError('Use a completed native lifecycle-only analysis of the original masks')
    inputs = {}
    for role in ('labels', 'unclaimed', 'raw'):
        path = declared_source(review, original, role)
        if path:
            inputs[role] = {'path': str(path), 'sha256': _fingerprint(path)}
    tables = {}
    for name in TABLES:
        path = table_path(review, original, name)
        if path is None:
            raise ValueError('The native lifecycle analysis did not save ' + name)
        tables[name] = {'path': path.relative_to(root).as_posix(), 'sha256': _fingerprint(path),
                        'rows': len(read_rows(path))}
    cells = read_rows(table_path(review, original, 'lifecycle_cells'))
    if {number(c['identity']) for c in cells} != set(observations):
        raise ValueError('Lifecycle review identities do not match this recording')
    for cell in cells:
        indices = sorted(observations[int(cell['identity'])])
        if (number(cell['start_frame']) != indices[0] or number(cell['end_frame']) != indices[-1]
                or number(cell['frames_named']) != len(indices)):
            raise ValueError('Lifecycle review identity continuity does not match this recording')
    for event in read_rows(table_path(review, original, 'lifecycle_events')):
        identity, frame = int(event['identity']), int(event['frame_index'])
        if frame not in observations.get(identity, {}) or not np.isclose(number(event['hours']), frames[frame]['hours']):
            raise ValueError('Lifecycle event identity or clock disagrees with saved observations')
    record = {'schema': SCHEMA, 'stem': stem, 'scope': _scope(root, movie), 'inputs': inputs,
              'tables': tables, 'review_run': review.relative_to(root).as_posix(),
              'method_version': module['method_version'], 'parameters': module['parameters'],
              'tracking_sha256': expected_tracking_sha256, 'source_frame_offset': movie['spec'].get('source_frame_offset', 0),
              'producer': 'pymicroglia.measure.modules.lifecycle',
              'classification_masks': 'original accepted tracking masks',
              'candidate_status_preserved': True}
    folder = _folder(root, stem)
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / 'reproduce.py'
    replay_inputs = {**original['spec'], 'sha256': {role: source['sha256'] for role, source in inputs.items()}}
    script.write_text('from pathlib import Path\nfrom pymicroglia.measure import measure\n'
                      f'movies = {[replay_inputs]!r}\n'
                      f'print(measure(movies, output_dir=Path(__file__).parent / "replayed", enabled_modules=["lifecycle"], frame_interval_min={float(interval*60)!r}, run_label="lifecycle_review"))\n', encoding='utf-8')
    _atomic_json(folder / 'review.json', record)
    return {'path': str(_folder(root, stem) / 'review.json'), 'review_run': str(review),
            'tables': tables, 'method_version': record['method_version'], 'script': str(script)}


def calculate_lifecycle_review(run, stem, *, tracking_labels=None,
                               expected_tracking_sha256=None, unclaimed=None):
    """Run the native lifecycle action, preserving the selected measurements.

    Original Motion inputs may come from the checksum-pinned history attachment,
    or be provided explicitly. Missing original masks are an error; the refined
    display masks are never substituted as classification inputs.
    """
    from pymicroglia.measure import measure
    from pymicroglia.tracking.contract import MOTION_OUTPUTS
    from .exploration_history import _folder as history_folder

    root, _, movie = recording(run, stem)
    if tracking_labels is None:
        record_path = history_folder(root, stem) / 'history.json'
        if not record_path.is_file():
            raise ValueError('Declare the original tracking labels and checksum for lifecycle review')
        record = json.loads(record_path.read_text(encoding='utf-8'))
        scope = _scope(root, movie)
        if (record.get('schema') != 'pymicroglia.exploration-history.v1' or record.get('stem') != stem
                or record.get('source_sha256') != scope['measurements']
                or record.get('movie_sha256') != scope['movie']
                or record.get('labels_sha256') != scope['refined_labels']):
            raise ValueError('The original tracking attachment does not match this recording')
        tracking_labels, expected_tracking_sha256 = record['tracking_labels'], record['tracking_sha256']
        # Use Motion's declared output role beside its saved original labels.
        candidate = Path(tracking_labels).parent.parent / MOTION_OUTPUTS['unclaimed'].format(stem=stem)
        if unclaimed is None and candidate.is_file():
            unclaimed = candidate
    if not expected_tracking_sha256:
        raise ValueError('An explicit original tracking checksum is required')
    root, movie, _, _, frames, _, base = verify_tracking_alignment(
        root, stem, tracking_labels, expected_tracking_sha256=expected_tracking_sha256)
    intervals = np.diff([f['hours'] for f in frames])
    if not len(intervals) or not np.isfinite(intervals).all() or not np.allclose(intervals, intervals[0]) or intervals[0] <= 0:
        raise ValueError('Lifecycle review needs the original regular recording clock')
    offset = int(movie['spec'].get('source_frame_offset', 0))
    if not np.allclose([f['hours'] for f in frames], (np.arange(len(frames))+offset)*intervals[0]):
        raise ValueError('The recording clock does not match its declared source-frame offset')
    raw = declared_source(root, movie, 'raw')
    inputs = {'labels': str(base), 'raw': str(raw) if raw else None,
              'unclaimed': str(Path(unclaimed).resolve()) if unclaimed else None,
              'source_frame_offset': offset, 'stem': stem,
              'sha256': {'labels': expected_tracking_sha256}}
    if unclaimed:
        inputs['sha256']['unclaimed'] = _fingerprint(Path(unclaimed))
    # Three decoded stacks plus the native measurement working arrays fit comfortably.
    work_bytes = sum(Path(p).stat().st_size for p in (base, raw, unclaimed) if p)
    if shutil.disk_usage(root).free < max(256 * 1024**2, work_bytes * 8):
        raise ValueError('There is insufficient local space for lifecycle review')
    folder = _folder(root, stem)
    folder.mkdir(parents=True, exist_ok=True)
    existing = folder / 'review.json'
    if existing.is_file():
        prior = json.loads(existing.read_text(encoding='utf-8'))
        from pymicroglia.measure.modules.lifecycle import METHOD_VERSION, DEFAULTS
        if (prior.get('tracking_sha256') == expected_tracking_sha256
                and prior.get('method_version') == METHOD_VERSION and prior.get('parameters') == DEFAULTS
                and prior.get('inputs', {}).get('unclaimed', {}).get('sha256') == inputs['sha256'].get('unclaimed')):
            for name in TABLES:
                attached_table(root, movie, name)
            return {'reused': True, 'path': str(existing), 'review_run': str(root / prior['review_run']),
                    'tables': prior['tables'], 'method_version': prior['method_version'],
                    'script': str(folder / 'reproduce.py')}
    result = measure([inputs], output_dir=folder / 'analysis', enabled_modules=['lifecycle'],
                     frame_interval_min=float(intervals[0]*60), run_label=f'lifecycle_{stem}_original_motion',
                     claim='Classify lifecycle events on original accepted Motion masks; preserve refined masks, intensity measurements and identities.')
    attached = attach_lifecycle_review(root, stem, result['run']['folder'],
                                      expected_tracking_sha256=expected_tracking_sha256)
    return {**attached, 'reused': False}
