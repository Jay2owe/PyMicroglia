"""Content-verified raw-evidence reuse. Ownership is recomputed for each label input."""
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import scipy
STAGES = ('25_raw_physical_hypotheses/out', '26_latent_body_tracks/out')
FILES = ('25_raw_physical_hypotheses/out/raw_physical_hypotheses.csv', '25_raw_physical_hypotheses/out/frame_evidence_thresholds.csv', '25_raw_physical_hypotheses/out/producer_metrics.json', '26_latent_body_tracks/out/latent_track_points.csv', '26_latent_body_tracks/out/physical_tracks.csv', '26_latent_body_tracks/out/encounter_episodes.csv', '26_latent_body_tracks/out/encounter_frames.csv', '26_latent_body_tracks/out/reconnection_hypotheses.csv', '26_latent_body_tracks/out/producer_metrics.json')

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_signature(value):
    data = np.ascontiguousarray(value)
    return dict(shape=list(data.shape), dtype=data.dtype.str, sha256=hashlib.sha256(memoryview(data).cast('B')).hexdigest())

def raw_producer_signature(history):
    """Hash parameter derivation and raw production, excluding later ownership."""
    tree = ast.parse(Path(history.__file__).read_text(encoding='utf8'))
    function = next(v for v in tree.body if isinstance(v, ast.FunctionDef) and v.name == '_owner_consensus')
    stop = next(i for i,v in enumerate(function.body) if isinstance(v, ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'production_points' for t in v.targets))
    function.body = function.body[:stop]
    # A new configuration dependency needs review before it can share evidence.
    allowed = {'enabled','targeting_mode','raw_detection','latent_tracking','owner_consensus','version','motion_frame_offset'}
    for node in ast.walk(function):
        if isinstance(node,ast.Name) and node.id == 'configured' and isinstance(node.ctx,ast.Load):
            parents = [p for p in ast.walk(function) if node in ast.iter_child_nodes(p)]
            if len(parents) != 1 or not isinstance(parents[0],ast.Attribute) or parents[0].attr != 'get':
                raise ValueError('Unreviewed raw configuration access')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id == 'configured':
            if node.func.attr != 'get' or not node.args or not isinstance(node.args[0],ast.Constant) or node.args[0].value not in allowed:
                raise ValueError('Unreviewed raw configuration key')
    imports = [v for v in tree.body if isinstance(v,(ast.Import,ast.ImportFrom))]
    payload = ast.dump(ast.Module(body=imports+[function],type_ignores=[]),include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def raw_configuration(configured, source_frame_offset=None):
    if 'motion_frame_offset' in configured:
        offset = int(configured['motion_frame_offset'])
    elif source_frame_offset is not None:
        offset = int(source_frame_offset)
    else:
        raise ValueError('Effective motion offset required for raw evidence reuse')
    return dict(raw_detection=deepcopy(configured.get('raw_detection',{})),
                latent_tracking=deepcopy(configured.get('latent_tracking',{})),motion_frame_offset=offset)


def fingerprint(history, raw, motion, configured, source_frame_offset=None):
    sources = {name: file_hash(getattr(history, name).__file__) for name in ('raw_physical_hypotheses', 'latent_body_tracks')}
    promotion = getattr(np, '_get_promotion_state', None)
    sources['numpy_promotion_state'] = promotion() if promotion is not None else 'not_exposed'
    sources['raw_producer_prefix'] = raw_producer_signature(history)
    sources['relative_visibility_mode'] = bool(getattr(history.raw_physical_hypotheses.detect_frame, '_motion_relative_visibility', False))
    sources['fractional_raw_peak_mode'] = bool(getattr(history.raw_physical_hypotheses.detect_frame, '_motion_fractional_raw_peaks', False))
    sources['relative_latent_visibility_mode'] = bool(getattr(history.latent_body_tracks.expand_track_points, '_motion_relative_latent_visibility', False))
    sources['relative_valley_mode'] = bool(getattr(history.latent_body_tracks._valley_ratio, '_motion_relative_valley', False))
    sources['cache_implementation'] = file_hash(__file__)
    sources['speed_implementation'] = file_hash(Path(history.raw_physical_hypotheses.__file__).with_name('physical_tracking_speed.py'))
    for name in ('encounter_speed.py', 'valley_sampling_speed.py', 'reconnection_speed_spatial.py', 'motion_sampling_speed.py', 'relative_detection_visibility.py', 'relative_latent_visibility.py', 'relative_valley_ratio.py', 'fractional_raw_peaks.py', 'peak_search_speed.py', 'parallel_detection_context.py', 'adaptive_link_search.py', 'detector_core_radius_units.py', 'contact_spatial_units.py'):
        sources[name] = file_hash(Path(history.raw_physical_hypotheses.__file__).with_name(name))
    return dict(schema='motion.raw-evidence.v2', raw=array_signature(raw), motion=array_signature(motion), configured=raw_configuration(configured, source_frame_offset), sources=sources, versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))

def record_evidence(history, raw, motion, configured, native_directory, proof, source_frame_offset=None):
    native = Path(native_directory)
    raw_params = deepcopy(configured.get('raw_detection', {}))
    latent_params = {**raw_params, **deepcopy(configured.get('latent_tracking', {}))}
    assert json.loads((native / FILES[2]).read_text())['parameters'] == raw_params
    assert json.loads((native / FILES[-1]).read_text())['parameters'] == latent_params
    return dict(fingerprint=fingerprint(history, raw, motion, configured, source_frame_offset), directory=str(native), files={name: file_hash(native / name) for name in FILES}, proof=proof)

def _owner_function_with_evidence(history, entry):
    source = Path(history.__file__)
    tree = ast.parse(source.read_text(encoding='utf8'), filename=str(source))
    original = next((v for v in tree.body if isinstance(v, ast.FunctionDef) and v.name == '_owner_consensus'))
    first = next((i for i, v in enumerate(original.body) if isinstance(v, ast.AnnAssign) and isinstance(v.target, ast.Name) and (v.target.id == 'hypothesis_rows')))
    last = next((i for i, v in enumerate(original.body) if isinstance(v, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'production_points' for t in v.targets))))
    removed = original.body[first:last]
    names = {v.id for node in removed for v in ast.walk(node) if isinstance(v, ast.Name)}
    assert not names & {'labels', 'unclaimed'}, 'Raw evidence block now depends on assigned identities'
    replacement = ast.parse('hypotheses, track_summaries, points_path = _copy_cached_evidence(output_root)').body
    original.body = original.body[:first] + replacement + original.body[last:]

    def copy_evidence(output_root):
        old, new = (Path(entry['directory']), Path(output_root))
        assert old.resolve() != new.resolve()
        for relative, expected in entry['files'].items():
            source_path = old / relative
            assert file_hash(source_path) == expected, source_path
            target = new / relative
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            assert file_hash(target) == expected, target
        return (pd.read_csv(new / FILES[0]), pd.read_csv(new / '26_latent_body_tracks/out/physical_tracks.csv'), new / '26_latent_body_tracks/out/latent_track_points.csv')
    namespace = dict(vars(history))
    namespace['_copy_cached_evidence'] = copy_evidence
    module = ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[]))
    exec(compile(module, str(source) + '::cached_raw_evidence', 'exec'), namespace)
    return namespace[original.name]

@contextmanager
def reuse_independent_evidence(history, entries=()):
    """One context may span both variants; cache misses use the original code."""
    original = history._owner_consensus
    cache = list(entries)
    statistics = dict(hits=0, misses=0, events=[])

    def owner(labels, unclaimed, raw, motion_full, config, output_root):
        configured = config.values['accepted_postprocessing'].get('field_wide_owner_consensus', {})
        if not configured.get('enabled', False):
            return original(labels, unclaimed, raw, motion_full, config, output_root)
        signature = fingerprint(history, raw, motion_full, configured, config.source_frame_offset)
        entry = next((v for v in cache if v['fingerprint'] == signature), None)
        if entry is None:
            result = original(labels, unclaimed, raw, motion_full, config, output_root)
            entry = record_evidence(history, raw, motion_full, configured, output_root, dict(kind='same_process_original_producer', source=file_hash(history.__file__)), source_frame_offset=config.source_frame_offset)
            cache.append(entry)
            statistics['misses'] += 1
            statistics['events'].append(dict(action='computed_original', directory=str(output_root)))
            return result
        result = _owner_function_with_evidence(history, entry)(labels, unclaimed, raw, motion_full, config, output_root)
        statistics['hits'] += 1
        statistics['events'].append(dict(action='reused_raw_only', source=entry['directory'], destination=str(output_root)))
        return result
    history._owner_consensus = owner
    try:
        yield (statistics, cache)
    finally:
        history._owner_consensus = original
import os
import tempfile

def cache_key(entry):
    serialized = json.dumps(entry['fingerprint'], sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(serialized.encode('utf8')).hexdigest()

def validate_entry(entry):
    if set(entry['files']) != set(FILES):
        raise ValueError('Incomplete raw evidence entry')
    directory = Path(entry['directory'])
    for relative in FILES:
        if file_hash(directory / relative) != entry['files'][relative]:
            raise ValueError('Changed raw evidence: ' + relative)
    return cache_key(entry)

def load_entries(directory):
    """Reject damaged entries explicitly; never return stale evidence."""
    directory = Path(directory)
    entries, rejected = ([], [])
    if not directory.exists():
        return (entries, rejected)
    for path in sorted(directory.glob('*.json')):
        try:
            entry = json.loads(path.read_text(encoding='utf8'))
            key = validate_entry(entry)
            if path.stem != key:
                raise ValueError('Entry name does not match its content fingerprint')
            entries.append(entry)
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejected.append(dict(path=str(path), reason=str(error)))
    return (entries, rejected)

def publish_entry(directory, entry):
    """Atomically publish a verified entry; readers never see partial JSON."""
    directory = Path(directory)
    key = validate_entry(entry)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (key + '.json')
    if target.exists():
        try:
            previous = json.loads(target.read_text(encoding='utf8'))
            validate_entry(previous)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        else:
            if previous['files'] != entry['files']:
                raise ValueError('Same fingerprint produced different raw evidence')
            return target
    descriptor, name = tempfile.mkstemp(prefix=key + '.', suffix='.tmp', dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf8') as stream:
            json.dump(entry, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target

@contextmanager
def persistent_evidence(history, directory, seed_entries=()):
    entries, rejected = load_entries(directory)
    by_key = {cache_key(entry): entry for entry in entries}
    for entry in seed_entries:
        key = validate_entry(entry)
        if key in by_key and by_key[key]['files'] != entry['files']:
            raise ValueError('Conflicting evidence for one fingerprint')
        by_key.setdefault(key, entry)
    with reuse_independent_evidence(history, list(by_key.values())) as (statistics, cache):
        statistics['rejected_persistent_entries'] = rejected
        yield (statistics, cache)
    for entry in cache:
        try:
            publish_entry(directory, entry)
        except OSError as error:
            statistics.setdefault('cache_write_failures', []).append(str(error))
